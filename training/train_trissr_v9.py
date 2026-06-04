"""Train TriSSR-V9 — the tri-scan SSM logit-refinement head — on top of a
frozen fine-tuned SwinUNETR proposer.

Stage 2.5 (novel, 2026-04-23): proposer + OrganFlow refiner remain frozen;
TriSSR learns to refine the proposer's 16-way softmax logits using
(proposer_logits ⊕ CT_image) as its input. Loss is Dice + BCE against the
volume-patch GT mask — same supervision as Stage 1.

Run:
    python training/train_trissr_v9.py --config configs/trissr_v9.yaml

Inputs:
    - --proposer-ckpt : the fine-tuned Stage 1 checkpoint (default = latest in
                        checkpoints/v9_stage1_ft/). TriSSR is trained with
                        this proposer's weights frozen.
    - --epochs        : number of epochs (default 8, ~4 min/epoch on a 4090)
    - --output-dir    : checkpoint dir for TriSSR (default checkpoints/trissr_v9)

Does NOT modify the proposer. Does NOT touch OrganFlow. Only emits a new
state dict containing TriSSR weights.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.amos22_v9 import AMOS22V9Dataset
from models.voluformer_v9 import VoluFormerV9
from models.trissr_v9 import TriSSRV9


def soft_dice_bce_loss(
    logits: torch.Tensor,          # (B, K+1, D, H, W)
    onehot: torch.Tensor,          # (B, K+1, D, H, W)
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Sum-of-squares Dice + BCE on 16-way logits. Same form as Stage 1's
    `stage1_loss` so TriSSR sees a familiar target landscape."""
    probs = F.softmax(logits, dim=1)
    # Dice excluding bg (channel 0)
    p = probs[:, 1:]
    g = onehot[:, 1:]
    dims = (0, 2, 3, 4)
    num = 2.0 * (p * g).sum(dim=dims)
    den = (p * p).sum(dim=dims) + (g * g).sum(dim=dims) + eps
    dice = (num / den).mean()
    dice_loss = 1.0 - dice

    bce = F.cross_entropy(logits, onehot.argmax(dim=1))

    total = dice_loss + bce
    return total, {"dice": float(dice.detach()), "bce": float(bce.detach())}


def build_onehot(mask_volume: torch.Tensor, K: int) -> torch.Tensor:
    """(B, K, D, H, W) binary masks -> (B, K+1, D, H, W) one-hot (bg prepended)."""
    fg_any = mask_volume.sum(dim=1, keepdim=True).clamp(0, 1)
    bg = 1.0 - fg_any
    return torch.cat([bg, mask_volume], dim=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg",           default="configs/v9_tierB.yaml",
                    help="base config — only data/model.proposer fields are read")
    ap.add_argument("--proposer-ckpt", default="checkpoints/v9_stage1_ft/last.pt")
    ap.add_argument("--output-dir",    default="checkpoints/trissr_v9")
    ap.add_argument("--epochs",        type=int,   default=8)
    ap.add_argument("--batch-size",    type=int,   default=2)
    ap.add_argument("--lr",            type=float, default=2.0e-4)
    ap.add_argument("--embed-dim",     type=int,   default=48)
    ap.add_argument("--n-blocks",      type=int,   default=3)
    ap.add_argument("--d-state",       type=int,   default=8)
    ap.add_argument("--num-workers",   type=int,   default=2)
    ap.add_argument("--resume", action="store_true",
                    help="Resume from latest epoch_NNN.pt in --output-dir if present")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(args.cfg)

    # ---- Data ----
    train_ds = AMOS22V9Dataset(
        data_root=cfg.data.data_root,
        split="train",
        img_size=cfg.data.img_size,
        depth=cfg.data.depth,
        volume_patch=cfg.data.volume_patch,
        modality="ct",
        hu_clip=tuple(cfg.data.hu_clip),
        pos_frac=cfg.data.pos_frac,
        neg_frac=cfg.data.neg_frac,
        mix_frac=cfg.data.mix_frac,
        slabs_per_volume=cfg.data.slabs_per_volume,
        seed=42,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    print(f"[trissr] train vols={len(train_ds.volume_ids)} samples={len(train_ds)}")

    # ---- Frozen proposer ----
    proposer_full = VoluFormerV9(cfg).to(device)
    sd = torch.load(args.proposer_ckpt, map_location=device, weights_only=False)
    state = sd.get("model", sd)
    miss, unex = proposer_full.load_state_dict(state, strict=False)
    print(f"[trissr] loaded {args.proposer_ckpt} miss={len(miss)} unex={len(unex)}")
    proposer = proposer_full.proposer
    for p in proposer.parameters():
        p.requires_grad = False
    proposer.eval()

    # ---- TriSSR head ----
    K = cfg.model.n_organs
    trissr = TriSSRV9(
        n_organs=K,
        embed_dim=args.embed_dim,
        n_blocks=args.n_blocks,
        d_state=args.d_state,
    ).to(device)
    n_params = sum(p.numel() for p in trissr.parameters() if p.requires_grad)
    print(f"[trissr] TriSSR params={n_params / 1e6:.2f}M "
          f"embed_dim={args.embed_dim} n_blocks={args.n_blocks}")

    opt = torch.optim.AdamW(trissr.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(ROOT / "logs" / "trissr_v9"))

    # ---- Optional resume ----
    start_epoch = 0
    if args.resume:
        ckpts = sorted(out_dir.glob("epoch_*.pt"))
        if ckpts:
            latest = ckpts[-1]
            sd = torch.load(str(latest), map_location=device, weights_only=False)
            trissr.load_state_dict(sd["model"])
            start_epoch = int(sd.get("epoch", 0))
            print(f"[trissr] resumed from {latest} (epoch {start_epoch})")
            for _ in range(start_epoch):
                sched.step()

    # ---- Train loop ----
    global_step = 0
    for epoch in range(start_epoch, args.epochs):
        trissr.train()
        t0 = time.time()
        loss_sum, n = 0.0, 0
        dice_sum, bce_sum = 0.0, 0.0
        for batch in train_loader:
            vol  = batch["volume"].to(device)          # (B, 1, P, P, P)
            mask = batch["mask_volume"].to(device)      # (B, K, P, P, P)

            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=True):
                    p_out = proposer(vol)
                # TriSSRV9 expects K+1 channels (bg + K organs) matching its
                # head/residual. Use full_logits (pre-softmax, K+1) not logits
                # (which is K organ-only channels).
                prop_logits = p_out["full_logits"].float() if "full_logits" in p_out else (
                    torch.log(p_out["full_probs"].clamp(min=1e-8)).float()
                )

            onehot = build_onehot(mask, K).float()
            with torch.amp.autocast("cuda", enabled=True):
                refined = trissr(prop_logits.to(device), vol)
                total, comp = soft_dice_bce_loss(refined, onehot)

            opt.zero_grad()
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trissr.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            loss_sum += float(total.detach()); n += 1
            dice_sum += comp["dice"];          bce_sum += comp["bce"]
            global_step += 1

        sched.step()
        avg, avg_dice, avg_bce = loss_sum / max(n, 1), dice_sum / max(n, 1), bce_sum / max(n, 1)
        dt = time.time() - t0
        print(f"[trissr] ep {epoch+1}/{args.epochs}  loss={avg:.4f}  "
              f"dice={avg_dice:.4f}  bce={avg_bce:.4f}  dt={dt/60:.1f}min")
        writer.add_scalar("train/loss", avg, epoch + 1)
        writer.add_scalar("train/dice", avg_dice, epoch + 1)
        writer.add_scalar("train/bce",  avg_bce,  epoch + 1)
        torch.save(
            {"model": trissr.state_dict(), "epoch": epoch + 1, "cfg": vars(args)},
            str(out_dir / f"epoch_{epoch+1:03d}.pt"),
        )

    torch.save(
        {"model": trissr.state_dict(), "epoch": args.epochs, "cfg": vars(args)},
        str(out_dir / "last.pt"),
    )
    writer.close()
    print(f"[trissr] done -> {out_dir/'last.pt'}")


if __name__ == "__main__":
    main()
