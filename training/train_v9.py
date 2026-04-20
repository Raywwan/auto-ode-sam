"""V9 training driver.

Stage-aware launcher for the V9 Tier-B cascade. The stage is read from the
config (`training.stage`). Each stage drives a different subset of the model:

  - Stage 1: SwinUNETR proposer alone on the 96^3 volume patch. Loss = CE + soft
    Dice on `full_logits` (K+1 classes).
  - Stage 2: refiner + fusion on a frozen proposer. Loss = V7-style 2D losses
    on the refiner masks + novel-loss contributions routed through V9Trainer.
  - Stage 3: joint fine-tune; same loss surface as Stage 2, all params trainable.

Usage (GPU box):
    python training/train_v9.py --config configs/v9_tierB.yaml
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.amos22_v9 import AMOS22V9Dataset
from models.voluformer_v9 import VoluFormerV9
from training.trainer_v9 import V9Trainer


# ----------------------------------------------------------------------
# Label helpers
# ----------------------------------------------------------------------

def masks_to_class_label(mask_onehot: torch.Tensor) -> torch.Tensor:
    """(B, K, D, H, W) binary -> (B, D, H, W) long in [0, K] with 0=bg."""
    B, K = mask_onehot.shape[:2]
    any_fg = mask_onehot.any(dim=1, keepdim=True).float()   # (B,1,D,H,W)
    bg = 1.0 - any_fg
    stacked = torch.cat([bg, mask_onehot.float()], dim=1)   # (B, K+1, D,H,W)
    return stacked.argmax(dim=1).long()                      # (B, D, H, W)


def soft_dice_loss(
    logits: torch.Tensor, target_onehot: torch.Tensor, eps: float = 1e-6,
) -> torch.Tensor:
    """logits: (B, K+1, ...), target_onehot: (B, K+1, ...) float in {0, 1}."""
    probs = F.softmax(logits, dim=1)
    dims = tuple(range(2, probs.dim()))
    inter = (probs * target_onehot).sum(dim=dims)
    denom = probs.pow(2).sum(dim=dims) + target_onehot.pow(2).sum(dim=dims)
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice[:, 1:].mean()  # exclude bg from the mean


# ----------------------------------------------------------------------
# Stage 1 proposer loss
# ----------------------------------------------------------------------

def stage1_loss(out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
    """CE + soft Dice on the 3D proposer output."""
    logits = out["proposer"]["full_logits"]          # (B, K+1, D, H, W)
    mask_vol = batch["mask_volume"]                   # (B, K, D, H, W)
    label = masks_to_class_label(mask_vol)            # (B, D, H, W)

    ce = F.cross_entropy(logits, label)

    K_plus = logits.shape[1]
    onehot = F.one_hot(label, K_plus).permute(0, 4, 1, 2, 3).float()
    dice = soft_dice_loss(logits, onehot)

    return {"ce": ce, "dice": dice, "total": ce + dice}


# ----------------------------------------------------------------------
# Train / val loops
# ----------------------------------------------------------------------

def _device(cfg) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _val_dice_3d(model: nn.Module, val_loader: DataLoader, device, max_batches: int = 20) -> float:
    """Cheap 3D dice estimate over a few val batches at patch-level. Used for
    stop-rule driving; final reporting uses `evaluation/eval_v9_3d.py`."""
    model.eval()
    dices = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            probs = model.proposer(batch["volume"])["probs"]   # (B, K, D, H, W)
            pred_bin = (probs > 0.5).float()
            gt = batch["mask_volume"]
            dims = (2, 3, 4)
            inter = (pred_bin * gt).sum(dim=dims)
            denom = pred_bin.sum(dim=dims) + gt.sum(dim=dims) + 1e-6
            dices.append((2 * inter / denom).mean().item())
    model.train()
    return float(sum(dices) / max(1, len(dices)))


def train(cfg) -> None:
    device = _device(cfg)
    print(f"[train_v9] device={device}  stage={cfg.training.stage}")

    torch.manual_seed(cfg.experiment.seed)

    # --- data ---
    train_ds = AMOS22V9Dataset(
        data_root=cfg.data.data_root,
        split="train",
        img_size=cfg.data.img_size,
        depth=cfg.data.depth,
        volume_patch=cfg.data.volume_patch,
        hu_clip=tuple(cfg.data.hu_clip),
        pos_frac=cfg.data.pos_frac,
        neg_frac=cfg.data.neg_frac,
        mix_frac=cfg.data.mix_frac,
        slabs_per_volume=cfg.data.slabs_per_volume,
        seed=cfg.experiment.seed,
    )
    val_ds = AMOS22V9Dataset(
        data_root=cfg.data.data_root,
        split="val",
        img_size=cfg.data.img_size,
        depth=cfg.data.depth,
        volume_patch=cfg.data.volume_patch,
        hu_clip=tuple(cfg.data.hu_clip),
        pos_frac=1.0, neg_frac=0.0, mix_frac=0.0,   # always POS for val signal
        slabs_per_volume=1,
        seed=cfg.experiment.seed + 1,
    )
    print(f"[train_v9] train vols={len(train_ds.volume_ids)}  val vols={len(val_ds.volume_ids)}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True,
        num_workers=cfg.training.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size, shuffle=False,
        num_workers=cfg.training.num_workers, pin_memory=(device.type == "cuda"),
    )

    # --- model ---
    model = VoluFormerV9(cfg).to(device)
    trainer = V9Trainer(model, cfg)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.training.optimizer.lr,
        weight_decay=cfg.training.optimizer.weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.training.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.training.amp and device.type == "cuda"))

    out_dir = ROOT / cfg.experiment.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train_v9] checkpoints -> {out_dir}")

    stage = int(cfg.training.stage)
    print(f"[train_v9] starting stage {stage}, epochs={cfg.training.epochs}")

    for epoch in range(cfg.training.epochs):
        model.train()
        t0 = time.time()
        loss_sum, n = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(cfg.training.amp and device.type == "cuda")):
                out = model(
                    {
                        "volume": batch["volume"],
                        "slab": batch["slab"],
                        "organ_id": batch["organ_id"],
                        "slab_center_z": batch["slab_center_z"],
                    },
                    stage=stage,
                )
                if stage == 1:
                    losses = stage1_loss(out, batch)
                    total = losses["total"]
                else:
                    # Base loss: refiner logits against mask_slab at center slice.
                    ref_logits = out["refiner"]["masks"]     # (B, K, H, W)
                    center = int(batch["slab_center_z"][0].item())
                    gt_slab = batch["mask_slab"][:, :, center]
                    base_bce = F.binary_cross_entropy_with_logits(ref_logits, gt_slab)
                    extras = trainer.step_losses(out, batch)
                    total = base_bce + sum(v for v in extras.values())
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            scaler.step(opt)
            scaler.update()
            loss_sum += float(total.item())
            n += 1
        sched.step()
        dt = time.time() - t0
        avg = loss_sum / max(1, n)
        print(f"[train_v9] ep {epoch+1}/{cfg.training.epochs}  train_loss={avg:.4f}  dt={dt/60:.1f}min")

        if (epoch + 1) % cfg.eval.every_n_epochs == 0:
            v = _val_dice_3d(model, val_loader, device)
            print(f"[train_v9]   val/dice_patch={v:.4f}")
            aborted = trainer.stop_rule_update(v)
            if aborted:
                print(f"[train_v9] STOP-RULE tripped at epoch {epoch+1}, val={v:.4f}. Aborting.")
                break
            trainer.save_ckpt(str(out_dir / f"epoch_{epoch+1:03d}.pt"), epoch + 1, v)

    trainer.save_ckpt(str(out_dir / "last.pt"), epoch + 1, None)
    print(f"[train_v9] done. saved {out_dir / 'last.pt'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v9_tierB.yaml")
    ap.add_argument("--stage", type=int, default=None,
                    help="override training.stage from config")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfg_dict = yaml.safe_load(open(ROOT / args.config))
    if args.stage is not None:
        cfg_dict["training"]["stage"] = args.stage
    if args.epochs is not None:
        cfg_dict["training"]["epochs"] = args.epochs
    cfg = OmegaConf.create(cfg_dict)
    train(cfg)


if __name__ == "__main__":
    main()
