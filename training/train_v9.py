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
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.amos22_v9 import AMOS22V9Dataset
from datasets.copy_paste_bank import CopyPasteBank, OrganCrop  # OrganCrop re-exported so pickle (__main__.OrganCrop in cached bank) can resolve when train_v9 is __main__.
from models.voluformer_v9 import VoluFormerV9
from training.trainer_v9 import V9Trainer
from training.losses_small_organ import SmallOrganLoss


ORGAN_NAMES = [
    "spleen", "right_kidney", "left_kidney", "gallbladder", "esophagus",
    "liver", "stomach", "aorta", "ivc", "pancreas",
    "right_adrenal", "left_adrenal", "duodenum", "bladder", "prostate_uterus",
]


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

def stage1_loss(
    out: Dict,
    batch: Dict,
    small_organ: Optional[nn.Module] = None,
    deep_sup_weight: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Stage 1 proposer loss.

    Default: CE + soft Dice on the 3D proposer output (original V9 baseline).
    When `small_organ` is provided, uses weighted Dice + Tversky + focal CE
    targeted at weak organs (see `training.losses_small_organ.SmallOrganLoss`).
    When `deep_sup_weight > 0` and proposer emits `aux_logits`, each aux head
    contributes CE at its native resolution (label downsampled with
    area-mode nearest). Aux weight is the caller's responsibility to decay.
    """
    logits = out["proposer"]["full_logits"]          # (B, K+1, D, H, W)
    mask_vol = batch["mask_volume"]                   # (B, K, D, H, W)
    label = masks_to_class_label(mask_vol)            # (B, D, H, W)

    if small_organ is not None:
        total = small_organ(logits, label)
        parts: Dict[str, torch.Tensor] = {"small_organ": total}
    else:
        ce = F.cross_entropy(logits, label)
        K_plus = logits.shape[1]
        onehot = F.one_hot(label, K_plus).permute(0, 4, 1, 2, 3).float()
        dice = soft_dice_loss(logits, onehot)
        total = ce + dice
        parts = {"ce": ce, "dice": dice}

    aux_logits = out["proposer"].get("aux_logits", {}) or {}
    if deep_sup_weight > 0.0 and aux_logits:
        aux_total = logits.new_zeros(())
        n_aux = 0
        for name, al in aux_logits.items():
            target_size = al.shape[-3:]
            lbl_aux = F.interpolate(
                label.float().unsqueeze(1), size=target_size, mode="nearest",
            ).squeeze(1).long()
            aux_total = aux_total + F.cross_entropy(al, lbl_aux)
            n_aux += 1
        aux_total = aux_total / max(n_aux, 1)
        parts["deep_sup"] = aux_total
        total = total + deep_sup_weight * aux_total

    parts["total"] = total
    return parts


# ----------------------------------------------------------------------
# Train / val loops
# ----------------------------------------------------------------------

def _device(cfg) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def refiner_base_loss(
    ref_logits: torch.Tensor,   # (B, K, H, W)
    gt_slab: torch.Tensor,      # (B, K, H, W)
    dice_w: float = 1.0,
    tversky_w: float = 1.0,
    focal_alpha: float = 0.7,
    focal_beta: float = 0.3,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """Stage 2 base loss: BCE + soft-Dice + Tversky, skipping empty-GT channels
    in the Dice/Tversky numerator (BCE already handles background).
    Returns per-term dict; caller sums."""
    bce = F.binary_cross_entropy_with_logits(ref_logits, gt_slab)

    prob = torch.sigmoid(ref_logits)
    dims = (2, 3)
    gt_sum = gt_slab.sum(dim=dims)
    present = (gt_sum > 0).float()                       # (B, K)

    inter = (prob * gt_slab).sum(dim=dims)
    p_sum = prob.sum(dim=dims)
    dice_per = (2 * inter + eps) / (p_sum + gt_sum + eps)
    dice_loss = (1.0 - dice_per) * present
    dice = dice_loss.sum() / present.sum().clamp_min(1.0)

    fp = (prob * (1.0 - gt_slab)).sum(dim=dims)
    fn = ((1.0 - prob) * gt_slab).sum(dim=dims)
    tv_per = (inter + eps) / (inter + focal_alpha * fn + focal_beta * fp + eps)
    tv_loss = (1.0 - tv_per) * present
    tv = tv_loss.sum() / present.sum().clamp_min(1.0)

    return {"bce": bce, "dice": dice_w * dice, "tversky": tversky_w * tv}


def _soft_dice_per_channel(pred_bin: torch.Tensor, gt: torch.Tensor, dims) -> torch.Tensor:
    """Per-channel Dice. Returns tensor (B, K). NaN where GT has no foreground
    (so caller can ignore empty organs when computing means)."""
    inter = (pred_bin * gt).sum(dim=dims)
    denom = pred_bin.sum(dim=dims) + gt.sum(dim=dims)
    dice = (2 * inter) / denom.clamp_min(1e-6)
    empty_gt = gt.sum(dim=dims) == 0
    dice = dice.masked_fill(empty_gt, float("nan"))
    return dice


def _val_metrics(
    model: nn.Module,
    val_loader: DataLoader,
    stage: int,
    device,
    max_batches: int = 20,
) -> Dict[str, float]:
    """Full evaluation — per-organ Dice for proposer (3D patch) and, for
    Stage 2/3, refiner + fused (2D slab center). Empty-GT channels are
    excluded from means. Returns flat dict suitable for TensorBoard logging.
    """
    model.eval()
    K = len(ORGAN_NAMES)
    # accumulators of present-only dice per organ.
    prop_sum = [0.0] * K; prop_n = [0] * K
    ref_sum  = [0.0] * K; ref_n  = [0] * K
    fus_sum  = [0.0] * K; fus_n  = [0] * K

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            if stage == 1:
                probs = model.proposer(batch["volume"])["probs"]    # (B, K, D, H, W)
                pred_bin = (probs > 0.5).float()
                gt = batch["mask_volume"]                           # (B, K, D, H, W)
                d = _soft_dice_per_channel(pred_bin, gt, dims=(2, 3, 4))  # (B, K)
                for k in range(K):
                    vals = d[:, k]
                    keep = ~torch.isnan(vals)
                    if keep.any():
                        prop_sum[k] += float(vals[keep].sum().item())
                        prop_n[k]   += int(keep.sum().item())
            else:
                out = model(
                    {
                        "volume": batch["volume"],
                        "slab": batch["slab"],
                        "organ_id": batch["organ_id"],
                        "slab_center_z": batch["slab_center_z"],
                    },
                    stage=stage,
                )
                # Proposer patch-level 3D (always available).
                probs = out["proposer"]["probs"]
                pred_bin = (probs > 0.5).float()
                d = _soft_dice_per_channel(pred_bin, batch["mask_volume"], dims=(2, 3, 4))
                for k in range(K):
                    keep = ~torch.isnan(d[:, k])
                    if keep.any():
                        prop_sum[k] += float(d[:, k][keep].sum().item())
                        prop_n[k]   += int(keep.sum().item())
                # Refiner 2D at slab centre.
                ref_logits = out["refiner"]["masks"]                # (B, K, H, W)
                ref_prob   = torch.sigmoid(ref_logits)
                ref_bin    = (ref_prob > 0.5).float()
                # NOTE: `slab_center_z` in the batch is the coordinate INTO the
                # proposer's volume patch (volume_patch // 2 = 48); for the 2D
                # slab mask we want the slab's own midplane, which is always
                # depth // 2 by construction.
                center     = batch["mask_slab"].shape[2] // 2
                gt2d       = batch["mask_slab"][:, :, center]       # (B, K, H, W)
                if gt2d.shape[-2:] != ref_bin.shape[-2:]:
                    gt2d_lr = F.interpolate(gt2d, size=ref_bin.shape[-2:], mode="nearest")
                else:
                    gt2d_lr = gt2d
                d_ref = _soft_dice_per_channel(ref_bin, gt2d_lr, dims=(2, 3))
                for k in range(K):
                    keep = ~torch.isnan(d_ref[:, k])
                    if keep.any():
                        ref_sum[k] += float(d_ref[:, k][keep].sum().item())
                        ref_n[k]   += int(keep.sum().item())
                # Fused (if available).
                fused = out.get("fused")
                if fused is not None and "fused" in fused:
                    fus = fused["fused"]                             # (B, K, H, W)
                    if fus.shape[-2:] != gt2d.shape[-2:]:
                        gt2d_fus = F.interpolate(gt2d, size=fus.shape[-2:], mode="nearest")
                    else:
                        gt2d_fus = gt2d
                    fus_bin = (fus > 0.5).float()
                    d_fus = _soft_dice_per_channel(fus_bin, gt2d_fus, dims=(2, 3))
                    for k in range(K):
                        keep = ~torch.isnan(d_fus[:, k])
                        if keep.any():
                            fus_sum[k] += float(d_fus[:, k][keep].sum().item())
                            fus_n[k]   += int(keep.sum().item())
    model.train()

    def _means(sums, ns):
        per = [(sums[k] / ns[k]) if ns[k] > 0 else float("nan") for k in range(K)]
        vals = [v for v in per if v == v]  # drop NaNs
        mean = float(sum(vals) / len(vals)) if vals else float("nan")
        return per, mean

    prop_per, prop_mean = _means(prop_sum, prop_n)
    metrics: Dict[str, float] = {"proposer_dice_mean": prop_mean}
    for k, name in enumerate(ORGAN_NAMES):
        metrics[f"proposer_dice/{name}"] = prop_per[k]
    if stage != 1:
        ref_per, ref_mean = _means(ref_sum, ref_n)
        fus_per, fus_mean = _means(fus_sum, fus_n)
        metrics["refiner_dice_mean"] = ref_mean
        metrics["fused_dice_mean"]   = fus_mean
        for k, name in enumerate(ORGAN_NAMES):
            metrics[f"refiner_dice/{name}"] = ref_per[k]
            metrics[f"fused_dice/{name}"]   = fus_per[k]
    return metrics


def train(cfg, train_ds=None, val_ds=None, train_sampler=None,
          loss_fn=None, warmstart_proposer_ckpt=None) -> None:
    device = _device(cfg)
    print(f"[train_v9] device={device}  stage={cfg.training.stage}")

    torch.manual_seed(cfg.experiment.seed)

    # --- data ---
    # Optional: copy-paste small-organ augmentation. Bank is built/loaded from
    # cache once (cfg.data.copy_paste_cache) and re-used every epoch.
    # Skipped when train_ds is injected (cross-dataset wrappers handle their own).
    cp_bank = None
    cp_prob = float(getattr(cfg.data, "copy_paste_prob", 0.0))
    if train_ds is None and cp_prob > 0.0:
        cache = Path(getattr(cfg.data, "copy_paste_cache",
                             str(Path(cfg.data.data_root) / "copy_paste_bank.pt")))
        if cache.exists():
            cp_bank = CopyPasteBank.build_from_preproc(
                preproc_dir=Path(cfg.data.data_root) / "preprocessed",
                volume_ids=[],  # unused when cache exists
                cache_path=cache,
            )
            print(f"[train_v9] copy-paste bank loaded (prob={cp_prob})")
        else:
            print(f"[train_v9] WARNING: copy_paste_cache {cache} missing; "
                  f"disabling copy-paste for this run. Build it with: "
                  f"python -m datasets.copy_paste_bank --cache {cache}")
            cp_prob = 0.0

    if train_ds is None:
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
            copy_paste_bank=cp_bank,
            copy_paste_prob=cp_prob,
        )
    if val_ds is None:
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

    if train_sampler is not None:
        train_loader = DataLoader(
            train_ds, batch_sampler=train_sampler,
            num_workers=cfg.training.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        print(f"[train_v9] using injected train_sampler "
              f"({type(train_sampler).__name__})")
    else:
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

    # Optional: warm-start proposer from a previous-stage checkpoint.
    # CLI override (warmstart_proposer_ckpt kwarg) wins over cfg.training.proposer_ckpt.
    # Stage 2 / 3 must load Stage 1 proposer weights before freezing.
    prop_ckpt = warmstart_proposer_ckpt or getattr(cfg.training, "proposer_ckpt", None)
    if prop_ckpt:
        ck_path = Path(prop_ckpt)
        if not ck_path.is_absolute():
            ck_path = ROOT / ck_path
        print(f"[train_v9] loading proposer warmstart from {ck_path}")
        sd = torch.load(str(ck_path), map_location="cpu", weights_only=False)
        full = sd.get("model", sd)
        keep = {k: v for k, v in full.items() if k.startswith("proposer.")}
        missing, unexpected = model.load_state_dict(keep, strict=False)
        prop_missing = [k for k in missing if k.startswith("proposer.")]
        print(
            f"[train_v9]   proposer keys: loaded={len(keep)} "
            f"missing_in_proposer={len(prop_missing)} unexpected={len(unexpected)}"
        )

    trainer = V9Trainer(model, cfg)

    # Optional: small-organ weighted loss for Stage 1. Behind
    # cfg.training.loss_small_organ flag; defaults off so the original CE+Dice
    # surface is preserved. Ablation path: flip this flag alone to measure lift.
    small_organ_crit = None
    if int(cfg.training.stage) == 1 and bool(
        getattr(cfg.training, "loss_small_organ", False)
    ):
        small_cfg = getattr(cfg.training, "loss_small_organ_cfg", {}) or {}
        small_organ_crit = SmallOrganLoss(
            n_classes=cfg.model.n_organs + 1,
            dice_weight=float(small_cfg.get("dice_weight", 1.0)),
            tversky_weight=float(small_cfg.get("tversky_weight", 1.0)),
            focal_weight=float(small_cfg.get("focal_weight", 1.0)),
            tversky_alpha=float(small_cfg.get("tversky_alpha", 0.3)),
            tversky_beta=float(small_cfg.get("tversky_beta", 0.7)),
            focal_gamma=float(small_cfg.get("focal_gamma", 1.5)),
        ).to(device)
        print(f"[train_v9] Stage 1 small-organ loss ENABLED {small_cfg}")

    # Deep-supervision aux-loss schedule. Linear decay from start->end over
    # training epochs, disabled at 0.0. Only active in Stage 1.
    ds_start = float(getattr(cfg.training, "deep_sup_weight_start", 0.0))
    ds_end = float(getattr(cfg.training, "deep_sup_weight_end", 0.0))
    if int(cfg.training.stage) == 1 and (ds_start > 0.0 or ds_end > 0.0):
        print(f"[train_v9] Stage 1 deep-supervision ENABLED (schedule {ds_start:.2f} -> {ds_end:.2f})")
        # Aux heads are created lazily on first forward — trigger one dry pass
        # so they exist BEFORE the optimizer is built (otherwise they'd never
        # collect grads). Done in eval mode + no_grad to avoid any side effects.
        if int(getattr(model.proposer, "deep_supervision", False)):
            P = int(cfg.model.proposer.get("patch_size", 96))
            with torch.no_grad():
                model.eval()
                dummy = torch.zeros(1, 1, P, P, P, device=device)
                _ = model.proposer(dummy)
                model.train()
            n_aux = sum(p.numel() for p in model.proposer.aux_heads.parameters())
            print(f"[train_v9] aux_heads materialised: {n_aux/1e3:.1f}k params across {len(model.proposer.aux_heads)} taps")

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.training.optimizer.lr,
        weight_decay=cfg.training.optimizer.weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.training.epochs)
    amp_dtype_str = str(getattr(cfg.training, "amp_dtype", "fp16")).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_str in ("bf16", "bfloat16") else torch.float16
    use_scaler = (cfg.training.amp and device.type == "cuda" and amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    print(f"[train_v9] AMP dtype={amp_dtype_str} grad_scaler={use_scaler}")

    out_dir = ROOT / cfg.experiment.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train_v9] checkpoints -> {out_dir}")

    stage = int(cfg.training.stage)
    tb_dir = ROOT / cfg.experiment.log_dir / f"stage{stage}"
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))
    print(f"[train_v9] tensorboard -> {tb_dir}")
    print(f"[train_v9] starting stage {stage}, epochs={cfg.training.epochs}")

    global_step = 0
    for epoch in range(cfg.training.epochs):
        model.train()
        t0 = time.time()
        loss_sum, n = 0.0, 0
        comp_sums: Dict[str, float] = {}
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            opt.zero_grad()
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(cfg.training.amp and device.type == "cuda")):
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
                    if cfg.training.epochs > 1:
                        frac = epoch / max(1, cfg.training.epochs - 1)
                    else:
                        frac = 0.0
                    ds_w = ds_start + (ds_end - ds_start) * frac
                    if loss_fn is not None:
                        seg_logits = out["proposer"]["full_logits"]
                        seg_target = masks_to_class_label(batch["mask_volume"])
                        presence_logits = getattr(model.proposer, "last_presence_logits", None)
                        gate_weights = (
                            model.proposer.collect_last_gates()
                            if hasattr(model.proposer, "collect_last_gates") else None
                        )
                        losses = loss_fn(
                            seg_logits=seg_logits,
                            seg_target=seg_target,
                            presence_logits=presence_logits,
                            gate_weights=gate_weights,
                        )
                    else:
                        losses = stage1_loss(
                            out, batch,
                            small_organ=small_organ_crit,
                            deep_sup_weight=ds_w,
                        )
                    total = losses["total"]
                    for k, v in losses.items():
                        if k == "total":
                            continue
                        comp_sums[k] = comp_sums.get(k, 0.0) + float(v.item())
                else:
                    # Base loss: BCE + Dice + Tversky on refiner at slab center.
                    ref_logits = out["refiner"]["masks"]     # (B, K, H, W)
                    # slab midplane lives at depth//2 (slab-local), not at
                    # slab_center_z (which is a volume-patch coordinate).
                    center = batch["mask_slab"].shape[2] // 2
                    gt_slab = batch["mask_slab"][:, :, center]
                    base = refiner_base_loss(
                        ref_logits, gt_slab,
                        dice_w=float(getattr(cfg.loss, "dice_weight", 1.0)),
                        tversky_w=float(getattr(cfg.loss, "tversky_weight", 1.0)),
                    )
                    extras = trainer.step_losses(out, batch)
                    total = base["bce"] + base["dice"] + base["tversky"] + sum(v for v in extras.values())
                    for k, v in base.items():
                        comp_sums[k] = comp_sums.get(k, 0.0) + float(v.item())
                    for k, v in extras.items():
                        comp_sums[k] = comp_sums.get(k, 0.0) + float(v.item())
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            scaler.step(opt)
            scaler.update()
            loss_sum += float(total.item())
            n += 1
            writer.add_scalar("train_step/loss", float(total.item()), global_step)
            global_step += 1
        sched.step()
        dt = time.time() - t0
        avg = loss_sum / max(1, n)
        current_lr = opt.param_groups[0]["lr"]
        writer.add_scalar("train/loss", avg, epoch + 1)
        writer.add_scalar("train/lr", current_lr, epoch + 1)
        writer.add_scalar("train/epoch_minutes", dt / 60.0, epoch + 1)
        for ck, cv in comp_sums.items():
            writer.add_scalar(f"train/component_{ck}", cv / max(1, n), epoch + 1)
        print(f"[train_v9] ep {epoch+1}/{cfg.training.epochs}  train_loss={avg:.4f}  dt={dt/60:.1f}min")

        # ---- Full eval every epoch ----
        metrics = _val_metrics(model, val_loader, stage, device)
        for key, val in metrics.items():
            if val == val:  # skip NaN
                writer.add_scalar(f"val/{key}", val, epoch + 1)
        # Human-readable summary line.
        parts = [f"prop={metrics['proposer_dice_mean']:.4f}"]
        if stage != 1:
            parts.append(f"ref={metrics['refiner_dice_mean']:.4f}")
            parts.append(f"fus={metrics['fused_dice_mean']:.4f}")
        print(f"[train_v9]   val  {'  '.join(parts)}")
        # Per-organ Dice — full breakdown so weak-organ effects are visible
        # in the log without opening TensorBoard.
        per_organ_line = "  ".join(
            f"{name}={metrics[f'proposer_dice/{name}']:.3f}"
            for name in ORGAN_NAMES
            if metrics.get(f'proposer_dice/{name}', float('nan')) == metrics.get(f'proposer_dice/{name}', float('nan'))
        )
        print(f"[train_v9]   val  per-organ  {per_organ_line}")
        # Dump full metrics dict to JSON (one file per epoch, plus a rolling
        # `metrics.json` list). Survives crashes; allows offline analysis
        # without tensorboard event files.
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": float(avg),
            "lr": float(current_lr),
            "epoch_minutes": float(dt / 60.0),
            **{k: (None if v != v else float(v)) for k, v in metrics.items()},
        }
        metrics_dir = out_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        with open(metrics_dir / f"epoch_{epoch+1:03d}.json", "w") as fh:
            json.dump(epoch_metrics, fh, indent=2)
        rolling_path = out_dir / "metrics.json"
        rolling = []
        if rolling_path.exists():
            try:
                rolling = json.load(open(rolling_path))
            except Exception:
                rolling = []
        rolling.append(epoch_metrics)
        with open(rolling_path, "w") as fh:
            json.dump(rolling, fh, indent=2)

        # Stop-rule uses the best available signal per stage.
        driver = (
            metrics["proposer_dice_mean"] if stage == 1
            else (metrics.get("fused_dice_mean") or metrics.get("refiner_dice_mean"))
        )
        if driver == driver:  # not NaN
            aborted = trainer.stop_rule_update(float(driver))
            if aborted:
                print(f"[train_v9] STOP-RULE tripped at epoch {epoch+1}, val={driver:.4f}. Aborting.")
                trainer.save_ckpt(str(out_dir / f"epoch_{epoch+1:03d}.pt"), epoch + 1, float(driver))
                break
        if (epoch + 1) % cfg.eval.every_n_epochs == 0:
            trainer.save_ckpt(
                str(out_dir / f"epoch_{epoch+1:03d}.pt"),
                epoch + 1,
                float(driver) if driver == driver else None,
            )

    writer.close()
    trainer.save_ckpt(str(out_dir / "last.pt"), epoch + 1, None)
    print(f"[train_v9] done. saved {out_dir / 'last.pt'}")


# Public alias used by OrganMoE pretrain/FT entrypoints. Keeps a stable name
# even if the underlying function is later split into stage1/stage2 paths.
train_stage1 = train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v9_tierB.yaml")
    ap.add_argument("--stage", type=int, default=None,
                    help="override training.stage from config")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--proposer_ckpt", type=str, default=None,
                    help="warm-start proposer from this checkpoint (Stage 2/3)")
    ap.add_argument("--output_dir", type=str, default=None,
                    help="override experiment.output_dir (per-stage ckpt dir)")
    args = ap.parse_args()

    cfg_dict = yaml.safe_load(open(ROOT / args.config))
    if args.stage is not None:
        cfg_dict["training"]["stage"] = args.stage
    if args.epochs is not None:
        cfg_dict["training"]["epochs"] = args.epochs
    if args.proposer_ckpt is not None:
        cfg_dict["training"]["proposer_ckpt"] = args.proposer_ckpt
    if args.output_dir is not None:
        cfg_dict["experiment"]["output_dir"] = args.output_dir
    cfg = OmegaConf.create(cfg_dict)
    train(cfg)


if __name__ == "__main__":
    main()
