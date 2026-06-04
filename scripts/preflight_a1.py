"""Path A1 -- Pre-flight checks before launching the 14-epoch run.

Per feedback_preflight_long_runs.md, every >30 min run must pass:
  1. Schema check       -- model builds + warmstart loads strict=True
  2. Label-map audit    -- AMOS22 IDs {1,2,3,6} resolve to expected names
  3. Config sanity      -- no obvious typos, units, etc.
  4. Dataset dry-run    -- one (image, mask, organ_id) sample loads correctly
  5. Forward+backward 1 step -- loss is finite, grads non-zero on key params
  6. First-epoch sentinel target -- compute expected loss range / time

ASCII-only output (Windows cp1252).
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf

from datasets import get_dataset
from datasets.amos22 import AMOS22_ORGAN_NAMES
from models import build_model

A1_CONFIG = ROOT / "configs" / "path_a1_big_solid.yaml"
BASE_CONFIG = ROOT / "configs" / "base.yaml"
WARMSTART = ROOT / "checkpoints" / "path_a1_big_solid" / "path_a1_big_solid_warmstart.pt"

EXPECTED_ORGAN_NAMES = {
    1: "spleen",
    2: "right_kidney",
    3: "left_kidney",
    6: "liver",
}


def load_a1_cfg():
    cfg = OmegaConf.load(str(BASE_CONFIG)) if BASE_CONFIG.exists() else OmegaConf.create({})
    a1 = OmegaConf.load(str(A1_CONFIG))
    if "defaults" in a1:
        a1 = OmegaConf.masked_copy(a1, [k for k in a1 if k != "defaults"])
    return OmegaConf.merge(cfg, a1)


def check_1_schema(cfg, device):
    print("[1/6] Schema check (build + warmstart strict load)")
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      model built: {n_params/1e6:.2f}M params")
    if not WARMSTART.exists():
        raise FileNotFoundError(f"warmstart missing: {WARMSTART}")
    ckpt = torch.load(str(WARMSTART), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"      strict=True load OK from {WARMSTART.name}")
    print(f"      warmstart V2 epoch={ckpt.get('warmstart_v2_epoch')}  "
          f"keys_loaded={ckpt.get('warmstart_keys_loaded')}")
    return model


def check_2_labelmap(cfg):
    print("[2/6] Label-map audit (AMOS22 IDs map to expected organs)")
    target = list(cfg.data.target_organs)
    if set(target) != set(EXPECTED_ORGAN_NAMES.keys()):
        raise RuntimeError(
            f"target_organs {target} != expected {sorted(EXPECTED_ORGAN_NAMES)}"
        )
    for oid, expected in EXPECTED_ORGAN_NAMES.items():
        actual = AMOS22_ORGAN_NAMES.get(oid)
        if actual != expected:
            raise RuntimeError(
                f"AMOS22 ID {oid}: expected '{expected}' got '{actual}' -- "
                f"check datasets/amos22.py AMOS22_ORGAN_NAMES dict"
            )
        print(f"      {oid} -> {actual} OK")


def check_3_config_sanity(cfg):
    print("[3/6] Config sanity")
    asserts = [
        (cfg.model.architecture == "auto_ode_sam", "architecture must be auto_ode_sam"),
        (cfg.model.img_size == 256, "img_size must be 256 for V2 encoder warmstart"),
        (cfg.data.dataset == "amos22_3d", "dataset must be amos22_3d"),
        (cfg.data.modality == "ct", "modality must be ct (V2 trained on CT)"),
        (cfg.training.epochs == 14, "epochs must be 14 per A1 plan"),
        (cfg.training.batch_size == 2, "batch_size must be 2 per A1 plan"),
        (cfg.training.warmup_epochs == 2, "warmup_epochs must be 2"),
        (cfg.training.boundary_ramp_epoch < cfg.training.epochs,
         "boundary_ramp_epoch must trigger before training ends"),
        (cfg.training.mixed_precision is True, "mixed_precision must be on"),
        (cfg.training.cutmix_prob == 0.0,
         "cutmix_prob must be 0 for AutoODESAM (would corrupt per-organ GT)"),
        (cfg.checkpoint.monitor_metric == "val_dice",
         "monitor_metric must be val_dice (trainer hard-codes this key)"),
        (cfg.validation.val_every_n_epochs <= 2,
         "val_every_n_epochs should be <=2 for 14-ep run"),
    ]
    for ok, msg in asserts:
        if not ok:
            raise RuntimeError(f"config sanity FAIL: {msg}")
    print(f"      all {len(asserts)} config assertions PASS")


def check_4_dataset_dryrun(cfg):
    print("[4/6] Dataset dry-run (load 2 samples)")
    train_ds = get_dataset(cfg, split="train", copypaste_prob=0.0)
    val_ds = get_dataset(cfg, split="val", copypaste_prob=0.0)
    print(f"      train samples={len(train_ds)}  val samples={len(val_ds)}")
    if len(train_ds) < 100:
        raise RuntimeError(f"train_ds too small: {len(train_ds)}")
    if len(val_ds) < 20:
        raise RuntimeError(f"val_ds too small: {len(val_ds)}")

    # Sample 0
    s0 = train_ds[0]
    expected_keys = {"image", "mask", "organ_id"}
    have = set(s0.keys())
    if not expected_keys.issubset(have):
        raise RuntimeError(f"sample missing keys: have={have} need={expected_keys}")
    img = s0["image"]
    mask = s0["mask"]
    organ_id = s0["organ_id"]
    print(f"      sample[0]: image={tuple(img.shape)} dtype={img.dtype}  "
          f"mask={tuple(mask.shape)} dtype={mask.dtype}  organ_id={int(organ_id)}")

    # Verify image shape: should be (D, 3, H, W) per AMOS22_3D_Dataset
    if img.dim() != 4 or img.shape[1] != 3:
        raise RuntimeError(f"image shape unexpected: {tuple(img.shape)}")
    D, C, H, W = img.shape
    if D != cfg.data.slices_per_volume:
        raise RuntimeError(f"D={D} != cfg.slices_per_volume={cfg.data.slices_per_volume}")
    if H != cfg.model.img_size or W != cfg.model.img_size:
        raise RuntimeError(f"H,W={H},{W} != img_size={cfg.model.img_size}")

    # Verify organ_id distribution across target_organs
    seen_organs = set()
    n_check = min(40, len(train_ds))
    for i in range(0, n_check, max(1, n_check // 20)):
        s = train_ds[i]
        oid = int(s["organ_id"]) if torch.is_tensor(s["organ_id"]) else int(s["organ_id"])
        seen_organs.add(oid)
    target = set(int(x) for x in cfg.data.target_organs)
    if not seen_organs.issubset(target):
        raise RuntimeError(f"sampler emitted organ {seen_organs - target} not in target")
    print(f"      organ_id sampling check: seen={sorted(seen_organs)} subset of {sorted(target)} OK")
    return train_ds, val_ds


def check_5_forward_backward(cfg, model, train_ds, device):
    print("[5/6] Forward + backward 1 step (B=2)")
    from torch.utils.data import DataLoader
    loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
    batch = next(iter(loader))

    images = batch["image"].to(device)            # (B, D, 3, H, W)
    masks = batch["mask"].to(device)              # (B, H, W) center-slice
    organ_id = batch["organ_id"]
    if organ_id.dim() > 1:
        organ_id = organ_id[:, organ_id.shape[1] // 2]
    organ_id = organ_id.to(device)
    print(f"      batch shapes: images={tuple(images.shape)}  "
          f"mask={tuple(masks.shape)}  organ_id={organ_id.tolist()}")

    model.train()
    out = model(images, organ_id, is_3d=True)
    pm = out["masks"]
    iou = out["iou_pred"]
    print(f"      forward OK: masks={tuple(pm.shape)}  iou_pred={tuple(iou.shape)}")

    # Per-item gather (mirror trainer's auto_ode path)
    organ_idxs = (organ_id - 1).long()
    H_out, W_out = pm.shape[-2:]
    best = pm.gather(1, organ_idxs.view(-1, 1, 1, 1).expand(-1, 1, H_out, W_out)).squeeze(1)

    # Resize mask to decoder output
    import torch.nn.functional as F
    masks_rs = F.interpolate(masks.unsqueeze(1).float(),
                             size=best.shape[-2:], mode="nearest").squeeze(1)

    # Simple BCE+Dice loss for the smoke step
    bce = F.binary_cross_entropy_with_logits(best, masks_rs)
    pred_sig = torch.sigmoid(best)
    inter = (pred_sig * masks_rs).sum(dim=[1, 2])
    dice = 1 - (2 * inter + 1e-5) / (pred_sig.sum([1, 2]) + masks_rs.sum([1, 2]) + 1e-5)
    loss = bce + dice.mean()
    print(f"      smoke loss (BCE+Dice): {loss.item():.4f}  "
          f"(finite={torch.isfinite(loss).item()})")
    if not torch.isfinite(loss):
        raise RuntimeError("Smoke loss is non-finite!")

    loss.backward()

    # Verify grads on key params
    enc_grad = sum(p.grad.norm().item() for n, p in model.named_parameters()
                   if p.grad is not None and n.startswith("encoder."))
    ode_grad = sum(p.grad.norm().item() for n, p in model.named_parameters()
                   if p.grad is not None and n.startswith("ode."))
    dec_grad = sum(p.grad.norm().item() for n, p in model.named_parameters()
                   if p.grad is not None and n.startswith("decoder."))
    print(f"      grad norms: encoder={enc_grad:.4f}  ode={ode_grad:.4f}  decoder={dec_grad:.4f}")
    if enc_grad == 0 or ode_grad == 0 or dec_grad == 0:
        raise RuntimeError("Some component has zero gradient -- training would not learn it")


def check_6_sentinel(cfg, train_ds):
    print("[6/6] First-epoch sentinel estimate")
    eff_batch = cfg.training.batch_size * cfg.training.grad_accumulation_steps
    n_train = len(train_ds)
    n_steps_per_epoch = n_train // cfg.training.batch_size
    print(f"      train samples={n_train}  batch={cfg.training.batch_size}  "
          f"accum={cfg.training.grad_accumulation_steps}  effective={eff_batch}")
    print(f"      steps/epoch={n_steps_per_epoch}  total over 14 ep={n_steps_per_epoch * 14}")
    print(f"      val_every_n_epochs={cfg.validation.val_every_n_epochs}  "
          f"save_every={cfg.checkpoint.save_every_n_epochs}")
    print(f"      sentinel rule: abort if loss > 10x ep1-mean OR loss==NaN")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print("=" * 70)
    print("Path A1 -- Pre-flight (per feedback_preflight_long_runs.md)")
    print("=" * 70)
    print(f"device={args.device}  torch={torch.__version__}")
    if args.device == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")
    print()

    cfg = load_a1_cfg()
    t0 = time.time()
    n_pass = 0
    try:
        model = check_1_schema(cfg, args.device);                   n_pass += 1
        check_2_labelmap(cfg);                                      n_pass += 1
        check_3_config_sanity(cfg);                                 n_pass += 1
        train_ds, val_ds = check_4_dataset_dryrun(cfg);             n_pass += 1
        check_5_forward_backward(cfg, model, train_ds, args.device); n_pass += 1
        check_6_sentinel(cfg, train_ds);                            n_pass += 1
    except Exception as e:
        print()
        print("=" * 70)
        print(f"FAIL at check {n_pass+1}/6: {type(e).__name__}: {e}")
        print("-" * 70)
        traceback.print_exc()
        print("=" * 70)
        sys.exit(1)

    print()
    print("=" * 70)
    print(f"PRE-FLIGHT PASS - all 6/6 gates clean in {time.time()-t0:.1f}s")
    print("=" * 70)
    print()
    print("Next: launch training with")
    print("  python train.py --config configs/path_a1_big_solid.yaml")


if __name__ == "__main__":
    main()
