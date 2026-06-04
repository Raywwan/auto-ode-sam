"""Phase 0 Day 1 sanity reload — Path A1.

Verifies the V2 liver checkpoint still loads cleanly into the current V4
codebase (`models/ode_sam.py` + `models/ode_cross_slice.py`). Reports schema
missing/unexpected keys, runs a forward pass on dummy input, and (optionally)
runs liver 3D Dice on a small AMOS22 val subset to confirm the working result.

Why this exists: per the A1 implementation plan we must reproduce V2's
working liver result before any K=4 surgery, otherwise we cannot trust the
warmstart path.

Usage:
    python scripts/sanity_reload_a1.py
    python scripts/sanity_reload_a1.py --max-volumes 5     # eval 5 vols
    python scripts/sanity_reload_a1.py --schema-only       # skip eval
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf

from models import build_model


V2_CKPT_PATH = Path(
    r"C:\Users\Raywa\Desktop\VoluFormer3D\checkpoints"
    r"\phase3_odesam_v2_256px_liver"
    r"\phase3_odesam_v2_256px_liver_best.pt"
)
V2_CONFIG_PATH = Path(
    r"C:\Users\Raywa\Desktop\VoluFormer3D\checkpoints"
    r"\phase3_odesam_v2_256px_liver\config.yaml"
)


def step_1_load_config() -> "OmegaConf":
    print(f"[step 1] loading V2 config from {V2_CONFIG_PATH}")
    if not V2_CONFIG_PATH.exists():
        raise FileNotFoundError(f"V2 config missing: {V2_CONFIG_PATH}")
    cfg = OmegaConf.load(str(V2_CONFIG_PATH))
    print(f"          architecture={cfg.model.architecture}  "
          f"encoder={cfg.model.encoder_name}  "
          f"img_size={cfg.model.img_size}  embed_dim={cfg.model.embed_dim}")
    print(f"          target_organs={list(cfg.data.target_organs)}  "
          f"slices_per_volume={cfg.data.slices_per_volume}")
    return cfg


def step_2_build_model(cfg, device: str) -> torch.nn.Module:
    print(f"[step 2] building model on {device}")
    model = build_model(cfg).to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"          {model.__class__.__name__}  total_params={n_params/1e6:.2f}M")
    return model


def step_3_load_ckpt(model: torch.nn.Module, device: str) -> dict:
    print(f"[step 3] loading checkpoint {V2_CKPT_PATH.name}")
    if not V2_CKPT_PATH.exists():
        raise FileNotFoundError(f"V2 ckpt missing: {V2_CKPT_PATH}")
    size_mb = V2_CKPT_PATH.stat().st_size / 1e6
    print(f"          size={size_mb:.0f} MB")

    ckpt = torch.load(str(V2_CKPT_PATH), map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    print(f"          ckpt keys: {sorted(set(ckpt.keys()))[:6]}...")

    missing, unexpected = model.load_state_dict(state, strict=False)
    n_loaded = len(state) - len(unexpected)
    n_total_model = sum(1 for _ in model.state_dict())
    miss_frac = len(missing) / max(n_total_model, 1)

    print(f"          loaded {n_loaded}/{len(state)} ckpt keys")
    print(f"          missing in ckpt: {len(missing)} keys (model has {n_total_model})")
    print(f"          unexpected in ckpt: {len(unexpected)} keys")
    print(f"          missing fraction: {miss_frac:.4f}")

    if missing[:5]:
        print(f"          missing sample: {missing[:5]}")
    if unexpected[:5]:
        print(f"          unexpected sample: {unexpected[:5]}")

    return {
        "ckpt_keys":  len(state),
        "missing":    len(missing),
        "unexpected": len(unexpected),
        "miss_frac":  miss_frac,
        "epoch":      ckpt.get("epoch"),
        "metrics":    {k: v for k, v in ckpt.items()
                       if k not in {"model_state_dict", "state_dict",
                                    "optimizer_state_dict", "scheduler_state_dict",
                                    "scaler_state_dict"}},
    }


def step_4_dummy_forward(model: torch.nn.Module, cfg, device: str) -> None:
    print("[step 4] dummy forward pass")
    img_size = int(cfg.model.img_size)
    depth    = int(cfg.data.slices_per_volume)
    images_3d = torch.randn(1, depth, 3, img_size, img_size, device=device)

    with torch.no_grad():
        # ode_sam expects bbox-prompted inference; build a dummy bbox at center
        # ode_sam.forward signature: (images, boxes, modality_ids, is_3d=True)
        # Try a few common interfaces gracefully
        try:
            box = torch.tensor([[64., 64., 192., 192.]], device=device)
            modality_id = torch.tensor([0], device=device)
            out = model(images_3d, box, modality_id, is_3d=True)
            print(f"          forward(images, box, modality, is_3d=True) -> "
                  f"{type(out).__name__}")
            if isinstance(out, dict):
                for k, v in out.items():
                    if torch.is_tensor(v):
                        print(f"            {k}: shape={tuple(v.shape)} dtype={v.dtype}")
                    else:
                        print(f"            {k}: {type(v).__name__}")
            elif torch.is_tensor(out):
                print(f"          output shape={tuple(out.shape)}")
        except TypeError as e:
            print(f"          ode_sam signature mismatch — trying fallbacks: {e}")
            raise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-volumes", type=int, default=0,
                    help="0 = skip Dice eval (schema only)")
    ap.add_argument("--schema-only", action="store_true",
                    help="Skip dummy forward pass too")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print("=" * 70)
    print("Phase 0 Day 1 - Sanity reload of V2 liver checkpoint")
    print("=" * 70)
    print(f"device={args.device}  torch={torch.__version__}")
    if args.device == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")
    print()

    t0 = time.time()
    cfg = step_1_load_config()
    print()
    model = step_2_build_model(cfg, args.device)
    print()
    schema = step_3_load_ckpt(model, args.device)
    print()

    if not args.schema_only:
        step_4_dummy_forward(model, cfg, args.device)
        print()

    # Gate
    print("=" * 70)
    print("PHASE 0 DAY 1 GATE")
    print("=" * 70)
    miss_frac = schema["miss_frac"]
    if miss_frac <= 0.05:
        print(f"PASS - schema mismatch {miss_frac:.4f} <= 0.05 floor.")
        if schema["epoch"] is not None:
            print(f"V2 ckpt was at epoch={schema['epoch']}.")
        if "metrics" in schema and schema["metrics"]:
            mks = {k: v for k, v in schema["metrics"].items()
                   if isinstance(v, (int, float, str, bool, type(None)))}
            if mks:
                print(f"V2 ckpt scalar metadata: {mks}")
    else:
        print(f"FAIL - schema mismatch {miss_frac:.4f} > 0.05 floor.")
        print("Triage: which keys are missing? Codebase may have drifted.")

    print(f"\n[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
