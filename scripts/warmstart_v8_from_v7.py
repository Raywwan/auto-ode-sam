"""Prepare a V8 warm-start checkpoint from a trained V7 checkpoint.

V8 adds modules (`text_encoder`, `dino`, `dino_fuse`) that V7 lacks. We:
  1. Build V8 fresh.
  2. Load V7's state_dict with strict=False — V8 keeps init for new modules.
  3. Save a *full V8* state_dict so the trainer's strict=True `.load` works.

Usage:
    python scripts/warmstart_v8_from_v7.py \
        --v7-ckpt checkpoints/v7_rescue_320/v7_rescue_320_best.pt \
        --v8-config configs/v8_mcp_killer.yaml \
        --out checkpoints/v8_mcp_killer/v8_warmstart.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf

from models import build_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v7-ckpt", required=True)
    ap.add_argument("--v8-config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = OmegaConf.load(args.v8_config)

    print(f"[warmstart] building V8 ({cfg.model.architecture}) on {device}")
    model = build_model(cfg).to(device)

    v7 = torch.load(args.v7_ckpt, map_location=device, weights_only=False)
    v7_state = v7.get("model_state_dict", v7.get("state_dict", v7))

    missing, unexpected = model.load_state_dict(v7_state, strict=False)
    print(f"[warmstart] loaded V7 weights into V8")
    print(f"[warmstart]   missing keys   (V8-new, stay at init): {len(missing)}")
    for k in missing[:8]:
        print(f"[warmstart]     - {k}")
    if len(missing) > 8:
        print(f"[warmstart]     ... +{len(missing)-8} more")
    print(f"[warmstart]   unexpected keys (V7-only, dropped):    {len(unexpected)}")
    for k in unexpected[:8]:
        print(f"[warmstart]     - {k}")

    # Trainer expects dict with "model_state_dict" + epoch + metrics. We set
    # epoch = 0 so training starts fresh but weights are V7's ep-N.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Omit optimizer/scheduler/scaler keys — trainer skips them when absent.
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": -1,                           # resume +1 → epoch 0 (fresh start)
        "metrics": {"warmstart_from": args.v7_ckpt, "v7_epoch": v7.get("epoch", -1)},
    }, str(out_path))
    print(f"[warmstart] saved V8 warm-start checkpoint to {out_path}")
    print(f"[warmstart] next: python train.py --config {args.v8_config} checkpoint.resume_from={out_path}")


if __name__ == "__main__":
    main()
