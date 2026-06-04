"""Path A1 - Prepare warmstart checkpoint for K=4 multi-organ training.

Strategy:
  V2 ckpt (ode_sam, single-organ liver) was trained on the SAME MultiScaleEncoder
  (TinyViT 21M + Stage-2 tap + skip_proj) that V3 (auto_ode_sam) uses.
  We carry encoder.* weights only:
    - V2 has non-organ-conditioned ODE (different shape from V3's organ-conditioned ODE)
    - V2 has prompt_encoder + mask_decoder (V3 has organ_query_decoder instead)
  All non-encoder modules in V3 stay at fresh-init.

Output:
  checkpoints/path_a1_big_solid/path_a1_big_solid_warmstart.pt

  Saved in CheckpointManager-compatible format:
    {
      "epoch": -1,                # so trainer's start_epoch becomes 0
      "model_state_dict": {...},  # full V3 state, encoder.* from V2, rest fresh
      "metrics": {},
      "monitor_metric": "val_dice",
      "warmstart_source": <path>,
      "warmstart_keys_loaded": <int>,
    }

  No optimizer / scheduler / scaler keys -- trainer's load() guards on them so
  the model loads cleanly via strict=True without restoring opt state.

Pre-flight gates:
  - V2 encoder key count >= 100 (sanity)
  - All V2 encoder.* keys must exist in V3 model and have matching shapes
  - Final model state_dict must round-trip through strict=True load
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf

from models import build_model

V2_CKPT = Path(
    r"C:\Users\Raywa\Desktop\VoluFormer3D\checkpoints"
    r"\phase3_odesam_v2_256px_liver"
    r"\phase3_odesam_v2_256px_liver_best.pt"
)
A1_CONFIG = ROOT / "configs" / "path_a1_big_solid.yaml"
BASE_CONFIG = ROOT / "configs" / "base.yaml"
OUT_DIR = ROOT / "checkpoints" / "path_a1_big_solid"
OUT_PATH = OUT_DIR / "path_a1_big_solid_warmstart.pt"


def load_a1_cfg():
    """Load A1 config merged with base.yaml (matches train.py behaviour)."""
    cfg = OmegaConf.load(str(BASE_CONFIG)) if BASE_CONFIG.exists() else OmegaConf.create({})
    a1 = OmegaConf.load(str(A1_CONFIG))
    if "defaults" in a1:
        a1 = OmegaConf.masked_copy(a1, [k for k in a1 if k != "defaults"])
    return OmegaConf.merge(cfg, a1)


def main() -> None:
    print("=" * 70)
    print("Path A1 -- Warmstart prep")
    print("=" * 70)

    # ---- 1. Build fresh V3 model from A1 config ----
    print(f"[1/6] loading A1 config from {A1_CONFIG.name}")
    if not A1_CONFIG.exists():
        raise FileNotFoundError(f"A1 config missing: {A1_CONFIG}")
    cfg = load_a1_cfg()
    print(f"      arch={cfg.model.architecture}  encoder={cfg.model.encoder_name}  "
          f"img_size={cfg.model.img_size}  embed_dim={cfg.model.embed_dim}")
    print(f"      target_organs={list(cfg.data.target_organs)}  "
          f"slices_per_volume={cfg.data.slices_per_volume}")

    print(f"[2/6] building V3 AutoODESAM (CPU init)")
    model = build_model(cfg).cpu()
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    n_keys = len(model.state_dict())
    print(f"      total params={n_params/1e6:.2f}M  state_dict keys={n_keys}")

    # ---- 2. Load V2 ckpt ----
    print(f"[3/6] loading V2 ckpt {V2_CKPT.name}")
    if not V2_CKPT.exists():
        raise FileNotFoundError(f"V2 ckpt missing: {V2_CKPT}")
    print(f"      size={V2_CKPT.stat().st_size/1e6:.0f} MB")
    v2_ckpt = torch.load(str(V2_CKPT), map_location="cpu", weights_only=False)
    v2_state = v2_ckpt.get("model_state_dict", v2_ckpt.get("state_dict", v2_ckpt))
    print(f"      V2 state_dict keys: {len(v2_state)}")
    print(f"      V2 epoch: {v2_ckpt.get('epoch')}")

    # ---- 3. Filter V2 -> encoder.* only ----
    print(f"[4/6] filtering V2 -> encoder.* keys")
    v2_encoder = {k: v for k, v in v2_state.items() if k.startswith("encoder.")}
    print(f"      V2 encoder.* keys: {len(v2_encoder)}")
    if len(v2_encoder) < 100:
        raise RuntimeError(
            f"V2 encoder key count too low ({len(v2_encoder)}); "
            f"check V2 ckpt structure"
        )

    # Verify shapes match V3's encoder
    v3_state = model.state_dict()
    mismatched = []
    missing_in_v3 = []
    for k, v in v2_encoder.items():
        if k not in v3_state:
            missing_in_v3.append(k)
        elif v3_state[k].shape != v.shape:
            mismatched.append((k, tuple(v.shape), tuple(v3_state[k].shape)))
    if missing_in_v3:
        print(f"      WARNING: {len(missing_in_v3)} V2 encoder keys NOT in V3 model")
        for k in missing_in_v3[:5]:
            print(f"        - {k}")
    if mismatched:
        print(f"      ERROR: {len(mismatched)} shape mismatches")
        for k, vs, ms in mismatched[:5]:
            print(f"        - {k}: V2={vs} V3={ms}")
        raise RuntimeError("Shape mismatch in encoder.* keys -- aborting")
    print(f"      shape audit OK: {len(v2_encoder)-len(missing_in_v3)} keys map cleanly")

    # ---- 4. Load filtered into V3 with strict=False ----
    print(f"[5/6] loading V2 encoder.* into V3 model (strict=False)")
    missing, unexpected = model.load_state_dict(v2_encoder, strict=False)
    n_loaded = len(v2_encoder) - len(unexpected)
    print(f"      loaded {n_loaded}/{len(v2_encoder)} encoder keys into V3")
    print(f"      missing in V3 model after load: {len(missing)} (non-encoder modules)")
    print(f"      unexpected (in v2_encoder but not V3): {len(unexpected)}")
    if unexpected:
        for k in unexpected[:5]:
            print(f"        unexpected: {k}")

    # Check that V3 state_dict has expected non-encoder keys (decoder, ode, etc.)
    decoder_keys = [k for k in v3_state if k.startswith("decoder.")]
    ode_keys = [k for k in v3_state if k.startswith("ode.")]
    print(f"      V3 has {len(decoder_keys)} decoder.* keys (fresh init)")
    print(f"      V3 has {len(ode_keys)} ode.* keys (fresh init -- organ-conditioned)")

    # ---- 5. Save warmstart in CheckpointManager-compatible format ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[6/6] saving warmstart to {OUT_PATH}")
    out_ckpt = {
        "epoch": -1,                      # trainer adds 1 -> start_epoch=0
        "model_state_dict": model.state_dict(),
        "metrics": {},
        "monitor_metric": "val_dice",
        "warmstart_source": str(V2_CKPT),
        "warmstart_v2_epoch": v2_ckpt.get("epoch"),
        "warmstart_keys_loaded": n_loaded,
    }
    torch.save(out_ckpt, str(OUT_PATH))
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"      saved {size_mb:.0f} MB")

    # ---- 6. Round-trip verify: reload with strict=True ----
    print()
    print("Round-trip verification (strict=True load):")
    verify_model = build_model(cfg).cpu()
    verify_ckpt = torch.load(str(OUT_PATH), map_location="cpu", weights_only=False)
    try:
        verify_model.load_state_dict(verify_ckpt["model_state_dict"])  # strict default True
        print("  PASS - strict=True load succeeded")
    except RuntimeError as e:
        print(f"  FAIL - strict=True load failed: {e}")
        raise

    # Verify encoder weights actually came from V2 (not fresh init)
    same_count = 0
    for k, v in v2_encoder.items():
        if torch.equal(verify_model.state_dict()[k], v):
            same_count += 1
    print(f"  encoder weight match: {same_count}/{len(v2_encoder)} keys equal V2")
    if same_count < len(v2_encoder):
        raise RuntimeError(
            f"Encoder warmstart incomplete: only {same_count}/{len(v2_encoder)} "
            f"keys match V2. Aborting."
        )

    print()
    print("=" * 70)
    print(f"PASS - warmstart ready at {OUT_PATH}")
    print(f"  Encoder ({n_loaded} keys) carries V2 fine-tuned weights.")
    print(f"  ODE + decoder fresh-init (V3 architecture is different).")
    print(f"  Trainer config resume_from -> this file.")
    print("=" * 70)


if __name__ == "__main__":
    main()
