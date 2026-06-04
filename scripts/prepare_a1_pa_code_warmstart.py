"""PA-CODE -- Prepare warmstart from the failed A1 run's best ckpt.

Source: checkpoints/path_a1_big_solid/path_a1_big_solid_epoch001.pt
  (empirical best per 2026-04-30 smoke 3D eval: mean DSC 0.24, liver 0.787 --
   the only A1 ckpt where V2 liver knowledge is mostly intact.)

Filter: KEEP encoder.*, decoder.*, deep_sup_head.*, dense_pe
        DROP ode.*  (PA-CODE has different ODE-module keys: gamma_head,
                     beta_head, position-aware net. Fresh init is the
                     identity-init guarantee from the design doc.)

Output: checkpoints/path_a1_pa_code/path_a1_pa_code_warmstart.pt
  (CheckpointManager-compatible: epoch=-1 so trainer's start_epoch=0)

Round-trip verify: strict=True load into a fresh PA-CODE model must succeed.

ASCII-only output (Windows cp1252).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf

from models import build_model

A1_BEST_CKPT = (
    ROOT / "checkpoints" / "path_a1_big_solid" / "path_a1_big_solid_epoch001.pt"
)
PA_CODE_CONFIG = ROOT / "configs" / "path_a1_pa_code.yaml"
BASE_CONFIG    = ROOT / "configs" / "base.yaml"
OUT_DIR        = ROOT / "checkpoints" / "path_a1_pa_code"
OUT_PATH       = OUT_DIR / "path_a1_pa_code_warmstart.pt"

# Prefixes we want to carry from the source ckpt.
KEEP_PREFIXES = ("encoder.", "decoder.", "deep_sup_head.", "dense_pe")
# Prefix we explicitly drop (incompatible -- PA-CODE ode keys differ).
DROP_PREFIXES = ("ode.",)
# Specific keys to drop even when their prefix is in KEEP_PREFIXES.
# decoder.organ_queries: V2/A1 used std=0.02 init (L2 ~0.32); diag ep_013 showed
# queries never escaped init magnitude (audit Concern 5). PA-CODE uses
# dim**-0.5 init (~0.0625) for stronger gradient signal — re-init from fresh.
# decoder.hq_token: trivially small, also re-init for consistency.
DROP_EXACT_KEYS = ("decoder.organ_queries.weight", "decoder.hq_token.weight")


def load_pa_code_cfg():
    cfg = OmegaConf.load(str(BASE_CONFIG)) if BASE_CONFIG.exists() else OmegaConf.create({})
    pa = OmegaConf.load(str(PA_CODE_CONFIG))
    if "defaults" in pa:
        pa = OmegaConf.masked_copy(pa, [k for k in pa if k != "defaults"])
    return OmegaConf.merge(cfg, pa)


def keep_key(k: str) -> bool:
    if any(k.startswith(p) for p in DROP_PREFIXES):
        return False
    if k in DROP_EXACT_KEYS:
        return False
    return any(k.startswith(p) for p in KEEP_PREFIXES) or (k == "dense_pe")


def main() -> None:
    print("=" * 70)
    print("PA-CODE -- Warmstart prep")
    print("=" * 70)

    print(f"[1/6] loading PA-CODE config from {PA_CODE_CONFIG.name}")
    if not PA_CODE_CONFIG.exists():
        raise FileNotFoundError(f"PA-CODE config missing: {PA_CODE_CONFIG}")
    cfg = load_pa_code_cfg()
    use_pa = bool(getattr(cfg.model.ode, "use_pa_code", False))
    if not use_pa:
        raise RuntimeError("cfg.model.ode.use_pa_code must be true for PA-CODE warmstart")
    print(f"      arch={cfg.model.architecture}  use_pa_code={use_pa}  "
          f"img_size={cfg.model.img_size}  embed_dim={cfg.model.embed_dim}")

    print(f"[2/6] building fresh PA-CODE model (CPU init)")
    model = build_model(cfg).cpu()
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    target_state = model.state_dict()
    n_keys_target = len(target_state)
    print(f"      total params={n_params/1e6:.2f}M  state_dict keys={n_keys_target}")

    print(f"[3/6] loading source ckpt {A1_BEST_CKPT.name}")
    if not A1_BEST_CKPT.exists():
        raise FileNotFoundError(f"source ckpt missing: {A1_BEST_CKPT}")
    print(f"      size={A1_BEST_CKPT.stat().st_size/1e6:.0f} MB")
    src_ckpt  = torch.load(str(A1_BEST_CKPT), map_location="cpu", weights_only=False)
    src_state = src_ckpt.get("model_state_dict", src_ckpt.get("state_dict", src_ckpt))
    print(f"      source state_dict keys: {len(src_state)}  "
          f"epoch={src_ckpt.get('epoch')}")

    print(f"[4/6] filter src -> keep {KEEP_PREFIXES} drop {DROP_PREFIXES}")
    kept = {k: v for k, v in src_state.items() if keep_key(k)}
    dropped = [k for k in src_state if not keep_key(k)]
    print(f"      kept {len(kept)} keys, dropped {len(dropped)}")
    drop_by_prefix = {}
    for k in dropped:
        head = k.split(".", 1)[0] + "."
        drop_by_prefix[head] = drop_by_prefix.get(head, 0) + 1
    for head, n in sorted(drop_by_prefix.items()):
        print(f"        dropped: {head}* ({n} keys)")

    # Shape audit against the new PA-CODE model
    mismatched = []
    not_in_target = []
    for k, v in kept.items():
        if k not in target_state:
            not_in_target.append(k)
        elif target_state[k].shape != v.shape:
            mismatched.append((k, tuple(v.shape), tuple(target_state[k].shape)))
    if not_in_target:
        print(f"      WARNING: {len(not_in_target)} kept keys NOT in PA-CODE model:")
        for k in not_in_target[:5]:
            print(f"        - {k}")
    if mismatched:
        print(f"      ERROR: {len(mismatched)} shape mismatches")
        for k, vs, ms in mismatched[:5]:
            print(f"        - {k}: src={vs} target={ms}")
        raise RuntimeError("Shape mismatch -- aborting")
    print(f"      shape audit OK: "
          f"{len(kept)-len(not_in_target)}/{len(kept)} keys map cleanly")

    print(f"[5/6] loading filtered keys into PA-CODE model (strict=False)")
    missing, unexpected = model.load_state_dict(kept, strict=False)
    n_loaded = len(kept) - len(unexpected)
    print(f"      loaded {n_loaded}/{len(kept)} keys")
    print(f"      missing in PA-CODE after load: {len(missing)} (PA-CODE-fresh ode.*)")
    print(f"      unexpected in src: {len(unexpected)}")
    if unexpected:
        for k in unexpected[:5]:
            print(f"        unexpected: {k}")
    # "Missing" keys allowed: ode.* (the new FiLM/position machinery) +
    # the explicit DROP_EXACT_KEYS we re-init on purpose (queries, hq_token).
    allowed_missing = lambda m: m.startswith("ode.") or m in DROP_EXACT_KEYS
    bad_missing = [m for m in missing if not allowed_missing(m)]
    if bad_missing:
        print(f"      ERROR: {len(bad_missing)} unexpected missing keys -- bug in filter")
        for k in bad_missing[:10]:
            print(f"        - {k}")
        raise RuntimeError("unexpected missing keys after warmstart -- aborting")
    n_ode_keys = len([k for k in target_state if k.startswith("ode.")])
    print(f"      ode.* keys staying at PA-CODE identity-init: {n_ode_keys}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[6/6] saving warmstart to {OUT_PATH}")
    out_ckpt = {
        "epoch": -1,                          # trainer adds 1 -> start_epoch=0
        "model_state_dict": model.state_dict(),
        "metrics": {},
        "monitor_metric": "val_dice_3d",
        "warmstart_source": str(A1_BEST_CKPT),
        "warmstart_src_epoch": src_ckpt.get("epoch"),
        "warmstart_keys_loaded": n_loaded,
        "warmstart_pa_code": True,
    }
    torch.save(out_ckpt, str(OUT_PATH))
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"      saved {size_mb:.0f} MB")

    # Round-trip verify
    print()
    print("Round-trip verification (strict=True load):")
    verify_model = build_model(cfg).cpu()
    verify_ckpt = torch.load(str(OUT_PATH), map_location="cpu", weights_only=False)
    verify_model.load_state_dict(verify_ckpt["model_state_dict"])  # strict default True
    print("  PASS - strict=True load succeeded")

    # Check that carried weights actually came from epoch_001 (not fresh)
    same = 0
    for k, v in kept.items():
        if k in not_in_target:
            continue
        if torch.equal(verify_model.state_dict()[k], v):
            same += 1
    print(f"  carried weight match: {same}/{len(kept)-len(not_in_target)} keys equal source")
    if same < (len(kept) - len(not_in_target)):
        raise RuntimeError(
            f"Warmstart carry incomplete: only {same} of "
            f"{len(kept)-len(not_in_target)} keys match source."
        )

    # Identity-init verification: forward(features) == features at step 0
    # using the PA-CODE ODE module alone.
    import torch as _t
    with _t.no_grad():
        feats = _t.randn(1, cfg.data.slices_per_volume, cfg.model.embed_dim, 16, 16)
        organ_id = _t.tensor([6])  # liver
        out = verify_model.ode(feats, organ_id)
        diff = (out - feats).abs().max().item()
    print(f"  identity-init check: |ode(x) - x|_inf = {diff:.3e}")
    if diff > 1e-5:
        raise RuntimeError(f"PA-CODE not identity at init: diff={diff}")

    print()
    print("=" * 70)
    print(f"PASS - PA-CODE warmstart ready at {OUT_PATH}")
    print(f"  Encoder + decoder + deep_sup_head + dense_pe carry epoch_001 weights.")
    print(f"  ODE module (FiLM gamma/beta/position-aug) at identity-init.")
    print(f"  Trainer config resume_from -> this file.")
    print("=" * 70)


if __name__ == "__main__":
    main()
