"""
build_swa_seed43.py — CPU-only SWA (Stochastic Weight Averaging) checkpoint builder.

Averages model_state_dict across the top-K (or late-N) checkpoints from the
seed=43 40-epoch run. Outputs a single ckpt file in the same directory.

Two variants by default:
  * swa_top5.pt  = mean of {ep10, ep15, ep25, ep30, ep35}   (the saved top-5 by val_dice)
  * swa_late4.pt = mean of {ep25, ep30, ep35, latest (=ep40)}

Pre-flight:
  --dry-run prints shapes / dtypes per ckpt and confirms key-match without writing.

Usage:
  python scripts/build_swa_seed43.py --dry-run
  python scripts/build_swa_seed43.py
  python scripts/build_swa_seed43.py --variant late4
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, List

import torch

DEFAULT_DIR = Path("C:/Users/Raywa/Desktop/VoluFormer3D_V4/checkpoints/phase3_odesam_v2_seed43_40ep")

TOP5_FILES = [
    "phase3_odesam_v2_seed43_40ep_epoch010.pt",
    "phase3_odesam_v2_seed43_40ep_epoch015.pt",
    "phase3_odesam_v2_seed43_40ep_epoch025.pt",
    "phase3_odesam_v2_seed43_40ep_epoch030.pt",
    "phase3_odesam_v2_seed43_40ep_epoch035.pt",
]
LATE4_FILES = [
    "phase3_odesam_v2_seed43_40ep_epoch025.pt",
    "phase3_odesam_v2_seed43_40ep_epoch030.pt",
    "phase3_odesam_v2_seed43_40ep_epoch035.pt",
    "phase3_odesam_v2_seed43_40ep_latest.pt",
]


def get_state_dict(ckpt: dict) -> Dict[str, torch.Tensor]:
    if "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


def average_state_dicts(state_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    n = len(state_dicts)
    keys0 = set(state_dicts[0].keys())
    for i, sd in enumerate(state_dicts[1:], start=1):
        if set(sd.keys()) != keys0:
            missing = keys0 - set(sd.keys())
            extra = set(sd.keys()) - keys0
            raise RuntimeError(f"ckpt #{i} key mismatch: missing={missing}, extra={extra}")
    avg = {}
    for k in keys0:
        t0 = state_dicts[0][k]
        if not torch.is_tensor(t0) or not t0.is_floating_point():
            # Keep integer buffers (e.g., num_batches_tracked) from first ckpt as-is.
            # Averaging them as floats is meaningless; SWA convention is to keep first.
            avg[k] = t0.clone()
            continue
        acc = t0.float().clone()
        for sd in state_dicts[1:]:
            acc += sd[k].float()
        acc /= n
        avg[k] = acc.to(t0.dtype)
    return avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=str, default=str(DEFAULT_DIR))
    ap.add_argument("--variant", choices=["top5", "late4"], default="top5")
    ap.add_argument("--out-name", type=str, default=None,
                    help="Output filename (default: swa_<variant>.pt)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print key/shape audit, do not write output.")
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.is_dir():
        print(f"ABORT: ckpt_dir does not exist: {ckpt_dir}", file=sys.stderr)
        sys.exit(2)

    files = TOP5_FILES if args.variant == "top5" else LATE4_FILES
    paths = [ckpt_dir / f for f in files]
    for p in paths:
        if not p.is_file():
            print(f"ABORT: missing ckpt {p}", file=sys.stderr)
            sys.exit(2)

    print(f"[swa] variant={args.variant}  averaging {len(paths)} ckpts:")
    for p in paths:
        print(f"  - {p.name}")

    # Load all on CPU
    raw = [torch.load(str(p), map_location="cpu", weights_only=False) for p in paths]
    sds = [get_state_dict(r) for r in raw]

    # Dry-run audit
    print(f"\n[swa] first ckpt key count: {len(sds[0])}")
    print(f"[swa] sample of first ckpt keys:")
    for k in list(sds[0].keys())[:6]:
        t = sds[0][k]
        print(f"    {k}  shape={tuple(t.shape) if torch.is_tensor(t) else type(t).__name__}  "
              f"dtype={t.dtype if torch.is_tensor(t) else '-'}")

    avg_sd = average_state_dicts(sds)
    print(f"\n[swa] averaging done — {len(avg_sd)} keys")

    # Build output ckpt: reuse first ckpt's outer structure, replace state_dict
    out_ckpt = {}
    if isinstance(raw[0], dict):
        out_ckpt.update({k: v for k, v in raw[0].items() if k not in ("model_state_dict", "state_dict")})
    out_ckpt["model_state_dict"] = avg_sd
    out_ckpt["swa_metadata"] = {
        "variant": args.variant,
        "source_files": files,
        "n_averaged": len(paths),
    }

    if args.dry_run:
        print("\n[swa] --dry-run: no write")
        return

    out_name = args.out_name or f"swa_{args.variant}.pt"
    out_path = ckpt_dir / out_name
    torch.save(out_ckpt, str(out_path))
    sz_mb = out_path.stat().st_size / (1024 ** 2)
    print(f"\n[swa] WROTE  {out_path}  ({sz_mb:.1f} MB)")


if __name__ == "__main__":
    main()
