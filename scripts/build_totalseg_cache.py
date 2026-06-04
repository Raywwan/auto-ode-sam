"""Build a fast-disk cache of TotalSegmentator volumes.

Each volume is normalized once (HU clip + scale to [0,1]) and the 15-class
label is remapped from per-organ binary masks. Both are saved as a single
.pt dict per volume:
    {"volume": fp16 (D,H,W), "label": uint8 (D,H,W)}

This shifts ~30-50 GB of preprocessing off the slow HDD onto the fast system
disk so the per-batch read drops from ~2.5 s (16 nii.gz file opens on HDD)
to ~50 ms (one .pt load on SSD).

Usage:
    python scripts/build_totalseg_cache.py \
        --data-root D:/data/totalsegmentator \
        --cache-dir  C:/cache/totalsegmentator \
        --hu-clip -200 250
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import nibabel as nib
import numpy as np
import torch

from datasets.totalsegmentator import _TS_FILENAME_TO_AMOS


def normalize_to_fp16(ct: np.ndarray, hu_lo: int, hu_hi: int) -> np.ndarray:
    ct = np.clip(ct, hu_lo, hu_hi)
    ct = (ct - hu_lo) / max(hu_hi - hu_lo, 1)
    return ct.astype(np.float16)


def build_one(seg_root: Path, vol_id: str, hu_lo: int, hu_hi: int):
    ct_path = seg_root / vol_id / "ct.nii.gz"
    seg_dir = seg_root / vol_id / "segmentations"
    ct = nib.load(str(ct_path)).get_fdata().astype(np.float32)
    label = np.zeros_like(ct, dtype=np.uint8)
    for ts_name, amos_id in _TS_FILENAME_TO_AMOS.items():
        f = seg_dir / f"{ts_name}.nii.gz"
        if not f.exists():
            continue
        m = nib.load(str(f)).get_fdata() > 0.5
        label[m] = int(amos_id)
    # Match dataset transpose: (X, Y, Z) -> (D, H, W) = (Z, X, Y).
    ct = ct.transpose(2, 0, 1)
    label = label.transpose(2, 0, 1)
    ct_fp16 = normalize_to_fp16(ct, hu_lo, hu_hi)
    return torch.from_numpy(ct_fp16), torch.from_numpy(label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--hu-clip", nargs=2, type=int, default=[-200, 250])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    args = parser.parse_args()

    manifest_path = args.data_root / "totalsegmentator_manifest.json"
    if not manifest_path.exists():
        print(f"[FAIL] manifest missing at {manifest_path}")
        sys.exit(1)
    manifest = json.loads(manifest_path.read_text())
    base = Path(manifest["data_root"])
    all_ids = sorted(manifest["volume_ids"])
    if args.limit:
        all_ids = all_ids[: args.limit]

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = args.cache_dir / "_meta.json"
    meta = {
        "hu_clip": list(args.hu_clip),
        "n_organs": 15,
        "label_map": "datasets.totalsegmentator._TS_FILENAME_TO_AMOS",
        "format": "fp16 volume + uint8 label, transposed to (D,H,W)",
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"[cache] build {len(all_ids)} vols -> {args.cache_dir}")
    print(f"[cache] hu_clip={args.hu_clip}")

    t0 = time.time()
    total_bytes = 0
    skipped = 0
    for i, vid in enumerate(all_ids):
        out = args.cache_dir / f"{vid}.pt"
        if args.skip_existing and out.exists():
            skipped += 1
            total_bytes += out.stat().st_size
            continue
        vol_t, lab_t = build_one(base, vid, args.hu_clip[0], args.hu_clip[1])
        torch.save({"volume": vol_t, "label": lab_t}, out)
        total_bytes += out.stat().st_size
        if (i + 1) % 25 == 0 or (i + 1) == len(all_ids):
            dt = time.time() - t0
            done = i + 1 - skipped
            rate = max(done, 1) / max(dt, 1e-6)
            eta = (len(all_ids) - (i + 1)) / max(rate, 1e-6)
            mean_mb = total_bytes / max(i + 1, 1) / (1024 ** 2)
            print(f"  [{i+1:4d}/{len(all_ids)}]  built {done}  skipped {skipped}  "
                  f"{dt:.0f}s elapsed  {rate:.2f} vol/s  eta {eta:.0f}s  "
                  f"avg {mean_mb:.1f} MB/vol")

    total_gb = total_bytes / (1024 ** 3)
    print(f"\n[cache] saved {len(all_ids)} vols  ({total_gb:.1f} GB total)")
    print(f"[cache] meta -> {meta_path}")


if __name__ == "__main__":
    main()
