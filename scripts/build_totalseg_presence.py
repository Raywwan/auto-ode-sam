"""Build a TotalSeg presence manifest for fast BalancedBatchSampler init.

For each volume, probe the per-organ binary mask files (no CT load) and
record which AMOS22 organ ids are present. Saves to:
    <data_root>/totalsegmentator_presence.json   { vol_id: [organ_ids...] }

This is a one-time cost (~8-15 min on a 1228-vol release). Sampler init
afterwards is a JSON read in <100 ms instead of a 55-min full-label pass.

Usage:
    python scripts/build_totalseg_presence.py --data-root D:/data/totalsegmentator
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

from datasets.totalsegmentator import _TS_FILENAME_TO_AMOS


def probe_volume(seg_dir: Path) -> list:
    present: set = set()
    for ts_name, amos_id in _TS_FILENAME_TO_AMOS.items():
        f = seg_dir / f"{ts_name}.nii.gz"
        if not f.exists():
            continue
        m = nib.load(str(f)).get_fdata()
        if (m > 0.5).any():
            present.add(int(amos_id))
    return sorted(present)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0,
                        help="Stop after N volumes (debug only; 0 = all).")
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

    out_path = args.data_root / "totalsegmentator_presence.json"
    print(f"[presence] {len(all_ids)} vols  ->  {out_path}")

    presence: dict = {}
    t0 = time.time()
    for i, vid in enumerate(all_ids):
        seg_dir = base / vid / "segmentations"
        presence[vid] = probe_volume(seg_dir)
        if (i + 1) % 50 == 0 or (i + 1) == len(all_ids):
            dt = time.time() - t0
            rate = (i + 1) / dt
            eta = (len(all_ids) - (i + 1)) / max(rate, 1e-6)
            print(f"  [{i+1:4d}/{len(all_ids)}]  {dt:.0f}s elapsed  "
                  f"{rate:.2f} vol/s  eta {eta:.0f}s")

    out_path.write_text(json.dumps(presence, indent=2))

    # Quick summary by class.
    K = 15
    counts = {k: 0 for k in range(1, K + 1)}
    for organs in presence.values():
        for o in organs:
            if 1 <= int(o) <= K:
                counts[int(o)] += 1
    print(f"\n[presence] saved {out_path}")
    print(f"[presence] per-class volume counts (out of {len(presence)}):")
    for k in range(1, K + 1):
        n = counts[k]
        rate = n / max(len(presence), 1)
        print(f"  class {k:2d}: {n:4d}/{len(presence)} ({rate*100:.0f}%)")


if __name__ == "__main__":
    main()
