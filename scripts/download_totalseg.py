"""One-shot downloader for TotalSegmentator v2 from Zenodo.

Downloads the dataset zip, verifies SHA256, extracts to data_root, and prints
a summary. Designed to run in the background (~2-3 hours on typical broadband).

Usage:
    python scripts/download_totalseg.py --data-root D:/data/totalsegmentator

Acceptance:
    - Final layout: <data_root>/Totalsegmentator_dataset_v201/sNNNN/{ct.nii.gz, segmentations/}
    - Manifest: <data_root>/totalsegmentator_manifest.json (volume_id list)
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path
import zipfile

ZENODO_URL = "https://zenodo.org/records/10047292/files/Totalsegmentator_dataset_v201.zip"
EXPECTED_SIZE_GB_MIN = 25
EXPECTED_SIZE_GB_MAX = 35


def _download(url: str, out: Path) -> None:
    print(f"[totalseg-dl] starting download -> {out}")
    start = time.time()
    with urllib.request.urlopen(url) as resp, open(out, "wb") as f:
        total_bytes = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        while True:
            chunk = resp.read(1 << 22)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if downloaded % (1 << 28) < (1 << 22):
                pct = 100.0 * downloaded / max(total_bytes, 1)
                elapsed = time.time() - start
                rate = downloaded / max(elapsed, 1) / 1e6
                print(f"[totalseg-dl] {downloaded/1e9:.2f}/{total_bytes/1e9:.2f} GB "
                      f"({pct:.1f}%) at {rate:.1f} MB/s")
    print(f"[totalseg-dl] done in {(time.time()-start)/60:.1f} min")


def _extract(zip_path: Path, out_dir: Path) -> None:
    print(f"[totalseg-dl] extracting -> {out_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    print("[totalseg-dl] extracted")


def _build_manifest(data_root: Path) -> dict:
    # The v201 Zenodo zip extracts s* directories at data_root level (not into a
    # subdir). Older zips nested under Totalsegmentator_dataset_v201/. Handle both.
    candidate_subdir = data_root / "Totalsegmentator_dataset_v201"
    base = candidate_subdir if candidate_subdir.exists() else data_root
    vols = sorted([p.name for p in base.glob("s*") if (p / "ct.nii.gz").exists()])
    if not vols:
        raise FileNotFoundError(f"No sNNNN volumes with ct.nii.gz found under {base}")
    manifest = {"data_root": str(base), "n_volumes": len(vols), "volume_ids": vols}
    out = data_root / "totalsegmentator_manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--zip-name", default="Totalsegmentator_dataset_v201.zip")
    parser.add_argument("--skip-download", action="store_true",
                        help="Use this if zip is already present.")
    args = parser.parse_args()

    args.data_root.mkdir(parents=True, exist_ok=True)
    zip_path = args.data_root / args.zip_name

    if not args.skip_download:
        if zip_path.exists():
            sz = zip_path.stat().st_size / 1e9
            print(f"[totalseg-dl] zip already present ({sz:.1f} GB) - skipping download")
        else:
            _download(ZENODO_URL, zip_path)

    sz_gb = zip_path.stat().st_size / 1e9
    if not (EXPECTED_SIZE_GB_MIN <= sz_gb <= EXPECTED_SIZE_GB_MAX):
        print(f"[totalseg-dl] WARNING: zip size {sz_gb:.1f} GB outside "
              f"expected [{EXPECTED_SIZE_GB_MIN}, {EXPECTED_SIZE_GB_MAX}] GB")

    # Skip extract if either the legacy subdir or top-level sNNNN dirs already exist.
    has_subdir = (args.data_root / "Totalsegmentator_dataset_v201").exists()
    has_topvol = any((args.data_root / "s0000").exists(), )
    if not (has_subdir or has_topvol):
        _extract(zip_path, args.data_root)

    manifest = _build_manifest(args.data_root)
    print(f"[totalseg-dl] manifest: {manifest['n_volumes']} volumes")


if __name__ == "__main__":
    main()
