"""
build_nnunet_amos22_liver.py — Convert AMOS22 to nnU-Net liver-binary dataset.

Builds Dataset511_AMOS22Liver/ with:
  imagesTr/   — 240 train CT (id<500), suffix _0000.nii.gz
  labelsTr/   — liver-binary (class 6 → 1; everything else → 0)
  imagesVa/   — first-100 val CT with non-empty liver (matches V2 eval)
  dataset.json

Usage:
    python build_nnunet_amos22_liver.py \
        --amos_root C:/Users/Raywa/Desktop/LiteSAM3D/data/amos22 \
        --dataset_dir <nnUNet_raw>/Dataset511_AMOS22Liver \
        --max_val 100
"""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import nibabel as nib


def is_ct_case(case_id: str) -> bool:
    try:
        num = int(case_id.split("_")[1])
        return num < 500
    except Exception:
        return False


def liver_binary_save(src_lbl: Path, dst_lbl: Path) -> int:
    """Load AMOS22 multi-class label, keep class 6 (liver) only, save as binary.

    Returns the count of liver voxels (for validation filtering).
    """
    img = nib.load(str(src_lbl))
    arr = img.get_fdata().astype(np.int16)
    bin_arr = (arr == 6).astype(np.uint8)
    out = nib.Nifti1Image(bin_arr, img.affine, img.header)
    out.set_data_dtype(np.uint8)
    nib.save(out, str(dst_lbl))
    return int(bin_arr.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amos_root", required=True)
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--max_val", type=int, default=100)
    args = ap.parse_args()

    amos = Path(args.amos_root)
    dset = Path(args.dataset_dir)
    (dset / "imagesTr").mkdir(parents=True, exist_ok=True)
    (dset / "labelsTr").mkdir(parents=True, exist_ok=True)
    (dset / "imagesVa").mkdir(parents=True, exist_ok=True)
    (dset / "labelsVa").mkdir(parents=True, exist_ok=True)

    # ---- Train set (240 CT volumes from imagesTr) ----
    n_train = 0
    for src_img in sorted((amos / "imagesTr").glob("amos_*.nii.gz")):
        case_id = src_img.stem.replace(".nii", "")  # amos_0001
        if not is_ct_case(case_id):
            continue
        src_lbl = amos / "labelsTr" / src_img.name
        if not src_lbl.exists():
            continue
        # Skip if liver is absent
        liver_vox = nib.load(str(src_lbl)).get_fdata().astype(np.int16)
        if (liver_vox == 6).sum() < 50:
            continue
        dst_img = dset / "imagesTr" / f"{case_id}_0000.nii.gz"
        dst_lbl = dset / "labelsTr" / f"{case_id}.nii.gz"
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)
        if not dst_lbl.exists():
            liver_binary_save(src_lbl, dst_lbl)
        n_train += 1

    # ---- Val set (first-N CT volumes with non-empty liver from imagesVa) ----
    n_val = 0
    val_ids = []
    for src_img in sorted((amos / "imagesVa").glob("amos_*.nii.gz")):
        case_id = src_img.stem.replace(".nii", "")
        if not is_ct_case(case_id):
            continue
        src_lbl = amos / "labelsVa" / src_img.name
        if not src_lbl.exists():
            continue
        liver_vox = nib.load(str(src_lbl)).get_fdata().astype(np.int16)
        if (liver_vox == 6).sum() < 50:
            continue
        dst_img = dset / "imagesVa" / f"{case_id}_0000.nii.gz"
        dst_lbl = dset / "labelsVa" / f"{case_id}.nii.gz"
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)
        if not dst_lbl.exists():
            liver_binary_save(src_lbl, dst_lbl)
        val_ids.append(case_id)
        n_val += 1
        if n_val >= args.max_val:
            break

    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "liver": 1},
        "numTraining": n_train,
        "file_ending": ".nii.gz",
        "name": "AMOS22Liver",
        "description": "AMOS22 liver-binary single-fold baseline (matches V2 N=100 val).",
    }
    with open(dset / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    # Save the val-id list so the eval-bridge can match exactly
    with open(dset / "val_ids.json", "w") as f:
        json.dump(val_ids, f, indent=2)

    print(f"[build] Dataset511 built at {dset}")
    print(f"[build]   train: {n_train}, val: {n_val}")
    print(f"[build]   val_ids: {val_ids[:3]}...{val_ids[-3:]}")


if __name__ == "__main__":
    main()
