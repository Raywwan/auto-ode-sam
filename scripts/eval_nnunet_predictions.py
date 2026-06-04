"""
eval_nnunet_predictions.py — Apply V2's 3D metric pipeline to nnU-Net predictions.

Loads nnU-Net Dataset511 imagesVa predictions (binary liver) + matching ground
truth, applies the same largest-CC postproc as V2, and feeds them through the
SAME `evaluation.metrics_3d.VolumetricMetrics` accumulator as V2's eval — so
the resulting per_volume_metrics.csv is column-for-column comparable with V2.

Output:
    <out_dir>/per_volume_metrics.csv     same schema as V2
    <out_dir>/summary.json
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evaluation.metrics_3d import VolumetricMetrics  # noqa: E402

try:
    from scipy import ndimage
except ImportError as e:
    raise SystemExit("scipy required: " + str(e))


def largest_cc(mask: np.ndarray) -> np.ndarray:
    if mask.sum() == 0:
        return mask
    lbl, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum_labels(mask, lbl, index=range(1, n + 1))
    keep = int(np.argmax(sizes)) + 1
    return (lbl == keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--amos_root", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    amos = Path(args.amos_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # The build script writes val_ids.json next to dataset.json
    raw_dset = pred_dir.parent.parent / "nnUNet_workspace" / "nnUNet_raw" / "Dataset511_AMOS22Liver"
    val_ids_path = raw_dset / "val_ids.json"
    if val_ids_path.exists():
        val_ids = json.loads(val_ids_path.read_text())
    else:
        val_ids = sorted(p.stem.replace(".nii", "") for p in pred_dir.glob("amos_*.nii.gz"))

    metrics = VolumetricMetrics()
    per_volume = []
    t0 = time.time()

    for i, case_id in enumerate(val_ids):
        pred_path = pred_dir / f"{case_id}.nii.gz"
        gt_path = amos / "labelsVa" / f"{case_id}.nii.gz"
        if not (pred_path.exists() and gt_path.exists()):
            print(f"[eval] SKIP {case_id} — missing pred or gt")
            continue

        pred_img = nib.load(str(pred_path))
        gt_img = nib.load(str(gt_path))
        spacing_xyz = pred_img.header.get_zooms()[:3]   # (x, y, z) mm
        # V2 uses (z, x, y) ordering for spacing in metrics — match that.
        spacing_zhw = (float(spacing_xyz[2]), float(spacing_xyz[0]), float(spacing_xyz[1]))

        pred = pred_img.get_fdata().astype(np.uint8)
        pred = (pred > 0).astype(bool)
        pred = largest_cc(pred)

        gt = gt_img.get_fdata().astype(np.int16)
        gt = (gt == 6).astype(bool)   # liver class in original AMOS22

        # nnU-Net keeps native orientation (X, Y, Z) — convert to (Z, H, W) for V2's metric layout.
        pred_zhw = np.transpose(pred, (2, 0, 1))
        gt_zhw = np.transpose(gt, (2, 0, 1))

        rec = metrics.update(
            pred_vol=pred_zhw, gt_vol=gt_zhw,
            spacing_mm=spacing_zhw, organ_id=6, patient_id=case_id,
        )
        per_volume.append({
            "case_id": case_id,
            "n_slices": int(pred_zhw.shape[0]),
            "dsc": float(rec["dsc"]),
            "hd95_mm": float(rec["hd95_mm"]),
            "nsd": float(rec["nsd"]),
            "iou": float(rec.get("iou", float("nan"))),
            "assd_mm": float(rec.get("assd_mm", float("nan"))),
            "sensitivity": float(rec.get("sensitivity", float("nan"))),
            "precision": float(rec.get("precision", float("nan"))),
            "specificity": float(rec.get("specificity", float("nan"))),
            "vol_sim": float(rec.get("vol_sim", float("nan"))),
            "vol_pred_ml": float(rec.get("vol_pred_ml", float("nan"))),
            "vol_gt_ml": float(rec.get("vol_gt_ml", float("nan"))),
            "nsd_at_2.0mm": float(rec.get("nsd_at_2.0mm", float("nan"))),
        })
        print(f"[eval] ({i+1}/{len(val_ids)}) {case_id} dsc={rec['dsc']:.4f} hd95={rec['hd95_mm']:.2f}mm")

    summary = metrics.summary()
    overall = summary.get("overall", {})
    print("\n=== nnU-Net AMOS22 liver results ===")
    print(f"  N volumes: {len(per_volume)}")
    print(f"  Mean DSC : {overall.get('dsc_mean', float('nan')):.4f} ± {overall.get('dsc_std', float('nan')):.4f}")
    print(f"  Mean HD95: {overall.get('hd95_mm_mean', float('nan')):.2f} ± {overall.get('hd95_mm_std', float('nan')):.2f} mm")
    print(f"  Mean NSD : {overall.get('nsd_mean', float('nan')):.4f}")
    print(f"  Mean IoU : {overall.get('iou_mean', float('nan')):.4f}")
    print(f"  Mean ASSD: {overall.get('assd_mm_mean', float('nan')):.2f} mm")
    print(f"  Total time: {(time.time() - t0) / 60:.1f} min")

    csv_path = out_dir / "per_volume_metrics.csv"
    with open(csv_path, "w") as f:
        f.write(
            "case_id,n_slices,dsc,hd95_mm,nsd,iou,assd_mm,sensitivity,"
            "precision,specificity,vol_sim,vol_pred_ml,vol_gt_ml,nsd_at_2mm\n"
        )
        for r in per_volume:
            f.write(
                f"{r['case_id']},{r['n_slices']},"
                f"{r['dsc']:.6f},{r['hd95_mm']:.4f},{r['nsd']:.6f},"
                f"{r['iou']:.6f},{r['assd_mm']:.4f},{r['sensitivity']:.6f},"
                f"{r['precision']:.6f},{r['specificity']:.6f},"
                f"{r['vol_sim']:.6f},{r['vol_pred_ml']:.4f},"
                f"{r['vol_gt_ml']:.4f},{r['nsd_at_2.0mm']:.6f}\n"
            )

    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "n_volumes": len(per_volume),
            "model": "nnUNetv2 3d_fullres fold-0",
            "summary": summary,
        }, f, indent=2)

    print(f"[eval] wrote: {csv_path}")
    print(f"[eval] wrote: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
