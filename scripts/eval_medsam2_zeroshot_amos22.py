"""Zero-shot MedSAM-2 on AMOS22 liver — apples-to-apples baseline for V2.

Loads the public MedSAM2_hiera_tiny.pt checkpoint via the official sam2.1
hiera-tiny config, runs single-image bbox-prompted prediction slice by
slice, applies the same largest-CC post-processing as the V2 pipeline,
and computes DSC/HD95/NSD@1mm with the canonical VolumetricMetrics path.

Preprocessing is bit-for-bit aligned with eval_amos22_liver_indomain.py:
  - LAS orientation (AMOS22 is already LAS; defensive flip-to-LAS otherwise)
  - HU clip [-175, 250], normalize to [0, 1]
  - Resize to 256x256 with bilinear (img) / nearest (mask)
  - GT bbox + 10px padding (oracle bbox, same as V2 in-domain eval)
  - Per-volume largest-3D-CC post-processing
  - Spacing scaled by H_native / 256 (x), W_native / 256 (y), src_z (z)

Differences vs V2:
  - SAM-2 is a 2D single-image model; no 8-slice context window
  - sigmoid output is binarised at 0.5 (no per-thresh sweep here)
  - no TTA (zero-shot baseline)

Usage:
    python scripts/eval_medsam2_zeroshot_amos22.py \
        --medsam2-ckpt checkpoints/medsam2/MedSAM2_hiera_tiny.pt \
        --amos-root C:/Users/Raywa/Desktop/LiteSAM3D/data/amos22 \
        --max-volumes 100 \
        --output thesis/results/medsam2_zeroshot_amos22
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.metrics_3d import VolumetricMetrics

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

AMOS22_ORIENT = ("L", "A", "S")
LIVER_LABEL = 6
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"


def load_amos22_case(img_path: Path, lbl_path: Path):
    img_nib = nib.load(str(img_path))
    lbl_nib = nib.load(str(lbl_path))
    img_orient = tuple(nib.aff2axcodes(img_nib.affine))
    if img_orient != AMOS22_ORIENT:
        img_nib = nib.as_closest_canonical(img_nib)
        lbl_nib = nib.as_closest_canonical(lbl_nib)
        img = np.asarray(img_nib.dataobj)[::-1, :, :]
        lbl = np.asarray(lbl_nib.dataobj)[::-1, :, :]
    else:
        img = np.asarray(img_nib.dataobj)
        lbl = np.asarray(lbl_nib.dataobj)
    spacing = tuple(float(s) for s in img_nib.header.get_zooms()[:3])
    img = img.astype(np.float32)
    liver = (lbl == LIVER_LABEL).astype(np.uint8)
    return np.ascontiguousarray(img), np.ascontiguousarray(liver), spacing


def find_eligible_cases(amos_root: Path, max_n: int):
    img_dir = amos_root / "imagesVa"
    lbl_dir = amos_root / "labelsVa"
    cases = []
    for img_path in sorted(img_dir.glob("amos_*.nii.gz")):
        case_id = img_path.stem.replace(".nii", "")
        try:
            num = int(case_id.split("_")[1])
        except Exception:
            continue
        if num >= 500:
            continue
        lbl_path = lbl_dir / img_path.name
        if not lbl_path.exists():
            continue
        cases.append((case_id, img_path, lbl_path))
        if len(cases) >= max_n:
            break
    return cases


def preprocess_slice_uint8(slice_HW: np.ndarray, img_size: int,
                           hu_clip=(-175.0, 250.0)) -> np.ndarray:
    """Return uint8 RGB image (H, W, 3) ready for SAM2ImagePredictor.set_image."""
    lo, hi = hu_clip
    s = np.clip(slice_HW, lo, hi).astype(np.float32)
    s = (s - lo) / max(hi - lo, 1e-8)
    t = torch.from_numpy(s)[None, None]
    t = F.interpolate(t, size=(img_size, img_size),
                      mode="bilinear", align_corners=False)
    s_re = t[0, 0].numpy()
    s_u8 = (np.clip(s_re, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.stack([s_u8, s_u8, s_u8], axis=-1)


def get_bounding_box_resized(mask_native: np.ndarray, img_size: int,
                             padding: int = 10):
    mt = torch.from_numpy(mask_native.astype(np.float32))[None, None]
    m_re = F.interpolate(mt, size=(img_size, img_size),
                         mode="nearest")[0, 0].numpy().astype(np.uint8)
    if m_re.sum() == 0:
        return None, m_re.astype(bool)
    rows = np.any(m_re, axis=1)
    cols = np.any(m_re, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    H, W = m_re.shape
    x1 = max(0, x1 - padding); x2 = min(W - 1, x2 + padding)
    y1 = max(0, y1 - padding); y2 = min(H - 1, y2 + padding)
    return (np.array([x1, y1, x2, y2], dtype=np.float32),
            m_re.astype(bool))


def keep_largest_cc(pred: np.ndarray) -> np.ndarray:
    from scipy import ndimage
    if pred.sum() == 0:
        return pred
    lbls, n_cc = ndimage.label(pred, structure=ndimage.generate_binary_structure(3, 1))
    if n_cc <= 1:
        return pred
    sizes = ndimage.sum(pred, lbls, index=range(1, n_cc + 1))
    largest = int(np.argmax(sizes)) + 1
    return (lbls == largest)


@torch.no_grad()
def evaluate_case(predictor: SAM2ImagePredictor,
                  image_DHW: np.ndarray, label_DHW: np.ndarray,
                  img_size: int, hu_clip=(-175.0, 250.0)):
    D, H, W = image_DHW.shape
    pred_DHW = np.zeros((D, img_size, img_size), dtype=bool)
    gt_DHW = np.zeros((D, img_size, img_size), dtype=bool)
    n_processed = 0

    for z in range(D):
        gt_slice = label_DHW[z]
        if gt_slice.sum() < 50:
            continue
        bbox, gt_re = get_bounding_box_resized(gt_slice, img_size, padding=10)
        if bbox is None:
            continue
        gt_DHW[z] = gt_re

        img_u8 = preprocess_slice_uint8(image_DHW[z], img_size, hu_clip)
        predictor.set_image(img_u8)
        masks, scores, _ = predictor.predict(
            box=bbox, multimask_output=False,
        )
        # masks shape (1, H, W) bool
        pred_DHW[z] = masks[0].astype(bool)
        n_processed += 1

    pred_DHW = keep_largest_cc(pred_DHW)
    return pred_DHW, gt_DHW, n_processed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--medsam2-ckpt", type=str,
                    default="checkpoints/medsam2/MedSAM2_hiera_tiny.pt")
    ap.add_argument("--amos-root", type=str,
                    default=r"C:/Users/Raywa/Desktop/LiteSAM3D/data/amos22")
    ap.add_argument("--max-volumes", type=int, default=100)
    ap.add_argument("--output", type=str,
                    default="thesis/results/medsam2_zeroshot_amos22")
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[medsam2] device={args.device}")
    print(f"[medsam2] checkpoint={args.medsam2_ckpt}")
    print(f"[medsam2] config={SAM2_CONFIG}")
    print(f"[medsam2] amos_root={args.amos_root}")
    print(f"[medsam2] max_volumes={args.max_volumes}  img_size={args.img_size}")

    sam2_model = build_sam2(SAM2_CONFIG, args.medsam2_ckpt, device=args.device)
    predictor = SAM2ImagePredictor(sam2_model)
    print(f"[medsam2] model loaded ({type(sam2_model).__name__})")

    cases = find_eligible_cases(Path(args.amos_root), args.max_volumes)
    print(f"[medsam2] found {len(cases)} eligible AMOS22 val cases")

    metrics = VolumetricMetrics()
    per_volume = []
    t_start = time.time()

    for i, (case_id, img_path, lbl_path) in enumerate(cases):
        try:
            img_HWD, lbl_HWD, src_spacing = load_amos22_case(img_path, lbl_path)
        except Exception as e:
            print(f"[medsam2] ({i+1}/{len(cases)}) {case_id} LOAD FAILED: {e}")
            continue
        if lbl_HWD.sum() == 0:
            print(f"[medsam2] ({i+1}/{len(cases)}) {case_id} SKIP (no liver)")
            continue

        t0 = time.time()
        img_DHW = np.transpose(img_HWD, (2, 0, 1))
        lbl_DHW = np.transpose(lbl_HWD, (2, 0, 1))
        pred, gt, n_proc = evaluate_case(
            predictor, img_DHW, lbl_DHW,
            img_size=args.img_size,
        )

        H_n, W_n = img_HWD.shape[0], img_HWD.shape[1]
        sp_x = float(src_spacing[0]) * (H_n / args.img_size)
        sp_y = float(src_spacing[1]) * (W_n / args.img_size)
        sp_z = float(src_spacing[2])
        spacing_eval = (sp_z, sp_x, sp_y)

        rec = metrics.update(
            pred_vol=pred, gt_vol=gt,
            spacing_mm=spacing_eval, organ_id=6, patient_id=case_id,
        )
        per_volume.append({
            "case_id":  case_id,
            "n_slices": n_proc,
            "dsc":      float(rec["dsc"]),
            "hd95_mm":  float(rec["hd95_mm"]),
            "nsd":      float(rec["nsd"]),
            "iou":         float(rec.get("iou", float("nan"))),
            "assd_mm":     float(rec.get("assd_mm", float("nan"))),
            "sensitivity": float(rec.get("sensitivity", float("nan"))),
            "precision":   float(rec.get("precision", float("nan"))),
            "specificity": float(rec.get("specificity", float("nan"))),
            "vol_sim":     float(rec.get("vol_sim", float("nan"))),
            "vol_pred_ml": float(rec.get("vol_pred_ml", float("nan"))),
            "vol_gt_ml":   float(rec.get("vol_gt_ml", float("nan"))),
            "nsd_at_2.0mm": float(rec.get("nsd_at_2.0mm", float("nan"))),
        })
        dt = time.time() - t0
        print(f"[medsam2] ({i+1}/{len(cases)}) {case_id} "
              f"D={img_DHW.shape[0]} liver_slices={n_proc} "
              f"dsc={rec['dsc']:.4f} hd95={rec['hd95_mm']:.2f}mm "
              f"nsd={rec['nsd']:.4f} {dt:.1f}s",
              flush=True)

    summary = metrics.summary()
    overall = summary.get("overall", {})
    print("\n=== MedSAM-2 zero-shot AMOS22 results ===")
    print(f"  N volumes: {len(per_volume)}")
    print(f"  Mean DSC : {overall.get('dsc_mean', float('nan')):.4f} "
          f"± {overall.get('dsc_std', float('nan')):.4f}")
    print(f"  Mean HD95: {overall.get('hd95_mm_mean', float('nan')):.2f} "
          f"± {overall.get('hd95_mm_std', float('nan')):.2f} mm")
    print(f"  Mean NSD : {overall.get('nsd_mean', float('nan')):.4f} "
          f"± {overall.get('nsd_std', float('nan')):.4f}")
    print(f"  Total time: {(time.time() - t_start) / 60:.1f} min")

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
            "checkpoint": args.medsam2_ckpt,
            "config": SAM2_CONFIG,
            "img_size": args.img_size,
            "summary": summary,
        }, f, indent=2)
    print(f"[medsam2] wrote: {csv_path}")
    print(f"[medsam2] wrote: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
