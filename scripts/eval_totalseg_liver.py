"""Cross-dataset evaluation: V2 Auto-ODE-SAM (AMOS22-trained, liver) on
TotalSegmentator held-out CT volumes.

Mirrors V2's exact training preprocessing (datasets/amos22.py:__getitem__):
  - NO physical-spacing resampling: V2 was trained on native-spacing AMOS22
    slice grids (npy_cache verified at 0.616mm × 0.616mm × 5mm, NOT 1.5 iso).
  - Orientation canonicalised LAS to match AMOS22 (TotalSeg ships RAS).
  - HU clip [-175, 250], normalise to [0, 1], stack 3 channels, F.interpolate
    bilinear to 256×256 — same operations as the trainer's val path.
  - Bbox extracted from the *resized* mask with padding=10 (matches V2's
    ValTransforms / get_bounding_box defaults).

Usage:
    python scripts/eval_totalseg_liver.py \
        --checkpoint C:/Users/Raywa/Desktop/VoluFormer3D/checkpoints/phase3_odesam_v2_256px_liver/phase3_odesam_v2_256px_liver_best.pt \
        --totalseg-root D:/data/totalsegmentator \
        --max-volumes 25 \
        --output thesis/results/totalseg_liver
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

from omegaconf import OmegaConf
from models import build_model
from evaluation.metrics_3d import VolumetricMetrics


# AMOS22 reference orientation, matching the LAS layout the V2 model was
# trained on (verified via nib.aff2axcodes on imagesVa/amos_0008.nii.gz).
AMOS22_ORIENT = ("L", "A", "S")


def load_to_amos_orientation(path: Path):
    """Load NIfTI and reorient to AMOS22's LAS layout.

    Returns (volume_HWD float32, spacing_HWD tuple).
    """
    nib_obj = nib.load(str(path))
    src_orient = tuple(nib.aff2axcodes(nib_obj.affine))
    if src_orient != AMOS22_ORIENT:
        nib_obj = nib.as_closest_canonical(nib_obj)               # → RAS
        # Convert RAS → LAS by flipping the L/R axis (axis 0)
        arr = np.asarray(nib_obj.dataobj)
        spacing = nib_obj.header.get_zooms()[:3]
        arr = arr[::-1, :, :]                                     # R→L flip
    else:
        arr = np.asarray(nib_obj.dataobj)
        spacing = nib_obj.header.get_zooms()[:3]
    return np.ascontiguousarray(arr), tuple(float(s) for s in spacing)


def load_totalseg_case(case_dir: Path):
    """Return (image_HWD float32 HU, label_HWD uint8 binary, spacing_HWD)."""
    img_path = case_dir / "ct.nii.gz"
    lbl_path = case_dir / "segmentations" / "liver.nii.gz"
    if not img_path.exists() or not lbl_path.exists():
        return None
    img, sp_img = load_to_amos_orientation(img_path)
    lbl, sp_lbl = load_to_amos_orientation(lbl_path)
    img = img.astype(np.float32)
    lbl = (lbl > 0).astype(np.uint8)
    return img, lbl, sp_img


def find_eligible_cases(root: Path, max_n: int):
    """Iterate s**** dirs, return first max_n that contain liver+ct files
       AND a non-empty liver mask (TotalSeg includes leg/head/thorax-only
       scans that have an empty liver.nii.gz)."""
    out = []
    for case_dir in sorted(root.iterdir()):
        if not case_dir.is_dir() or not case_dir.name.startswith("s"):
            continue
        ct = case_dir / "ct.nii.gz"
        lv = case_dir / "segmentations" / "liver.nii.gz"
        if not (ct.exists() and lv.exists()):
            continue
        try:
            lab = np.asarray(nib.load(str(lv)).dataobj, dtype=np.uint8)
        except Exception:
            continue
        if lab.sum() < 1000:
            continue
        out.append(case_dir)
        if len(out) >= max_n:
            break
    return out


def preprocess_slice_window(window_DHW: np.ndarray, img_size: int,
                            hu_clip=(-175.0, 250.0)) -> torch.Tensor:
    """Apply V2's exact training preprocessing to a (D, H, W) slice window.

    Mirrors `AMOS22_3D_Dataset.__getitem__` lines 773–791:
      clip → normalise [0,1] → expand to 3 channels by repeat → batched
      F.interpolate(bilinear) to (img_size, img_size).

    Returns (D, 3, img_size, img_size) float32 tensor.
    """
    lo, hi = hu_clip
    win = np.clip(window_DHW, lo, hi).astype(np.float32)
    win = (win - lo) / max(hi - lo, 1e-8)
    t = torch.from_numpy(win).unsqueeze(1).repeat(1, 3, 1, 1)        # (D, 3, H, W)
    t = F.interpolate(t, size=(img_size, img_size),
                      mode="bilinear", align_corners=False)
    return t


def get_bounding_box_resized(mask_native: np.ndarray, img_size: int,
                             padding: int = 10) -> np.ndarray:
    """Resize mask via F.interpolate-nearest (V2's path) then derive bbox at
    img_size resolution with padding=10. Matches V2's ValTransforms behaviour
    via `get_bounding_box(mask, padding=10)` after resize_and_pad — for square
    sources the two paths produce identical bboxes.
    """
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


def build_window(volume_DHW: np.ndarray, center: int, n_slices: int):
    """Edge-padded slice window."""
    half = n_slices // 2
    D = volume_DHW.shape[0]
    sl_start = max(0, center - half)
    sl_end = min(D, center + half)
    win = volume_DHW[sl_start:sl_end]
    n = win.shape[0]
    pad_before = max(0, half - (center - sl_start))
    pad_after = n_slices - n - pad_before
    if pad_before or pad_after:
        win = np.pad(win, ((pad_before, pad_after), (0, 0), (0, 0)), mode="edge")
    return win  # (n_slices, H, W)


def keep_largest_cc(pred: np.ndarray) -> np.ndarray:
    """Standard 3D-segmentation postproc: retain only the single largest
    foreground connected component. Removes spurious far-from-organ blobs
    that inflate HD95 / wreck NSD without meaningfully affecting Dice."""
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
def evaluate_case(model, image_DHW: np.ndarray, label_DHW: np.ndarray,
                  n_slices: int, img_size: int, device,
                  modality_id: int = 0, hu_clip=(-175.0, 250.0),
                  apply_cc: bool = True):
    """Per-slice inference with 8-slice context window. Predictions are
    accumulated at img_size resolution to match the GT mask resolution. Slices
    with no GT liver are left as zeros (oracle-bbox regime — no prompt would be
    issued for empty slices). Largest-CC postproc applied to the final volume."""
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

        win = build_window(image_DHW, z, n_slices)                # (D, H, W)
        win_t = preprocess_slice_window(win, img_size, hu_clip)   # (D, 3, h, w)
        win_t = win_t.unsqueeze(0).to(device)                     # (1, D, 3, h, w)

        box_t = torch.from_numpy(bbox)[None].to(device)           # (1, 4)
        mod_t = torch.tensor([modality_id], dtype=torch.long, device=device)
        mod_t = mod_t.unsqueeze(1).expand(1, n_slices)            # (1, D)

        prob = model.predict(win_t, box_t, mod_t, is_3d=True)     # (1, 1, h, w)
        if prob.shape[-1] != img_size:
            prob = F.interpolate(prob, size=(img_size, img_size),
                                 mode="bilinear", align_corners=False)
        pred_DHW[z] = prob.squeeze().cpu().numpy() > 0.5
        n_processed += 1

    if apply_cc:
        pred_DHW = keep_largest_cc(pred_DHW)

    return pred_DHW, gt_DHW, n_processed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument(
        "--config", type=str,
        default=r"C:/Users/Raywa/Desktop/VoluFormer3D/checkpoints/phase3_odesam_v2_256px_liver/config.yaml",
    )
    ap.add_argument("--totalseg-root", type=str, default="D:/data/totalsegmentator")
    ap.add_argument("--max-volumes", type=int, default=25)
    ap.add_argument("--output", type=str, default="thesis/results/totalseg_liver")
    ap.add_argument("--n-slices", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(args.config)

    print(f"[eval] device={args.device}")
    print(f"[eval] config={args.config}")
    print(f"[eval] checkpoint={args.checkpoint}")
    print(f"[eval] totalseg_root={args.totalseg_root}")
    print(f"[eval] preprocessing = V2 native (no spacing resample, LAS reorient)")

    model = build_model(cfg).to(args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[eval] loaded ckpt | missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    cases = find_eligible_cases(Path(args.totalseg_root), args.max_volumes)
    print(f"[eval] found {len(cases)} eligible TotalSeg cases")

    metrics = VolumetricMetrics()
    per_volume = []
    t_start = time.time()

    for i, case_dir in enumerate(cases):
        case_id = case_dir.name
        try:
            res = load_totalseg_case(case_dir)
            if res is None:
                print(f"[eval] ({i+1}/{len(cases)}) {case_id} SKIP (missing files)")
                continue
            img_HWD, lbl_HWD, src_spacing = res
        except Exception as e:
            print(f"[eval] ({i+1}/{len(cases)}) {case_id} LOAD FAILED: {e}")
            continue
        if lbl_HWD.sum() == 0:
            print(f"[eval] ({i+1}/{len(cases)}) {case_id} SKIP (no liver in mask)")
            continue

        t0 = time.time()
        # Transpose to axial-major (D, H, W). LAS = (L, A, S); axial = S (axis 2).
        img_DHW = np.transpose(img_HWD, (2, 0, 1))
        lbl_DHW = np.transpose(lbl_HWD, (2, 0, 1))

        pred, gt, n_proc = evaluate_case(
            model, img_DHW, lbl_DHW,
            n_slices=args.n_slices, img_size=args.img_size,
            device=args.device,
        )

        # Compute volumetric metrics. The pred/gt volumes are at native z (no
        # z-resampling) and resized in-plane to img_size; physical spacing is
        # original_z and original_x*(H/img_size), original_y*(W/img_size).
        # VolumetricMetrics expects (Z, H, W) shaped arrays.
        H_n, W_n = img_HWD.shape[0], img_HWD.shape[1]
        sp_x = float(src_spacing[0]) * (H_n / args.img_size)
        sp_y = float(src_spacing[1]) * (W_n / args.img_size)
        sp_z = float(src_spacing[2])
        spacing_eval = (sp_z, sp_x, sp_y)

        rec = metrics.update(
            pred_vol=pred,
            gt_vol=gt,
            spacing_mm=spacing_eval,
            organ_id=6,                      # liver id (matches AMOS22 mapping)
            patient_id=case_id,
        )

        per_volume.append({
            "case_id":   case_id,
            "n_slices":  n_proc,
            "dsc":       float(rec["dsc"]),
            "hd95_mm":   float(rec["hd95_mm"]),
            "nsd":       float(rec["nsd"]),
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
        print(f"[eval] ({i+1}/{len(cases)}) {case_id} "
              f"D={img_DHW.shape[0]} liver_slices={n_proc} "
              f"dsc={rec['dsc']:.4f} hd95={rec['hd95_mm']:.2f}mm "
              f"nsd={rec['nsd']:.4f} assd={rec.get('assd_mm', float('nan')):.2f}mm "
              f"iou={rec.get('iou', float('nan')):.4f} {dt:.1f}s")

    summary = metrics.summary()
    print("\n=== TotalSeg cross-dataset results ===")
    overall = summary.get("overall", {})
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
    with open(out_dir / "per_volume_dsc.json", "w") as f:
        json.dump(per_volume, f, indent=2)
    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "n_volumes": len(per_volume),
            "checkpoint": args.checkpoint,
            "n_slices": args.n_slices,
            "img_size": args.img_size,
            "preprocessing": "V2 native (no spacing resample, LAS reorient)",
            "summary": summary,
        }, f, indent=2)
    print(f"[eval] wrote: {out_dir / 'per_volume_metrics.csv'}")
    print(f"[eval] wrote: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
