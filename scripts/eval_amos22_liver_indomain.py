"""In-domain reproduction: V2 Auto-ODE-SAM on AMOS22 val split, liver only.

Mirrors the cross-dataset eval pipeline (eval_totalseg_liver.py) so the
in-domain and cross-dataset numbers are computed in the same metric
namespace. Differences vs cross-dataset:

  - No orientation flip (AMOS22 is already LAS).
  - Liver mask = (label == 6); we only evaluate volumes that contain liver.
  - CT-only filter (AMOS22 also contains MRI; we skip those).
  - Reads from imagesVa/labelsVa NIfTI directly.

Usage:
    python scripts/eval_amos22_liver_indomain.py \
        --checkpoint C:/Users/Raywa/Desktop/VoluFormer3D/checkpoints/phase3_odesam_v2_256px_liver/phase3_odesam_v2_256px_liver_best.pt \
        --amos-root C:/Users/Raywa/Desktop/LiteSAM3D/data/amos22 \
        --output thesis/results/amos22_liver_indomain
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

# AMOS22 = LAS already; no flip needed.
AMOS22_ORIENT = ("L", "A", "S")
LIVER_LABEL = 6


def load_amos22_case(img_path: Path, lbl_path: Path):
    """Return (image_HWD float32 HU, liver_mask_HWD uint8, spacing_HWD)."""
    img_nib = nib.load(str(img_path))
    lbl_nib = nib.load(str(lbl_path))
    img_orient = tuple(nib.aff2axcodes(img_nib.affine))
    if img_orient != AMOS22_ORIENT:
        # Defensive: if any val volume isn't LAS, canonicalise to RAS then flip.
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
    """Iterate AMOS22 val volumes (imagesVa/labelsVa). Filter:
        - file ID < 500 (CT cases; AMOS22 IDs >=500 are MRI)
        - non-empty liver label
        - matching label file exists
    """
    img_dir = amos_root / "imagesVa"
    lbl_dir = amos_root / "labelsVa"
    cases = []
    for img_path in sorted(img_dir.glob("amos_*.nii.gz")):
        case_id = img_path.stem.replace(".nii", "")  # amos_0008
        try:
            num = int(case_id.split("_")[1])
        except Exception:
            continue
        if num >= 500:                      # MRI cases
            continue
        lbl_path = lbl_dir / img_path.name
        if not lbl_path.exists():
            continue
        cases.append((case_id, img_path, lbl_path))
        if len(cases) >= max_n:
            break
    return cases


def preprocess_slice_window(window_DHW: np.ndarray, img_size: int,
                            hu_clip=(-175.0, 250.0)) -> torch.Tensor:
    lo, hi = hu_clip
    win = np.clip(window_DHW, lo, hi).astype(np.float32)
    win = (win - lo) / max(hi - lo, 1e-8)
    t = torch.from_numpy(win).unsqueeze(1).repeat(1, 3, 1, 1)
    t = F.interpolate(t, size=(img_size, img_size),
                      mode="bilinear", align_corners=False)
    return t


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


def build_window(volume_DHW: np.ndarray, center: int, n_slices: int):
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
    return win


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
def evaluate_case(model, image_DHW: np.ndarray, label_DHW: np.ndarray,
                  n_slices: int, img_size: int, device,
                  modality_id: int = 0, hu_clip=(-175.0, 250.0),
                  apply_cc: bool = True, tta_flips: bool = False):
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

        win = build_window(image_DHW, z, n_slices)
        win_t = preprocess_slice_window(win, img_size, hu_clip)
        win_t = win_t.unsqueeze(0).to(device)
        box_t = torch.from_numpy(bbox)[None].to(device)
        mod_t = torch.tensor([modality_id], dtype=torch.long, device=device)
        mod_t = mod_t.unsqueeze(1).expand(1, n_slices)

        if not tta_flips:
            prob = model.predict(win_t, box_t, mod_t, is_3d=True)
            if prob.shape[-1] != img_size:
                prob = F.interpolate(prob, size=(img_size, img_size),
                                     mode="bilinear", align_corners=False)
            pred_DHW[z] = prob.squeeze().cpu().numpy() > 0.5
        else:
            # 4-flip TTA: identity, H-flip, V-flip, HV-flip
            probs_acc = torch.zeros(1, 1, img_size, img_size, device=device)
            for flip_h, flip_v in [(False, False), (True, False),
                                    (False, True), (True, True)]:
                w_in = win_t
                b_in = box_t.clone()
                if flip_h:
                    w_in = torch.flip(w_in, dims=[-1])
                    # box: [x1, y1, x2, y2] -> mirror x
                    new_x1 = img_size - 1 - b_in[:, 2]
                    new_x2 = img_size - 1 - b_in[:, 0]
                    b_in[:, 0] = new_x1; b_in[:, 2] = new_x2
                if flip_v:
                    w_in = torch.flip(w_in, dims=[-2])
                    new_y1 = img_size - 1 - b_in[:, 3]
                    new_y2 = img_size - 1 - b_in[:, 1]
                    b_in[:, 1] = new_y1; b_in[:, 3] = new_y2
                p = model.predict(w_in, b_in, mod_t, is_3d=True)
                if p.shape[-1] != img_size:
                    p = F.interpolate(p, size=(img_size, img_size),
                                      mode="bilinear", align_corners=False)
                if flip_h: p = torch.flip(p, dims=[-1])
                if flip_v: p = torch.flip(p, dims=[-2])
                probs_acc = probs_acc + p
            probs_acc = probs_acc / 4.0
            pred_DHW[z] = probs_acc.squeeze().cpu().numpy() > 0.5
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
    ap.add_argument("--amos-root", type=str,
                    default=r"C:/Users/Raywa/Desktop/LiteSAM3D/data/amos22")
    ap.add_argument("--max-volumes", type=int, default=999)
    ap.add_argument("--output", type=str,
                    default="thesis/results/amos22_liver_indomain")
    ap.add_argument("--n-slices", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--tta", action="store_true",
                    help="Enable 4-flip test-time augmentation")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(args.config)
    print(f"[eval] device={args.device}")
    print(f"[eval] checkpoint={args.checkpoint}")
    print(f"[eval] amos_root={args.amos_root}")
    print(f"[eval] tta={args.tta}")

    model = build_model(cfg).to(args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[eval] loaded ckpt | missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    cases = find_eligible_cases(Path(args.amos_root), args.max_volumes)
    print(f"[eval] found {len(cases)} eligible AMOS22 val cases (CT, with liver)")

    metrics = VolumetricMetrics()
    per_volume = []
    t_start = time.time()

    for i, (case_id, img_path, lbl_path) in enumerate(cases):
        try:
            img_HWD, lbl_HWD, src_spacing = load_amos22_case(img_path, lbl_path)
        except Exception as e:
            print(f"[eval] ({i+1}/{len(cases)}) {case_id} LOAD FAILED: {e}")
            continue
        if lbl_HWD.sum() == 0:
            print(f"[eval] ({i+1}/{len(cases)}) {case_id} SKIP (no liver)")
            continue

        t0 = time.time()
        img_DHW = np.transpose(img_HWD, (2, 0, 1))
        lbl_DHW = np.transpose(lbl_HWD, (2, 0, 1))
        pred, gt, n_proc = evaluate_case(
            model, img_DHW, lbl_DHW,
            n_slices=args.n_slices, img_size=args.img_size,
            device=args.device, tta_flips=args.tta,
        )

        H_n, W_n = img_HWD.shape[0], img_HWD.shape[1]
        sp_x = float(src_spacing[0]) * (H_n / args.img_size)
        sp_y = float(src_spacing[1]) * (W_n / args.img_size)
        sp_z = float(src_spacing[2])
        spacing_eval = (sp_z, sp_x, sp_y)

        rec = metrics.update(
            pred_vol=pred,
            gt_vol=gt,
            spacing_mm=spacing_eval,
            organ_id=6,
            patient_id=case_id,
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
        print(f"[eval] ({i+1}/{len(cases)}) {case_id} "
              f"D={img_DHW.shape[0]} liver_slices={n_proc} "
              f"dsc={rec['dsc']:.4f} hd95={rec['hd95_mm']:.2f}mm "
              f"nsd={rec['nsd']:.4f} assd={rec.get('assd_mm', float('nan')):.2f}mm "
              f"iou={rec.get('iou', float('nan')):.4f} {dt:.1f}s")

    summary = metrics.summary()
    overall = summary.get("overall", {})
    print("\n=== AMOS22 in-domain results ===")
    print(f"  N volumes: {len(per_volume)}")
    print(f"  Mean DSC : {overall.get('dsc_mean', float('nan')):.4f} "
          f"± {overall.get('dsc_std', float('nan')):.4f}")
    print(f"  Mean HD95: {overall.get('hd95_mm_mean', float('nan')):.2f} "
          f"± {overall.get('hd95_mm_std', float('nan')):.2f} mm")
    print(f"  Mean NSD : {overall.get('nsd_mean', float('nan')):.4f} "
          f"± {overall.get('nsd_std', float('nan')):.4f}")
    print(f"  Total time: {(time.time() - t_start) / 60:.1f} min")

    csv_path = out_dir / ("per_volume_metrics_tta.csv" if args.tta
                          else "per_volume_metrics.csv")
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
    summary_name = "summary_tta.json" if args.tta else "summary.json"
    with open(out_dir / summary_name, "w") as f:
        json.dump({
            "n_volumes": len(per_volume),
            "checkpoint": args.checkpoint,
            "tta": args.tta,
            "n_slices": args.n_slices,
            "img_size": args.img_size,
            "summary": summary,
        }, f, indent=2)
    print(f"[eval] wrote: {csv_path}")
    print(f"[eval] wrote: {out_dir / summary_name}")


if __name__ == "__main__":
    main()
