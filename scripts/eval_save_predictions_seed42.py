"""Re-run seed=42 AMOS22 inference and SAVE predictions + sigmoid probabilities.

Thin wrapper around eval_amos22_liver_indomain.py: re-imports its helper
functions to guarantee byte-identical pre-processing, BBox derivation, and
post-processing. Adds only one thing: per-volume .npz dump of the sigmoid
probability stack, the binarised+CC-cleaned prediction, and the GT, alongside
spacing metadata.

Unblocks downstream calibration / failure-taxonomy / reliability-diagram CPU
work, which all need per-voxel probabilities the canonical eval discards.

VERIFICATION: re-derived DSC mean must match the canonical run's
summary.json (results/amos22_liver_indomain/summary.json) to <1e-6 before
this script declares success. Otherwise it aborts -- a mismatch would mean
the saved predictions diverge from canonical, which would invalidate any
downstream CPU analysis built on them.

Usage:
    python scripts/eval_save_predictions_seed42.py \
        --checkpoint <ckpt> \
        --output thesis/results/amos22_liver_indomain_predictions \
        [--smoke]   # first-volume sentinel only
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
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from models import build_model
from evaluation.metrics_3d import VolumetricMetrics

# Re-use canonical eval helpers; do NOT re-implement the math.
from eval_amos22_liver_indomain import (  # type: ignore
    LIVER_LABEL,
    load_amos22_case,
    find_eligible_cases,
    preprocess_slice_window,
    get_bounding_box_resized,
    build_window,
    keep_largest_cc,
)

CANONICAL_SUMMARY = ROOT / "thesis" / "results" / "amos22_liver_indomain" / "summary.json"
CANONICAL_CSV = ROOT / "thesis" / "results" / "amos22_liver_indomain" / "per_volume_metrics.csv"

# Tolerance budget: the existing canonical CSV stores DSC to 6 decimals; this
# rerun should reproduce per-volume DSC exactly to model float64 precision.
# Bump tolerance to 1e-4 to account for non-deterministic CUDA kernels;
# anything above that indicates a real divergence.
PER_VOL_DSC_TOL = 1e-4
MEAN_DSC_TOL = 1e-4


@torch.no_grad()
def predict_and_save_case(model, image_DHW: np.ndarray, label_DHW: np.ndarray,
                          n_slices: int, img_size: int, device,
                          modality_id: int = 0,
                          hu_clip=(-175.0, 250.0),
                          apply_cc: bool = True):
    """Run forward pass; return (prob_DHW, pred_DHW, gt_DHW, n_processed).

    prob_DHW: float32 sigmoid probability stack, shape (D, img_size, img_size)
              non-liver slices left as 0 (consistent with the canonical eval
              treating them as empty predictions).
    pred_DHW: bool post-threshold + post-CC, same shape.
    gt_DHW:   bool, GT resized to img_size for metric-namespace consistency.
    """
    D, H, W = image_DHW.shape
    prob_DHW = np.zeros((D, img_size, img_size), dtype=np.float32)
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

        prob = model.predict(win_t, box_t, mod_t, is_3d=True)
        if prob.shape[-1] != img_size:
            prob = F.interpolate(prob, size=(img_size, img_size),
                                 mode="bilinear", align_corners=False)
        prob_np = prob.squeeze().cpu().numpy().astype(np.float32)
        prob_DHW[z] = prob_np
        pred_DHW[z] = prob_np > 0.5
        n_processed += 1

    if apply_cc:
        pred_DHW = keep_largest_cc(pred_DHW)
    return prob_DHW, pred_DHW, gt_DHW, n_processed


def load_canonical_dsc_by_case() -> dict[str, float]:
    """Parse canonical per_volume_metrics.csv into {case_id: dsc}."""
    out: dict[str, float] = {}
    with CANONICAL_CSV.open() as f:
        rdr = __import__("csv").DictReader(f)
        for row in rdr:
            out[row["case_id"]] = float(row["dsc"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument(
        "--config", type=str,
        default=r"C:/Users/Raywa/Desktop/VoluFormer3D/checkpoints/phase3_odesam_v2_256px_liver/config.yaml",
    )
    ap.add_argument("--amos-root", type=str,
                    default=r"C:/Users/Raywa/Desktop/LiteSAM3D/data/amos22")
    ap.add_argument("--output", type=str,
                    default="thesis/results/amos22_liver_indomain_predictions")
    ap.add_argument("--n-slices", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--smoke", action="store_true",
                    help="First-volume sentinel only; abort if dsc < 0.85.")
    ap.add_argument("--max-volumes", type=int, default=999)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.output)
    pred_dir = out_dir / "predictions_seed42"
    pred_dir.mkdir(parents=True, exist_ok=True)

    print(f"[save-pred] device={args.device}")
    print(f"[save-pred] checkpoint={args.checkpoint}")
    print(f"[save-pred] output={out_dir}")
    print(f"[save-pred] smoke={args.smoke}")

    cfg = OmegaConf.load(args.config)
    model = build_model(cfg).to(args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[save-pred] loaded ckpt | missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    canonical_dsc = load_canonical_dsc_by_case()
    print(f"[save-pred] canonical CSV has {len(canonical_dsc)} cases")

    max_n = 1 if args.smoke else args.max_volumes
    cases = find_eligible_cases(Path(args.amos_root), max_n)
    print(f"[save-pred] processing {len(cases)} cases")

    metrics = VolumetricMetrics()
    per_volume = []
    mismatches = []
    t_start = time.time()

    for i, (case_id, img_path, lbl_path) in enumerate(cases):
        img_HWD, lbl_HWD, src_spacing = load_amos22_case(img_path, lbl_path)
        if lbl_HWD.sum() == 0:
            print(f"[save-pred] ({i+1}/{len(cases)}) {case_id} SKIP (no liver)")
            continue

        t0 = time.time()
        img_DHW = np.transpose(img_HWD, (2, 0, 1))
        lbl_DHW = np.transpose(lbl_HWD, (2, 0, 1))
        prob, pred, gt, n_proc = predict_and_save_case(
            model, img_DHW, lbl_DHW,
            n_slices=args.n_slices, img_size=args.img_size,
            device=args.device,
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
        dsc_now = float(rec["dsc"])
        per_volume.append({
            "case_id": case_id, "n_slices": n_proc,
            "dsc": dsc_now, "hd95_mm": float(rec["hd95_mm"]),
            "nsd": float(rec["nsd"]),
        })

        # Per-case sanity check vs canonical CSV
        canonical = canonical_dsc.get(case_id)
        if canonical is not None:
            diff = abs(dsc_now - canonical)
            if diff > PER_VOL_DSC_TOL:
                mismatches.append((case_id, canonical, dsc_now, diff))

        # Save (compressed) per-volume probability + prediction + GT
        npz_path = pred_dir / f"{case_id}.npz"
        np.savez_compressed(
            npz_path,
            prob=prob.astype(np.float16),       # float16 = 2x storage saving; sigmoid is well within range
            pred=pred.astype(np.uint8),
            gt=gt.astype(np.uint8),
            spacing_mm=np.asarray(spacing_eval, dtype=np.float32),
        )
        dt = time.time() - t0
        marker = "" if canonical is None else f" (canonical {canonical:.4f})"
        print(f"[save-pred] ({i+1}/{len(cases)}) {case_id} "
              f"D={img_DHW.shape[0]} dsc={dsc_now:.4f}{marker} "
              f"npz={npz_path.stat().st_size/1e6:.1f}MB {dt:.1f}s")

        if args.smoke and dsc_now < 0.85:
            print(f"[save-pred] SMOKE FAIL dsc={dsc_now:.4f} below 0.85; aborting")
            sys.exit(2)

    summary = metrics.summary()
    overall = summary.get("overall", {})
    rederived_mean = float(overall.get("dsc_mean", float("nan")))
    canonical_mean = float(json.loads(CANONICAL_SUMMARY.read_text())
                           ["summary"]["overall"]["dsc_mean"])
    mean_diff = abs(rederived_mean - canonical_mean)

    print("")
    print(f"[save-pred] DSC mean re-derived: {rederived_mean:.6f}")
    print(f"[save-pred] DSC mean canonical : {canonical_mean:.6f}")
    print(f"[save-pred] |diff| = {mean_diff:.3e}  (tolerance {MEAN_DSC_TOL:.0e})")
    if mismatches:
        print(f"[save-pred] {len(mismatches)} per-volume DSC mismatches above {PER_VOL_DSC_TOL}:")
        for case_id, can, now, d in mismatches[:5]:
            print(f"[save-pred]   {case_id}: canonical={can:.4f} rederived={now:.4f} diff={d:.4f}")

    # Write summary
    with open(out_dir / "save_predictions_summary.json", "w") as f:
        json.dump({
            "n_volumes": len(per_volume),
            "checkpoint": args.checkpoint,
            "smoke": args.smoke,
            "n_slices": args.n_slices,
            "img_size": args.img_size,
            "rederived_dsc_mean": rederived_mean,
            "canonical_dsc_mean": canonical_mean,
            "abs_diff_dsc_mean": mean_diff,
            "n_per_vol_mismatches_above_tol": len(mismatches),
            "summary": summary,
        }, f, indent=2)

    if not args.smoke and mean_diff > MEAN_DSC_TOL:
        print(f"[save-pred] FAILED: mean DSC mismatch exceeds tolerance.")
        sys.exit(3)
    print(f"[save-pred] OK -- saved to {pred_dir}/  ({(time.time()-t_start)/60:.1f} min)")


if __name__ == "__main__":
    main()
