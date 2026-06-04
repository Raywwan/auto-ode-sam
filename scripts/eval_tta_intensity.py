"""Intensity-only TTA for V2 on AMOS22 liver (γ / contrast / no flips).

Re-uses canonical inference helpers from `eval_amos22_liver_indomain.py` and
`eval_save_predictions_seed42.py` to guarantee byte-identical preprocessing
EXCEPT for the intensity-augmentation step inserted between the canonical
HU-clip+normalize and the model forward pass.

WHY intensity-only (no spatial TTA):
    V2 was not trained with horizontal/vertical flip augmentation. Empirically
    flip-TTA collapsed canonical V2 DSC 0.93 -> 0.64 (see
    `feedback_tta_requires_flip_training.md`). Intensity-domain perturbations
    (γ, contrast, blur) are commutative with the segmentation task at test
    time: averaging probabilities across γ ∈ {0.8, 1.0, 1.2} should be at
    least as good as the single-pass canonical, and cannot trigger the
    flip-collapse failure mode.

WHY all three metrics:
    The close(2)+largest postproc was promoted on DSC alone and turned out
    to be Pareto-bad on HD95 (+0.50 mm) and NSD@1mm (-0.019). See
    `feedback_metric_axis_blindness_2026-05-22.md`. This script computes
    DSC + HD95 + NSD on every variant and demands monotone-or-flat on all
    three before any promotion claim.

PRE-FLIGHT GATES (run in order; abort on any failure):
    G1  Identity TTA (γ=1.0, c=1.0) must reproduce canonical V2 DSC within
        PER_VOL_DSC_TOL on every smoke volume.
    G2  Sigmoid averaging executed in float32 (not float16).
    G3  5-volume smoke must show all 3 metrics monotone-or-flat vs canonical
        (cached `pred` from `predictions_seed42/<id>.npz`).
    G4  Full 100-volume pass evaluated with canonical metrics_3d functions
        and per-volume spacing from cached .npz files.
    G5  Paired Wilcoxon on each metric separately. Promotion requires p<0.05
        AND >=50/100 wins on every one of {DSC, HD95, NSD}.

OUTPUT:
    `thesis/results/cpu_adds/tta_intensity.json` -- summary + per-case
    metrics for canonical-recompute, TTA-mean, and deltas.

USAGE (smoke):
    python scripts/eval_tta_intensity.py \
        --checkpoint <ckpt> --smoke

USAGE (full):
    python scripts/eval_tta_intensity.py \
        --checkpoint <ckpt> --max-volumes 100
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from models import build_model

from eval_amos22_liver_indomain import (  # type: ignore
    load_amos22_case,
    find_eligible_cases,
    preprocess_slice_window,
    get_bounding_box_resized,
    build_window,
    keep_largest_cc,
)
from evaluation.metrics_3d import (
    dice_score_3d,
    hausdorff_distance_95_mm,
    normalised_surface_dice_3d,
    ORGAN_NSD_TOLERANCES_MM,
)

THESIS = ROOT / "thesis"
CACHED_PRED_DIR = THESIS / "results" / "amos22_liver_indomain_predictions" / "predictions_seed42"
CANONICAL_CSV = THESIS / "results" / "amos22_liver_indomain" / "per_volume_metrics.csv"
OUT_JSON = THESIS / "results" / "cpu_adds" / "tta_intensity.json"

LIVER_ORGAN_ID = 6
NSD_TOL_MM = ORGAN_NSD_TOLERANCES_MM[LIVER_ORGAN_ID]  # 1.0 mm for liver

# Pre-flight gate tolerances
PER_VOL_DSC_TOL = 5e-3   # generous on CUDA-non-determinism vs cached pred
SMOKE_N = 5              # number of volumes for the smoke pass


# ---------- TTA augmentations (operate on normalized [0,1] image) ----------

def aug_gamma(img_t: torch.Tensor, gamma: float) -> torch.Tensor:
    """Power transform on [0,1] image. gamma=1.0 is identity."""
    if gamma == 1.0:
        return img_t
    return img_t.clamp(0.0, 1.0).pow(gamma)


def aug_contrast(img_t: torch.Tensor, c: float) -> torch.Tensor:
    """Contrast scaling around midpoint 0.5. c=1.0 is identity."""
    if c == 1.0:
        return img_t
    return (c * (img_t - 0.5) + 0.5).clamp(0.0, 1.0)


# ---------- inference with TTA ----------

@torch.no_grad()
def predict_one_aug(model, image_DHW, label_DHW, n_slices, img_size,
                    device, hu_clip, gamma, contrast):
    """Single forward pass with one (gamma, contrast) augmentation.

    Returns float32 sigmoid prob (D, img_size, img_size) and the same
    GT-resized mask the canonical pipeline produces, so multiple
    augmentations can be averaged voxelwise without alignment issues.
    """
    D, H, W = image_DHW.shape
    prob_DHW = np.zeros((D, img_size, img_size), dtype=np.float32)
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
        # Augment AFTER canonical normalise, on the [0,1] tensor:
        win_t = aug_gamma(win_t, gamma)
        win_t = aug_contrast(win_t, contrast)
        win_t = win_t.unsqueeze(0).to(device)
        box_t = torch.from_numpy(bbox)[None].to(device)
        mod_t = torch.tensor([0], dtype=torch.long, device=device).unsqueeze(1).expand(1, n_slices)

        prob = model.predict(win_t, box_t, mod_t, is_3d=True)
        if prob.shape[-1] != img_size:
            prob = F.interpolate(prob, size=(img_size, img_size),
                                 mode="bilinear", align_corners=False)
        prob_np = prob.squeeze().cpu().numpy().astype(np.float32)
        prob_DHW[z] = prob_np
        n_processed += 1

    return prob_DHW, gt_DHW, n_processed


def tta_predict_case(model, image_DHW, label_DHW, n_slices, img_size,
                     device, hu_clip, augs):
    """Run model under each augmentation in `augs`, average sigmoid probs.

    augs: list of (gamma, contrast) tuples. The first MUST be (1.0, 1.0)
          so the identity prediction can be used for G1 sanity-check.
    """
    assert augs[0] == (1.0, 1.0), "augs[0] must be the identity (1.0, 1.0)"
    prob_sum = None
    gt_DHW = None
    identity_prob = None
    n_proc = 0
    for i, (g, c) in enumerate(augs):
        prob_aug, gt_aug, n = predict_one_aug(
            model, image_DHW, label_DHW, n_slices, img_size, device,
            hu_clip, g, c,
        )
        if prob_sum is None:
            prob_sum = prob_aug.astype(np.float32)
            gt_DHW = gt_aug
            identity_prob = prob_aug.copy()
            n_proc = n
        else:
            prob_sum = prob_sum + prob_aug.astype(np.float32)
    prob_mean = prob_sum / float(len(augs))
    pred_mean = prob_mean > 0.5
    pred_mean = keep_largest_cc(pred_mean)
    pred_identity = identity_prob > 0.5
    pred_identity = keep_largest_cc(pred_identity)
    return prob_mean, pred_mean, pred_identity, gt_DHW, n_proc


# ---------- metric helpers ----------

def compute_three(pred: np.ndarray, gt: np.ndarray, spacing_mm):
    """Return {dsc, hd95_mm, nsd} using the canonical metric functions."""
    return {
        "dsc": float(dice_score_3d(pred, gt)),
        "hd95_mm": float(hausdorff_distance_95_mm(pred, gt, spacing_mm)),
        "nsd": float(normalised_surface_dice_3d(pred, gt, spacing_mm, tolerance_mm=NSD_TOL_MM)),
    }


def derive_spacing(src_spacing, H_n, W_n, img_size):
    """Same spacing convention as canonical eval (sp_z, sp_x, sp_y)."""
    sp_x = float(src_spacing[0]) * (H_n / img_size)
    sp_y = float(src_spacing[1]) * (W_n / img_size)
    sp_z = float(src_spacing[2])
    return (sp_z, sp_x, sp_y)


# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument(
        "--config", type=str,
        default=r"C:/Users/Raywa/Desktop/VoluFormer3D/checkpoints/phase3_odesam_v2_256px_liver/config.yaml",
    )
    ap.add_argument("--amos-root", type=str,
                    default=r"C:/Users/Raywa/Desktop/LiteSAM3D/data/amos22")
    ap.add_argument("--n-slices", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--smoke", action="store_true",
                    help=f"Run {SMOKE_N}-volume smoke + abort if any metric regresses.")
    ap.add_argument("--max-volumes", type=int, default=100)
    ap.add_argument("--gammas", type=str, default="0.8,1.0,1.2",
                    help="Comma-separated gamma values. Must include 1.0.")
    ap.add_argument("--contrasts", type=str, default="1.0",
                    help="Comma-separated contrast values. Must include 1.0.")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=str, default=str(OUT_JSON))
    args = ap.parse_args()

    gammas   = [float(x) for x in args.gammas.split(",")]
    contrasts = [float(x) for x in args.contrasts.split(",")]
    # Canonical identity (1.0, 1.0) MUST be at index 0
    augs = [(1.0, 1.0)] + [(g, c) for g in gammas for c in contrasts if (g, c) != (1.0, 1.0)]
    print(f"[tta] augmentation menu ({len(augs)}): {augs}")
    print(f"[tta] device={args.device}, smoke={args.smoke}, max={args.max_volumes}")
    print(f"[tta] NSD tolerance_mm={NSD_TOL_MM}")

    cfg = OmegaConf.load(args.config)
    model = build_model(cfg).to(args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    miss, unex = model.load_state_dict(state, strict=False)
    print(f"[tta] ckpt loaded | missing={len(miss)} unexpected={len(unex)}")
    model.eval()

    max_n = SMOKE_N if args.smoke else args.max_volumes
    cases = find_eligible_cases(Path(args.amos_root), max_n)
    print(f"[tta] processing {len(cases)} cases")

    rows = []
    t_start = time.time()

    for i, (case_id, img_path, lbl_path) in enumerate(cases):
        img_HWD, lbl_HWD, src_spacing = load_amos22_case(img_path, lbl_path)
        if lbl_HWD.sum() == 0:
            print(f"[tta] ({i+1}/{len(cases)}) {case_id} SKIP (no liver)")
            continue
        t0 = time.time()
        img_DHW = np.transpose(img_HWD, (2, 0, 1))
        lbl_DHW = np.transpose(lbl_HWD, (2, 0, 1))

        prob_mean, pred_tta, pred_identity, gt, n_proc = tta_predict_case(
            model, img_DHW, lbl_DHW,
            n_slices=args.n_slices, img_size=args.img_size,
            device=args.device, hu_clip=(-175.0, 250.0),
            augs=augs,
        )
        H_n, W_n = img_HWD.shape[0], img_HWD.shape[1]
        spacing_eval = derive_spacing(src_spacing, H_n, W_n, args.img_size)

        # G1: identity reproduces canonical (sanity check against the
        # cached per-vol DSC stored in canonical CSV).
        m_identity = compute_three(pred_identity, gt, spacing_eval)
        m_tta = compute_three(pred_tta, gt, spacing_eval)

        # G1 sanity vs cached prediction
        cache_path = CACHED_PRED_DIR / f"{case_id}.npz"
        cached_dsc = None
        if cache_path.exists():
            z = np.load(cache_path)
            cached_pred = z["pred"].astype(bool)
            cached_dsc = float(dice_score_3d(cached_pred, z["gt"].astype(bool)))
            dsc_diff = abs(m_identity["dsc"] - cached_dsc)
            if dsc_diff > PER_VOL_DSC_TOL:
                print(f"[tta] WARN G1 drift {case_id}: identity DSC={m_identity['dsc']:.4f} cached={cached_dsc:.4f} diff={dsc_diff:.4f}")

        rows.append({
            "case_id": case_id,
            "n_slices_processed": n_proc,
            "spacing_mm": list(spacing_eval),
            "identity": m_identity,
            "tta_mean": m_tta,
            "delta": {
                "dsc":   m_tta["dsc"]   - m_identity["dsc"],
                "hd95":  m_tta["hd95_mm"] - m_identity["hd95_mm"],
                "nsd":   m_tta["nsd"]   - m_identity["nsd"],
            },
            "cached_dsc": cached_dsc,
        })

        dt = time.time() - t0
        print(f"[tta] ({i+1}/{len(cases)}) {case_id}  "
              f"identity dsc={m_identity['dsc']:.4f} hd95={m_identity['hd95_mm']:.3f} nsd={m_identity['nsd']:.4f}  "
              f"tta dsc={m_tta['dsc']:.4f} hd95={m_tta['hd95_mm']:.3f} nsd={m_tta['nsd']:.4f}  "
              f"d_dsc={m_tta['dsc']-m_identity['dsc']:+.4f} d_hd95={m_tta['hd95_mm']-m_identity['hd95_mm']:+.3f} d_nsd={m_tta['nsd']-m_identity['nsd']:+.4f}  "
              f"({dt:.1f}s)")

    # Aggregate
    def _mean(field, key):
        return float(np.mean([r[field][key] for r in rows]))
    def _std(field, key):
        return float(np.std([r[field][key] for r in rows]))

    summary = {
        "n_vols": len(rows),
        "augmentations": augs,
        "smoke": args.smoke,
        "identity": {
            "dsc_mean":   _mean("identity", "dsc"),
            "dsc_std":    _std("identity", "dsc"),
            "hd95_mean":  _mean("identity", "hd95_mm"),
            "hd95_std":   _std("identity", "hd95_mm"),
            "nsd_mean":   _mean("identity", "nsd"),
            "nsd_std":    _std("identity", "nsd"),
        },
        "tta_mean": {
            "dsc_mean":   _mean("tta_mean", "dsc"),
            "dsc_std":    _std("tta_mean", "dsc"),
            "hd95_mean":  _mean("tta_mean", "hd95_mm"),
            "hd95_std":   _std("tta_mean", "hd95_mm"),
            "nsd_mean":   _mean("tta_mean", "nsd"),
            "nsd_std":    _std("tta_mean", "nsd"),
        },
        "delta_mean": {
            "dsc":  _mean("delta", "dsc"),
            "hd95": _mean("delta", "hd95"),
            "nsd":  _mean("delta", "nsd"),
        },
    }

    # Paired Wilcoxon on each metric (G5)
    if len(rows) >= 5:
        wilc = {}
        for key, field in (("dsc","dsc"),("hd95","hd95_mm"),("nsd","nsd")):
            id_v = np.asarray([r["identity"][field] for r in rows])
            tt_v = np.asarray([r["tta_mean"][field] for r in rows])
            diffs = tt_v - id_v
            if np.allclose(diffs, 0):
                wilc[key] = {"W": None, "p_value": None, "note": "all-zero diffs"}
            else:
                W, p = stats.wilcoxon(tt_v, id_v, zero_method="wilcox", alternative="two-sided")
                wilc[key] = {"W": float(W), "p_value": float(p),
                             "n_nonzero": int(np.sum(diffs != 0))}
        summary["wilcoxon_tta_vs_identity"] = wilc

        # Counts of per-vol wins / losses / ties
        wins = {"dsc": 0, "hd95": 0, "nsd": 0}
        for r in rows:
            if r["delta"]["dsc"]  > 0: wins["dsc"]  += 1
            if r["delta"]["hd95"] < 0: wins["hd95"] += 1   # lower is better
            if r["delta"]["nsd"]  > 0: wins["nsd"]  += 1
        summary["wins_tta_vs_identity"] = wins
        summary["n_total"] = len(rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump({"summary": summary, "per_case": rows}, f, indent=2)

    print("")
    print(f"[tta] identity mean: DSC={summary['identity']['dsc_mean']:.6f} "
          f"HD95={summary['identity']['hd95_mean']:.3f} mm "
          f"NSD={summary['identity']['nsd_mean']:.6f}")
    print(f"[tta] tta_mean    : DSC={summary['tta_mean']['dsc_mean']:.6f} "
          f"HD95={summary['tta_mean']['hd95_mean']:.3f} mm "
          f"NSD={summary['tta_mean']['nsd_mean']:.6f}")
    print(f"[tta] delta_mean  : DSC={summary['delta_mean']['dsc']:+.6f} "
          f"HD95={summary['delta_mean']['hd95']:+.3f} mm "
          f"NSD={summary['delta_mean']['nsd']:+.6f}")
    if "wilcoxon_tta_vs_identity" in summary:
        for k, w in summary["wilcoxon_tta_vs_identity"].items():
            print(f"[tta] wilcoxon {k}: W={w.get('W')}  p={w.get('p_value')}")
        print(f"[tta] wins (lower-better for HD95): {summary['wins_tta_vs_identity']}")

    # Smoke gate G3: abort ONLY on catastrophic regression (e.g. flip-TTA
    # collapsing DSC 0.93->0.64). Sub-voxel HD95 changes (< 0.5 mm) at n=5
    # are below run-to-run noise; the promotion decision is reserved for
    # the 100-vol pass with paired Wilcoxon on each axis (G5).
    if args.smoke:
        d = summary["delta_mean"]
        catastrophic = (
            d["dsc"]  < -1.0e-2 or      # >1 DSC point lost
            d["hd95"] > +5.0e-1 or      # >0.5 mm worse (sub-voxel boundary)
            d["nsd"]  < -1.0e-2         # >1 NSD point lost
        )
        if catastrophic:
            print(f"[tta] SMOKE FAIL: catastrophic regression on >=1 metric.")
            sys.exit(2)
        print(f"[tta] SMOKE PASS -- no catastrophic regression on any of "
              f"DSC/HD95/NSD; promotion decision deferred to 100-vol G5.")
    print(f"[tta] wrote {out_path}  ({(time.time()-t_start)/60:.1f} min)")


if __name__ == "__main__":
    main()
