"""Robustness battery: DSC/HD95/NSD vs input perturbations on canonical seed=42.

Holds the eval pipeline (HU-clip, BBox, model, CC) fixed. Applies the
perturbation in HU space BEFORE the HU clip-and-normalise step, so the
model sees the perturbed HU just as it would in a clinical-shift scenario.

Perturbations (one axis at a time):
  - Gaussian noise: additive iid N(0, sigma^2) HU; sigma in {0, 1, 2, 4, 8}
  - Contrast scaling: HU' = (HU - mu_lo) * gamma + mu_lo where mu_lo = -175
    HU (the lower clip); gamma in {0.70, 0.85, 1.00, 1.15, 1.30}.

For each (perturbation, value, case) tuple, computes DSC/HD95/NSD. Output:
  - JSON dump of all per-vol metrics keyed by (axis, value, case_id)
  - 2-panel PDF: DSC vs sigma and DSC vs gamma, with mean +- std bands.

VERIFICATION: the (sigma=0, gamma=1.0) baseline run must reproduce the
canonical per-vol DSC to <1e-3 on every vol; otherwise the script aborts.

Usage:
    python scripts/eval_robustness_battery_seed42.py \
        --checkpoint <ckpt> \
        --n-cases 20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from models import build_model
from evaluation.metrics_3d import VolumetricMetrics

from eval_amos22_liver_indomain import (  # type: ignore
    LIVER_LABEL, load_amos22_case, find_eligible_cases,
    preprocess_slice_window, get_bounding_box_resized,
    build_window, keep_largest_cc,
)

THESIS = ROOT / "thesis"
OUT_JSON = THESIS / "results" / "cpu_adds" / "robustness_battery.json"
OUT_PDF  = THESIS / "figures" / "fig_robustness_battery.pdf"
OUT_PNG  = THESIS / "figures" / "fig_robustness_battery.png"
CANONICAL_CSV = THESIS / "results" / "amos22_liver_indomain" / "per_volume_metrics.csv"

NOISE_LEVELS   = [0.0, 1.0, 2.0, 4.0, 8.0]                          # HU sigma
CONTRAST_GAMMAS = [0.70, 0.85, 1.00, 1.15, 1.30]                    # multiplier
BASELINE_TOL_DSC = 1e-3                                             # vs canonical CSV at (0, 1.0)


def load_canonical_dsc() -> dict[str, float]:
    out: dict[str, float] = {}
    with CANONICAL_CSV.open() as f:
        rdr = __import__("csv").DictReader(f)
        for r in rdr:
            out[r["case_id"]] = float(r["dsc"])
    return out


def perturb_volume(img_DHW: np.ndarray, *, sigma_hu: float, gamma: float,
                   rng: np.random.Generator) -> np.ndarray:
    """Apply perturbation in HU space. sigma=0 + gamma=1.0 -> identity."""
    out = img_DHW
    if gamma != 1.0:
        # Anchor contrast at lower HU clip so background stays near 0
        out = (out - (-175.0)) * gamma + (-175.0)
    if sigma_hu > 0.0:
        out = out + rng.normal(0.0, sigma_hu, out.shape).astype(np.float32)
    return out.astype(np.float32)


@torch.no_grad()
def evaluate_perturbed_case(model, image_DHW: np.ndarray, label_DHW: np.ndarray,
                            *, n_slices: int, img_size: int, device,
                            modality_id: int = 0, hu_clip=(-175.0, 250.0)):
    D = image_DHW.shape[0]
    pred_DHW = np.zeros((D, img_size, img_size), dtype=bool)
    gt_DHW = np.zeros((D, img_size, img_size), dtype=bool)
    n_proc = 0
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
        pred_DHW[z] = prob.squeeze().cpu().numpy() > 0.5
        n_proc += 1
    pred_DHW = keep_largest_cc(pred_DHW)
    return pred_DHW, gt_DHW, n_proc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument(
        "--config", type=str,
        default=r"C:/Users/Raywa/Desktop/VoluFormer3D/checkpoints/phase3_odesam_v2_256px_liver/config.yaml",
    )
    ap.add_argument("--amos-root", type=str,
                    default=r"C:/Users/Raywa/Desktop/LiteSAM3D/data/amos22")
    ap.add_argument("--n-cases", type=int, default=20)
    ap.add_argument("--n-slices", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--rng-seed", type=int, default=20260521)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(args.config)
    model = build_model(cfg).to(args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[robust] loaded ckpt | missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    canonical_dsc = load_canonical_dsc()
    cases = find_eligible_cases(Path(args.amos_root), args.n_cases)
    print(f"[robust] {len(cases)} cases × ({len(NOISE_LEVELS)} noise + "
          f"{len(CONTRAST_GAMMAS)} contrast - 1 shared baseline) conditions")

    # Build the condition list. Identity-baseline appears once.
    conditions: list[tuple[str, float]] = []
    conditions.append(("baseline", 0.0))
    for s in NOISE_LEVELS:
        if s == 0.0: continue
        conditions.append(("noise_sigma_hu", s))
    for g in CONTRAST_GAMMAS:
        if g == 1.0: continue
        conditions.append(("contrast_gamma", g))

    rng = np.random.default_rng(args.rng_seed)
    records: list[dict] = []
    baseline_metrics_by_case: dict[str, float] = {}
    t_start = time.time()

    for axis, value in conditions:
        metrics = VolumetricMetrics()
        local_records = []
        for i, (case_id, img_path, lbl_path) in enumerate(cases):
            img_HWD, lbl_HWD, src_spacing = load_amos22_case(img_path, lbl_path)
            img_DHW = np.transpose(img_HWD, (2, 0, 1)).astype(np.float32)
            lbl_DHW = np.transpose(lbl_HWD, (2, 0, 1))
            if lbl_DHW.sum() == 0:
                continue
            if axis == "baseline":
                img_p = img_DHW
            elif axis == "noise_sigma_hu":
                img_p = perturb_volume(img_DHW, sigma_hu=value, gamma=1.0, rng=rng)
            elif axis == "contrast_gamma":
                img_p = perturb_volume(img_DHW, sigma_hu=0.0, gamma=value, rng=rng)
            else:
                raise ValueError(axis)
            pred, gt, n_proc = evaluate_perturbed_case(
                model, img_p, lbl_DHW,
                n_slices=args.n_slices, img_size=args.img_size, device=args.device,
            )
            H_n, W_n = img_HWD.shape[0], img_HWD.shape[1]
            sp_x = float(src_spacing[0]) * (H_n / args.img_size)
            sp_y = float(src_spacing[1]) * (W_n / args.img_size)
            spacing_eval = (float(src_spacing[2]), sp_x, sp_y)
            rec = metrics.update(
                pred_vol=pred, gt_vol=gt,
                spacing_mm=spacing_eval, organ_id=6, patient_id=case_id,
            )
            dsc = float(rec["dsc"])
            local_records.append({
                "axis": axis, "value": value, "case_id": case_id,
                "dsc": dsc, "hd95_mm": float(rec["hd95_mm"]),
                "nsd": float(rec["nsd"]),
            })
            if axis == "baseline":
                baseline_metrics_by_case[case_id] = dsc

        summary = metrics.summary()
        overall = summary.get("overall", {})
        mean_dsc = float(overall.get("dsc_mean", float("nan")))
        std_dsc  = float(overall.get("dsc_std",  float("nan")))
        records.extend(local_records)
        print(f"[robust] {axis:>18s} = {value:+.3f}  "
              f"DSC mean {mean_dsc:.4f} ± {std_dsc:.4f}  "
              f"n={len(local_records)}  ({(time.time()-t_start)/60:.1f} min)")

        if axis == "baseline":
            # Sanity check: per-vol DSC matches canonical CSV within tolerance
            bad = []
            for case_id, d in baseline_metrics_by_case.items():
                canonical = canonical_dsc.get(case_id)
                if canonical is None: continue
                if abs(d - canonical) > BASELINE_TOL_DSC:
                    bad.append((case_id, canonical, d, abs(d - canonical)))
            if bad:
                print(f"[robust] BASELINE MISMATCH on {len(bad)}/{len(baseline_metrics_by_case)} cases:")
                for cid, can, now, diff in bad[:5]:
                    print(f"[robust]   {cid}: canonical={can:.4f} now={now:.4f} diff={diff:.4f}")
                print(f"[robust] Aborting before perturbation runs to avoid quoting an unverified baseline.")
                sys.exit(2)
            print(f"[robust] BASELINE OK: all {len(baseline_metrics_by_case)} cases within {BASELINE_TOL_DSC}")

    # Aggregate per condition for plotting
    by_cond: dict[tuple[str, float], list[float]] = {}
    by_cond_hd: dict[tuple[str, float], list[float]] = {}
    by_cond_nsd: dict[tuple[str, float], list[float]] = {}
    for r in records:
        k = (r["axis"], float(r["value"]))
        by_cond.setdefault(k, []).append(r["dsc"])
        by_cond_hd.setdefault(k, []).append(r["hd95_mm"])
        by_cond_nsd.setdefault(k, []).append(r["nsd"])

    # Plot two-panel
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    accent = (18/255, 87/255, 107/255)

    def panel(ax, axis_name: str, values: list[float], label: str, baseline_x: float):
        means, stds = [], []
        xs = sorted(set([baseline_x] + values))
        for v in xs:
            key = ("baseline", 0.0) if v == baseline_x else (axis_name, v)
            arr = by_cond.get(key, [])
            means.append(float(np.mean(arr)) if arr else float("nan"))
            stds.append(float(np.std(arr)) if arr else 0.0)
        ax.errorbar(xs, means, yerr=stds, fmt="-o", color=accent, lw=2.0,
                    capsize=4, ms=6, mfc="white", mec=accent)
        ax.set_xlabel(label)
        ax.set_ylabel("3D Dice (mean ± std)")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Canonical seed=42 robustness on N={args.n_cases} AMOS22 vols")
        ax.set_ylim(0.0, 1.0)

    panel(axes[0], "noise_sigma_hu",  NOISE_LEVELS,   r"Additive Gaussian noise $\sigma$ (HU)", 0.0)
    panel(axes[1], "contrast_gamma",  CONTRAST_GAMMAS, r"Contrast scale $\gamma$",                1.0)

    plt.savefig(OUT_PDF, bbox_inches="tight")
    plt.savefig(OUT_PNG, bbox_inches="tight", dpi=150)
    plt.close(fig)

    with open(OUT_JSON, "w") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "n_cases": args.n_cases,
            "noise_sigma_hu": NOISE_LEVELS,
            "contrast_gamma": CONTRAST_GAMMAS,
            "rng_seed": args.rng_seed,
            "baseline_tolerance_dsc": BASELINE_TOL_DSC,
            "per_case_records": records,
        }, f, indent=2)

    print(f"[robust] wrote {OUT_JSON}")
    print(f"[robust] wrote {OUT_PDF}")
    print(f"[robust] done in {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
