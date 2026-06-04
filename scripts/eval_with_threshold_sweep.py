"""
eval_with_threshold_sweep.py — Internal threshold sweep wrapper.

Runs inference ONCE per volume, caches the per-slice sigmoid prob map,
then sweeps thresholds [thr_lo .. thr_hi step thr_step] and computes
DSC/HD95/NSD at each threshold.

Reuses the data-loading + preprocessing + model-build path from
scripts/eval_amos22_liver_indomain.py to guarantee bit-for-bit pipeline
parity with the existing eval.

Memory: per-volume probs are kept as float16 (~26 MB each) and freed
after that volume's full threshold sweep — never holds all 100 vols at
once.

Output: <out>/per_threshold_summary.csv  (cols: threshold, n_vols, dsc_mean, dsc_std, hd95_mean, nsd_mean)
        <out>/per_threshold_per_volume.csv  (long-format)

Usage:
  python scripts/eval_with_threshold_sweep.py \
      --checkpoint checkpoints/.../best.pt \
      --config configs/phase3_odesam_v2_seed43.yaml \
      --output thesis/results/threshold_sweep/seed43 \
      --img-size 256 --n-slices 8 \
      --thr-lo 0.30 --thr-hi 0.70 --thr-step 0.025
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

# Make the project root importable for `models`, `voluformer3d`, etc.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# Reuse everything from the canonical eval
from eval_amos22_liver_indomain import (  # type: ignore  # noqa: E402
    build_model,
    find_eligible_cases,
    load_amos22_case,
    preprocess_slice_window,
    build_window,
    get_bounding_box_resized,
    keep_largest_cc,
    VolumetricMetrics,
)


@torch.no_grad()
def compute_probs_DHW(model, image_DHW: np.ndarray, label_DHW: np.ndarray,
                     n_slices: int, img_size: int, device,
                     modality_id: int = 0, hu_clip=(-175.0, 250.0)):
    """Run inference once. Return:
        probs_DHW: float16 (D, img_size, img_size) — slice-level sigmoid probs (0 where skipped)
        gt_DHW:    bool    (D, img_size, img_size)
        liver_z:   list of z-indices that were processed
    """
    D, H, W = image_DHW.shape
    probs_DHW = np.zeros((D, img_size, img_size), dtype=np.float16)
    gt_DHW = np.zeros((D, img_size, img_size), dtype=bool)
    liver_z = []

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
        probs_DHW[z] = prob.squeeze().cpu().numpy().astype(np.float16)
        liver_z.append(z)

    return probs_DHW, gt_DHW, liver_z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--amos-root", type=str,
                    default=r"C:/Users/Raywa/Desktop/LiteSAM3D/data/amos22")
    ap.add_argument("--max-volumes", type=int, default=999)
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--n-slices", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--thr-lo", type=float, default=0.30)
    ap.add_argument("--thr-hi", type=float, default=0.70)
    ap.add_argument("--thr-step", type=float, default=0.025)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(args.config)
    print(f"[sweep] device={args.device}")
    print(f"[sweep] checkpoint={args.checkpoint}")
    print(f"[sweep] amos_root={args.amos_root}")
    print(f"[sweep] thresholds: [{args.thr_lo:.3f} .. {args.thr_hi:.3f}] step {args.thr_step:.4f}")

    thresholds = np.arange(args.thr_lo, args.thr_hi + 1e-9, args.thr_step)
    print(f"[sweep] {len(thresholds)} thresholds")

    model = build_model(cfg).to(args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[sweep] loaded ckpt | missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    cases = find_eligible_cases(Path(args.amos_root), args.max_volumes)
    print(f"[sweep] found {len(cases)} eligible AMOS22 val cases")

    # One VolumetricMetrics per threshold
    metrics_by_thr = {round(float(t), 4): VolumetricMetrics() for t in thresholds}
    per_volume_long = []   # (case_id, threshold, dsc, hd95, nsd)
    t_start = time.time()

    for i, (case_id, img_path, lbl_path) in enumerate(cases):
        try:
            img_HWD, lbl_HWD, src_spacing = load_amos22_case(img_path, lbl_path)
        except Exception as e:
            print(f"[sweep] ({i+1}/{len(cases)}) {case_id} LOAD FAILED: {e}")
            continue
        if lbl_HWD.sum() == 0:
            print(f"[sweep] ({i+1}/{len(cases)}) {case_id} SKIP (no liver)")
            continue

        img_DHW = np.transpose(img_HWD, (2, 0, 1))
        lbl_DHW = np.transpose(lbl_HWD, (2, 0, 1))
        t0 = time.time()
        probs_DHW, gt_DHW, liver_z = compute_probs_DHW(
            model, img_DHW, lbl_DHW,
            n_slices=args.n_slices, img_size=args.img_size,
            device=args.device,
        )

        H_n, W_n = img_HWD.shape[0], img_HWD.shape[1]
        sp_x = float(src_spacing[0]) * (H_n / args.img_size)
        sp_y = float(src_spacing[1]) * (W_n / args.img_size)
        sp_z = float(src_spacing[2])
        spacing_eval = (sp_z, sp_x, sp_y)

        # Sweep thresholds (per-volume — no all-vols-in-memory)
        probs_f32 = probs_DHW.astype(np.float32)
        for t in thresholds:
            t_key = round(float(t), 4)
            pred = probs_f32 > t
            pred = keep_largest_cc(pred)
            rec = metrics_by_thr[t_key].update(
                pred_vol=pred,
                gt_vol=gt_DHW,
                spacing_mm=spacing_eval,
                organ_id=6,
                patient_id=case_id,
            )
            per_volume_long.append({
                "case_id": case_id, "threshold": t_key,
                "dsc": float(rec["dsc"]),
                "hd95_mm": float(rec["hd95_mm"]),
                "nsd": float(rec["nsd"]),
            })
        dt = time.time() - t0
        print(f"[sweep] ({i+1}/{len(cases)}) {case_id} "
              f"D={img_DHW.shape[0]} liver_slices={len(liver_z)} {dt:.1f}s")

    print("\n=== Threshold sweep summary ===")
    summary_rows = []
    for t in thresholds:
        t_key = round(float(t), 4)
        s = metrics_by_thr[t_key].summary().get("overall", {})
        n = len(metrics_by_thr[t_key].records)
        summary_rows.append({
            "threshold": t_key,
            "n_vols": n,
            "dsc_mean": s.get("dsc_mean", float("nan")),
            "dsc_std":  s.get("dsc_std",  float("nan")),
            "hd95_mean": s.get("hd95_mm_mean", float("nan")),
            "hd95_std":  s.get("hd95_mm_std",  float("nan")),
            "nsd_mean": s.get("nsd_mean", float("nan")),
            "nsd_std":  s.get("nsd_std",  float("nan")),
        })
        print(f"  thr={t_key:.4f}  N={n:3d}  "
              f"DSC={s.get('dsc_mean', float('nan')):.4f}±{s.get('dsc_std', 0):.4f}  "
              f"HD95={s.get('hd95_mm_mean', float('nan')):.2f}  "
              f"NSD={s.get('nsd_mean', float('nan')):.4f}")

    # Write CSVs
    sum_path = out_dir / "per_threshold_summary.csv"
    with open(sum_path, "w") as f:
        f.write("threshold,n_vols,dsc_mean,dsc_std,hd95_mean,hd95_std,nsd_mean,nsd_std\n")
        for r in summary_rows:
            f.write(f"{r['threshold']:.4f},{r['n_vols']},"
                    f"{r['dsc_mean']:.6f},{r['dsc_std']:.6f},"
                    f"{r['hd95_mean']:.4f},{r['hd95_std']:.4f},"
                    f"{r['nsd_mean']:.6f},{r['nsd_std']:.6f}\n")
    print(f"\n[sweep] wrote {sum_path}")

    long_path = out_dir / "per_threshold_per_volume.csv"
    with open(long_path, "w") as f:
        f.write("case_id,threshold,dsc,hd95_mm,nsd\n")
        for r in per_volume_long:
            f.write(f"{r['case_id']},{r['threshold']:.4f},"
                    f"{r['dsc']:.6f},{r['hd95_mm']:.4f},{r['nsd']:.6f}\n")
    print(f"[sweep] wrote {long_path}")

    # Best threshold
    best = max(summary_rows, key=lambda r: r["dsc_mean"])
    print(f"\n[sweep] BEST: threshold={best['threshold']:.4f}  "
          f"DSC={best['dsc_mean']:.4f}±{best['dsc_std']:.4f}  "
          f"HD95={best['hd95_mean']:.2f}  NSD={best['nsd_mean']:.4f}")
    print(f"[sweep] total time: {(time.time() - t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
