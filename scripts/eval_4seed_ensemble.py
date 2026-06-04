"""
eval_4seed_ensemble.py — 4-seed output-probability ensemble eval on AMOS22.

Holds 4 V2 model replicas in GPU memory (each ~600 MB live), runs each
on the same window, averages the sigmoid outputs across seeds per
slice, binarises at a single threshold (default 0.5 — override with
the optimal threshold from the threshold sweep), applies largest-CC,
and reports DSC/HD95/NSD using V2's VolumetricMetrics.

Reuses the canonical AMOS22 eval path (find_eligible_cases /
load_amos22_case / preprocess_slice_window / build_window /
get_bounding_box_resized / build_model / keep_largest_cc) for
bit-for-bit pipeline parity with the single-seed evals.

Usage:
  python scripts/eval_4seed_ensemble.py \
      --checkpoints checkpoints/.../seed42_best.pt checkpoints/.../seed43_best.pt \
                   checkpoints/.../seed44_best.pt checkpoints/.../seed45_best.pt \
      --config configs/phase3_odesam_v2_seed43.yaml \
      --output thesis/results/amos22_liver_indomain_4seed_ensemble \
      --img-size 256 --n-slices 8 --threshold 0.5
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

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


def load_models(ckpt_paths, cfg, device):
    models = []
    for i, p in enumerate(ckpt_paths):
        m = build_model(cfg).to(device)
        ckpt = torch.load(str(p), map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        missing, unexpected = m.load_state_dict(state, strict=False)
        print(f"[ens] [{i}] loaded {Path(p).name} "
              f"| missing={len(missing)} unexpected={len(unexpected)}")
        m.eval()
        models.append(m)
    return models


@torch.no_grad()
def evaluate_case_ensemble(models, image_DHW, label_DHW,
                          n_slices, img_size, device,
                          threshold=0.5, modality_id=0,
                          hu_clip=(-175.0, 250.0)):
    D, H, W = image_DHW.shape
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

        prob_sum = None
        for m in models:
            prob = m.predict(win_t, box_t, mod_t, is_3d=True)
            if prob.shape[-1] != img_size:
                prob = F.interpolate(prob, size=(img_size, img_size),
                                     mode="bilinear", align_corners=False)
            prob_sum = prob if prob_sum is None else prob_sum + prob
        prob_avg = prob_sum / float(len(models))
        pred_DHW[z] = prob_avg.squeeze().cpu().numpy() > threshold
        n_proc += 1

    pred_DHW = keep_largest_cc(pred_DHW)
    return pred_DHW, gt_DHW, n_proc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", type=str, nargs="+", required=True,
                    help="One or more checkpoint .pt files to ensemble")
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--amos-root", type=str,
                    default=r"C:/Users/Raywa/Desktop/LiteSAM3D/data/amos22")
    ap.add_argument("--max-volumes", type=int, default=999)
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--n-slices", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(args.config)
    print(f"[ens] device={args.device}")
    print(f"[ens] threshold={args.threshold}")
    print(f"[ens] ensembling {len(args.checkpoints)} ckpts:")
    for p in args.checkpoints:
        print(f"  - {p}")

    models = load_models(args.checkpoints, cfg, args.device)

    cases = find_eligible_cases(Path(args.amos_root), args.max_volumes)
    print(f"[ens] found {len(cases)} eligible AMOS22 val cases")

    metrics = VolumetricMetrics()
    per_volume = []
    t_start = time.time()

    for i, (case_id, img_path, lbl_path) in enumerate(cases):
        try:
            img_HWD, lbl_HWD, src_spacing = load_amos22_case(img_path, lbl_path)
        except Exception as e:
            print(f"[ens] ({i+1}/{len(cases)}) {case_id} LOAD FAILED: {e}")
            continue
        if lbl_HWD.sum() == 0:
            print(f"[ens] ({i+1}/{len(cases)}) {case_id} SKIP (no liver)")
            continue

        t0 = time.time()
        img_DHW = np.transpose(img_HWD, (2, 0, 1))
        lbl_DHW = np.transpose(lbl_HWD, (2, 0, 1))
        pred, gt, n_proc = evaluate_case_ensemble(
            models, img_DHW, lbl_DHW,
            n_slices=args.n_slices, img_size=args.img_size,
            device=args.device, threshold=args.threshold,
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
        print(f"[ens] ({i+1}/{len(cases)}) {case_id} "
              f"D={img_DHW.shape[0]} liver_slices={n_proc} "
              f"dsc={rec['dsc']:.4f} hd95={rec['hd95_mm']:.2f}mm "
              f"nsd={rec['nsd']:.4f} {dt:.1f}s")

    summary = metrics.summary()
    overall = summary.get("overall", {})
    print("\n=== 4-seed ensemble results ===")
    print(f"  N volumes: {len(per_volume)}")
    print(f"  DSC : {overall.get('dsc_mean', float('nan')):.4f} "
          f"± {overall.get('dsc_std', float('nan')):.4f}")
    print(f"  HD95: {overall.get('hd95_mm_mean', float('nan')):.2f} mm")
    print(f"  NSD : {overall.get('nsd_mean', float('nan')):.4f}")
    print(f"  total time: {(time.time() - t_start)/60:.1f} min")

    # CSV
    csv_path = out_dir / "per_volume_metrics.csv"
    with open(csv_path, "w") as f:
        f.write("case_id,n_slices,dsc,hd95_mm,nsd,iou,assd_mm,sensitivity,"
                "precision,specificity,vol_sim,vol_pred_ml,vol_gt_ml,nsd_at_2mm\n")
        for r in per_volume:
            f.write(f"{r['case_id']},{r['n_slices']},"
                    f"{r['dsc']:.6f},{r['hd95_mm']:.4f},{r['nsd']:.6f},"
                    f"{r['iou']:.6f},{r['assd_mm']:.4f},"
                    f"{r['sensitivity']:.6f},{r['precision']:.6f},"
                    f"{r['specificity']:.6f},{r['vol_sim']:.6f},"
                    f"{r['vol_pred_ml']:.2f},{r['vol_gt_ml']:.2f},"
                    f"{r['nsd_at_2.0mm']:.6f}\n")

    # summary.json
    out_summary = {
        "n_volumes": len(per_volume),
        "checkpoints": list(args.checkpoints),
        "threshold": args.threshold,
        "n_slices": args.n_slices,
        "img_size": args.img_size,
        "summary": {"liver": {"organ_id": 6, **overall, "n_volumes": len(per_volume)}},
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(out_summary, f, indent=2)
    print(f"[ens] wrote {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
