# =============================================================================
# evaluate_3d.py — Volumetric 3D Evaluation Entry Point (v2)
#
# Computes STANDARD 3D metrics:
#   - DSC  — 3D volume-level Dice Similarity Coefficient
#   - HD95 — 95th percentile Hausdorff Distance in MILLIMETRES
#   - NSD  — Normalised Surface Dice with organ-specific mm tolerances
#
# FIXES vs v1:
#   - DSC on full 3D volumes (not per-slice averages)
#   - HD95 in mm using physical spacing (not pixels)
#   - NSD with organ-specific mm tolerances (not 1.0px)
#   - dataset always amos22_3d → ISA always active
#   - Results include Wilcoxon signed-rank test vs baseline
#
# Usage:
#   python evaluate_3d.py --checkpoint checkpoints/run3/best.pt --split val
#   python evaluate_3d.py --checkpoint v2.pt --baseline v1.pt --compare
#   python evaluate_3d.py --checkpoint best.pt --per_organ
#
# =============================================================================

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def parse_args():
    parser = argparse.ArgumentParser(
        description="3D volumetric evaluation for LiteSAM-3D v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--config", type=str, default=None,
                        help="Config YAML. Auto-discovers from checkpoint dir if omitted.")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--output", type=str, default="results",
                        help="Directory to save results CSV")
    parser.add_argument("--baseline", type=str, default=None,
                        help="Baseline checkpoint for Wilcoxon comparison")
    parser.add_argument("--per_organ", action="store_true",
                        help="Print per-organ breakdown table")
    parser.add_argument("--spacing_mm", type=float, nargs=3, default=[1.5, 1.5, 1.5],
                        help="Voxel spacing in mm (D H W). Must match preprocessing.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Inference batch size")
    return parser.parse_args()


def load_config(args):
    if args.config:
        cfg = OmegaConf.load(args.config)
        base = Path("configs/base.yaml")
        if base.exists():
            cfg = OmegaConf.merge(OmegaConf.load(str(base)), cfg)
        return cfg

    ckpt_dir = Path(args.checkpoint).parent
    for p in [ckpt_dir / "config.yaml", ckpt_dir.parent / "config.yaml",
              Path("configs/amos22.yaml")]:
        if p.exists():
            print(f"Config: {p}")
            cfg = OmegaConf.load(str(p))
            base = Path("configs/base.yaml")
            if base.exists():
                cfg = OmegaConf.merge(OmegaConf.load(str(base)), cfg)
            return cfg

    raise FileNotFoundError("No config found. Pass --config explicitly.")


def load_model(checkpoint_path: str, cfg, device):
    from models import build_model

    model = build_model(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=True)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {n_params:.1f}M params | ISA: {model.use_isa}")
    return model


@torch.no_grad()
def run_evaluation(model, dataloader, device, spacing_mm):
    """
    Run inference and compute 3D volumetric metrics.

    Groups per-slice predictions by (patient_id, organ_id), stacks them
    in slice order, then computes DSC / HD95 / NSD on the reconstructed volume.
    """
    from evaluation.metrics_3d import VolumetricMetrics

    metrics = VolumetricMetrics()
    spacing = tuple(spacing_mm)

    # Accumulate: {(patient_id, organ_id): {slice_idx: (pred, gt)}}
    patient_preds = defaultdict(dict)
    patient_gts = defaultdict(dict)

    for batch in tqdm(dataloader, desc="Inference", dynamic_ncols=True):
        images = batch["image"].to(device)          # (B, D, 3, H, W)
        boxes = batch["box"].to(device)             # (B, 4) center slice
        modality_ids = batch["modality_id"].to(device)  # (B, D) or (B,)
        masks_gt = batch["mask"].cpu().numpy()      # (B, H, W) center slice GT

        is_3d_raw = batch.get("is_3d", torch.tensor([False]))
        if isinstance(is_3d_raw, torch.Tensor):
            is_3d = bool(is_3d_raw[0].item())
        else:
            is_3d = bool(is_3d_raw)

        organ_ids = batch["organ_id"].cpu().numpy()          # (B,)
        center_slices = batch["center_slice"].cpu().numpy()  # (B,)
        image_paths = batch["image_path"]                    # list of str (B,)

        # model.predict handles (B,D) modality_ids internally
        pred_probs = model.predict(
            images, boxes, modality_ids, is_3d=is_3d
        )  # (B, 1, H_out, W_out)
        pred_probs_np = pred_probs.squeeze(1).cpu().numpy()  # (B, H_out, W_out)

        # Resize GT to match prediction spatial resolution if needed
        pred_H, pred_W = pred_probs_np.shape[-2:]
        gt_H, gt_W = masks_gt.shape[-2:]
        if pred_H != gt_H or pred_W != gt_W:
            import torch.nn.functional as F_nn
            gt_t = torch.from_numpy(masks_gt).unsqueeze(1).float()
            gt_t = F_nn.interpolate(gt_t, size=(pred_H, pred_W), mode="nearest")
            masks_gt = gt_t.squeeze(1).numpy()

        B = pred_probs_np.shape[0]
        for i in range(B):
            pid = str(image_paths[i])
            oid = int(organ_ids[i])
            sidx = int(center_slices[i])
            key = (pid, oid)
            patient_preds[key][sidx] = (pred_probs_np[i] > 0.5)  # binary
            patient_gts[key][sidx] = masks_gt[i].astype(bool)

    print(f"\nReconstructing {len(patient_preds)} patient-organ volumes...")

    for key, slice_preds in tqdm(patient_preds.items(), desc="3D metrics"):
        pid, oid = key
        slice_gts = patient_gts[key]

        sorted_idxs = sorted(slice_preds.keys())
        pred_vol = np.stack([slice_preds[i] for i in sorted_idxs], axis=0)  # (D, H, W)
        gt_vol = np.stack([slice_gts[i] for i in sorted_idxs], axis=0)

        metrics.update(
            pred_vol=pred_vol,
            gt_vol=gt_vol,
            spacing_mm=spacing,
            organ_id=oid,
            patient_id=pid,
        )

    return metrics


def print_summary(summary, spacing_mm):
    overall = summary.get("overall", {})
    print("\n" + "=" * 65)
    print("  LiteSAM-3D v2 — Volumetric 3D Evaluation")
    print(f"  Spacing: {spacing_mm} mm | HD95 and NSD in physical units")
    print("=" * 65)
    print(f"  Mean DSC:  {overall.get('dsc_mean', float('nan')):.4f} "
          f"± {overall.get('dsc_std', float('nan')):.4f}")
    print(f"  Mean HD95: {overall.get('hd95_mm_mean', float('nan')):.2f} "
          f"± {overall.get('hd95_mm_std', float('nan')):.2f} mm")
    print(f"  Mean NSD:  {overall.get('nsd_mean', float('nan')):.4f} "
          f"± {overall.get('nsd_std', float('nan')):.4f}")
    print("=" * 65)


def print_per_organ(summary):
    print(f"\n{'Organ':<20} {'DSC':>8} {'±':>6} {'HD95mm':>8} {'±':>6} {'NSD':>8}")
    print("-" * 60)
    for organ_name, vals in sorted(summary.items(), key=lambda x: -x[1].get("dsc_mean", 0)):
        if organ_name == "overall":
            continue
        print(f"{organ_name:<20} "
              f"{vals['dsc_mean']:>8.4f} "
              f"{vals['dsc_std']:>6.4f} "
              f"{vals['hd95_mm_mean']:>8.2f} "
              f"{vals['hd95_mm_std']:>6.2f} "
              f"{vals['nsd_mean']:>8.4f}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = load_config(args)
    cfg.data.dataset = "amos22_3d"  # force 3D — ISA always active

    model = load_model(args.checkpoint, cfg, device)

    from datasets import get_dataset
    from torch.utils.data import DataLoader

    dataset = get_dataset(cfg, split=args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    print(f"Dataset: {len(dataset)} samples ({args.split})")

    metrics = run_evaluation(model, loader, device, args.spacing_mm)
    summary = metrics.summary()

    print_summary(summary, args.spacing_mm)
    if args.per_organ:
        print_per_organ(summary)

    os.makedirs(args.output, exist_ok=True)
    df = metrics.to_dataframe()
    out_path = Path(args.output) / f"3d_results_{args.split}.csv"
    df.to_csv(str(out_path), index=False)
    print(f"\nResults saved: {out_path}")

    if args.baseline:
        print(f"\nBaseline evaluation: {args.baseline}")
        model_bl = load_model(args.baseline, cfg, device)
        metrics_bl = run_evaluation(model_bl, loader, device, args.spacing_mm)

        wilcoxon_results = metrics.wilcoxon_test(metrics_bl, metric="dsc")
        overall_w = wilcoxon_results.get("overall", {})
        p_val = overall_w.get("p_value", float("nan"))
        mean_diff = overall_w.get("mean_diff", float("nan"))

        print(f"\nWilcoxon test (v2 vs baseline) — DSC:")
        print(f"  p-value:        {p_val:.4f} {'*' if p_val < 0.05 else '(n.s.)'}")
        print(f"  Mean delta DSC: {mean_diff:+.4f}")

        bl_out = Path(args.output) / "baseline_3d_results.csv"
        metrics_bl.to_dataframe().to_csv(str(bl_out), index=False)
        print(f"Baseline results saved: {bl_out}")


if __name__ == "__main__":
    main()
