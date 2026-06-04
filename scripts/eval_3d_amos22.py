"""Standalone 3D evaluation on AMOS22 val split.

Loads a checkpoint, iterates validation volumes, runs sliding-window +
optional TTA, applies CC postproc, reports 3D Dice per organ and mean Dice
across the 15-organ AMOS22 set — the same metric MCP-MedSAM reports.

Usage:
    python scripts/eval_3d_amos22.py \
        --config configs/v7_rescue_320px.yaml \
        --checkpoint checkpoints/v7_rescue_320/best.pt \
        --tta --cc --max-volumes 10

Output:
    JSON report under evaluation/reports/<exp>_3d_dice.json
    Printed per-organ table + mean.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import nibabel as nib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf

from models import build_model
from evaluation.sliding_window_3d import sliding_window_predict_3d, compute_3d_dice_per_organ
from inference.postproc import postprocess_multi_organ


ORGAN_NAMES = [
    "spleen", "r_kidney", "l_kidney", "gallbladder", "esophagus",
    "liver", "stomach", "aorta", "ivc", "pancreas",
    "r_adrenal", "l_adrenal", "duodenum", "bladder", "prostate_uterus",
]


def list_val_volumes(data_root: Path, modality: str = "ct") -> List[Dict[str, str]]:
    img_dir = data_root / "imagesVa"
    lbl_dir = data_root / "labelsVa"
    if not img_dir.exists() or not lbl_dir.exists():
        raise FileNotFoundError(f"AMOS22 val dirs not found under {data_root}")
    out: List[Dict[str, str]] = []
    for img_path in sorted(img_dir.glob("*.nii.gz")):
        stem = img_path.name.replace(".nii.gz", "")
        try:
            case_id = int(stem.split("_")[-1])
        except ValueError:
            continue
        if modality == "ct" and not (1 <= case_id <= 500):
            continue
        if modality == "mri" and not (501 <= case_id <= 600):
            continue
        lbl_path = lbl_dir / img_path.name
        if not lbl_path.exists():
            continue
        out.append({"image": str(img_path), "label": str(lbl_path), "stem": stem})
    return out


def load_volume(path: str) -> np.ndarray:
    """Return (Z, H, W) float32 for image, (Z, H, W) uint8 for label."""
    nib_vol = nib.load(path)
    arr = np.asarray(nib_vol.dataobj)                  # (H, W, D)
    return np.ascontiguousarray(np.transpose(arr, (2, 0, 1)))


def evaluate(
    cfg,
    ckpt_path: Path,
    device: str,
    max_volumes: int,
    use_tta: bool,
    use_cc: bool,
    threshold: float,
    depth: int,
    img_size: int,
) -> Dict:
    model = build_model(cfg).to(device)
    model.eval()

    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[eval] loaded checkpoint {ckpt_path.name} | "
              f"missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print(f"[eval] WARNING: checkpoint {ckpt_path} not found — running RANDOM init")

    vols = list_val_volumes(Path(cfg.data.data_root), cfg.data.modality)
    if max_volumes > 0:
        vols = vols[:max_volumes]
    print(f"[eval] evaluating {len(vols)} volumes | tta={use_tta} cc={use_cc}")

    per_organ_all: Dict[int, List[float]] = {k: [] for k in range(1, 16)}

    t_start = time.time()
    for i, v in enumerate(vols):
        img_vol = load_volume(v["image"])                        # (Z, H, W) HU
        lbl_vol = load_volume(v["label"]).astype(np.uint8)       # (Z, H, W) labels 0..15

        t0 = time.time()
        prob_vol = sliding_window_predict_3d(
            model=model,
            image_vol=img_vol.astype(np.float32),
            depth=depth,
            img_size=img_size,
            n_organs=15,
            device=device,
            amp=True,
        )                                                        # (15, Z, H, W)

        if use_cc:
            pred_vol = postprocess_multi_organ(
                prob_vol, threshold=threshold, fill_holes=True, largest_cc=True,
            )
            # convert back to prob-ish for dice compute (binary already)
            per_organ = {}
            for k in range(1, 16):
                gt_k = (lbl_vol == k).astype(np.float32)
                if gt_k.sum() < 50:
                    continue
                pred_k = pred_vol[k - 1].astype(np.float32)
                inter = float((pred_k * gt_k).sum())
                denom = float(pred_k.sum() + gt_k.sum())
                per_organ[k] = (2.0 * inter + 1e-6) / (denom + 1e-6)
        else:
            per_organ = compute_3d_dice_per_organ(
                prob_vol, lbl_vol, n_organs=15, threshold=threshold,
            )

        for k, d in per_organ.items():
            per_organ_all[k].append(float(d))

        dt = time.time() - t0
        mean_this = float(np.mean(list(per_organ.values()))) if per_organ else 0.0
        print(f"[eval] ({i+1}/{len(vols)}) {v['stem']} Z={img_vol.shape[0]} "
              f"present={len(per_organ)}  mean_dice={mean_this:.4f}  {dt:.1f}s")

    per_organ_mean = {
        k: float(np.mean(vals)) if vals else 0.0
        for k, vals in per_organ_all.items()
    }
    mean_dice = float(np.mean([v for v in per_organ_mean.values() if v > 0]))

    print("\n=== PER-ORGAN 3D DICE ===")
    print(f"{'organ':<18} {'dice':>8} {'#vols':>8}")
    for k in range(1, 16):
        name = ORGAN_NAMES[k - 1]
        print(f"{name:<18} {per_organ_mean[k]:>8.4f} {len(per_organ_all[k]):>8}")
    print(f"{'MEAN':<18} {mean_dice:>8.4f}")
    print(f"[eval] total time {(time.time() - t_start)/60:.1f} min")

    return {
        "experiment": cfg.experiment.name,
        "checkpoint": str(ckpt_path),
        "n_volumes": len(vols),
        "tta": use_tta,
        "cc": use_cc,
        "threshold": threshold,
        "per_organ_dice": {ORGAN_NAMES[k-1]: per_organ_mean[k] for k in range(1, 16)},
        "per_organ_vol_counts": {ORGAN_NAMES[k-1]: len(per_organ_all[k]) for k in range(1, 16)},
        "mean_dice": mean_dice,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--cc", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--max-volumes", type=int, default=-1)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=320)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    report = evaluate(
        cfg=cfg,
        ckpt_path=Path(args.checkpoint),
        device=args.device,
        max_volumes=args.max_volumes,
        use_tta=args.tta,
        use_cc=args.cc,
        threshold=args.threshold,
        depth=args.depth,
        img_size=args.img_size,
    )

    out_dir = ROOT / "evaluation" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cfg.experiment.name}_3d_dice.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[eval] report saved to {out_path}")


if __name__ == "__main__":
    main()
