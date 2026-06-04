"""Path A1 -- Smoke 3D eval across ALL epoch checkpoints.

Builds the model + dataset ONCE, then loads each ckpt and reuses the same
2-volume eval. Produces a single comparison table.

Why: best.pt was overwritten by the resumed run, so we lost the actual
global best. This script tells us which saved epoch_*.pt has the highest
3D mean DSC.

Output: table to stdout + JSON at evaluation/reports/path_a1_all_epochs.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from omegaconf import OmegaConf

from models import build_model
from evaluation.metrics_3d import VolumetricMetrics, ORGAN_NSD_TOLERANCES_MM

from eval_a1_per_organ import (
    TARGET_ORGANS,
    load_a1_cfg,
    list_val_volumes,
    load_volume_zhw,
    normalize_hu,
    resize_z_stack,
    sliding_predict_one_organ,
)

CKPT_DIRS = [
    ("a1",   ROOT / "checkpoints" / "path_a1_big_solid",      "path_a1_big_solid_epoch"),
    ("v2hp", ROOT / "checkpoints" / "path_a1_big_solid_v2hp", "path_a1_big_solid_v2hp_epoch"),
]

MAX_VOLUMES = 2
THRESHOLD = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def list_epoch_ckpts() -> List[Tuple[str, Path]]:
    out = []
    for run_tag, d, prefix in CKPT_DIRS:
        if not d.exists():
            continue
        for p in sorted(d.glob(f"{prefix}*.pt")):
            out.append((f"{run_tag}_{p.stem.split('_epoch')[-1]}", p))
    return out


@torch.no_grad()
def eval_one_ckpt(model, ckpt_path: Path, vols, cfg) -> Dict:
    ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()

    img_size = int(cfg.model.img_size)
    depth = int(cfg.data.slices_per_volume)
    n_organs = int(getattr(cfg.model.ode, "n_organs", 15))
    hu_clip = tuple(cfg.data.clip_range)

    vm = VolumetricMetrics()
    for v in vols:
        img_zhw, spacing_zhw = load_volume_zhw(v["image"])
        lbl_zhw, _ = load_volume_zhw(v["label"])
        lbl_zhw = lbl_zhw.astype(np.uint8)
        Z, H, W = img_zhw.shape
        img_norm = normalize_hu(img_zhw, hu_clip)
        img_resized = resize_z_stack(img_norm, img_size, img_size)

        for oid, oname in TARGET_ORGANS.items():
            gt_native = (lbl_zhw == oid)
            if gt_native.sum() < 50:
                vm._records.append({
                    "patient_id": v["stem"], "organ_id": oid, "organ_name": oname,
                    "dsc": float("nan"), "hd95_mm": float("nan"), "nsd": float("nan"),
                    "tolerance_mm": ORGAN_NSD_TOLERANCES_MM.get(oid, 2.0),
                })
                continue
            prob_resized = sliding_predict_one_organ(
                model=model, img_resized=img_resized,
                organ_id_value=oid, n_organs_total=n_organs,
                depth=depth, img_size=img_size, device=DEVICE,
            )
            if (H, W) != (img_size, img_size):
                t = torch.from_numpy(prob_resized).unsqueeze(1)
                t = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
                prob_native = t.squeeze(1).numpy()
            else:
                prob_native = prob_resized
            pred_bin = (prob_native > THRESHOLD).astype(bool)
            vm.update(pred_vol=pred_bin, gt_vol=gt_native,
                      spacing_mm=spacing_zhw, organ_id=oid, patient_id=v["stem"])

    summary = vm.summary()
    per_organ = {}
    for oid, oname in TARGET_ORGANS.items():
        if oname not in summary:
            per_organ[oname] = 0.0
            continue
        s = summary[oname]
        per_organ[oname] = float(s["dsc_mean"]) if not np.isnan(s["dsc_mean"]) else 0.0
    mean_dsc = float(np.mean(list(per_organ.values())))
    return {
        "ckpt": str(ckpt_path.name),
        "epoch_field": ckpt.get("epoch"),
        "per_organ_dsc": per_organ,
        "mean_dsc": mean_dsc,
    }


def main() -> None:
    print(f"device={DEVICE}  torch={torch.__version__}")
    cfg = load_a1_cfg()
    print("[all-eval] building model once")
    model = build_model(cfg).to(DEVICE)
    print("[all-eval] listing val volumes")
    all_vols = list_val_volumes(Path(cfg.data.data_root), cfg.data.modality)
    vols = all_vols[:MAX_VOLUMES]
    print(f"[all-eval] eval set: {[v['stem'] for v in vols]}")

    ckpts = list_epoch_ckpts()
    print(f"[all-eval] found {len(ckpts)} epoch ckpts to score")
    for tag, p in ckpts:
        print(f"  {tag:>14s}  {p.name}")

    rows: List[Dict] = []
    t_total = time.time()
    for tag, p in ckpts:
        t0 = time.time()
        r = eval_one_ckpt(model, p, vols, cfg)
        r["tag"] = tag
        rows.append(r)
        print(f"[all-eval] {tag:>14s}  mean_dsc={r['mean_dsc']:.4f}  "
              f"liver={r['per_organ_dsc']['liver']:.3f}  "
              f"spleen={r['per_organ_dsc']['spleen']:.3f}  "
              f"r_kid={r['per_organ_dsc']['r_kidney']:.3f}  "
              f"l_kid={r['per_organ_dsc']['l_kidney']:.3f}  "
              f"({time.time()-t0:.0f}s)")

    rows.sort(key=lambda r: r["mean_dsc"], reverse=True)
    print()
    print("=" * 88)
    print("RANKING by mean DSC (smoke=2 vols)")
    print("=" * 88)
    print(f"{'rank':<5}{'tag':<16}{'mean':>8}{'liver':>8}{'spleen':>8}{'r_kid':>8}{'l_kid':>8}{'ckpt':<40}")
    for i, r in enumerate(rows):
        po = r["per_organ_dsc"]
        print(f"{i+1:<5}{r['tag']:<16}{r['mean_dsc']:>8.4f}"
              f"{po['liver']:>8.3f}{po['spleen']:>8.3f}"
              f"{po['r_kidney']:>8.3f}{po['l_kidney']:>8.3f}  {r['ckpt']}")
    print(f"\nTotal eval wall: {(time.time()-t_total)/60:.1f} min")

    out = {"smoke_max_volumes": MAX_VOLUMES, "rows": rows}
    out_path = ROOT / "evaluation" / "reports" / "path_a1_all_epochs.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[all-eval] saved {out_path}")


if __name__ == "__main__":
    main()
