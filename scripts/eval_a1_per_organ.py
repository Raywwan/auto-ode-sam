"""Path A1 -- Per-organ 3D DSC + HD95 + NSD evaluation on AMOS22 val.

Used at every gate (ep 4 / 8 / 12 / 14). Reports per-organ DSC, HD95 (mm),
NSD (mm tolerance per AMOS22) for the K=4 target organs and prints
PASS/FAIL against A1 targets.

Metrics come from evaluation/metrics_3d.VolumetricMetrics so paper numbers
are identical to what we report in the writeup. Voxel spacing is read from
each nifti header so HD95/NSD are in millimetres (not pixels).

Critical design point (different from scripts/eval_3d_amos22.py):
  AutoODESAM's ODE is organ-conditioned. The cross-slice dynamics depend on
  the organ_id passed in. To compute per-organ DSC correctly we MUST run a
  separate forward pass for each target organ with that organ's id, then
  extract only that organ's channel. K=4 organs -> 4x forward passes per
  sliding-window position. Slower but correct.

Usage:
    python scripts/eval_a1_per_organ.py \
        --ckpt checkpoints/path_a1_big_solid/path_a1_big_solid_epoch003.pt \
        --max-volumes 30 --tag ep4

Output:
  evaluation/reports/path_a1_<tag>_per_organ.json
  evaluation/reports/path_a1_<tag>_per_organ_records.json (full per-volume
    records for Wilcoxon stats vs baselines)
  Pretty per-organ table to stdout (ASCII only).
"""
from __future__ import annotations

import argparse
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

from omegaconf import OmegaConf

from models import build_model
from evaluation.metrics_3d import VolumetricMetrics, ORGAN_NSD_TOLERANCES_MM

A1_CONFIG = ROOT / "configs" / "path_a1_big_solid.yaml"
BASE_CONFIG = ROOT / "configs" / "base.yaml"

# AMOS22 organ ids (1-indexed) for A1 K=4 target organs. These names must
# match evaluation.metrics_3d.ORGAN_NAMES so VolumetricMetrics groups correctly.
TARGET_ORGANS: Dict[int, str] = {
    1: "spleen",
    2: "r_kidney",
    3: "l_kidney",
    6: "liver",
}

# A1 acceptance gates per project_path_a1_big_solid_organs memory.
GATES = {
    # gate name : (epoch hint, mean_dsc_floor, liver_floor)
    "ep_4_warmup":  (4,  0.78, 0.85),
    "ep_8_mid":     (8,  0.86, 0.92),
    "ep_12_late":   (12, 0.90, 0.93),
    "ep_14_final":  (14, 0.91, 0.94),  # floor; stretch is 0.94 / 0.96
}


def load_a1_cfg():
    cfg = OmegaConf.load(str(BASE_CONFIG)) if BASE_CONFIG.exists() else OmegaConf.create({})
    a1 = OmegaConf.load(str(A1_CONFIG))
    if "defaults" in a1:
        a1 = OmegaConf.masked_copy(a1, [k for k in a1 if k != "defaults"])
    return OmegaConf.merge(cfg, a1)


def list_val_volumes(data_root: Path, modality: str) -> List[Dict[str, str]]:
    img_dir = data_root / "imagesVa"
    lbl_dir = data_root / "labelsVa"
    if not img_dir.exists() or not lbl_dir.exists():
        raise FileNotFoundError(f"AMOS22 val dirs not found under {data_root}")
    out = []
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


def load_volume_zhw(path: str) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Load a nifti, return (Z,H,W) volume + voxel spacing in mm as (Dz,Dy,Dx).

    The nifti header reports zooms in (X,Y,Z); after transposing to (Z,H,W)
    we reorder to (Z_spacing, H_spacing, W_spacing) so callers downstream
    (HD95/NSD) get spacing in the same axis order as the array.
    """
    import nibabel as nib
    nii = nib.load(path)
    arr = np.asarray(nii.dataobj)
    arr_zhw = np.ascontiguousarray(np.transpose(arr, (2, 0, 1)))  # (Z, H, W)
    zooms = nii.header.get_zooms()  # (X,Y,Z) in mm
    # zooms axes are (X,Y,Z) -> array axes after transpose are (Z,Y,X)
    spacing_zhw = (float(zooms[2]), float(zooms[1]), float(zooms[0]))
    return arr_zhw, spacing_zhw


def normalize_hu(image_vol: np.ndarray, hu_clip: Tuple[float, float]) -> np.ndarray:
    lo, hi = hu_clip
    x = np.clip(image_vol, lo, hi).astype(np.float32)
    return (x - lo) / max(hi - lo, 1.0)


def resize_z_stack(stack: np.ndarray, h: int, w: int) -> np.ndarray:
    """stack: (Z, H, W) -> (Z, h, w) bilinear via torch (faster than scipy)."""
    Z = stack.shape[0]
    t = torch.from_numpy(stack).unsqueeze(1).float()              # (Z, 1, H, W)
    t = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
    return t.squeeze(1).numpy()                                   # (Z, h, w)


@torch.no_grad()
def sliding_predict_one_organ(
    model: torch.nn.Module,
    img_resized: np.ndarray,           # (Z, h, w) float32 in [0,1]
    organ_id_value: int,               # 1-indexed AMOS22 ID
    n_organs_total: int,               # 15 (AutoODESAM decoder slots)
    depth: int,
    img_size: int,
    device: str,
    amp: bool = True,
) -> np.ndarray:
    """Return (Z, h, w) probability volume for the requested organ only.

    Mirrors evaluation/sliding_window_3d.py's reflect-pad strategy: for each
    target z, run a D-window with mid-slot aligned at z, then read the
    decoder's center-slice output.
    """
    Z, h, w = img_resized.shape
    mid = depth // 2
    pad_lo = mid
    pad_hi = depth - mid - 1
    img_padded = np.concatenate(
        [np.tile(img_resized[:1], (pad_lo, 1, 1)),
         img_resized,
         np.tile(img_resized[-1:], (pad_hi, 1, 1))],
        axis=0,
    )
    organ_id = torch.tensor([organ_id_value], dtype=torch.long, device=device)
    organ_idx = organ_id_value - 1  # 0-indexed channel

    out_prob = np.zeros((Z, h, w), dtype=np.float32)

    for z in range(Z):
        chunk = img_padded[z:z + depth]                            # (D, h, w)
        chunk_t = torch.from_numpy(chunk).to(device)
        chunk_t = chunk_t.unsqueeze(0).unsqueeze(2).repeat(1, 1, 3, 1, 1)  # (1, D, 3, h, w)
        with torch.amp.autocast("cuda", enabled=amp):
            out = model(chunk_t, organ_id, is_3d=True)
        masks = out["masks"][:, organ_idx:organ_idx + 1].float()    # (1, 1, H_out, W_out)
        if masks.shape[-2:] != (img_size, img_size):
            masks = F.interpolate(masks, size=(img_size, img_size),
                                  mode="bilinear", align_corners=False)
        out_prob[z] = torch.sigmoid(masks)[0, 0].cpu().numpy()
    return out_prob


def evaluate(
    cfg,
    ckpt_path: Path,
    device: str,
    max_volumes: int,
    threshold: float,
) -> Tuple[Dict, List[Dict]]:
    print(f"[eval-a1] building model")
    model = build_model(cfg).to(device)
    model.eval()

    if not ckpt_path.exists():
        raise FileNotFoundError(f"ckpt missing: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[eval-a1] loaded {ckpt_path.name} | missing={len(missing)} "
          f"unexpected={len(unexpected)} | epoch={ckpt.get('epoch')}")

    img_size = int(cfg.model.img_size)
    depth = int(cfg.data.slices_per_volume)
    n_organs = int(getattr(cfg.model.ode, "n_organs", 15))
    hu_clip = tuple(cfg.data.clip_range)

    vols = list_val_volumes(Path(cfg.data.data_root), cfg.data.modality)
    if max_volumes > 0:
        vols = vols[:max_volumes]
    print(f"[eval-a1] evaluating {len(vols)} volumes x {len(TARGET_ORGANS)} organs "
          f"= {len(vols) * len(TARGET_ORGANS)} per-organ inferences")
    print(f"[eval-a1] metrics: DSC + HD95 (mm) + NSD (mm-tolerance per organ)")

    vm = VolumetricMetrics()

    t_start = time.time()
    for i, v in enumerate(vols):
        img_zhw, spacing_zhw = load_volume_zhw(v["image"])              # (Z,H,W) HU + spacing (mm)
        lbl_zhw, _           = load_volume_zhw(v["label"])
        lbl_zhw = lbl_zhw.astype(np.uint8)
        Z, H, W = img_zhw.shape
        img_norm = normalize_hu(img_zhw, hu_clip)
        img_resized = resize_z_stack(img_norm, img_size, img_size)

        t0 = time.time()
        per_organ_this: Dict[str, Tuple[float, float, float]] = {}
        for oid, oname in TARGET_ORGANS.items():
            gt_native = (lbl_zhw == oid)
            if gt_native.sum() < 50:
                # organ absent in this volume -- record DSC=NaN via empty pred
                # so VolumetricMetrics returns NaN (its internal handling). We
                # skip the forward pass to save GPU time.
                vm._records.append({
                    "patient_id": v["stem"],
                    "organ_id": oid,
                    "organ_name": oname,
                    "dsc": float("nan"),
                    "hd95_mm": float("nan"),
                    "nsd": float("nan"),
                    "tolerance_mm": ORGAN_NSD_TOLERANCES_MM.get(oid, 2.0),
                })
                continue

            prob_resized = sliding_predict_one_organ(
                model=model,
                img_resized=img_resized,
                organ_id_value=oid,
                n_organs_total=n_organs,
                depth=depth,
                img_size=img_size,
                device=device,
            )                                                    # (Z, h, w)

            # Resize back to native (H, W)
            if (H, W) != (img_size, img_size):
                t = torch.from_numpy(prob_resized).unsqueeze(1)
                t = F.interpolate(t, size=(H, W), mode="bilinear",
                                  align_corners=False)
                prob_native = t.squeeze(1).numpy()
            else:
                prob_native = prob_resized

            pred_bin = (prob_native > threshold).astype(bool)
            rec = vm.update(
                pred_vol=pred_bin,
                gt_vol=gt_native,
                spacing_mm=spacing_zhw,
                organ_id=oid,
                patient_id=v["stem"],
            )
            per_organ_this[oname] = (rec["dsc"], rec["hd95_mm"], rec["nsd"])

        # Per-volume console line: DSC for quick read, HD95+NSD packed compact
        present = {k: t for k, t in per_organ_this.items() if not np.isnan(t[0])}
        mean_dsc_this = float(np.mean([t[0] for t in present.values()])) if present else float("nan")
        dt = time.time() - t0
        per_str = "  ".join(
            f"{n}=D{t[0]:.3f}/H{t[1]:.1f}/N{t[2]:.3f}" for n, t in per_organ_this.items()
        )
        print(f"[eval-a1] ({i+1}/{len(vols)}) {v['stem']} Z={Z}  "
              f"mean_dsc={mean_dsc_this:.4f}  {per_str}  {dt:.1f}s")

    # ---- aggregate via VolumetricMetrics ----
    summary = vm.summary()  # {organ_name: {...}, "overall": {...}}

    per_organ_dsc: Dict[str, float] = {}
    per_organ_hd95: Dict[str, float] = {}
    per_organ_nsd: Dict[str, float] = {}
    per_organ_count: Dict[str, int] = {}
    for oid, oname in TARGET_ORGANS.items():
        if oname not in summary:
            per_organ_dsc[oname] = 0.0
            per_organ_hd95[oname] = float("nan")
            per_organ_nsd[oname] = 0.0
            per_organ_count[oname] = 0
            continue
        s = summary[oname]
        per_organ_dsc[oname] = float(s["dsc_mean"]) if not np.isnan(s["dsc_mean"]) else 0.0
        per_organ_hd95[oname] = float(s["hd95_mm_mean"])
        per_organ_nsd[oname] = float(s["nsd_mean"]) if not np.isnan(s["nsd_mean"]) else 0.0
        per_organ_count[oname] = int(s["n_volumes"])

    overall_mean_dsc = float(np.mean(list(per_organ_dsc.values())))
    overall_mean_nsd = float(np.mean(list(per_organ_nsd.values())))
    valid_hd = [v for v in per_organ_hd95.values() if not np.isnan(v)]
    overall_mean_hd95 = float(np.mean(valid_hd)) if valid_hd else float("nan")
    liver_dsc = per_organ_dsc["liver"]

    print()
    print("=" * 78)
    print(f"PATH A1 PER-ORGAN 3D METRICS  (ckpt={ckpt_path.name})")
    print("=" * 78)
    print(f"{'organ':<14} {'DSC':>8} {'HD95(mm)':>10} {'NSD':>8} {'#vols':>7} {'tol(mm)':>9}")
    for oid in sorted(TARGET_ORGANS):
        oname = TARGET_ORGANS[oid]
        tol = ORGAN_NSD_TOLERANCES_MM.get(oid, 2.0)
        print(f"{oname:<14} {per_organ_dsc[oname]:>8.4f} "
              f"{per_organ_hd95[oname]:>10.2f} {per_organ_nsd[oname]:>8.4f} "
              f"{per_organ_count[oname]:>7d} {tol:>9.1f}")
    print(f"{'MEAN(K=4)':<14} {overall_mean_dsc:>8.4f} "
          f"{overall_mean_hd95:>10.2f} {overall_mean_nsd:>8.4f}")
    print()

    # Gate evaluation -- show pass/fail vs each gate the user might be at
    print("Gate status (informational; pick the right one for current epoch):")
    print(f"{'gate':<18} {'mean_floor':>10} {'liver_floor':>12} {'mean_status':>12} {'liver_status':>13}")
    for gate_name, (_, mean_floor, liver_floor) in GATES.items():
        ms = "PASS" if overall_mean_dsc >= mean_floor else "FAIL"
        ls = "PASS" if liver_dsc >= liver_floor else "FAIL"
        print(f"{gate_name:<18} {mean_floor:>10.2f} {liver_floor:>12.2f} {ms:>12} {ls:>13}")
    print()
    print(f"Total eval time: {(time.time()-t_start)/60:.1f} min")

    report = {
        "checkpoint": str(ckpt_path),
        "ckpt_epoch": ckpt.get("epoch"),
        "n_volumes": len(vols),
        "threshold": threshold,
        "per_organ_dsc": per_organ_dsc,
        "per_organ_hd95_mm": per_organ_hd95,
        "per_organ_nsd": per_organ_nsd,
        "per_organ_n_vols": per_organ_count,
        "mean_dsc_K4": overall_mean_dsc,
        "mean_hd95_mm_K4": overall_mean_hd95,
        "mean_nsd_K4": overall_mean_nsd,
        "liver_dsc": liver_dsc,
        "gates": {
            gn: {
                "mean_floor": mf,
                "liver_floor": lf,
                "mean_pass": overall_mean_dsc >= mf,
                "liver_pass": liver_dsc >= lf,
            } for gn, (_, mf, lf) in GATES.items()
        },
    }
    return report, vm.records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True,
                    help="path to a path_a1_big_solid_*.pt checkpoint")
    ap.add_argument("--tag", type=str, default="run",
                    help="tag for the output JSON (e.g. ep4)")
    ap.add_argument("--max-volumes", type=int, default=-1,
                    help="cap eval volumes; -1 = all CT val")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print("=" * 70)
    print(f"Path A1 -- Per-organ 3D eval ({args.tag})")
    print("=" * 70)
    print(f"device={args.device}  torch={torch.__version__}")
    print(f"target organs: {list(TARGET_ORGANS.values())}  (AMOS22 IDs {sorted(TARGET_ORGANS)})")
    print()

    cfg = load_a1_cfg()
    report, records = evaluate(
        cfg=cfg,
        ckpt_path=Path(args.ckpt),
        device=args.device,
        max_volumes=args.max_volumes,
        threshold=args.threshold,
    )

    out_dir = ROOT / "evaluation" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"path_a1_{args.tag}_per_organ.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[eval-a1] report saved to {out_path}")

    rec_path = out_dir / f"path_a1_{args.tag}_per_organ_records.json"
    # Replace NaN with None for valid JSON
    clean_records = []
    for r in records:
        clean_records.append({
            k: (None if isinstance(v, float) and np.isnan(v) else v)
            for k, v in r.items()
        })
    with open(rec_path, "w") as f:
        json.dump(clean_records, f, indent=2)
    print(f"[eval-a1] per-volume records saved to {rec_path}")


if __name__ == "__main__":
    main()
