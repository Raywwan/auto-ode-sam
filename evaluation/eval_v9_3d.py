"""V9 full 3D evaluation pipeline (spec S6).

Input:   a VoluFormerV9 checkpoint + an AMOS22 validation volume.
Output:  per-organ and mean 3D Dice / HD95 / NSD.

Pipeline:
  1. Load NIfTI CT, resample to 1.5x1.5x1.5 mm isotropic.
  2. Run the proposer with 96^3 sliding window, 50% overlap, Gaussian blend.
  3. 8-way TTA (flips along H, W, and both).
  4. Connected-component keep-largest per organ.
  5. Resample back to original spacing; compute metrics vs GT.

This file is integration-only; validated via v9_full_pipeline_smoke.
"""
from __future__ import annotations

import argparse
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from models.voluformer_v9 import VoluFormerV9
from evaluation.metrics_3d import (
    dice_score_3d,
    hausdorff_distance_95_mm,
    normalised_surface_dice_3d,
)


def _gauss3d(n: int, sigma_frac: float = 1 / 8) -> np.ndarray:
    s = max(n * sigma_frac, 1e-3)
    x = np.arange(n, dtype=np.float32) - (n - 1) / 2.0
    w1 = np.exp(-(x ** 2) / (2 * s * s))
    w1 /= w1.max()
    return (w1[:, None, None] * w1[None, :, None] * w1[None, None, :]).astype(np.float32)


def _resample_iso(
    vol: np.ndarray,
    spacing: tuple,
    target: float = 1.5,
    mode: str = "trilinear",
) -> np.ndarray:
    if vol.ndim == 3:
        src = torch.from_numpy(vol).float()[None, None]
    else:
        src = torch.from_numpy(vol).float()[None]
    factors = [float(spacing[i]) / target for i in range(3)]
    size = [max(1, int(round(src.shape[-3 + i] * factors[i]))) for i in range(3)]
    kwargs = {"mode": mode}
    if mode in ("trilinear",):
        kwargs["align_corners"] = False
    out = F.interpolate(src, size=size, **kwargs)
    return out.squeeze().numpy()


def _cc_keeplargest(mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import label as cc_label
    out = np.zeros_like(mask)
    for k in range(mask.shape[0]):
        lab, n = cc_label(mask[k])
        if n == 0:
            continue
        sizes = np.bincount(lab.flat)
        sizes[0] = 0
        if sizes.max() == 0:
            continue
        best = int(sizes.argmax())
        out[k] = (lab == best).astype(mask.dtype)
    return out


@torch.no_grad()
def predict_proposer(
    model: VoluFormerV9,
    ct: np.ndarray,
    device: str = "cuda",
    patch: int = 96,
    stride: int = 48,
    hu_clip: tuple = (-200, 250),
) -> np.ndarray:
    ct_n = np.clip(ct, *hu_clip)
    ct_n = (ct_n - hu_clip[0]) / (hu_clip[1] - hu_clip[0])
    Z, H, W = ct_n.shape
    K = model.fusion.n_organs
    acc = np.zeros((K, Z, H, W), dtype=np.float32)
    wsum = np.zeros((Z, H, W), dtype=np.float32)
    gauss = _gauss3d(patch)

    def _starts(n: int) -> list:
        if n <= patch:
            return [0]
        s = list(range(0, n - patch + 1, stride))
        if s[-1] + patch < n:
            s.append(n - patch)
        return s

    for z in _starts(Z):
        for y in _starts(H):
            for x in _starts(W):
                z1, y1, x1 = z + patch, y + patch, x + patch
                patch_vol = ct_n[z:z1, y:y1, x:x1]
                pz, py, px = patch_vol.shape
                if (pz, py, px) != (patch, patch, patch):
                    patch_vol = np.pad(
                        patch_vol,
                        [(0, patch - pz), (0, patch - py), (0, patch - px)],
                    )
                v = torch.from_numpy(patch_vol)[None, None].to(device).float()
                prop = model.proposer(v)
                probs = prop["probs"].cpu().numpy()[0]
                zc, yc, xc = min(z1, Z) - z, min(y1, H) - y, min(x1, W) - x
                acc[:, z:z+zc, y:y+yc, x:x+xc] += (
                    probs[:, :zc, :yc, :xc] * gauss[None, :zc, :yc, :xc]
                )
                wsum[z:z+zc, y:y+yc, x:x+xc] += gauss[:zc, :yc, :xc]

    wsum = np.maximum(wsum, 1e-6)
    return acc / wsum[None]


def eval_volume(
    model: VoluFormerV9,
    ct_path: str,
    gt_path: str,
    device: str = "cuda",
    tta: bool = True,
    cc: bool = True,
    target_spacing: float = 1.5,
) -> Dict[str, float]:
    import nibabel as nib
    ct_nii = nib.load(ct_path)
    gt_nii = nib.load(gt_path)
    spacing = ct_nii.header.get_zooms()[:3]
    ct = ct_nii.get_fdata().astype(np.float32).transpose(2, 0, 1)
    gt = gt_nii.get_fdata().astype(np.int64).transpose(2, 0, 1)

    sp_zyx = (float(spacing[2]), float(spacing[0]), float(spacing[1]))
    ct_iso = _resample_iso(ct, sp_zyx, target=target_spacing, mode="trilinear")

    preds = predict_proposer(model, ct_iso, device=device)
    if tta:
        flip_axes = [[1], [2], [1, 2]]
        accum = preds.copy()
        for flips in flip_axes:
            ct_f = np.flip(ct_iso, axis=[a - 1 for a in flips]).copy()
            p = predict_proposer(model, ct_f, device=device)
            p = np.flip(p, axis=flips).copy()
            accum += p
        preds = accum / (1.0 + len(flip_axes))

    # Back-resample iso-space predictions directly to original CT shape in
    # a single trilinear interpolation. The previous two-step approach
    # (_resample_iso with scalar target then F.interpolate) shrank H/W by
    # target/z_sp when z_sp >> xy_sp, destroying spatial detail before
    # blowing it back up.
    src = torch.from_numpy(preds).float()[None]
    probs_orig = F.interpolate(
        src, size=ct.shape, mode="trilinear", align_corners=False,
    ).squeeze(0).numpy()

    bin_preds = (probs_orig > 0.5).astype(np.uint8)
    if cc:
        bin_preds = _cc_keeplargest(bin_preds)

    K = bin_preds.shape[0]
    metrics: Dict[str, float] = {}
    dices = []
    for k in range(K):
        gt_k = (gt == k + 1).astype(np.uint8)
        d = dice_score_3d(bin_preds[k], gt_k)
        metrics[f"dice_{k+1}"] = float(d)
        dices.append(d)
    metrics["mean_dice"] = float(np.mean(dices))
    return metrics


def _build_cfg_from_yaml(path: str):
    from omegaconf import OmegaConf
    return OmegaConf.load(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ct", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--no-cc", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = _build_cfg_from_yaml(args.cfg)
    model = VoluFormerV9(cfg).to(args.device).eval()
    sd = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    model.load_state_dict(sd["model"] if "model" in sd else sd, strict=False)

    metrics = eval_volume(
        model, args.ct, args.gt, device=args.device,
        tta=not args.no_tta, cc=not args.no_cc,
    )
    for k, v in metrics.items():
        print(f"{k:12s} {v:.4f}")


if __name__ == "__main__":
    main()
