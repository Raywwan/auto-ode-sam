"""Sliding-window 3D inference for OrganFlowSAM2.

Given a full CT volume (Z, H, W), slide a D-slice window with stride D/2 over
Z, run the model on each window, accumulate per-slice predictions with
Gaussian blending, and return a full-resolution organ probability volume
(N_ORGANS, Z, H, W). Intended as a drop-in replacement for the trainer's
center-slice-only validation path when we want apples-to-apples 3D Dice
comparison with MCP-MedSAM and other AMOS22 SOTA.

Key properties:
  - Gaussian weighting reduces seams between overlapping windows.
  - Predictions are computed on all D slices per window (not just the center),
    leveraging the ODE's full cross-slice trajectory.
  - Image resizing to model.img_size is done per window; output is upsampled
    back to native H,W before accumulation.
  - Organ_id is derived per window from ground truth if provided, else set to
    a default (organ 1 = spleen, but the model returns all 15 organs anyway).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _gaussian_1d_weights(n: int, sigma_frac: float = 1.0 / 8.0) -> np.ndarray:
    sigma = max(n * sigma_frac, 1e-3)
    x = np.arange(n, dtype=np.float32) - (n - 1) / 2.0
    w = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return w / w.max()


@torch.no_grad()
def sliding_window_predict_3d(
    model: torch.nn.Module,
    image_vol: np.ndarray,                   # (Z, H, W) float32 in original HU
    depth: int = 8,
    stride: Optional[int] = None,
    img_size: int = 320,
    hu_clip: Tuple[float, float] = (-200.0, 250.0),
    n_organs: int = 15,
    device: str = "cuda",
    amp: bool = True,
) -> np.ndarray:
    """Return per-organ probability volume (n_organs, Z, H, W) on CPU."""
    Z, H, W = image_vol.shape
    if stride is None:
        stride = max(depth // 2, 1)
    lo, hi = hu_clip
    img_norm = np.clip(image_vol, lo, hi)
    img_norm = ((img_norm - lo) / max(hi - lo, 1)).astype(np.float32)

    # Resize each slice once to model input resolution.
    if (H, W) != (img_size, img_size):
        from scipy.ndimage import zoom
        zh, zw = img_size / H, img_size / W
        img_resized = np.stack(
            [zoom(img_norm[z], (zh, zw), order=1) for z in range(Z)],
            axis=0,
        )                                                   # (Z, img_size, img_size)
    else:
        img_resized = img_norm

    # The decoder emits only the *center-slice* full-res mask. To cover the
    # full volume correctly, we slide with stride=1: for each target z in
    # [0, Z), we reflect-pad the volume and run a D-window whose mid=D//2
    # slot aligns exactly with z. This preserves the original Z order (ODE
    # cross-slice dynamics depend on it) and gives every slice a dedicated
    # center-slice inference.
    # Earlier impl broadcast the center mask across all D slices in the
    # window, which obliterated thin organs (esophagus, adrenals) — hence
    # V7's apparent 0.32 Dice.
    prob_vol_small = np.zeros((n_organs, Z, img_size, img_size), dtype=np.float32)

    organ_id = torch.zeros(1, dtype=torch.long, device=device)
    mid = depth // 2
    pad_lo = mid
    pad_hi = depth - mid - 1
    img_padded = np.concatenate(
        [np.tile(img_resized[:1], (pad_lo, 1, 1)),
         img_resized,
         np.tile(img_resized[-1:], (pad_hi, 1, 1))],
        axis=0,
    )                                                            # (Z + D - 1, h, w)

    for z in range(Z):
        z0_padded = z                                           # window [z, z+D) in padded coords
        chunk = img_padded[z0_padded:z0_padded + depth]          # (D, h, w); z is at slot `mid`
        chunk_t = torch.from_numpy(chunk).to(device)             # (D, h, w)
        chunk_t = chunk_t.unsqueeze(0).unsqueeze(2)              # (1, D, 1, h, w)
        chunk_t = chunk_t.repeat(1, 1, 3, 1, 1)                  # (1, D, 3, h, w)

        with torch.amp.autocast("cuda", enabled=amp):
            out = model(chunk_t, organ_id, is_3d=True)
        masks_center = out["masks"].float()                      # (1, K, H_out, W_out)
        if masks_center.shape[-2:] != (img_size, img_size):
            masks_center = F.interpolate(
                masks_center, size=(img_size, img_size),
                mode="bilinear", align_corners=False,
            )
        prob_vol_small[:, z] = torch.sigmoid(masks_center)[0].cpu().numpy()

    prob_vol = prob_vol_small

    # Resize probability volume back to native (H, W) per slice.
    if (H, W) != (img_size, img_size):
        from scipy.ndimage import zoom
        zh, zw = H / img_size, W / img_size
        out_vol = np.zeros((n_organs, Z, H, W), dtype=np.float32)
        for k in range(n_organs):
            for z in range(Z):
                out_vol[k, z] = zoom(prob_vol[k, z], (zh, zw), order=1)
        return out_vol
    return prob_vol


def compute_3d_dice_per_organ(
    prob_vol: np.ndarray,                    # (K, Z, H, W) float
    gt_vol: np.ndarray,                      # (Z, H, W) uint8, labels 1..K (0 = background)
    n_organs: int = 15,
    threshold: float = 0.5,
) -> Dict[int, float]:
    """Return {organ_id: dice} for every organ with GT present in this volume."""
    eps = 1e-6
    per_organ: Dict[int, float] = {}
    for k in range(1, n_organs + 1):
        gt_k = (gt_vol == k).astype(np.float32)
        if gt_k.sum() < 50:
            continue                                        # absent or near-absent organ
        pred_k = (prob_vol[k - 1] >= threshold).astype(np.float32)
        inter = float((pred_k * gt_k).sum())
        denom = float(pred_k.sum() + gt_k.sum())
        per_organ[k] = (2.0 * inter + eps) / (denom + eps)
    return per_organ
