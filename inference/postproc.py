"""Post-processing: connected-component filtering, hole filling.

Applied after sigmoid+threshold to final 3D predictions. Keeps the largest
connected component per organ (AMOS22 organs are anatomically contiguous, so
spurious small islands are almost always noise). Optional hole filling via
morphological closing.
"""
from __future__ import annotations

import numpy as np

try:
    from scipy.ndimage import label, binary_fill_holes
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """mask: (Z, H, W) or (H, W) binary. Return same shape with only the
    largest connected component retained."""
    if not _HAS_SCIPY:
        return mask
    if mask.sum() == 0:
        return mask
    lab, n = label(mask)
    if n <= 1:
        return mask
    # Argmax over components 1..n (skip 0 = background)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    keep = int(np.argmax(sizes))
    return (lab == keep).astype(mask.dtype)


def postprocess_multi_organ(
    prob_vol: np.ndarray,                   # (K, Z, H, W) or (K, H, W) float
    threshold: float = 0.5,
    fill_holes: bool = True,
    largest_cc: bool = True,
) -> np.ndarray:
    """Binarise + optional CC + hole-fill. Return uint8 mask same shape."""
    K = prob_vol.shape[0]
    out = (prob_vol >= threshold).astype(np.uint8)
    if not _HAS_SCIPY:
        return out
    for k in range(K):
        m = out[k]
        if fill_holes and m.ndim == 3:
            # Fill holes slice-wise in 2D (volumetric fill can merge organs)
            for z in range(m.shape[0]):
                m[z] = binary_fill_holes(m[z]).astype(np.uint8)
        elif fill_holes and m.ndim == 2:
            m = binary_fill_holes(m).astype(np.uint8)
        if largest_cc:
            m = keep_largest_component(m)
        out[k] = m
    return out
