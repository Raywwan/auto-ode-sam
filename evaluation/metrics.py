# =============================================================================
# evaluation/metrics.py — Segmentation Evaluation Metrics
#
# Implements the standard medical image segmentation metrics:
#
#   DSC  — Dice Similarity Coefficient (overlap, primary metric)
#   IoU  — Intersection over Union / Jaccard Index
#   HD95 — 95th Percentile Hausdorff Distance (surface distance, shape accuracy)
#   NSD  — Normalised Surface Dice (boundary accuracy, CVPR 2024 challenge metric)
#
# Also implements the SegmentationMetrics class for accumulating metrics
# over a full validation/test epoch.
# =============================================================================

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional
import numpy as np
from scipy.ndimage import distance_transform_edt


def _compute_sample_metrics(args):
    """Compute all four metrics for a single sample. Module-level for thread pool."""
    p, t, threshold, nsd_tolerance = args
    return (
        dice_score(p, t, threshold),
        iou_score(p, t, threshold),
        hausdorff_distance_95(p, t, threshold),
        normalised_surface_dice(p, t, threshold, nsd_tolerance),
    )


# =============================================================================
# Metric Functions (operate on numpy arrays)
# =============================================================================

def dice_score(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5,
    smooth: float = 1e-5,
) -> float:
    """
    Dice Similarity Coefficient (DSC).

    DSC = 2 * |P ∩ T| / (|P| + |T|)

    Range: [0, 1]. Higher is better.
    0 = no overlap, 1 = perfect overlap.

    Args:
        pred: Predicted probability map or binary mask. Any shape.
        target: Ground truth binary mask. Same shape as pred.
        threshold: Binarisation threshold for probabilities.
        smooth: Numerical stability constant.

    Returns:
        float: Dice score in [0, 1].
    """
    pred_bin = (pred > threshold).astype(float).flatten()
    target_flat = target.astype(float).flatten()

    intersection = (pred_bin * target_flat).sum()
    return float(
        (2.0 * intersection + smooth) / (pred_bin.sum() + target_flat.sum() + smooth)
    )


def iou_score(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5,
    smooth: float = 1e-5,
) -> float:
    """
    Intersection over Union (IoU) / Jaccard Index.

    IoU = |P ∩ T| / |P ∪ T| = |P ∩ T| / (|P| + |T| - |P ∩ T|)

    Range: [0, 1]. Higher is better.

    Args:
        pred: Predicted probability map. Any shape.
        target: Ground truth binary mask. Same shape.
        threshold: Binarisation threshold.
        smooth: Numerical stability constant.

    Returns:
        float: IoU score in [0, 1].
    """
    pred_bin = (pred > threshold).astype(float).flatten()
    target_flat = target.astype(float).flatten()

    intersection = (pred_bin * target_flat).sum()
    union = pred_bin.sum() + target_flat.sum() - intersection
    return float((intersection + smooth) / (union + smooth))


def hausdorff_distance_95(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """
    95th Percentile Hausdorff Distance (HD95).

    Measures the maximum distance between the two surfaces, but uses
    the 95th percentile to be robust to outliers.

    HD(P, T) = max(h(P→T), h(T→P))
    where h(A→B) = max_{a∈A} min_{b∈B} d(a, b)

    HD95 replaces max with 95th percentile — much more robust.

    Range: [0, inf). Lower is better.
    Unit: pixels (not mm, unless input is in mm).

    Args:
        pred: Predicted probability map or binary mask.
        target: Ground truth binary mask.
        threshold: Binarisation threshold.

    Returns:
        float: HD95 in pixel units. Returns inf if either mask is empty.
    """
    pred_bin = (pred > threshold).astype(bool)
    target_bin = target.astype(bool)

    # Handle empty masks
    if not pred_bin.any() or not target_bin.any():
        return float("inf")

    # Compute distance transforms from the surfaces
    # distance_transform_edt gives distance from EACH pixel to the nearest True pixel
    # We need the surface (boundary) of each mask
    dist_pred = distance_transform_edt(~pred_bin)    # Distance from each pixel to pred surface
    dist_target = distance_transform_edt(~target_bin)  # Distance from each pixel to GT surface

    # Distances from pred surface to GT: d(pred_surface → GT)
    dist_pred_to_gt = dist_target[pred_bin]   # Values AT pred surface pixels

    # Distances from GT surface to pred: d(GT_surface → pred)
    dist_gt_to_pred = dist_pred[target_bin]   # Values AT GT surface pixels

    # 95th percentile of the combined surface distances
    all_distances = np.concatenate([dist_pred_to_gt, dist_gt_to_pred])
    return float(np.percentile(all_distances, 95))


def normalised_surface_dice(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5,
    tolerance: float = 1.0,
) -> float:
    """
    Normalised Surface Dice (NSD).

    Measures what fraction of surface points from each mask are within
    `tolerance` pixels of the other mask's surface.

    NSD = (|pred_surf within tol of GT_surf| + |GT_surf within tol of pred_surf|)
          / (|pred_surf| + |GT_surf|)

    Used as the secondary metric in the CVPR 2024 medical segmentation challenge.
    Range: [0, 1]. Higher is better.

    Args:
        pred: Predicted probability map or binary mask.
        target: Ground truth binary mask.
        threshold: Binarisation threshold.
        tolerance: Surface tolerance in pixels (default=1.0).

    Returns:
        float: NSD score in [0, 1].
    """
    pred_bin = (pred > threshold).astype(bool)
    target_bin = target.astype(bool)

    if not pred_bin.any() or not target_bin.any():
        return 0.0

    # Compute distance transforms
    dist_from_pred = distance_transform_edt(~pred_bin)    # (H, W) — distance to pred surface
    dist_from_target = distance_transform_edt(~target_bin)  # (H, W) — distance to GT surface

    # Extract surface masks (use erosion: surface = mask XOR eroded_mask)
    # Simplified: any foreground pixel with at least one background neighbour = surface
    from scipy.ndimage import binary_erosion
    pred_surface = pred_bin & ~binary_erosion(pred_bin)
    target_surface = target_bin & ~binary_erosion(target_bin)

    if not pred_surface.any() or not target_surface.any():
        return 0.0

    # Fraction of pred surface within tolerance of GT surface
    pred_surf_within_tol = (dist_from_target[pred_surface] <= tolerance).mean()

    # Fraction of GT surface within tolerance of pred surface
    target_surf_within_tol = (dist_from_pred[target_surface] <= tolerance).mean()

    # Normalised Surface Dice
    nsd = (pred_surf_within_tol + target_surf_within_tol) / 2.0
    return float(nsd)


# =============================================================================
# Metric Accumulator
# =============================================================================

class SegmentationMetrics:
    """
    Accumulates segmentation metrics over a validation/test epoch.

    Usage:
        metrics = SegmentationMetrics(spacing_mm=1.5)
        for pred, gt in val_loader:
            metrics.update(pred.numpy(), gt.numpy())
        summary = metrics.summary()
        print(summary)
    """

    def __init__(self, threshold: float = 0.5, nsd_tolerance: float = 1.0, spacing_mm: float = 1.0):
        self.threshold = threshold
        self.nsd_tolerance = nsd_tolerance
        self.spacing_mm = spacing_mm  # pixel → mm conversion for HD95
        self._pool = ThreadPoolExecutor(max_workers=os.cpu_count())
        self.reset()

    def reset(self):
        """Clear all accumulated values."""
        self.dice_scores: List[float] = []
        self.iou_scores: List[float] = []
        self.hd95_scores: List[float] = []
        self.nsd_scores: List[float] = []
        self.soft_dice_scores: List[float] = []  # V4 multi-organ soft-Dice (threshold-free)
        self.n_samples: int = 0

    def add_soft_dice(self, values):
        """Append pre-computed per-volume soft-Dice scores (V4 multi-organ path)."""
        self.soft_dice_scores.extend(float(v) for v in values)

    def update(
        self,
        pred: np.ndarray,    # Probability map or binary mask, any batch shape
        target: np.ndarray,  # Binary ground truth, same shape
    ):
        """
        Update metrics with a batch of predictions.

        Handles both single samples and batches.
        Each sample is processed independently.

        Args:
            pred: (H, W) or (B, H, W) or (B, 1, H, W)
            target: Same shape as pred
        """
        # Normalise to (B, H, W)
        pred = np.squeeze(pred)
        target = np.squeeze(target)

        if pred.ndim == 2:
            # Single sample — wrap in batch dim
            pred = pred[np.newaxis]
            target = target[np.newaxis]

        args = [(pred[i], target[i], self.threshold, self.nsd_tolerance) for i in range(pred.shape[0])]
        for d, iou, h, n in self._pool.map(_compute_sample_metrics, args):
            self.dice_scores.append(d)
            self.iou_scores.append(iou)
            self.hd95_scores.append(h)
            self.nsd_scores.append(n)
            self.n_samples += 1

    def summary(self) -> Dict[str, float]:
        """
        Returns mean ± std of each metric.

        Returns:
            dict: {
                "dice_mean": ..., "dice_std": ...,
                "iou_mean":  ..., "iou_std":  ...,
                "hd95_mean": ..., "hd95_std": ...,
                "nsd_mean":  ..., "nsd_std":  ...,
                "n_samples": ...,
            }
        """
        if self.n_samples == 0:
            return {"error": "No samples accumulated. Call update() first."}

        # Filter out inf values for HD95 (empty masks give inf)
        hd95_valid = [h for h in self.hd95_scores if np.isfinite(h)]
        hd95_mm = [h * self.spacing_mm for h in hd95_valid]

        return {
            "dice_mean": float(np.mean(self.dice_scores)),
            "dice_std": float(np.std(self.dice_scores)),
            "iou_mean": float(np.mean(self.iou_scores)),
            "iou_std": float(np.std(self.iou_scores)),
            "hd95_mean": float(np.mean(hd95_mm)) if hd95_mm else float("inf"),
            "hd95_std": float(np.std(hd95_mm)) if hd95_mm else 0.0,
            "nsd_mean": float(np.mean(self.nsd_scores)),
            "nsd_std": float(np.std(self.nsd_scores)),
            "soft_dice_mean": float(np.mean(self.soft_dice_scores)) if self.soft_dice_scores else 0.0,
            "soft_dice_std": float(np.std(self.soft_dice_scores)) if self.soft_dice_scores else 0.0,
            "n_samples": self.n_samples,
        }

    def print_summary(self):
        """Pretty-print the metrics summary to console."""
        s = self.summary()
        if "error" in s:
            print(s["error"])
            return

        print("\n" + "="*55)
        print(f"  Segmentation Metrics (n={s['n_samples']} samples)")
        print("="*55)
        print(f"  DSC:  {s['dice_mean']:.4f} ± {s['dice_std']:.4f}")
        print(f"  IoU:  {s['iou_mean']:.4f} ± {s['iou_std']:.4f}")
        print(f"  HD95: {s['hd95_mean']:.2f} ± {s['hd95_std']:.2f} mm")
        print(f"  NSD:  {s['nsd_mean']:.4f} ± {s['nsd_std']:.4f}")
        print("="*55 + "\n")
