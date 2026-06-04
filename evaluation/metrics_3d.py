# C:\Users\Raywa\Desktop\LiteSAM3D_v2\evaluation\metrics_3d.py
# =============================================================================
# 3D Volumetric Metrics for Medical Image Segmentation
#
# All metrics operate on 3D numpy arrays (D, H, W) in PHYSICAL space.
# Distances are computed in millimetres using voxel spacing.
#
# Fixes over LiteSAM-3D v1:
#   - v1 bug: DSC computed on 2D slices (inflated by 1-3%)
#   - v1 bug: HD95 in pixels, not mm (incomparable to literature)
#   - v1 bug: NSD tolerance in pixels, not mm
#
# This module provides:
#   - dice_score_3d: Volumetric Dice Similarity Coefficient
#   - hausdorff_distance_95_mm: 95th percentile Hausdorff distance in mm
#   - normalised_surface_dice_3d: Normalised Surface Dice with organ-specific
#     tolerance in mm
#   - VolumetricMetrics: Accumulator for per-volume, per-organ results with
#     statistical testing support
# =============================================================================

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

try:
    from scipy import ndimage, stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Organ-specific NSD tolerances (mm) from AMOS22 challenge
# ---------------------------------------------------------------------------
ORGAN_NSD_TOLERANCES_MM: Dict[int, float] = {
    1: 1.0,    # spleen
    2: 1.0,    # right kidney
    3: 1.0,    # left kidney
    4: 2.0,    # gallbladder (small, variable shape)
    5: 2.0,    # esophagus
    6: 1.0,    # liver
    7: 2.0,    # stomach
    8: 1.5,    # aorta
    9: 1.5,    # inferior vena cava
    10: 2.0,   # pancreas
    11: 2.0,   # right adrenal gland
    12: 2.0,   # left adrenal gland
    13: 3.0,   # duodenum
    14: 2.0,   # bladder
    15: 2.0,   # prostate/uterus
}

ORGAN_NAMES: Dict[int, str] = {
    1: "spleen", 2: "r_kidney", 3: "l_kidney", 4: "gallbladder",
    5: "esophagus", 6: "liver", 7: "stomach", 8: "aorta",
    9: "ivc", 10: "pancreas", 11: "r_adrenal", 12: "l_adrenal",
    13: "duodenum", 14: "bladder", 15: "prostate_uterus",
}


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def dice_score_3d(
    pred_vol: np.ndarray,
    gt_vol: np.ndarray,
    spacing_mm: Tuple[float, ...] = (1.5, 1.5, 1.5),
) -> float:
    """
    Compute 3D volumetric Dice Similarity Coefficient.

    DSC = 2 * |P intersection G| / (|P| + |G|)

    Operates on full 3D volumes, not individual slices.

    Args:
        pred_vol: Binary prediction array of shape (D, H, W).
        gt_vol: Binary ground truth array of shape (D, H, W).
        spacing_mm: Voxel spacing in mm (unused for DSC but kept for
            consistent API). DSC is dimensionless.

    Returns:
        DSC value in [0, 1]. Returns NaN if both pred and gt are empty.

    >>> dice_score_3d(np.ones((3,3,3), dtype=bool), np.ones((3,3,3), dtype=bool))
    1.0
    >>> dice_score_3d(np.zeros((3,3,3), dtype=bool), np.zeros((3,3,3), dtype=bool))
    nan
    >>> round(dice_score_3d(np.ones((3,3,3), dtype=bool), np.zeros((3,3,3), dtype=bool)), 4)
    0.0
    """
    pred_bool = pred_vol.astype(bool)
    gt_bool = gt_vol.astype(bool)

    pred_sum = pred_bool.sum()
    gt_sum = gt_bool.sum()

    # Both empty: undefined (NaN, not 1.0 — avoids inflating mean DSC)
    if pred_sum == 0 and gt_sum == 0:
        return float("nan")

    # One empty: DSC = 0
    if pred_sum == 0 or gt_sum == 0:
        return 0.0

    intersection = np.logical_and(pred_bool, gt_bool).sum()
    return float(2.0 * intersection / (pred_sum + gt_sum))


def hausdorff_distance_95_mm(
    pred_vol: np.ndarray,
    gt_vol: np.ndarray,
    spacing_mm: Tuple[float, ...] = (1.5, 1.5, 1.5),
) -> float:
    """
    Compute 95th percentile Hausdorff Distance in millimetres.

    Uses scipy.ndimage.distance_transform_edt with physical spacing to
    compute surface-to-surface distances in mm.

    HD95 = max(d95(P->G), d95(G->P))
    where d95(A->B) = 95th percentile of {min_b d(a,b) : a in surface(A)}

    Args:
        pred_vol: Binary prediction array of shape (D, H, W).
        gt_vol: Binary ground truth array of shape (D, H, W).
        spacing_mm: Voxel spacing in mm as (D_spacing, H_spacing, W_spacing).

    Returns:
        HD95 in millimetres. Returns NaN if either volume is empty.

    >>> hd = hausdorff_distance_95_mm(
    ...     np.ones((5,5,5), dtype=bool), np.ones((5,5,5), dtype=bool),
    ...     spacing_mm=(1.0, 1.0, 1.0))
    >>> hd == 0.0
    True
    """
    if not _HAS_SCIPY:
        raise ImportError("scipy is required for hausdorff_distance_95_mm")

    pred_bool = pred_vol.astype(bool)
    gt_bool = gt_vol.astype(bool)

    # Handle empty volumes
    if pred_bool.sum() == 0 or gt_bool.sum() == 0:
        return float("nan")

    # Extract surfaces (border voxels)
    pred_surface = pred_bool ^ ndimage.binary_erosion(
        pred_bool, structure=ndimage.generate_binary_structure(3, 1)
    )
    gt_surface = gt_bool ^ ndimage.binary_erosion(
        gt_bool, structure=ndimage.generate_binary_structure(3, 1)
    )

    # Handle single-voxel case (erosion removes everything)
    if pred_surface.sum() == 0:
        pred_surface = pred_bool
    if gt_surface.sum() == 0:
        gt_surface = gt_bool

    # Distance transforms with physical spacing (mm)
    # dist_to_pred[voxel] = distance in mm from voxel to nearest pred surface
    dist_to_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing_mm)
    dist_to_gt = ndimage.distance_transform_edt(~gt_surface, sampling=spacing_mm)

    # Surface-to-surface distances
    # For each gt surface voxel, distance to nearest pred surface voxel
    gt_to_pred_dists = dist_to_pred[gt_surface]
    pred_to_gt_dists = dist_to_gt[pred_surface]

    # HD95 = max of 95th percentiles in both directions
    hd95 = max(
        np.percentile(gt_to_pred_dists, 95),
        np.percentile(pred_to_gt_dists, 95),
    )
    return float(hd95)


def normalised_surface_dice_3d(
    pred_vol: np.ndarray,
    gt_vol: np.ndarray,
    spacing_mm: Tuple[float, ...] = (1.5, 1.5, 1.5),
    tolerance_mm: float = 2.0,
) -> float:
    """
    Compute Normalised Surface Dice (NSD) with tolerance in millimetres.

    NSD measures the fraction of the surface that is within a given physical
    distance tolerance. It is the standard metric for the AMOS22 challenge.

    NSD = (|S_p intersect B_tau(S_g)| + |S_g intersect B_tau(S_p)|) /
          (|S_p| + |S_g|)

    where S_p, S_g are prediction/GT surfaces and B_tau is the tau-ball
    (all voxels within tolerance_mm of the surface).

    Args:
        pred_vol: Binary prediction array of shape (D, H, W).
        gt_vol: Binary ground truth array of shape (D, H, W).
        spacing_mm: Voxel spacing in mm.
        tolerance_mm: Surface distance tolerance in mm.

    Returns:
        NSD in [0, 1]. Returns NaN if both are empty.

    >>> nsd = normalised_surface_dice_3d(
    ...     np.ones((5,5,5), dtype=bool), np.ones((5,5,5), dtype=bool),
    ...     spacing_mm=(1.0, 1.0, 1.0), tolerance_mm=1.0)
    >>> nsd == 1.0
    True
    """
    if not _HAS_SCIPY:
        raise ImportError("scipy is required for normalised_surface_dice_3d")

    pred_bool = pred_vol.astype(bool)
    gt_bool = gt_vol.astype(bool)

    # Both empty: undefined
    if pred_bool.sum() == 0 and gt_bool.sum() == 0:
        return float("nan")

    # One empty: NSD = 0
    if pred_bool.sum() == 0 or gt_bool.sum() == 0:
        return 0.0

    # Extract surfaces
    struct = ndimage.generate_binary_structure(3, 1)
    pred_surface = pred_bool ^ ndimage.binary_erosion(pred_bool, structure=struct)
    gt_surface = gt_bool ^ ndimage.binary_erosion(gt_bool, structure=struct)

    # Handle single-voxel case
    if pred_surface.sum() == 0:
        pred_surface = pred_bool
    if gt_surface.sum() == 0:
        gt_surface = gt_bool

    # Distance transforms in mm
    dist_to_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing_mm)
    dist_to_gt = ndimage.distance_transform_edt(~gt_surface, sampling=spacing_mm)

    # Count surface voxels within tolerance
    pred_surface_count = pred_surface.sum()
    gt_surface_count = gt_surface.sum()

    # Pred surface voxels within tolerance of GT surface
    pred_within_tol = (dist_to_gt[pred_surface] <= tolerance_mm).sum()
    # GT surface voxels within tolerance of pred surface
    gt_within_tol = (dist_to_pred[gt_surface] <= tolerance_mm).sum()

    nsd = float((pred_within_tol + gt_within_tol) / (pred_surface_count + gt_surface_count))
    return nsd


def extra_metrics_3d(
    pred_vol: np.ndarray,
    gt_vol: np.ndarray,
    spacing_mm: Tuple[float, ...] = (1.5, 1.5, 1.5),
    nsd_tolerances_mm: Tuple[float, ...] = (1.0, 2.0),
) -> Dict[str, float]:
    """
    Compute IoU, ASSD, sensitivity/precision/specificity, volume similarity,
    and NSD at multiple tolerances. Pure numpy + scipy.ndimage.
    """
    try:
        from scipy import ndimage
    except ImportError:
        raise ImportError("scipy is required for extra_metrics_3d")

    p = pred_vol.astype(bool)
    g = gt_vol.astype(bool)
    out: Dict[str, float] = {}

    inter = int(np.logical_and(p, g).sum())
    union = int(np.logical_or(p, g).sum())
    out["iou"] = float(inter / union) if union else float("nan")

    tp = inter
    fp = int(p.sum() - tp)
    fn = int(g.sum() - tp)
    tn = int(p.size - tp - fp - fn)
    out["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) else float("nan")
    out["precision"]   = float(tp / (tp + fp)) if (tp + fp) else float("nan")
    out["specificity"] = float(tn / (tn + fp)) if (tn + fp) else float("nan")

    vox_mm3 = float(np.prod(spacing_mm))
    vp = float(p.sum()) * vox_mm3
    vg = float(g.sum()) * vox_mm3
    out["vol_pred_ml"] = vp / 1000.0
    out["vol_gt_ml"]   = vg / 1000.0
    out["vol_sim"]     = float(1.0 - abs(vp - vg) / (vp + vg)) if (vp + vg) else float("nan")

    if p.sum() == 0 or g.sum() == 0:
        out["assd_mm"] = float("nan")
        for t in nsd_tolerances_mm:
            out[f"nsd_at_{t}mm"] = float("nan")
        return out

    struct = ndimage.generate_binary_structure(3, 1)
    sp = p ^ ndimage.binary_erosion(p, structure=struct)
    sg = g ^ ndimage.binary_erosion(g, structure=struct)
    if sp.sum() == 0:
        sp = p
    if sg.sum() == 0:
        sg = g
    d_to_p = ndimage.distance_transform_edt(~sp, sampling=spacing_mm)
    d_to_g = ndimage.distance_transform_edt(~sg, sampling=spacing_mm)
    out["assd_mm"] = float(0.5 * (d_to_p[sg].mean() + d_to_g[sp].mean()))
    for t in nsd_tolerances_mm:
        sp_count = int(sp.sum())
        sg_count = int(sg.sum())
        within = int((d_to_g[sp] <= t).sum() + (d_to_p[sg] <= t).sum())
        denom = sp_count + sg_count
        out[f"nsd_at_{t}mm"] = float(within / denom) if denom else float("nan")

    return out


# ---------------------------------------------------------------------------
# VolumetricMetrics accumulator
# ---------------------------------------------------------------------------

class VolumetricMetrics:
    """
    Accumulates per-volume, per-organ 3D segmentation metrics.

    Stores DSC, HD95 (mm), and NSD (mm) for each (patient_id, organ_id) pair.
    Provides summary statistics, DataFrame export, and statistical testing.

    Usage:
        >>> vm = VolumetricMetrics()
        >>> pred = np.ones((10, 10, 10), dtype=bool)
        >>> gt = np.ones((10, 10, 10), dtype=bool)
        >>> vm.update(pred, gt, spacing_mm=(1.0, 1.0, 1.0), organ_id=1, patient_id="p001")
        >>> s = vm.summary()
        >>> round(s['overall']['dsc_mean'], 1)
        1.0
    """

    def __init__(self) -> None:
        # Storage: list of dicts, one per (patient, organ) pair
        self._records: List[Dict[str, Any]] = []

    def update(
        self,
        pred_vol: np.ndarray,
        gt_vol: np.ndarray,
        spacing_mm: Tuple[float, ...],
        organ_id: int,
        patient_id: str,
    ) -> Dict[str, float]:
        """
        Compute and store all 3D metrics for one (patient, organ) volume.

        Args:
            pred_vol: Binary prediction (D, H, W).
            gt_vol: Binary ground truth (D, H, W).
            spacing_mm: Physical voxel spacing in mm.
            organ_id: Integer organ label (1-15 for AMOS22).
            patient_id: Unique patient identifier string.

        Returns:
            Dict with computed metric values for this volume.
        """
        tolerance_mm = ORGAN_NSD_TOLERANCES_MM.get(organ_id, 2.0)

        dsc = dice_score_3d(pred_vol, gt_vol, spacing_mm)
        hd95 = hausdorff_distance_95_mm(pred_vol, gt_vol, spacing_mm)
        nsd = normalised_surface_dice_3d(pred_vol, gt_vol, spacing_mm, tolerance_mm)

        record = {
            "patient_id": patient_id,
            "organ_id": organ_id,
            "organ_name": ORGAN_NAMES.get(organ_id, f"organ_{organ_id}"),
            "dsc": dsc,
            "hd95_mm": hd95,
            "nsd": nsd,
            "tolerance_mm": tolerance_mm,
        }
        try:
            record.update(extra_metrics_3d(pred_vol, gt_vol, spacing_mm))
        except Exception as e:
            warnings.warn(f"extra_metrics_3d failed for {patient_id}/{organ_id}: {e}")
        self._records.append(record)
        return record

    @property
    def records(self) -> List[Dict[str, Any]]:
        """Access raw records list."""
        return self._records

    def summary(self) -> Dict[str, Dict[str, float]]:
        """
        Compute mean/std per organ and overall mean.

        Returns:
            Dict with keys = organ names + 'overall', values = dict of
            metric_name -> value. Metric names: dsc_mean, dsc_std,
            hd95_mm_mean, hd95_mm_std, nsd_mean, nsd_std, n_volumes.
        """
        if not self._records:
            return {}

        result: Dict[str, Dict[str, float]] = {}

        # Group by organ
        organ_groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for rec in self._records:
            organ_groups[rec["organ_id"]].append(rec)

        # Extra metric keys to aggregate (mean/std).  Skipped silently when absent.
        EXTRA_KEYS = (
            "iou", "assd_mm", "sensitivity", "precision", "specificity",
            "vol_sim", "vol_pred_ml", "vol_gt_ml",
            "nsd_at_1.0mm", "nsd_at_2.0mm",
        )

        all_dsc: List[float] = []
        all_hd95: List[float] = []
        all_nsd: List[float] = []
        all_extra: Dict[str, List[float]] = {k: [] for k in EXTRA_KEYS}

        for organ_id in sorted(organ_groups.keys()):
            records = organ_groups[organ_id]
            organ_name = ORGAN_NAMES.get(organ_id, f"organ_{organ_id}")

            dscs = np.array([r["dsc"] for r in records], dtype=np.float64)
            hd95s = np.array([r["hd95_mm"] for r in records], dtype=np.float64)
            nsds = np.array([r["nsd"] for r in records], dtype=np.float64)

            organ_summary = {
                "organ_id": organ_id,
                "dsc_mean": float(np.nanmean(dscs)),
                "dsc_std": float(np.nanstd(dscs)),
                "hd95_mm_mean": float(np.nanmean(hd95s)),
                "hd95_mm_std": float(np.nanstd(hd95s)),
                "nsd_mean": float(np.nanmean(nsds)),
                "nsd_std": float(np.nanstd(nsds)),
                "n_volumes": len(records),
            }

            for k in EXTRA_KEYS:
                vals = np.array(
                    [r[k] for r in records if k in r], dtype=np.float64
                )
                if vals.size > 0:
                    organ_summary[f"{k}_mean"] = float(np.nanmean(vals))
                    organ_summary[f"{k}_std"]  = float(np.nanstd(vals))
                    finite = vals[~np.isnan(vals)].tolist()
                    all_extra[k].extend(finite)

            result[organ_name] = organ_summary

            # Accumulate for overall (only non-NaN)
            all_dsc.extend(dscs[~np.isnan(dscs)].tolist())
            all_hd95.extend(hd95s[~np.isnan(hd95s)].tolist())
            all_nsd.extend(nsds[~np.isnan(nsds)].tolist())

        # Overall
        overall = {
            "dsc_mean": float(np.mean(all_dsc)) if all_dsc else float("nan"),
            "dsc_std": float(np.std(all_dsc)) if all_dsc else float("nan"),
            "hd95_mm_mean": float(np.mean(all_hd95)) if all_hd95 else float("nan"),
            "hd95_mm_std": float(np.std(all_hd95)) if all_hd95 else float("nan"),
            "nsd_mean": float(np.mean(all_nsd)) if all_nsd else float("nan"),
            "nsd_std": float(np.std(all_nsd)) if all_nsd else float("nan"),
            "n_volumes": len(self._records),
        }
        for k, vals in all_extra.items():
            if vals:
                overall[f"{k}_mean"] = float(np.mean(vals))
                overall[f"{k}_std"]  = float(np.std(vals))
        result["overall"] = overall

        return result

    def to_dataframe(self) -> "pd.DataFrame":
        """
        Export all records as a pandas DataFrame.

        Returns:
            DataFrame with columns: patient_id, organ_id, organ_name,
            dsc, hd95_mm, nsd, tolerance_mm.

        Raises:
            ImportError: If pandas is not installed.
        """
        if not _HAS_PANDAS:
            raise ImportError("pandas is required for to_dataframe()")
        return pd.DataFrame(self._records)

    def wilcoxon_test(
        self,
        other: "VolumetricMetrics",
        metric: str = "dsc",
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare this result set against another using Wilcoxon signed-rank test.

        Both must have results for the same (patient_id, organ_id) pairs.
        This is a non-parametric paired test suitable for comparing two
        segmentation methods on the same patients.

        Args:
            other: Another VolumetricMetrics instance (e.g., baseline).
            metric: Which metric to compare ('dsc', 'hd95_mm', or 'nsd').

        Returns:
            Dict with keys = organ names + 'overall', values = dict with
            'statistic', 'p_value', 'n_pairs', 'mean_diff'.

        Raises:
            ImportError: If scipy is not installed.
            ValueError: If paired samples cannot be matched.
        """
        if not _HAS_SCIPY:
            raise ImportError("scipy is required for wilcoxon_test()")

        # Build lookup: (patient_id, organ_id) -> metric value
        def _build_lookup(records: List[Dict[str, Any]]) -> Dict[Tuple[str, int], float]:
            lookup = {}
            for rec in records:
                key = (rec["patient_id"], rec["organ_id"])
                lookup[key] = rec[metric]
            return lookup

        self_lookup = _build_lookup(self._records)
        other_lookup = _build_lookup(other._records)

        # Find common keys
        common_keys = set(self_lookup.keys()) & set(other_lookup.keys())
        if len(common_keys) == 0:
            raise ValueError(
                "No matching (patient_id, organ_id) pairs between the two result sets"
            )

        result: Dict[str, Dict[str, float]] = {}

        # Per-organ tests
        organ_groups: Dict[int, List[Tuple[str, int]]] = defaultdict(list)
        for key in common_keys:
            organ_groups[key[1]].append(key)

        all_self_vals: List[float] = []
        all_other_vals: List[float] = []

        for organ_id in sorted(organ_groups.keys()):
            keys = organ_groups[organ_id]
            organ_name = ORGAN_NAMES.get(organ_id, f"organ_{organ_id}")

            self_vals = np.array([self_lookup[k] for k in keys], dtype=np.float64)
            other_vals = np.array([other_lookup[k] for k in keys], dtype=np.float64)

            # Remove NaN pairs
            valid = ~(np.isnan(self_vals) | np.isnan(other_vals))
            self_valid = self_vals[valid]
            other_valid = other_vals[valid]

            if len(self_valid) < 5:
                # Too few samples for reliable test
                result[organ_name] = {
                    "statistic": float("nan"),
                    "p_value": float("nan"),
                    "n_pairs": int(valid.sum()),
                    "mean_diff": float(np.mean(self_valid - other_valid))
                    if len(self_valid) > 0 else float("nan"),
                }
            else:
                try:
                    stat_result = stats.wilcoxon(
                        self_valid, other_valid, alternative="two-sided"
                    )
                    result[organ_name] = {
                        "statistic": float(stat_result.statistic),
                        "p_value": float(stat_result.pvalue),
                        "n_pairs": int(valid.sum()),
                        "mean_diff": float(np.mean(self_valid - other_valid)),
                    }
                except ValueError:
                    # All differences are zero
                    result[organ_name] = {
                        "statistic": 0.0,
                        "p_value": 1.0,
                        "n_pairs": int(valid.sum()),
                        "mean_diff": 0.0,
                    }

            all_self_vals.extend(self_valid.tolist())
            all_other_vals.extend(other_valid.tolist())

        # Overall test
        all_self = np.array(all_self_vals)
        all_other = np.array(all_other_vals)
        if len(all_self) >= 5:
            try:
                stat_result = stats.wilcoxon(all_self, all_other, alternative="two-sided")
                result["overall"] = {
                    "statistic": float(stat_result.statistic),
                    "p_value": float(stat_result.pvalue),
                    "n_pairs": len(all_self),
                    "mean_diff": float(np.mean(all_self - all_other)),
                }
            except ValueError:
                result["overall"] = {
                    "statistic": 0.0,
                    "p_value": 1.0,
                    "n_pairs": len(all_self),
                    "mean_diff": 0.0,
                }
        else:
            result["overall"] = {
                "statistic": float("nan"),
                "p_value": float("nan"),
                "n_pairs": len(all_self),
                "mean_diff": float(np.mean(all_self - all_other))
                if len(all_self) > 0 else float("nan"),
            }

        return result
