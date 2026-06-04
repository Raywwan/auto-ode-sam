# C:\Users\Raywa\Desktop\LiteSAM3D_v2\evaluation\volume_reconstructor.py
# =============================================================================
# Volume Reconstructor — Assembles 2D slice predictions into 3D volumes
#
# LiteSAM-3D processes volumes slice-by-slice. During evaluation, we must
# reconstruct the full 3D volume from individual slice predictions before
# computing volumetric metrics (DSC, HD95, NSD).
#
# This module:
#   1. Accumulates 2D slice predictions indexed by (patient_id, slice_idx).
#   2. Reconstructs 3D volumes sorted by slice index.
#   3. Computes all 3D metrics per patient.
#   4. Optionally resamples predictions back to original NIfTI voxel space.
# =============================================================================

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from evaluation.metrics_3d import (
    ORGAN_NSD_TOLERANCES_MM,
    VolumetricMetrics,
    dice_score_3d,
    hausdorff_distance_95_mm,
    normalised_surface_dice_3d,
)

try:
    import SimpleITK as sitk
    _HAS_SITK = True
except ImportError:
    _HAS_SITK = False


class VolumeReconstructor:
    """
    Accumulates 2D slice predictions and reconstructs 3D volumes.

    Slices are stored indexed by (patient_id, slice_idx) and assembled into
    contiguous (D, H, W) arrays sorted by slice_idx when requested.

    Usage:
        >>> recon = VolumeReconstructor()
        >>> for i in range(5):
        ...     pred = np.ones((64, 64), dtype=np.uint8)
        ...     gt = np.ones((64, 64), dtype=np.uint8)
        ...     recon.add_slice("patient_001", i, pred, gt,
        ...                     spacing_mm=(1.5, 1.5, 1.5), organ_id=6)
        >>> pred_3d, gt_3d = recon.reconstruct_patient("patient_001")
        >>> pred_3d.shape
        (5, 64, 64)
    """

    def __init__(self) -> None:
        # {patient_id: {slice_idx: {"pred": 2d_array, "gt": 2d_array}}}
        self._slices: Dict[str, Dict[int, Dict[str, np.ndarray]]] = defaultdict(dict)
        # {patient_id: {"spacing_mm": tuple, "organ_id": int}}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def add_slice(
        self,
        patient_id: str,
        slice_idx: int,
        pred_2d: np.ndarray,
        gt_2d: np.ndarray,
        spacing_mm: Tuple[float, ...] = (1.5, 1.5, 1.5),
        organ_id: int = 0,
    ) -> None:
        """
        Add one 2D slice prediction/ground-truth pair.

        Args:
            patient_id: Unique patient identifier.
            slice_idx: Integer depth index of this slice in the volume.
                Must be unique per patient. Slices are sorted by this index
                during reconstruction.
            pred_2d: Binary prediction of shape (H, W).
            gt_2d: Binary ground truth of shape (H, W).
            spacing_mm: Physical voxel spacing (D, H, W) in mm.
            organ_id: Integer organ label for this slice.

        Raises:
            ValueError: If slice_idx already exists for this patient.
        """
        if slice_idx in self._slices[patient_id]:
            raise ValueError(
                f"Duplicate slice_idx {slice_idx} for patient {patient_id}"
            )

        self._slices[patient_id][slice_idx] = {
            "pred": pred_2d.astype(np.uint8),
            "gt": gt_2d.astype(np.uint8),
        }

        # Store metadata (use first slice's metadata; assume consistent)
        if patient_id not in self._metadata:
            self._metadata[patient_id] = {
                "spacing_mm": tuple(spacing_mm),
                "organ_id": organ_id,
            }

    def get_patient_ids(self) -> List[str]:
        """Return list of all patient IDs with stored slices."""
        return list(self._slices.keys())

    def reconstruct_patient(
        self, patient_id: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct 3D volume from stored 2D slices for one patient.

        Slices are stacked in sorted order by slice_idx to form (D, H, W).

        Args:
            patient_id: Patient identifier.

        Returns:
            Tuple of (pred_3d, gt_3d), each of shape (D, H, W) as uint8.

        Raises:
            KeyError: If patient_id has no stored slices.
            ValueError: If patient has zero slices stored.
        """
        if patient_id not in self._slices or len(self._slices[patient_id]) == 0:
            raise KeyError(f"No slices stored for patient '{patient_id}'")

        patient_slices = self._slices[patient_id]
        sorted_indices = sorted(patient_slices.keys())

        # Check for gaps in slice indices (warning, not error)
        expected = list(range(sorted_indices[0], sorted_indices[-1] + 1))
        if sorted_indices != expected:
            import warnings
            missing = set(expected) - set(sorted_indices)
            warnings.warn(
                f"Patient {patient_id}: missing slice indices {missing}. "
                f"Gaps will be filled with zeros.",
                stacklevel=2,
            )
            # Fill gaps with zeros matching the spatial shape of existing slices
            sample_shape = patient_slices[sorted_indices[0]]["pred"].shape
            for idx in missing:
                patient_slices[idx] = {
                    "pred": np.zeros(sample_shape, dtype=np.uint8),
                    "gt": np.zeros(sample_shape, dtype=np.uint8),
                }
            sorted_indices = expected

        pred_slices = [patient_slices[i]["pred"] for i in sorted_indices]
        gt_slices = [patient_slices[i]["gt"] for i in sorted_indices]

        pred_3d = np.stack(pred_slices, axis=0)  # (D, H, W)
        gt_3d = np.stack(gt_slices, axis=0)  # (D, H, W)

        return pred_3d, gt_3d

    def compute_patient_metrics(
        self,
        patient_id: str,
        organ_id: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Compute all 3D volumetric metrics for one patient.

        Args:
            patient_id: Patient identifier.
            organ_id: Override organ ID (otherwise uses stored metadata).

        Returns:
            Dict with keys: dsc, hd95_mm, nsd, tolerance_mm.
        """
        pred_3d, gt_3d = self.reconstruct_patient(patient_id)
        meta = self._metadata.get(patient_id, {})
        spacing_mm = meta.get("spacing_mm", (1.5, 1.5, 1.5))
        oid = organ_id if organ_id is not None else meta.get("organ_id", 0)
        tolerance_mm = ORGAN_NSD_TOLERANCES_MM.get(oid, 2.0)

        dsc = dice_score_3d(pred_3d, gt_3d, spacing_mm)
        hd95 = hausdorff_distance_95_mm(pred_3d, gt_3d, spacing_mm)
        nsd = normalised_surface_dice_3d(pred_3d, gt_3d, spacing_mm, tolerance_mm)

        return {
            "patient_id": patient_id,
            "organ_id": oid,
            "dsc": dsc,
            "hd95_mm": hd95,
            "nsd": nsd,
            "tolerance_mm": tolerance_mm,
        }

    def compute_all_metrics(self) -> VolumetricMetrics:
        """
        Compute 3D metrics for ALL stored patients.

        Returns:
            VolumetricMetrics instance with all results accumulated.
        """
        vm = VolumetricMetrics()

        for patient_id in self.get_patient_ids():
            pred_3d, gt_3d = self.reconstruct_patient(patient_id)
            meta = self._metadata.get(patient_id, {})
            spacing_mm = meta.get("spacing_mm", (1.5, 1.5, 1.5))
            organ_id = meta.get("organ_id", 0)

            vm.update(
                pred_vol=pred_3d,
                gt_vol=gt_3d,
                spacing_mm=spacing_mm,
                organ_id=organ_id,
                patient_id=patient_id,
            )

        return vm

    def clear(self) -> None:
        """Remove all stored slices and metadata."""
        self._slices.clear()
        self._metadata.clear()


def resample_to_original_space(
    pred_resampled: np.ndarray,
    original_nifti_path: str,
    resampled_spacing: Tuple[float, float, float] = (1.5, 1.5, 1.5),
) -> np.ndarray:
    """
    Resample a prediction volume back to the original NIfTI voxel space.

    During preprocessing, volumes are resampled to isotropic spacing (e.g.,
    1.5mm). For challenge submission and fair comparison, predictions must be
    mapped back to the original resolution using nearest-neighbour
    interpolation (to preserve label integrity).

    Args:
        pred_resampled: Prediction volume at resampled spacing, shape (D, H, W).
            Values should be integer labels (0 = background, 1+ = organs).
        original_nifti_path: Path to the original NIfTI file (.nii.gz).
            Used to extract original size, spacing, direction, and origin.
        resampled_spacing: The isotropic spacing (mm) used during preprocessing.

    Returns:
        Prediction volume at original resolution, shape matching the original
        NIfTI file dimensions.

    Raises:
        ImportError: If SimpleITK is not installed.
        FileNotFoundError: If original_nifti_path does not exist.
    """
    if not _HAS_SITK:
        raise ImportError(
            "SimpleITK is required for resample_to_original_space(). "
            "Install with: pip install SimpleITK"
        )

    # Load original NIfTI to get reference geometry
    original_img = sitk.ReadImage(original_nifti_path)
    original_size = original_img.GetSize()          # (W, H, D) in SimpleITK
    original_spacing = original_img.GetSpacing()    # (W, H, D)
    original_origin = original_img.GetOrigin()
    original_direction = original_img.GetDirection()

    # Convert prediction to SimpleITK image
    # numpy (D, H, W) -> SimpleITK expects (D, H, W) via GetImageFromArray
    pred_sitk = sitk.GetImageFromArray(pred_resampled.astype(np.uint8))
    # Set the resampled spacing: SimpleITK expects (W, H, D)
    pred_sitk.SetSpacing(
        (float(resampled_spacing[2]), float(resampled_spacing[1]), float(resampled_spacing[0]))
    )
    pred_sitk.SetOrigin(original_origin)
    pred_sitk.SetDirection(original_direction)

    # Resample to original space with nearest-neighbour interpolation
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(original_img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetOutputPixelType(sitk.sitkUInt8)

    resampled_pred = resampler.Execute(pred_sitk)

    return sitk.GetArrayFromImage(resampled_pred)  # (D, H, W)
