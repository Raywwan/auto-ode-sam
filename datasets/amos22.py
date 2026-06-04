# =============================================================================
# datasets/amos22.py — AMOS22 Multi-Organ Segmentation Dataset
#
# AMOS22: A Large-Scale Abdominal Multi-Organ Benchmark for Versatile Medical
# Image Segmentation (NeurIPS 2022 Dataset Track)
#
# Download: https://amos22.grand-challenge.org/
# Paper: https://arxiv.org/abs/2206.08023
#
# Structure after download:
#   data/amos22/
#     imagesTr/   — 500 CT + 100 MRI training images (.nii.gz)
#     labelsTr/   — Corresponding segmentation masks
#     imagesVa/   — Validation images
#     labelsVa/   — Validation masks
#     imagesTs/   — Test images (no labels, for challenge submission)
#
# File naming:
#   CT  images: amos_0001.nii.gz — amos_0500.nii.gz
#   MRI images: amos_0501.nii.gz — amos_0600.nii.gz
#
# Labels (15 organs):
#   1=spleen, 2=right kidney, 3=left kidney, 4=gallbladder, 5=esophagus,
#   6=liver, 7=stomach, 8=aorta, 9=inferior vena cava, 10=pancreas,
#   11=right adrenal gland, 12=left adrenal gland, 13=duodenum,
#   14=bladder, 15=prostate/uterus
#
# This dataset class:
#   - Processes 3D NIfTI volumes slice-by-slice
#   - Supports training on specific organs or all 15
#   - Supports CT only, MRI only, or both
# =============================================================================

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import nibabel as nib
import torch

from datasets.base_dataset import MedicalImageDataset

# Human-readable organ names for logging
AMOS22_ORGAN_NAMES = {
    1: "spleen", 2: "right_kidney", 3: "left_kidney", 4: "gallbladder",
    5: "esophagus", 6: "liver", 7: "stomach", 8: "aorta",
    9: "inferior_vena_cava", 10: "pancreas", 11: "right_adrenal_gland",
    12: "left_adrenal_gland", 13: "duodenum", 14: "bladder", 15: "prostate_uterus",
}


class AMOS22Dataset(MedicalImageDataset):
    """
    AMOS22 slice-level dataset for organ segmentation.

    Each sample is a 2D axial slice from a 3D CT or MRI volume,
    paired with a binary mask for one (or all) organs.

    For 3D training (with ISA), use the AMOS22_3D_Dataset variant below
    which returns stacks of consecutive slices.

    Args:
        cfg: OmegaConf config
        split: "train", "val", or "test"
        modality: "ct", "mri", or "all"
        target_organs: List of organ IDs to include. None = all 15 organs.
                       E.g., [6] = liver only, [2, 3] = both kidneys.
    """

    def __init__(
        self,
        cfg,
        split: str = "train",
        modality: str = "ct",
        target_organs: Optional[List[int]] = None,
    ):
        super().__init__(cfg, split=split, modality=modality)

        self.data_root = Path(cfg.data.data_root)
        self.target_organs = target_organs or list(range(1, 16))  # Default: all 15

        # Load slice index: list of (image_path, label_path, slice_idx, organ_id, modality)
        self.samples = self._build_slice_index()

        print(f"  AMOS22 [{split}] {modality.upper()}: "
              f"{len(self.samples)} slices from organs {self.target_organs}")

    def _build_slice_index(self) -> List[Dict]:
        """
        Build an index of all valid slices.

        A slice is "valid" if it contains at least one foreground pixel
        for at least one target organ. This prevents training on mostly
        empty slices (which would bias the model toward empty masks).

        Returns:
            List of dicts: {"image_path", "label_path", "slice_idx", "organ_id", "modality"}
        """
        # Determine which image/label directories to use
        if self.split in ["train", "train_val"]:
            img_dir = self.data_root / "imagesTr"
            lbl_dir = self.data_root / "labelsTr"
        elif self.split == "val":
            img_dir = self.data_root / "imagesVa"
            lbl_dir = self.data_root / "labelsVa"
        else:
            img_dir = self.data_root / "imagesTs"
            lbl_dir = None  # No labels for test

        if not img_dir.exists():
            raise FileNotFoundError(
                f"AMOS22 data not found at {img_dir}.\n"
                f"Please download from https://amos22.grand-challenge.org/ "
                f"and place in {self.data_root}/"
            )

        # Find all image files
        all_files = sorted(img_dir.glob("*.nii.gz"))

        # Filter by modality:
        #   CT: amos_0001 to amos_0500
        #   MRI: amos_0501 to amos_0600
        filtered_files = []
        for f in all_files:
            num = int(f.stem.replace("amos_", "").replace(".nii", ""))
            if self.modality == "ct" and num <= 500:
                filtered_files.append((f, "ct"))
            elif self.modality == "mri" and num > 500:
                filtered_files.append((f, "mri"))
            elif self.modality == "all":
                mod = "ct" if num <= 500 else "mri"
                filtered_files.append((f, mod))

        # Build slice index
        samples = []
        for img_path, mod in filtered_files:
            if lbl_dir is None:
                # Test set — no labels
                lbl_path = None
                # Can't filter by foreground without labels
                # Load volume just to count slices
                try:
                    img_nib = nib.load(str(img_path))
                    n_slices = img_nib.shape[2]
                    for sl in range(n_slices):
                        for organ_id in self.target_organs:
                            samples.append({
                                "image_path": str(img_path),
                                "label_path": None,
                                "slice_idx": sl,
                                "organ_id": organ_id,
                                "modality": mod,
                            })
                except Exception:
                    continue
            else:
                lbl_path = lbl_dir / img_path.name
                if not lbl_path.exists():
                    print(f"  Warning: Label not found for {img_path.name}, skipping.")
                    continue

                # Load label volume to find slices with foreground
                try:
                    lbl_nib = nib.load(str(lbl_path))
                    label_vol = lbl_nib.get_fdata().astype(np.uint8)  # (H, W, D)
                    n_slices = label_vol.shape[2]

                    for organ_id in self.target_organs:
                        # Binary mask for this organ
                        organ_mask_vol = (label_vol == organ_id)

                        # Only include slices where this organ is present
                        # (at least 50 foreground pixels to avoid tiny edge slices)
                        for sl in range(n_slices):
                            if organ_mask_vol[:, :, sl].sum() >= 50:
                                samples.append({
                                    "image_path": str(img_path),
                                    "label_path": str(lbl_path),
                                    "slice_idx": sl,
                                    "organ_id": organ_id,
                                    "modality": mod,
                                })
                except Exception as e:
                    print(f"  Warning: Error loading {img_path.name}: {e}")
                    continue

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Load a single 2D axial slice from a 3D volume.

        Returns:
            image: (H, W) float32 — CT or MRI slice
            mask: (H, W) uint8 — binary organ mask
            metadata: dict with "modality", "organ_id", "slice_idx", etc.
        """
        sample = self.samples[idx]

        img_path = sample["image_path"]
        lbl_path = sample["label_path"]
        sl = sample["slice_idx"]
        organ_id = sample["organ_id"]
        mod = sample["modality"]

        # Load image slice — use .npy cache if available (100x faster than .nii.gz)
        img_path = Path(img_path)
        stem = img_path.name.replace(".nii.gz", "")
        data_root = img_path.parents[1]
        split_dir = img_path.parent.name  # e.g. "imagesTr"
        npy_img = data_root / "npy_cache" / split_dir / stem / f"slice_{sl:04d}.npy"

        if npy_img.exists():
            image = np.load(str(npy_img))
        else:
            img_nib = nib.load(str(img_path))
            image = np.asarray(img_nib.dataobj[:, :, sl], dtype=np.float32)
            del img_nib

        # Load mask slice
        if lbl_path is not None:
            lbl_path = Path(lbl_path)
            lbl_split_dir = lbl_path.parent.name  # e.g. "labelsTr"
            npy_lbl = data_root / "npy_cache" / lbl_split_dir / stem / f"slice_{sl:04d}.npy"

            if npy_lbl.exists():
                full_mask = np.load(str(npy_lbl))
            else:
                lbl_nib = nib.load(str(lbl_path))
                full_mask = np.asarray(lbl_nib.dataobj[:, :, sl], dtype=np.uint8)
                del lbl_nib

            mask = (full_mask == organ_id).astype(np.uint8)  # Binary for this organ
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        metadata = {
            "modality": mod,
            "organ_id": organ_id,
            "organ_name": AMOS22_ORGAN_NAMES.get(organ_id, "unknown"),
            "slice_idx": sl,
            "image_path": img_path,
        }

        return image, mask, metadata


class AMOS22_3D_Dataset(MedicalImageDataset):
    """
    AMOS22 3D slice-stack dataset for LiteSAM-3D with ISA module.

    Instead of single slices, returns stacks of D consecutive slices.
    This is the input format needed for the Inter-Slice Attention module.

    The ISA module needs to see multiple slices to learn inter-slice context.

    Args:
        cfg: OmegaConf config
        split: "train", "val", or "test"
        modality: "ct", "mri", or "all"
        target_organs: List of organ IDs (None = all 15)
        n_slices: Number of consecutive slices per stack (D in the architecture)
        center_slice_only: If True, only supervise the center slice mask.
                           If False, supervise all slices in the stack.
    """

    def __init__(
        self,
        cfg,
        split: str = "train",
        modality: str = "ct",
        target_organs: Optional[List[int]] = None,
        n_slices: int = 16,
        center_slice_only: bool = True,
        copypaste_prob: float = 0.0,
        random_windowing_prob: float = 0.0,
        rw_center_jitter: float = 0.10,
        rw_width_jitter: float = 0.15,
    ):
        super().__init__(cfg, split=split, modality=modality)

        self.data_root = Path(cfg.data.data_root)
        self.target_organs = target_organs or list(range(1, 16))
        self.n_slices = n_slices
        self.center_slice_only = center_slice_only
        self.copypaste_prob = copypaste_prob
        # ---- RandomWindowing (training-only) knobs ----
        # Read from cfg.training.random_windowing_prob if not set explicitly.
        self.random_windowing_prob = float(
            random_windowing_prob
            or getattr(getattr(cfg, "training", cfg), "random_windowing_prob", 0.0)
        )
        self.rw_center_jitter = float(rw_center_jitter)
        self.rw_width_jitter  = float(rw_width_jitter)

        # Index: list of (image_path, label_path, center_slice, organ_id, modality)
        self.samples = self._build_volume_index()

        # Small organs: gallbladder, esophagus, adrenal glands, duodenum.
        # These are 5-50× smaller than liver/spleen by voxel count.
        SMALL_ORGAN_IDS = {4, 5, 11, 12, 13}

        # Foreground oversampling index — guarantees 33% of batches contain
        # small organ foreground. Pre-computed once at dataset init.
        self._fg_indices       = []  # regular organ slices
        self._small_fg_indices = []  # small organ slices
        for i, s in enumerate(self.samples):
            oid = s.get("organ_id", 0)
            if oid in SMALL_ORGAN_IDS:
                self._small_fg_indices.append(i)
            else:
                self._fg_indices.append(i)
        print(f"  Foreground oversampling index: "
              f"{len(self._small_fg_indices)} small-organ slices, "
              f"{len(self._fg_indices)} regular slices")

        # Copy-paste donor index — maps each small organ_id to its list of
        # sample indices. Built for all splits but only used when copypaste_prob > 0.
        # Only small organs are donors: pasting a large organ (liver) onto another
        # sample would corrupt context far more than it helps.
        self._donor_index: Dict[int, List[int]] = {oid: [] for oid in SMALL_ORGAN_IDS}
        for i, s in enumerate(self.samples):
            oid = s.get("organ_id", 0)
            if oid in self._donor_index:
                self._donor_index[oid].append(i)

        print(f"  AMOS22-3D [{split}] {modality.upper()}: "
              f"{len(self.samples)} slice stacks "
              f"(D={n_slices}, organs={self.target_organs})")

    def _build_volume_index(self) -> List[Dict]:
        """Build index of valid (volume, center_slice, organ) combinations."""
        if self.split in ["train"]:
            img_dir = self.data_root / "imagesTr"
            lbl_dir = self.data_root / "labelsTr"
        elif self.split == "val":
            img_dir = self.data_root / "imagesVa"
            lbl_dir = self.data_root / "labelsVa"
        else:
            img_dir = self.data_root / "imagesTs"
            lbl_dir = None

        if not img_dir.exists():
            raise FileNotFoundError(
                f"AMOS22 data not found at {img_dir}."
            )

        all_files = sorted(img_dir.glob("*.nii.gz"))
        filtered_files = []
        for f in all_files:
            num = int(f.stem.replace("amos_", "").replace(".nii", ""))
            if self.modality == "ct" and num <= 500:
                filtered_files.append((f, "ct"))
            elif self.modality == "mri" and num > 500:
                filtered_files.append((f, "mri"))
            elif self.modality == "all":
                filtered_files.append((f, "ct" if num <= 500 else "mri"))

        samples = []
        half = self.n_slices // 2

        for img_path, mod in filtered_files:
            lbl_path = (lbl_dir / img_path.name) if lbl_dir else None
            if lbl_path and not lbl_path.exists():
                continue

            try:
                if lbl_path:
                    lbl_nib = nib.load(str(lbl_path))
                    label_vol = lbl_nib.get_fdata().astype(np.uint8)
                    n_total = label_vol.shape[2]

                    for organ_id in self.target_organs:
                        organ_mask = (label_vol == organ_id)
                        # Find valid center slices (organ present, enough context)
                        for center in range(half, n_total - half):
                            if organ_mask[:, :, center].sum() >= 50:
                                samples.append({
                                    "image_path": str(img_path),
                                    "label_path": str(lbl_path),
                                    "center_slice": center,
                                    "organ_id": organ_id,
                                    "modality": mod,
                                    "n_slices": n_total,  # cache for __getitem__
                                })
            except Exception as e:
                print(f"  Warning: {img_path.name}: {e}")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def get_oversampled_indices(self, n_samples: int, small_organ_fraction: float = 0.33) -> List[int]:
        """
        Return a list of `n_samples` indices with `small_organ_fraction`
        guaranteed to come from small-organ slices (foreground oversampling).

        Used to build a custom sampler in the trainer.
        """
        import random as _random
        n_small   = int(n_samples * small_organ_fraction)
        n_regular = n_samples - n_small

        small_pool   = self._small_fg_indices if self._small_fg_indices else self._fg_indices
        regular_pool = self._fg_indices if self._fg_indices else self._small_fg_indices

        chosen_small   = _random.choices(small_pool,   k=n_small)
        chosen_regular = _random.choices(regular_pool, k=n_regular)
        indices = chosen_small + chosen_regular
        _random.shuffle(indices)
        return indices

    # ------------------------------------------------------------------
    # Copy-paste augmentation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bbox_with_margin(mask: np.ndarray, margin: int = 4) -> Tuple[int, int, int, int]:
        """
        Compute tight bounding box of non-zero region in a 2D binary mask,
        expanded by `margin` pixels on all sides.

        Returns:
            (y1, y2, x1, x2) — row/col slices into the mask.
            Returns (0, 0, 0, 0) if the mask is empty.
        """
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any():
            return 0, 0, 0, 0
        H, W = mask.shape
        y1 = max(0,  int(rows.argmax())               - margin)
        y2 = min(H,  int(H - rows[::-1].argmax())     + margin)
        x1 = max(0,  int(cols.argmax())               - margin)
        x2 = min(W,  int(W - cols[::-1].argmax())     + margin)
        return y1, y2, x1, x2

    def _copypaste_aug(
        self,
        image_stack: np.ndarray,      # (D, H, W) float32 — modified in-place
        all_masks:   List[np.ndarray], # D × (H, W) uint8  — modified in-place
        organ_id:    int,
        current_idx: int,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Full-stack copy-paste augmentation: paste a donor organ across all D
        slices where it is present.

        The donor is a different sample with the same organ_id, loaded from the
        pre-built _donor_index.  For each depth slice d:
          - If the donor has the organ present (>= 50 px): paste its bounding-box
            region (image + mask) into the corresponding position of image_stack / all_masks.
          - Otherwise: skip (organ absent from this slice).

        Pasting at the same spatial position is anatomically valid because all
        volumes are resampled to 1.5mm³ isotropic spacing.

        Args:
            image_stack:  (D, H, W) raw (not yet normalised) CT values.
            all_masks:    List of D binary numpy masks (H, W), uint8.
            organ_id:     1-indexed organ ID for this sample.
            current_idx:  Index of the current sample (to avoid self-paste).

        Returns:
            (image_stack, all_masks) — same objects, modified in-place, returned
            for clarity.
        """
        donors = self._donor_index.get(organ_id, [])
        if not donors:
            return image_stack, all_masks

        # Pick a donor that is not the current sample (max 3 retries)
        donor_idx = current_idx
        for _ in range(3):
            donor_idx = random.choice(donors)
            if donor_idx != current_idx:
                break
        if donor_idx == current_idx:
            return image_stack, all_masks   # all donors are self — skip

        donor_sample = self.samples[donor_idx]
        donor_center = donor_sample["center_slice"]
        donor_img_path = donor_sample["image_path"]
        donor_lbl_path = donor_sample["label_path"]
        if donor_lbl_path is None:
            return image_stack, all_masks

        half = self.n_slices // 2
        n_total_donor = donor_sample.get("n_slices", None)

        # Determine donor slice window
        donor_img_split = Path(donor_img_path).parent.name
        donor_lbl_split = Path(donor_lbl_path).parent.name
        donor_img_cache = self._npy_cache_dir(donor_img_path, donor_img_split)
        donor_lbl_cache = self._npy_cache_dir(donor_lbl_path, donor_lbl_split)

        if n_total_donor is None:
            if donor_img_cache.exists():
                n_total_donor = len(list(donor_img_cache.glob("slice_*.npy")))
            else:
                n_total_donor = nib.load(donor_img_path).shape[2]
            donor_sample["n_slices"] = n_total_donor

        d_sl_start = max(0, donor_center - half)
        d_sl_end   = min(n_total_donor, donor_center + half)
        n_donor_actual = d_sl_end - d_sl_start

        # Load donor image slices (same npy_cache logic as __getitem__)
        _fallback_donor_img = None
        donor_imgs = []
        for sl in range(d_sl_start, d_sl_end):
            npy = donor_img_cache / f"slice_{sl:04d}.npy"
            if npy.exists():
                donor_imgs.append(np.load(str(npy)))
            else:
                if _fallback_donor_img is None:
                    _fallback_donor_img = nib.load(donor_img_path).get_fdata(dtype=np.float32)
                donor_imgs.append(_fallback_donor_img[:, :, sl])

        # Load donor label slices
        _fallback_donor_lbl = None
        donor_lbls = []
        for sl in range(d_sl_start, d_sl_end):
            npy = donor_lbl_cache / f"slice_{sl:04d}.npy"
            if npy.exists():
                donor_lbls.append(np.load(str(npy)))
            else:
                if _fallback_donor_lbl is None:
                    _fallback_donor_lbl = nib.load(donor_lbl_path).get_fdata().astype(np.uint8)
                donor_lbls.append(_fallback_donor_lbl[:, :, sl])

        # Pad donor stacks to n_slices if near volume boundary
        if n_donor_actual < self.n_slices:
            pad_b = half - (donor_center - d_sl_start)
            pad_a = self.n_slices - n_donor_actual - pad_b
            donor_imgs = [donor_imgs[0]] * pad_b + donor_imgs + [donor_imgs[-1]] * pad_a
            donor_lbls = [donor_lbls[0]] * pad_b + donor_lbls + [donor_lbls[-1]] * pad_a

        # Resize donor slices to match target spatial dimensions.
        # Different CT scanners produce different matrix sizes (e.g. 512×512 vs 768×768).
        # All volumes share the same isotropic spacing (1.5mm³) so a resize is safe —
        # the organ occupies the same physical size; only the pixel grid differs.
        H_t, W_t = image_stack.shape[1], image_stack.shape[2]
        H_d, W_d = donor_imgs[0].shape[0], donor_imgs[0].shape[1]
        if H_d != H_t or W_d != W_t:
            import torch.nn.functional as _F
            n_d = len(donor_imgs)
            # Resize images: (n_d, 1, H_d, W_d) → (n_d, 1, H_t, W_t)
            imgs_tensor = torch.from_numpy(
                np.stack(donor_imgs, axis=0)
            ).unsqueeze(1).float()
            imgs_tensor = _F.interpolate(
                imgs_tensor, size=(H_t, W_t), mode='bilinear', align_corners=False
            )
            donor_imgs = [imgs_tensor[i, 0].numpy() for i in range(n_d)]
            # Resize labels: nearest-neighbour to preserve binary values
            lbls_tensor = torch.from_numpy(
                np.stack(donor_lbls, axis=0).astype(np.float32)
            ).unsqueeze(1)
            lbls_tensor = _F.interpolate(
                lbls_tensor, size=(H_t, W_t), mode='nearest'
            )
            donor_lbls = [lbls_tensor[i, 0].numpy().astype(np.uint8) for i in range(n_d)]

        # Paste slice-by-slice across all D depth positions
        D = self.n_slices
        H, W = image_stack.shape[1], image_stack.shape[2]
        for d in range(min(D, len(donor_imgs))):
            donor_mask_d = (donor_lbls[d] == organ_id).astype(np.uint8)
            if donor_mask_d.sum() < 50:
                continue  # organ absent or too small in this slice — skip

            y1, y2, x1, x2 = self._bbox_with_margin(donor_mask_d, margin=4)
            if y1 == y2 or x1 == x2:
                continue  # degenerate bbox — skip

            # Clip bbox to image bounds (should already fit after resize, but be safe)
            y2 = min(y2, H); x2 = min(x2, W)

            # Paste image region and union mask
            image_stack[d, y1:y2, x1:x2] = donor_imgs[d][y1:y2, x1:x2]
            all_masks[d] = np.maximum(all_masks[d], donor_mask_d)

        return image_stack, all_masks

    def _load_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Load a stack of D slices centred on the sample's center_slice.

        Returns:
            image: (D, H, W) float32 — stack of D slices
            mask: (H, W) uint8 — binary mask for the center slice
            metadata: dict
        """
        sample = self.samples[idx]
        center = sample["center_slice"]
        organ_id = sample["organ_id"]
        mod = sample["modality"]
        half = self.n_slices // 2

        # Load full volume
        img_nib = nib.load(sample["image_path"])
        image_vol = img_nib.get_fdata(dtype=np.float32)  # (H, W, D)

        lbl_nib = nib.load(sample["label_path"])
        label_vol = lbl_nib.get_fdata().astype(np.uint8)  # (H, W, D)

        # Extract slice stack: slices [center-half, ..., center+half-1]
        n_total = image_vol.shape[2]
        sl_start = max(0, center - half)
        sl_end = min(n_total, center + half)

        image_stack = image_vol[:, :, sl_start:sl_end]  # (H, W, D_actual)
        image_stack = image_stack.transpose(2, 0, 1)    # (D_actual, H, W)

        # Compute local center index and pad symmetrically
        center_local = center - sl_start
        n_actual = image_stack.shape[0]
        if n_actual < self.n_slices:
            pad_before = max(0, half - center_local)
            pad_after = max(0, self.n_slices - n_actual - pad_before)
            image_stack = np.pad(
                image_stack, ((pad_before, pad_after), (0, 0), (0, 0)), mode="edge"
            )
            center_local = center_local + pad_before

        # Center slice mask only (or stack masks if center_slice_only=False)
        center_mask = (label_vol[:, :, center] == organ_id).astype(np.uint8)

        metadata = {
            "modality": mod,
            "organ_id": organ_id,
            "organ_name": AMOS22_ORGAN_NAMES.get(organ_id, "unknown"),
            "center_slice": center,
            "n_slices": self.n_slices,
        }

        # For the base class _load_sample, we return (image, mask, metadata)
        # For 3D stacks, image is (D, H, W) — transforms handle the center slice
        # The dataset __getitem__ needs to be overridden for 3D:
        return image_stack[center_local], center_mask, metadata

    def _npy_cache_dir(self, nifti_path: str, split_subdir: str) -> Path:
        """Return the npy_cache directory for a given NIfTI file."""
        p = Path(nifti_path)
        stem = p.name.replace(".nii.gz", "").replace(".nii", "")
        data_root = p.parents[1]
        return data_root / "npy_cache" / split_subdir / stem

    def _load_slice_npy(self, cache_dir: Path, sl: int, fallback_vol: np.ndarray) -> np.ndarray:
        """Load one slice from npy_cache; fall back to in-memory volume if missing."""
        npy = cache_dir / f"slice_{sl:04d}.npy"
        if npy.exists():
            return np.load(str(npy))
        # Cache miss — use pre-loaded volume
        return fallback_vol[:, :, sl]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Return full 3D stack using npy_cache (fast) with NIfTI fallback.

        Uses per-slice .npy files from npy_cache/ for 100x faster I/O vs
        loading full NIfTI volumes. Falls back to NIfTI only on cache miss.
        """
        sample = self.samples[idx]
        center = sample["center_slice"]
        organ_id = sample["organ_id"]
        half = self.n_slices // 2

        img_path = sample["image_path"]
        lbl_path = sample["label_path"]

        # Determine npy_cache directories
        img_split = Path(img_path).parent.name   # e.g. "imagesTr"
        lbl_split = Path(lbl_path).parent.name   # e.g. "labelsTr"
        img_cache = self._npy_cache_dir(img_path, img_split)
        lbl_cache = self._npy_cache_dir(lbl_path, lbl_split)

        # Determine slice range
        # We need total slice count for boundary clamping.
        # Try to infer from cache file count; fall back to NIfTI header.
        n_total = sample.get("n_slices", None)
        if n_total is None:
            if img_cache.exists():
                n_total = len(list(img_cache.glob("slice_*.npy")))
            else:
                img_nib = nib.load(img_path)
                n_total = img_nib.shape[2]
            # Cache n_slices back into sample for subsequent calls
            sample["n_slices"] = n_total

        sl_start = max(0, center - half)
        sl_end = min(n_total, center + half)
        center_local = center - sl_start

        # Load image slices — use npy_cache, fall back to full NIfTI only on miss
        _fallback_img_vol = None
        slices = []
        for sl in range(sl_start, sl_end):
            npy = img_cache / f"slice_{sl:04d}.npy"
            if npy.exists():
                slices.append(np.load(str(npy)))
            else:
                # Cache miss — load full NIfTI once and reuse
                if _fallback_img_vol is None:
                    _fallback_img_vol = nib.load(img_path).get_fdata(dtype=np.float32)
                slices.append(_fallback_img_vol[:, :, sl])

        image_stack = np.stack(slices, axis=0)  # (D_actual, H, W)

        # Pad symmetrically if near volume boundary
        n_actual = image_stack.shape[0]
        if n_actual < self.n_slices:
            pad_before = max(0, half - center_local)
            pad_after = max(0, self.n_slices - n_actual - pad_before)
            image_stack = np.pad(
                image_stack, ((pad_before, pad_after), (0, 0), (0, 0)), mode="edge"
            )
            center_local = center_local + pad_before

        # Load center slice label — npy_cache first
        lbl_npy = lbl_cache / f"slice_{center:04d}.npy"
        if lbl_npy.exists():
            center_label_full = np.load(str(lbl_npy))
        else:
            lbl_nib = nib.load(lbl_path)
            center_label_full = lbl_nib.get_fdata().astype(np.uint8)[:, :, center]

        center_mask = (center_label_full == organ_id).astype(np.uint8)

        # Load ALL D slice labels for all-slice supervision
        # Uses npy_cache (fast); falls back to NIfTI on cache miss.
        _fallback_lbl_vol = None
        all_masks = []
        for sl in range(sl_start, sl_end):
            lbl_npy_d = lbl_cache / f"slice_{sl:04d}.npy"
            if lbl_npy_d.exists():
                lbl_d = np.load(str(lbl_npy_d))
            else:
                if _fallback_lbl_vol is None:
                    _fallback_lbl_vol = nib.load(lbl_path).get_fdata().astype(np.uint8)
                lbl_d = _fallback_lbl_vol[:, :, sl]
            all_masks.append((lbl_d == organ_id).astype(np.float32))

        # Pad symmetrically to n_slices (edge padding — same as image stack)
        n_loaded = len(all_masks)
        if n_loaded < self.n_slices:
            pad_before = max(0, half - center_local)
            pad_after = self.n_slices - n_loaded - pad_before
            all_masks = [all_masks[0]] * pad_before + all_masks + [all_masks[-1]] * pad_after

        # ---- Copy-paste augmentation (small organs only) ----
        # Paste a donor organ's full D-slice region into this sample to expose
        # the ODE cross-slice module to more small-organ 3D examples.
        # Only triggered for small organ samples where a donor pool exists.
        if (self.copypaste_prob > 0
                and organ_id in self._donor_index
                and self._donor_index[organ_id]
                and random.random() < self.copypaste_prob):
            image_stack, all_masks = self._copypaste_aug(
                image_stack, all_masks, organ_id, idx
            )
            # Sync center_mask with potentially updated all_masks[center_local]
            center_mask = np.asarray(all_masks[center_local], dtype=np.uint8)

        masks_all_np = np.stack(all_masks, axis=0)  # (D, H, W)
        masks_all_t = torch.from_numpy(masks_all_np).unsqueeze(1)  # (D, 1, H, W)
        masks_all_t = torch.nn.functional.interpolate(
            masks_all_t, size=(self.img_size, self.img_size), mode="nearest"
        ).squeeze(1)  # (D, img_size, img_size)

        # Apply transforms to the center slice only (for augmentation + box extraction)
        # Context slices get normalise + resize only (no geometric aug → spatial consistency)
        center_img = image_stack[center_local]
        transformed = self.transforms(center_img, center_mask)

        # Normalise + resize ALL slices in one batched F.interpolate call (fast)
        # Step 1: clip + normalise entire stack at once — (D, H, W) numpy
        lo, hi = (self.clip_range[0], self.clip_range[1]) if self.clip_range else (None, None)
        # ---- RandomWindowing (Eilertsen MIDL 2025) — train only ----
        # Jitter (lo, hi) per-sample during training so the network sees the
        # same anatomy under different HU display windows. The same (lo, hi)
        # is applied to ALL D slices so cross-slice intensity coherence is
        # preserved (line 769's spatial-consistency invariant generalises
        # to intensity-consistency here).
        if (lo is not None
                and self.split == "train"
                and getattr(self, "random_windowing_prob", 0.0) > 0.0):
            from datasets.transforms import random_windowing
            lo, hi = random_windowing(
                lo, hi,
                center_jitter_frac=getattr(self, "rw_center_jitter", 0.10),
                width_jitter_frac=getattr(self, "rw_width_jitter", 0.15),
                prob=self.random_windowing_prob,
            )
        if lo is not None:
            stack_norm = np.clip(image_stack, lo, hi).astype(np.float32)
            stack_norm = (stack_norm - lo) / max(hi - lo, 1e-8)
        else:
            lo_v = image_stack.min()
            hi_v = image_stack.max()
            denom = max(hi_v - lo_v, 1e-8)
            stack_norm = ((image_stack - lo_v) / denom).astype(np.float32)

        # Step 2: (D, H, W) → (D, 3, H, W) by repeating channel dim
        stack_3ch = torch.from_numpy(stack_norm).unsqueeze(1).repeat(1, 3, 1, 1)  # (D, 3, H, W)

        # Step 3: One batched resize — (D, 3, H, W) → (D, 3, img_size, img_size)
        images_3d = torch.nn.functional.interpolate(
            stack_3ch, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False
        )  # (D, 3, img_size, img_size)

        modality_id = torch.tensor(
            self.modality_id, dtype=torch.long
        ).expand(self.n_slices)  # (D,) — same modality for all slices

        return {
            "image": images_3d,                               # (D, 3, H, W)
            "mask": transformed["mask"].float(),              # (H, W) — center slice only
            "masks_all": masks_all_t,                         # (D, H, W) — all D slices
            "box": transformed["box"],                        # (4,)
            "modality_id": modality_id,                       # (D,) — all same modality
            "is_3d": torch.tensor(True),                      # scalar BoolTensor
            "organ_id": torch.tensor(organ_id, dtype=torch.long),
            "center_slice": torch.tensor(center, dtype=torch.long),
            "image_path": sample["image_path"],               # str — for patient grouping
            "idx": torch.tensor(idx, dtype=torch.long),
        }
