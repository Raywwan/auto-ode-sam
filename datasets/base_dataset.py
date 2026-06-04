# =============================================================================
# datasets/base_dataset.py — Abstract Base Dataset
#
# All dataset classes inherit from MedicalImageDataset and must implement:
#   - __len__()
#   - _load_sample(idx) → (image, mask, metadata)
#
# The base class handles:
#   - Transform application
#   - Bounding box generation from mask
#   - Modality ID assignment
#   - Caching (optional, speeds up repeated access)
# =============================================================================

import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.transforms import (
    TrainTransforms,
    ValTransforms,
    get_bounding_box,
)

# Modality name → integer ID mapping
# Must match the num_modalities in config and the prompt encoder
MODALITY_ID = {
    "ct":          0,
    "mri":         1,
    "pet":         2,
    "xray":        3,
    "ultrasound":  4,
    "mammography": 5,
    "oct":         6,
    "endoscopy":   7,
    "fundus":      8,
    "dermoscopy":  9,
    "microscopy":  10,
}


class MedicalImageDataset(Dataset, ABC):
    """
    Abstract base class for all medical image segmentation datasets.

    Subclasses must implement:
        __len__(): Return total number of samples.
        _load_sample(idx): Load and return raw (image, mask, metadata) for sample idx.
            - image: (H, W) or (H, W, C) numpy float32
            - mask: (H, W) numpy binary uint8 or bool
            - metadata: dict with at least "modality" key (string from MODALITY_ID)

    Args:
        cfg: OmegaConf config
        split: "train", "val", or "test"
        modality: Imaging modality string (e.g., "ct", "mri")
    """

    def __init__(
        self,
        cfg,
        split: str = "train",
        modality: str = "ct",
    ):
        self.cfg = cfg
        self.split = split
        self.modality = modality.lower()
        self.modality_id = MODALITY_ID.get(self.modality, 0)
        self.img_size = cfg.model.img_size

        # Clip range for intensity normalisation
        # CT: clip HU values; other modalities: None (normalise to min/max)
        clip_cfg = getattr(cfg.data, "clip_range", None)
        self.clip_range = tuple(clip_cfg) if clip_cfg else None

        # Build transforms
        if split == "train":
            self.transforms = TrainTransforms(
                img_size=self.img_size,
                clip_range=self.clip_range,
            )
        else:
            self.transforms = ValTransforms(
                img_size=self.img_size,
                clip_range=self.clip_range,
            )

        # Optional: in-memory cache for small datasets
        self.use_cache = getattr(cfg.data, "cache_rate", 0.0) > 0
        self._cache: Dict[int, Any] = {}

    @abstractmethod
    def __len__(self) -> int:
        """Return total number of samples in this split."""
        raise NotImplementedError

    @abstractmethod
    def _load_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Load raw image, mask, and metadata for sample at index idx.

        Returns:
            image: (H, W) or (H, W, C) float32 numpy array (unnormalised)
            mask: (H, W) binary uint8 or bool numpy array
            metadata: dict — must contain "modality" key (str)
        """
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load, augment, and return a single training sample.

        Returns dict with:
            "image": (3, H, W) float32 tensor
            "mask": (H, W) int64 tensor
            "box": (4,) float32 tensor [x1, y1, x2, y2]
            "modality_id": scalar int64 tensor
            "idx": scalar int64 tensor (for debugging)
        """
        # Load from cache if available
        if self.use_cache and idx in self._cache:
            image, mask, metadata = self._cache[idx]
        else:
            image, mask, metadata = self._load_sample(idx)
            if self.use_cache:
                self._cache[idx] = (image, mask, metadata)

        # Get modality ID (use per-sample if available, else dataset default)
        modality_str = metadata.get("modality", self.modality)
        modality_id = MODALITY_ID.get(modality_str.lower(), self.modality_id)

        # Get organ ID (0 if not available, e.g. non-organ datasets)
        organ_id = metadata.get("organ_id", 0)

        # Apply transforms (augmentation + normalisation + resize)
        transformed = self.transforms(image, mask)

        return {
            "image": transformed["image"],           # (3, H, W) float32
            "mask": transformed["mask"].float(),     # (H, W) float32
            "box": transformed["box"],               # (4,) float32
            "modality_id": torch.tensor(modality_id, dtype=torch.long),
            "organ_id": torch.tensor(organ_id, dtype=torch.long),
            "idx": torch.tensor(idx, dtype=torch.long),
        }

    def get_sample_info(self, idx: int) -> str:
        """Return a human-readable description of sample idx (for debugging)."""
        _, _, metadata = self._load_sample(idx)
        return str(metadata)
