"""V9 dataset: returns paired (3D volume patch, 2D slab) with three sampling
regimes for multi-organ AMOS22 CT:

  - POSITIVE (50 %): slab center contains the chosen organ.
  - NEGATIVE (30 %): slab center does NOT contain the chosen organ.
  - MIXED    (20 %): slab center contains >= 2 different organs.

The negative/mixed regimes fix V7's distribution shift (V7 trained only on
positive slabs -> hallucinated positives at inference on absent-organ slabs).

Volume patch (for SwinUNETR proposer) is cropped around the slab center with
spatial size volume_patch (default 96).
"""
from __future__ import annotations

from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class SlabType(IntEnum):
    POSITIVE = 0
    NEGATIVE = 1
    MIXED = 2


class AMOS22V9Dataset(Dataset):
    N_ORGANS = 15

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        img_size: int = 320,
        depth: int = 8,
        volume_patch: int = 96,
        modality: str = "ct",
        hu_clip: Tuple[int, int] = (-200, 250),
        pos_frac: float = 0.5,
        neg_frac: float = 0.3,
        mix_frac: float = 0.2,
        slabs_per_volume: int = 4,
        seed: int = 0,
    ) -> None:
        assert abs(pos_frac + neg_frac + mix_frac - 1.0) < 1e-4
        self.root = Path(data_root)
        self.split = split
        self.img_size = img_size
        self.depth = depth
        self.volume_patch = volume_patch
        self.modality = modality
        self.hu_clip = hu_clip
        self.pos_frac = pos_frac
        self.neg_frac = neg_frac
        self.mix_frac = mix_frac
        self.slabs_per_volume = slabs_per_volume
        self.n_organs = self.N_ORGANS
        self._seed = seed

        self.volume_ids: List[str] = self._discover_volumes()
        self.length = len(self.volume_ids) * slabs_per_volume

    def _discover_volumes(self) -> List[str]:
        imdir = self.root / ("imagesTr" if self.split == "train" else "imagesVa")
        if not imdir.exists():
            return []
        return sorted(p.name.replace(".nii.gz", "") for p in imdir.glob("*.nii.gz"))

    def _load_volume(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        import nibabel as nib
        vol_id = self.volume_ids[idx % len(self.volume_ids)]
        imdir = self.root / ("imagesTr" if self.split == "train" else "imagesVa")
        lbdir = self.root / ("labelsTr" if self.split == "train" else "labelsVa")
        img = nib.load(str(imdir / f"{vol_id}.nii.gz")).get_fdata().astype(np.float32)
        lab = nib.load(str(lbdir / f"{vol_id}.nii.gz")).get_fdata().astype(np.int64)
        img = img.transpose(2, 0, 1)
        lab = lab.transpose(2, 0, 1)
        return img, lab

    def _draw_slab_type(self, rng: np.random.Generator) -> SlabType:
        u = rng.random()
        if u < self.pos_frac:
            return SlabType.POSITIVE
        if u < self.pos_frac + self.neg_frac:
            return SlabType.NEGATIVE
        return SlabType.MIXED

    def _pick_center(
        self,
        lab: np.ndarray,
        organ_id: int,
        slab_type: SlabType,
        rng: np.random.Generator,
    ) -> int:
        Z = lab.shape[0]
        if slab_type == SlabType.POSITIVE:
            present = np.where((lab == organ_id).any(axis=(1, 2)))[0]
            if len(present) == 0:
                return int(rng.integers(self.depth // 2, max(self.depth // 2 + 1, Z - self.depth // 2)))
            return int(rng.choice(present))
        if slab_type == SlabType.NEGATIVE:
            absent = np.where(~((lab == organ_id).any(axis=(1, 2))))[0]
            if len(absent) == 0:
                return int(rng.integers(self.depth // 2, max(self.depth // 2 + 1, Z - self.depth // 2)))
            return int(rng.choice(absent))
        unique_per_z = np.array([len(np.unique(lab[z])) - 1 for z in range(Z)])
        mixed = np.where(unique_per_z >= 2)[0]
        if len(mixed) == 0:
            return int(rng.integers(self.depth // 2, max(self.depth // 2 + 1, Z - self.depth // 2)))
        return int(rng.choice(mixed))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = np.random.default_rng(self._seed + idx)
        vol_idx = idx // self.slabs_per_volume
        vol, lab = self._load_volume(vol_idx)

        slab_type = self._draw_slab_type(rng)
        organ_id = int(rng.integers(1, self.n_organs + 1))
        cz = self._pick_center(lab, organ_id, slab_type, rng)

        d2 = self.depth // 2
        z0, z1 = cz - d2, cz - d2 + self.depth
        z0c = max(0, z0)
        z1c = min(vol.shape[0], z1)
        pad_front = z0c - z0
        pad_back = z1 - z1c
        slab_img = vol[z0c:z1c]
        slab_lab = lab[z0c:z1c]
        if pad_front or pad_back:
            slab_img = np.pad(slab_img, ((pad_front, pad_back), (0, 0), (0, 0)))
            slab_lab = np.pad(slab_lab, ((pad_front, pad_back), (0, 0), (0, 0)))

        slab_img = np.clip(slab_img, *self.hu_clip)
        slab_img = (slab_img - self.hu_clip[0]) / (self.hu_clip[1] - self.hu_clip[0])
        slab_img_t = torch.from_numpy(slab_img).float().unsqueeze(0)
        slab_img_t = torch.nn.functional.interpolate(
            slab_img_t, size=(self.img_size, self.img_size), mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        slab_rgb = slab_img_t.unsqueeze(1).expand(-1, 3, -1, -1).contiguous()

        slab_mask = np.zeros(
            (self.n_organs, self.depth, slab_img.shape[1], slab_img.shape[2]),
            dtype=np.float32,
        )
        for k in range(1, self.n_organs + 1):
            slab_mask[k - 1] = (slab_lab == k).astype(np.float32)
        slab_mask_t = torch.from_numpy(slab_mask)
        slab_mask_t = torch.nn.functional.interpolate(
            slab_mask_t, size=(self.img_size, self.img_size), mode="nearest",
        )

        p = self.volume_patch
        H, W = vol.shape[1], vol.shape[2]
        y0 = max(0, min(H - p, H // 2 - p // 2))
        x0 = max(0, min(W - p, W // 2 - p // 2))
        zc = max(p // 2, min(vol.shape[0] - p // 2, cz))
        z0 = zc - p // 2
        vol_patch = vol[z0:z0 + p, y0:y0 + p, x0:x0 + p]
        lab_patch = lab[z0:z0 + p, y0:y0 + p, x0:x0 + p]
        vp = np.clip(vol_patch, *self.hu_clip)
        vp = (vp - self.hu_clip[0]) / (self.hu_clip[1] - self.hu_clip[0])
        vol_t = torch.from_numpy(vp).float().unsqueeze(0)
        mask_vol = np.zeros((self.n_organs, p, p, p), dtype=np.float32)
        for k in range(1, self.n_organs + 1):
            mask_vol[k - 1] = (lab_patch == k).astype(np.float32)
        mask_vol_t = torch.from_numpy(mask_vol)

        slab_center_local = self.depth // 2

        return {
            "slab":          slab_rgb,
            "mask_slab":     slab_mask_t,
            "volume":        vol_t,
            "mask_volume":   mask_vol_t,
            "organ_id":      torch.tensor(organ_id - 1, dtype=torch.long),
            "slab_center_z": torch.tensor(slab_center_local, dtype=torch.long),
            "slab_type":     torch.tensor(int(slab_type), dtype=torch.long),
        }
