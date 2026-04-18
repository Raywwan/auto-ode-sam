"""AMOS22 per-volume 15-organ dataset.

Returns a depth-D stack of axial slices centered on a slice that has the most
organs present (heuristic: pick centers that contain >=5 organs with >=50 voxels).

Each sample provides all 15 binary organ masks, making multi-organ supervision
one-forward-pass.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


class AMOS22MultiOrgan3D_Dataset(Dataset):
    N_ORGANS = 15

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        img_size: int = 256,
        depth: int = 8,
        modality: str = "ct",
        hu_clip: tuple = (-200, 250),
    ) -> None:
        self.root = Path(data_root)
        self.split = split
        self.img_size = img_size
        self.depth = depth
        self.modality = modality
        self.hu_clip = hu_clip
        self.samples: List[Dict] = self._build_index()

    def _build_index(self) -> List[Dict]:
        out: List[Dict] = []
        img_dir = self.root / ("imagesTr" if self.split == "train" else "imagesVa")
        lbl_dir = self.root / ("labelsTr" if self.split == "train" else "labelsVa")
        half = self.depth // 2
        if not img_dir.exists() or not lbl_dir.exists():
            raise FileNotFoundError(f"AMOS22 split dirs not found under {self.root}")
        for img_path in sorted(img_dir.glob("*.nii.gz")):
            stem = img_path.name.replace(".nii.gz", "")
            try:
                case_id = int(stem.split("_")[-1])
            except ValueError:
                continue
            if self.modality == "ct" and not (1 <= case_id <= 500):
                continue
            if self.modality == "mri" and not (501 <= case_id <= 600):
                continue
            lbl_path = lbl_dir / img_path.name
            if not lbl_path.exists():
                continue
            try:
                lbl = nib.load(str(lbl_path))
                n_slices = lbl.shape[2]
                if n_slices < self.depth:
                    continue
                out.append({
                    "image_path": str(img_path),
                    "label_path": str(lbl_path),
                    "n_slices": n_slices,
                })
            except Exception as e:
                print(f"  skip {img_path.name}: {e}")
                continue
        return out

    def __len__(self) -> int:
        return len(self.samples)

    def _normalize_image(self, x: np.ndarray) -> np.ndarray:
        lo, hi = self.hu_clip
        x = np.clip(x, lo, hi)
        x = (x - lo) / max(hi - lo, 1)
        return x.astype(np.float32)

    def _resize_2d(self, arr: np.ndarray, order: int) -> np.ndarray:
        from scipy.ndimage import zoom
        h, w = arr.shape
        zh = self.img_size / h
        zw = self.img_size / w
        return zoom(arr, (zh, zw), order=order)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        img_nib = nib.load(s["image_path"])
        lbl_nib = nib.load(s["label_path"])
        img_vol = np.asarray(img_nib.dataobj, dtype=np.float32)
        lbl_vol = np.asarray(lbl_nib.dataobj, dtype=np.uint8)

        H, W, Z = img_vol.shape
        half = self.depth // 2

        counts_per_slice = np.zeros(Z, dtype=np.int32)
        for k in range(1, self.N_ORGANS + 1):
            has_k = (lbl_vol == k).reshape(H * W, Z).sum(axis=0) >= 50
            counts_per_slice += has_k.astype(np.int32)
        valid = np.arange(half, Z - half)
        if valid.size == 0:
            center = Z // 2
        else:
            best = valid[np.argmax(counts_per_slice[valid])]
            center = int(best)

        z0, z1 = center - half, center + half
        img_stack = img_vol[..., z0:z1]
        lbl_stack = lbl_vol[..., z0:z1]

        img_resized = np.stack([self._resize_2d(img_stack[..., d], order=1) for d in range(self.depth)], axis=-1)
        lbl_resized = np.stack([self._resize_2d(lbl_stack[..., d], order=0) for d in range(self.depth)], axis=-1)

        img_norm = self._normalize_image(img_resized)
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(1)

        masks = np.zeros((self.N_ORGANS, self.depth, self.img_size, self.img_size), dtype=np.uint8)
        for k in range(1, self.N_ORGANS + 1):
            masks[k - 1] = (lbl_resized == k).transpose(2, 0, 1).astype(np.uint8)
        masks_tensor = torch.from_numpy(masks)

        present_mask = torch.tensor(
            [masks_tensor[k].sum() >= 50 for k in range(self.N_ORGANS)],
            dtype=torch.bool,
        )

        return {
            "image": img_tensor,
            "masks": masks_tensor,
            "present_mask": present_mask,
            "case_id": Path(s["image_path"]).name.replace(".nii.gz", ""),
            "center_slice": center,
        }
