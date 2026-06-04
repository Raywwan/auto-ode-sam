"""AMOS22 per-volume 15-organ dataset.

Returns a depth-D stack of axial slices centered on a slice that has the most
organs present (heuristic: pick centers that contain >=5 organs with >=50 voxels).

Each sample provides all 15 binary organ masks, making multi-organ supervision
one-forward-pass.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
        augment: Optional[bool] = None,
        slabs_per_volume: int = 1,
        light_aug: bool = False,
    ) -> None:
        self.root = Path(data_root)
        self.split = split
        self.img_size = img_size
        self.depth = depth
        self.modality = modality
        self.hu_clip = hu_clip
        self.augment = (split == "train") if augment is None else augment
        # V7: multi-slab sampling — K distinct Z-offsets per volume per epoch.
        # Only active on train split; val/test always use 1 center slab.
        self.slabs_per_volume = int(slabs_per_volume) if split == "train" else 1
        # V7: light_aug = H-flip + intensity only (no elastic/gamma/noise).
        self.light_aug = bool(light_aug)
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
        return len(self.samples) * self.slabs_per_volume

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

    def _augment(self, img: np.ndarray, lbl: np.ndarray) -> tuple:
        """Paired image + label augmentation (nnU-Net-inspired recipe).

        img: (H, W, D) float HU, lbl: (H, W, D) uint8 — both already at self.img_size.
        Augmentations: H-flip, small rotation, elastic deformation, gamma,
        Gaussian noise, intensity jitter. NO 90° rotation (collapses OAP priors).
        """
        # H-flip (axis=1)
        if np.random.rand() < 0.5:
            img = img[:, ::-1, :].copy()
            lbl = lbl[:, ::-1, :].copy()

        # Small random rotation ±15° — skipped in light_aug mode (V7).
        if (not self.light_aug) and np.random.rand() < 0.5:
            from scipy.ndimage import rotate
            angle = float(np.random.uniform(-15.0, 15.0))
            img = rotate(img, angle=angle, axes=(0, 1), reshape=False, order=1, mode="nearest")
            lbl = rotate(lbl, angle=angle, axes=(0, 1), reshape=False, order=0, mode="nearest")

        # Elastic deformation — skipped in light_aug mode.
        if (not self.light_aug) and np.random.rand() < 0.3:
            from scipy.ndimage import gaussian_filter, map_coordinates
            H, W, D = img.shape
            alpha = float(np.random.uniform(60.0, 120.0))
            sigma = float(np.random.uniform(8.0, 14.0))
            dx = gaussian_filter(np.random.rand(H, W) * 2 - 1, sigma) * alpha
            dy = gaussian_filter(np.random.rand(H, W) * 2 - 1, sigma) * alpha
            yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
            y_map = (yy + dy).astype(np.float32)
            x_map = (xx + dx).astype(np.float32)
            img_out = np.empty_like(img)
            lbl_out = np.empty_like(lbl)
            for d in range(D):
                img_out[..., d] = map_coordinates(img[..., d], [y_map, x_map], order=1, mode="nearest")
                lbl_out[..., d] = map_coordinates(lbl[..., d], [y_map, x_map], order=0, mode="nearest")
            img, lbl = img_out, lbl_out

        # Intensity jitter on HU window
        if np.random.rand() < 0.7:
            lo, hi = self.hu_clip
            rng = hi - lo
            img = img * float(np.random.uniform(0.85, 1.15)) + float(np.random.uniform(-0.08, 0.08)) * rng
            img = np.clip(img, lo, hi).astype(np.float32)

        # Gamma correction — skipped in light_aug mode.
        if (not self.light_aug) and np.random.rand() < 0.3:
            lo, hi = self.hu_clip
            norm = (img - lo) / max(hi - lo, 1)
            norm = np.clip(norm, 1e-6, 1.0)
            gamma = float(np.random.uniform(0.7, 1.5))
            norm = norm ** gamma
            img = (norm * (hi - lo) + lo).astype(np.float32)

        # Gaussian noise — skipped in light_aug mode.
        if (not self.light_aug) and np.random.rand() < 0.2:
            lo, hi = self.hu_clip
            rng = hi - lo
            img = img + np.random.normal(0.0, 0.01 * rng, img.shape).astype(np.float32)
            img = np.clip(img, lo, hi).astype(np.float32)

        return img, lbl

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        n_vols = len(self.samples)
        vol_idx = idx % n_vols
        slab_idx = idx // n_vols                                 # 0..slabs_per_volume-1
        s = self.samples[vol_idx]
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
            # V7 multi-slab: pick the top-K slice centers (most organs present),
            # then use slab_idx to select. Each slab gets a different high-ROI window.
            K = max(self.slabs_per_volume, 1)
            scored = counts_per_slice[valid]
            # Use argpartition for top-K, tie-break randomly in train split.
            top_k_within = np.argpartition(-scored, min(K, len(scored) - 1))[:K]
            # Sort top-K by score descending for deterministic ordering.
            top_k_within = top_k_within[np.argsort(-scored[top_k_within])]
            centers = valid[top_k_within]
            center = int(centers[min(slab_idx, len(centers) - 1)])

        # --- Train-only augmentation: z-jitter the center slice (high-ROI) ---
        if self.augment and valid.size > 0:
            # ±3 slice jitter on Z, clipped to valid window; each epoch picks a different 8-slice stack.
            shift = int(np.random.randint(-3, 4))
            center = int(np.clip(center + shift, half, Z - half - 1))

        z0, z1 = center - half, center + half
        img_stack = img_vol[..., z0:z1]
        lbl_stack = lbl_vol[..., z0:z1]

        img_resized = np.stack([self._resize_2d(img_stack[..., d], order=1) for d in range(self.depth)], axis=-1)
        lbl_resized = np.stack([self._resize_2d(lbl_stack[..., d], order=0) for d in range(self.depth)], axis=-1)

        # --- Train-only augmentation: spatial + intensity (paired image+mask) ---
        if self.augment:
            img_resized, lbl_resized = self._augment(img_resized, lbl_resized)

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
