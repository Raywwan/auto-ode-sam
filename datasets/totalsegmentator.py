"""TotalSegmentator dataset wrapper for cross-dataset pretraining.

Reads the Zenodo v201 release (1228 CT volumes), remaps labels to the AMOS22
13-shared-organ subset using `datasets.totalseg_label_map`, and emits the
same item shape as `AMOS22V9Dataset` so downstream training code is unchanged.

Layout expected (set up by `scripts/download_totalseg.py`):
    <data_root>/
      totalsegmentator_manifest.json
      Totalsegmentator_dataset_v201/
        s0001/
          ct.nii.gz
          segmentations/   (one file per class)
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets.totalseg_label_map import (
    AMOS22_TO_TOTALSEG, SHARED_AMOS_INDICES, remap_label_volume,
)

# TotalSeg seg files are per-class binary masks named e.g. liver.nii.gz.
# We need to map filename → AMOS22 organ id. Filenames are stable across v2.
_TS_FILENAME_TO_AMOS = {
    "spleen": 1, "kidney_right": 2, "kidney_left": 3, "gallbladder": 4,
    "esophagus": 5, "liver": 6, "stomach": 7, "aorta": 8,
    "inferior_vena_cava": 9, "pancreas": 10,
    "adrenal_gland_right": 11, "adrenal_gland_left": 12, "duodenum": 13,
    "urinary_bladder": 14,
    # AMOS22 class 15 = prostate_uterus. TotalSeg v2 has prostate but no
    # uterus annotation; female subjects therefore contribute no foreground to
    # this channel (consistent with AMOS22's mixed-sex labelling). Prostate
    # alone still gives the GATE-I-critical prostate channel real signal.
    "prostate": 15,
}


class TotalSegmentatorDataset(Dataset):
    N_ORGANS = 15  # match AMOS22; classes 14, 15 are always 0 here

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        volume_patch: int = 96,
        hu_clip: Tuple[int, int] = (-200, 250),
        slabs_per_volume: int = 4,
        seed: int = 0,
        train_frac: float = 0.95,
        cache_dir: str = None,
    ) -> None:
        self.root = Path(data_root)
        self.split = split
        self.volume_patch = int(volume_patch)
        self.hu_clip = hu_clip
        self.slabs_per_volume = int(slabs_per_volume)
        self.n_organs = self.N_ORGANS
        self._seed = int(seed)
        self._cache_dir = Path(cache_dir) if cache_dir else None

        manifest_path = self.root / "totalsegmentator_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"TotalSeg manifest missing at {manifest_path}. "
                f"Run scripts/download_totalseg.py first."
            )
        manifest = json.loads(manifest_path.read_text())
        self._base = Path(manifest["data_root"])
        all_ids = sorted(manifest["volume_ids"])

        cut = int(round(len(all_ids) * train_frac))
        self.volume_ids: List[str] = all_ids[:cut] if split == "train" else all_ids[cut:]
        self.length = len(self.volume_ids) * self.slabs_per_volume

    def _load_volume(self, vol_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (normalized_image, label) as (D, H, W) float32 / int64.

        Cache hit: dequantize fp16 from `<cache_dir>/<vol_id>.pt`.
        Cache miss: load raw nii.gz, remap labels, normalize.
        """
        if self._cache_dir is not None:
            cache_path = self._cache_dir / f"{vol_id}.pt"
            if cache_path.exists():
                data = torch.load(str(cache_path), weights_only=True)
                ct = data["volume"].to(torch.float32).numpy()
                label = data["label"].to(torch.int64).numpy()
                return ct, label
        import nibabel as nib
        ct = nib.load(str(self._base / vol_id / "ct.nii.gz")).get_fdata().astype(np.float32)
        seg_dir = self._base / vol_id / "segmentations"
        label = np.zeros_like(ct, dtype=np.int64)
        for ts_name, amos_id in _TS_FILENAME_TO_AMOS.items():
            f = seg_dir / f"{ts_name}.nii.gz"
            if not f.exists():
                continue
            m = nib.load(str(f)).get_fdata() > 0.5
            label[m] = amos_id
        ct = ct.transpose(2, 0, 1)
        label = label.transpose(2, 0, 1)
        ct = self._normalize(ct)
        return ct, label

    def organs_in_volume(self, volume_idx: int) -> set:
        """Presence lookup shared with BalancedBatchSampler.

        Reads from a precomputed disk cache when available (built by
        `scripts/build_totalseg_presence.py`); otherwise falls back to a
        per-organ-file probe (no CT load) and memoizes in-process.
        """
        if not hasattr(self, "_presence_cache"):
            self._presence_cache: dict = self._load_disk_presence_cache()
        if volume_idx in self._presence_cache:
            return self._presence_cache[volume_idx]
        present = self._probe_presence(self.volume_ids[volume_idx])
        self._presence_cache[volume_idx] = present
        return present

    def _load_disk_presence_cache(self) -> dict:
        cache_path = self.root / "totalsegmentator_presence.json"
        if not cache_path.exists():
            return {}
        raw = json.loads(cache_path.read_text())
        out: dict = {}
        id_to_idx = {vid: i for i, vid in enumerate(self.volume_ids)}
        for vid, organs in raw.items():
            if vid in id_to_idx:
                out[id_to_idx[vid]] = set(int(o) for o in organs)
        return out

    def _probe_presence(self, vol_id: str) -> set:
        """Fast per-organ-file presence probe — no CT load."""
        import nibabel as nib
        seg_dir = self._base / vol_id / "segmentations"
        present: set = set()
        for ts_name, amos_id in _TS_FILENAME_TO_AMOS.items():
            f = seg_dir / f"{ts_name}.nii.gz"
            if not f.exists():
                continue
            m = nib.load(str(f)).get_fdata()
            if (m > 0.5).any():
                present.add(int(amos_id))
        return present

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        lo, hi = self.hu_clip
        img = np.clip(img, lo, hi)
        img = (img - lo) / max(hi - lo, 1)
        return img.astype(np.float32)

    def _crop_patch(self, img: np.ndarray, lab: np.ndarray, rng: np.random.Generator):
        D, H, W = img.shape
        ps = self.volume_patch
        # Random crop with bounds.
        z0 = rng.integers(0, max(D - ps + 1, 1))
        y0 = rng.integers(0, max(H - ps + 1, 1))
        x0 = rng.integers(0, max(W - ps + 1, 1))
        z1, y1, x1 = z0 + ps, y0 + ps, x0 + ps
        img_p = img[z0:z1, y0:y1, x0:x1]
        lab_p = lab[z0:z1, y0:y1, x0:x1]
        # Pad if undersized (small volumes near edges).
        pad = [(0, ps - s) for s in img_p.shape]
        if any(p[1] > 0 for p in pad):
            img_p = np.pad(img_p, pad, mode="constant", constant_values=0)
            lab_p = np.pad(lab_p, pad, mode="constant", constant_values=0)
        return img_p, lab_p

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> dict:
        vol_idx = idx // self.slabs_per_volume
        rng = np.random.default_rng(self._seed + idx * 13)
        img, lab = self._load_volume(self.volume_ids[vol_idx])  # already normalized
        img_p, lab_p = self._crop_patch(img, lab, rng)

        K = self.N_ORGANS
        p = self.volume_patch
        vol_t = torch.from_numpy(img_p).unsqueeze(0).float()           # (1, p, p, p)
        mask_vol = np.zeros((K, p, p, p), dtype=np.float32)
        for k in range(1, K + 1):
            mask_vol[k - 1] = (lab_p == k).astype(np.float32)
        mask_vol_t = torch.from_numpy(mask_vol)

        # Stage-2 stand-ins; the model short-circuits at `if stage == 1: return out`
        # in voluformer_v9.forward, so these are only read for batch dict
        # construction in train_v9 and never reach GPU computation.
        slab_dummy = torch.zeros((1, 1, 1, 1), dtype=torch.float32)
        mask_slab_dummy = torch.zeros((1, 1, 1, 1), dtype=torch.float32)

        return {
            "volume":        vol_t,
            "mask_volume":   mask_vol_t,
            "slab":          slab_dummy,
            "mask_slab":     mask_slab_dummy,
            "organ_id":      torch.tensor(0, dtype=torch.long),
            "slab_center_z": torch.tensor(p // 2, dtype=torch.long),
        }
