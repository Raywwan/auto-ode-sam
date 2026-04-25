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
        copy_paste_bank=None,
        copy_paste_prob: float = 0.0,
    ) -> None:
        assert abs(pos_frac + neg_frac + mix_frac - 1.0) < 1e-4
        assert modality in ("ct", "mri", "joint"), f"modality must be ct|mri|joint, got {modality}"
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
        # Copy-paste small-organ augmentation is train-only. Bank is optional;
        # leave prob=0 (default) and the dataset behaves identically to V8.
        self.copy_paste_bank = copy_paste_bank if split == "train" else None
        self.copy_paste_prob = float(copy_paste_prob) if split == "train" else 0.0

        self.volume_ids: List[str] = self._discover_volumes()
        self.length = len(self.volume_ids) * slabs_per_volume

    def _discover_volumes(self) -> List[str]:
        """Enumerate CT-only volume IDs for this split.

        Prefers `preprocessed/*.pt` (1.5 mm isotropic, HU-normalised). Falls
        back to `imagesTr|Va/*.nii.gz` when preprocessed cache is missing.

        AMOS22 convention: amos_0001..amos_0500 are CT, amos_0501..amos_0600
        are MRI. We keep CT only for V9.

        Split policy when using preprocessed cache (no separate val dir for
        preprocessed files): deterministic 90 / 10 by sorted ID — last 10 %
        of CT IDs go to val. Upgrade to 5-fold CV in Stage 3.
        """
        import re
        pre = self.root / "preprocessed"
        if pre.exists():
            all_ids = []
            for p in pre.glob("*.pt"):
                m = re.match(r"amos_(\d+)", p.stem)
                if not m:
                    continue
                num = int(m.group(1))
                if num > 500:  # MRI — skip
                    continue
                all_ids.append((num, p.stem))
            all_ids.sort()
            cut = int(round(len(all_ids) * 0.9))
            chosen = all_ids[:cut] if self.split == "train" else all_ids[cut:]
            return [stem for _, stem in chosen]

        # Fallback: raw NIfTI under imagesTr / imagesVa.
        imdir = self.root / ("imagesTr" if self.split == "train" else "imagesVa")
        if not imdir.exists():
            return []
        out = []
        for p in imdir.glob("*.nii.gz"):
            stem = p.name.replace(".nii.gz", "")
            m = re.match(r"amos_(\d+)", stem)
            if m and int(m.group(1)) <= 500:
                out.append(stem)
        return sorted(out)

    def _load_volume(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        vol_id = self.volume_ids[idx % len(self.volume_ids)]
        pre_path = self.root / "preprocessed" / f"{vol_id}.pt"
        if pre_path.exists():
            import torch as _t
            d = _t.load(str(pre_path), map_location="cpu", weights_only=False)
            # Preprocessed layout: (1, H, W, D) float32 in [0, 1]; label int8 same.
            img = d["image"][0].numpy()                              # (H, W, D)
            lab = d["label"][0].numpy().astype(np.int64)              # (H, W, D)
            img = np.transpose(img, (2, 0, 1))                       # (D, H, W)
            lab = np.transpose(lab, (2, 0, 1))
            # Preprocessed is already HU-normalised to [0, 1]; undo to raw HU
            # so downstream hu_clip normalisation is consistent with nii fallback.
            img = img * (self.hu_clip[1] - self.hu_clip[0]) + self.hu_clip[0]
            return img.astype(np.float32), lab

        # Fallback: raw NIfTI. NOTE: this path does NOT resample to 1.5 mm.
        import nibabel as nib
        imdir = self.root / ("imagesTr" if self.split == "train" else "imagesVa")
        lbdir = self.root / ("labelsTr" if self.split == "train" else "labelsVa")
        img = nib.load(str(imdir / f"{vol_id}.nii.gz")).get_fdata().astype(np.float32)
        lab = nib.load(str(lbdir / f"{vol_id}.nii.gz")).get_fdata().astype(np.int64)
        img = img.transpose(2, 0, 1)
        lab = lab.transpose(2, 0, 1)
        return img, lab

    def organs_in_volume(self, volume_idx: int) -> set:
        """Return the set of organ ids present in volume `volume_idx`.

        Cached after first call per volume. Reads only the label volume.
        """
        if not hasattr(self, "_organ_presence_cache"):
            self._organ_presence_cache: dict = {}
        if volume_idx in self._organ_presence_cache:
            return self._organ_presence_cache[volume_idx]
        _, lab = self._load_volume(volume_idx)
        present = set(int(v) for v in np.unique(lab) if int(v) > 0)
        self._organ_presence_cache[volume_idx] = present
        return present

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

    def _pick_yx(
        self,
        lab: np.ndarray,
        organ_id: int,
        cz: int,
        slab_type: SlabType,
        rng: np.random.Generator,
    ) -> Tuple[int, int]:
        """Pick the top-left (y0, x0) of the H,W window for the volume patch
        and slab. Train-only diversification: previously this was always the
        volume's H,W center, so the model never saw lateralized patches in
        the (R, A) plane. POSITIVE slabs now jitter around the chosen organ's
        centroid; NEGATIVE/MIXED draw a random (y0, x0). Val still receives
        the deterministic center crop for stable val tracking.
        """
        p = self.volume_patch
        H, W = lab.shape[1], lab.shape[2]
        max_y0 = max(0, H - p)
        max_x0 = max(0, W - p)

        if self.split != "train":
            y0 = max(0, min(max_y0, H // 2 - p // 2))
            x0 = max(0, min(max_x0, W // 2 - p // 2))
            return y0, x0

        if slab_type == SlabType.POSITIVE:
            slc = (lab[cz] == organ_id)
            if slc.any():
                ys, xs = np.where(slc)
                yc, xc = int(ys.mean()), int(xs.mean())
            else:
                fg_any = (lab == organ_id).any(axis=0)
                if fg_any.any():
                    ys, xs = np.where(fg_any)
                    yc, xc = int(ys.mean()), int(xs.mean())
                else:
                    yc, xc = H // 2, W // 2
            jitter = max(p // 4, 8)
            yc += int(rng.integers(-jitter, jitter + 1))
            xc += int(rng.integers(-jitter, jitter + 1))
        else:
            yc = int(rng.integers(p // 2, max(p // 2 + 1, H - p // 2)))
            xc = int(rng.integers(p // 2, max(p // 2 + 1, W - p // 2)))

        y0 = int(np.clip(yc - p // 2, 0, max_y0))
        x0 = int(np.clip(xc - p // 2, 0, max_x0))
        return y0, x0

    # Label IDs that must swap when the L-R body axis is flipped. Without this
    # remap an L-R flipped GT trains the model to call the right kidney's anatomy
    # "left kidney" (and vice versa), which contaminates the lateralized channels.
    # Layout matches AMOS22 (1=spleen, 2=r_kidney, 3=l_kidney, ..., 11=r_adrenal,
    # 12=l_adrenal). Index 0 = background.
    _LR_LABEL_REMAP = None  # built lazily, cached on the class

    def _lr_label_remap(self) -> np.ndarray:
        if AMOS22V9Dataset._LR_LABEL_REMAP is None:
            r = np.arange(self.n_organs + 1, dtype=np.int64)
            r[2], r[3] = 3, 2     # r_kidney <-> l_kidney
            r[11], r[12] = 12, 11  # r_adrenal <-> l_adrenal
            AMOS22V9Dataset._LR_LABEL_REMAP = r
        return AMOS22V9Dataset._LR_LABEL_REMAP

    def _maybe_flip_lr(
        self,
        slab_img: np.ndarray,
        slab_lab: np.ndarray,
        vol_patch: np.ndarray,
        lab_patch: np.ndarray,
        organ_id: int,
        rng: np.random.Generator,
    ):
        """Train-only random L-R flip (axis 1 = R axis after RAS + (D,H,W)
        transpose). Keeps slab and volume in lockstep; remaps paired-organ
        labels (and the conditioning organ_id) so GT semantics stay correct.
        Returns (slab_img, slab_lab, vol_patch, lab_patch, organ_id).
        """
        if self.split != "train":
            return slab_img, slab_lab, vol_patch, lab_patch, organ_id
        if rng.random() >= 0.5:
            return slab_img, slab_lab, vol_patch, lab_patch, organ_id
        slab_img = np.ascontiguousarray(slab_img[:, ::-1, :])
        slab_lab = np.ascontiguousarray(slab_lab[:, ::-1, :])
        vol_patch = np.ascontiguousarray(vol_patch[:, ::-1, :])
        lab_patch = np.ascontiguousarray(lab_patch[:, ::-1, :])
        remap = self._lr_label_remap()
        slab_lab = remap[slab_lab]
        lab_patch = remap[lab_patch]
        organ_id = int(remap[organ_id])
        return slab_img, slab_lab, vol_patch, lab_patch, organ_id

    def __len__(self) -> int:
        return self.length

    def _modality_of(self, vol_id: str) -> int:
        """0 = CT (amos_XXXX with num <=500), 1 = MRI (num >500).

        The AMOS22 release uses amos_0001..0500 for CT and amos_5001..5100 for MRI.
        Used to branch normalization so CT gets HU-clip while MRI gets z-score.
        """
        import re as _re
        m = _re.match(r"amos_(\d+)", str(vol_id))
        return 1 if (m and int(m.group(1)) > 500) else 0

    def _normalize(self, x: np.ndarray, modality: int) -> np.ndarray:
        """Branch normalization per modality.

        CT (modality=0) -> HU-clip and scale to [0, 1].
        MRI (modality=1) -> per-volume z-score, clip to +-3 sigma, scale to [0, 1].
        """
        if modality == 0:
            x = np.clip(x, *self.hu_clip)
            return ((x - self.hu_clip[0]) / (self.hu_clip[1] - self.hu_clip[0])).astype(np.float32)
        mu = float(x.mean())
        sd = float(x.std()) + 1e-6
        x = np.clip((x - mu) / sd, -3.0, 3.0)
        return ((x + 3.0) / 6.0).astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = np.random.default_rng(self._seed + idx)
        vol_idx = idx // self.slabs_per_volume
        vol_id = self.volume_ids[vol_idx % len(self.volume_ids)]
        vol_modality = self._modality_of(vol_id)
        vol, lab = self._load_volume(vol_idx)

        slab_type = self._draw_slab_type(rng)
        organ_id = int(rng.integers(1, self.n_organs + 1))
        cz = self._pick_center(lab, organ_id, slab_type, rng)

        # --- Pick the shared (y0, x0) H,W window FIRST -----------------------
        # CRITICAL: the slab MUST share the same (y0, x0) window as the 3D
        # volume patch, otherwise cascade fusion sees a stretched proposer
        # slice pasted onto unrelated anatomy. Previously the slab used the
        # full H,W while the volume patch used a center crop -> fused Dice
        # collapsed near 0 even with a strong proposer.
        # 2026-04-22: train now diversifies (y0, x0) (positive: jitter around
        # the organ centroid; negative/mixed: random) so the model sees the
        # full (R, A) extent. Val keeps the deterministic center crop.
        p = self.volume_patch
        y0, x0 = self._pick_yx(lab, organ_id, cz, slab_type, rng)

        # --- Slab (D_slab slices, same H,W window as the 3D patch) ----------
        d2 = self.depth // 2
        zs0, zs1 = cz - d2, cz - d2 + self.depth
        zs0c = max(0, zs0)
        zs1c = min(vol.shape[0], zs1)
        pad_front = zs0c - zs0
        pad_back = zs1 - zs1c
        slab_img = vol[zs0c:zs1c, y0:y0 + p, x0:x0 + p]
        slab_lab = lab[zs0c:zs1c, y0:y0 + p, x0:x0 + p]
        if pad_front or pad_back:
            slab_img = np.pad(slab_img, ((pad_front, pad_back), (0, 0), (0, 0)))
            slab_lab = np.pad(slab_lab, ((pad_front, pad_back), (0, 0), (0, 0)))

        # Pre-flip extract of the volume patch so flip applies to both views in
        # lockstep. The slab and patch share (y0, x0); flipping after extract
        # keeps cascade fusion alignment intact.
        zc_pre = max(p // 2, min(vol.shape[0] - p // 2, cz))
        z0_pre = zc_pre - p // 2
        vol_patch_pre = vol[z0_pre:z0_pre + p, y0:y0 + p, x0:x0 + p]
        lab_patch_pre = lab[z0_pre:z0_pre + p, y0:y0 + p, x0:x0 + p]
        slab_img, slab_lab, vol_patch_pre, lab_patch_pre, organ_id = (
            self._maybe_flip_lr(
                slab_img, slab_lab, vol_patch_pre, lab_patch_pre, organ_id, rng,
            )
        )

        slab_img = self._normalize(slab_img, vol_modality)
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

        # --- 3D volume patch (same H,W window as the slab) ------------------
        # Reuse the pre-flip extract so the volume view is consistent with the
        # slab view (both flipped or both not). Recompute z0 only for the
        # comment trail; the actual data already lives in vol_patch_pre.
        zc = max(p // 2, min(vol.shape[0] - p // 2, cz))
        z0 = zc - p // 2
        vol_patch = vol_patch_pre
        lab_patch = lab_patch_pre
        vp = self._normalize(vol_patch, vol_modality)
        # Copy-paste small-organ augmentation (train-only, CT-only). Operates
        # on the [0, 1] normalised volume patch + int label patch in lockstep.
        # Bank crops are stored [0, 1] HU-normalised (built from preprocessed
        # volumes that were already HU-normalised), so intensity spaces match.
        # getattr guards against pre-edit pickled instances in workers.
        _cp_bank = getattr(self, "copy_paste_bank", None)
        _cp_prob = getattr(self, "copy_paste_prob", 0.0)
        if _cp_bank is not None and _cp_prob > 0.0 and vol_modality == 0:
            vp, lab_patch = _cp_bank.maybe_paste(
                vp, lab_patch, rng, _cp_prob,
            )
        vol_t = torch.from_numpy(vp).float().unsqueeze(0)
        mask_vol = np.zeros((self.n_organs, p, p, p), dtype=np.float32)
        for k in range(1, self.n_organs + 1):
            mask_vol[k - 1] = (lab_patch == k).astype(np.float32)
        mask_vol_t = torch.from_numpy(mask_vol)

        # slab_center_z is the z-index INTO THE VOLUME PATCH where the slab is
        # centered. Since the volume patch (shape=volume_patch^3) is cropped to
        # be centered on cz (z0 = cz - p//2), the slab's mid-slice lands at
        # patch-slice p // 2. This is NOT depth // 2 (that would be the slab's
        # own local centre, not the volume-patch coordinate).
        slab_center_in_volume_patch = self.volume_patch // 2

        return {
            "slab":          slab_rgb,
            "mask_slab":     slab_mask_t,
            "volume":        vol_t,
            "mask_volume":   mask_vol_t,
            "organ_id":      torch.tensor(organ_id - 1, dtype=torch.long),
            "slab_center_z": torch.tensor(slab_center_in_volume_patch, dtype=torch.long),
            "slab_type":     torch.tensor(int(slab_type), dtype=torch.long),
            "modality":      torch.tensor(vol_modality, dtype=torch.long),
        }
