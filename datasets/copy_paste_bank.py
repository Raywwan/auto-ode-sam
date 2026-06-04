"""Copy-paste small-organ augmentation for 3D CT multi-organ segmentation.

Rationale (2026-04-24): the FT proposer 3D eval showed weak organs at
<0.30 Dice (stomach, pancreas, adrenals, duodenum, gallbladder) because they
are rare or small in the 270-volume training set — each batch rarely sees a
good example. Copy-paste augments small organs by sampling bbox crops from
other training volumes and pasting them into the current patch, giving the
proposer many more diverse exposures per epoch.

Design:
  - Build a per-organ crop bank from training volumes ONCE (cached to disk).
    Each bank entry stores a tight bbox crop of a specific organ instance
    + its binary mask.
  - At train time, with probability p_paste, select a random rare-organ
    crop and paste it into the current volume patch at a random location
    (avoiding the subject organ if already present). Image = crop values;
    label = organ_id.
  - Novel for 3D CT multi-organ: instance-segmentation copy-paste exists
    in 2D (Ghiasi et al. 2021 "Simple copy-paste") but not standardised
    for 3D medical volumes.

Interface:
  - `CopyPasteBank.build(dataset, rare_organ_ids, cache_path)` — one-time build
  - `bank.maybe_paste(vol_patch, lab_patch, rng, p_paste, rare_ids)` — called
    from `AMOS22V9Dataset.__getitem__`; returns (possibly modified) patch/label
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


# AMOS22 organ ids 1..15; rare / weak organs default list.
DEFAULT_RARE_ORGAN_IDS: Tuple[int, ...] = (
    4,   # gallbladder
    7,   # stomach — actually large but highly variable; borderline rare
    10,  # pancreas
    11,  # r_adrenal
    12,  # l_adrenal
    13,  # duodenum
)


class OrganCrop:
    """One organ-instance crop. Stored in the bank."""
    __slots__ = ("image", "mask", "organ_id", "vol_id")

    def __init__(
        self,
        image: np.ndarray,     # (d, h, w) HU-normalised [0, 1]
        mask:  np.ndarray,     # (d, h, w) uint8 {0, 1}
        organ_id: int,
        vol_id: str,
    ) -> None:
        self.image = image
        self.mask  = mask
        self.organ_id = int(organ_id)
        self.vol_id = vol_id

    def __repr__(self) -> str:
        return (
            f"OrganCrop(organ={self.organ_id}, vol={self.vol_id}, "
            f"shape={self.image.shape})"
        )


class CopyPasteBank:
    """Per-organ crop bank. Build once, reuse across epochs."""

    def __init__(self, crops_by_organ: Dict[int, List[OrganCrop]]) -> None:
        self.crops_by_organ = crops_by_organ

    # ------------------------------------------------------------------
    @classmethod
    def build_from_preproc(
        cls,
        preproc_dir: Path,
        volume_ids: List[str],
        rare_organ_ids: Iterable[int] = DEFAULT_RARE_ORGAN_IDS,
        margin_vox: int = 4,
        cache_path: Optional[Path] = None,
        max_crops_per_organ: int = 300,
    ) -> "CopyPasteBank":
        """Scan preprocessed .pt volumes, extract bboxes for each rare organ,
        and return a populated bank. If `cache_path` exists, loads from cache.
        """
        if cache_path is not None and cache_path.exists():
            print(f"[copy_paste] loading cached bank {cache_path}")
            d = torch.load(str(cache_path), map_location="cpu", weights_only=False)
            return cls(d["crops_by_organ"])

        rare_set = set(int(o) for o in rare_organ_ids)
        crops: Dict[int, List[OrganCrop]] = {o: [] for o in rare_set}

        n_scanned = 0
        for vid in volume_ids:
            pt = preproc_dir / f"{vid}.pt"
            if not pt.exists():
                continue
            d = torch.load(str(pt), map_location="cpu", weights_only=False)
            img = d["image"][0].numpy()                                # (H, W, D) in [0, 1]
            lab = d["label"][0].numpy().astype(np.int64)                # (H, W, D)
            img = np.transpose(img, (2, 0, 1)).astype(np.float32)       # (Z, H, W)
            lab = np.transpose(lab, (2, 0, 1))
            for organ in rare_set:
                if len(crops[organ]) >= max_crops_per_organ:
                    continue
                mask = (lab == organ)
                if mask.sum() < 50:
                    continue
                # bbox + margin
                zs, ys, xs = np.where(mask)
                z0, z1 = max(0, zs.min() - margin_vox), min(lab.shape[0], zs.max() + margin_vox + 1)
                y0, y1 = max(0, ys.min() - margin_vox), min(lab.shape[1], ys.max() + margin_vox + 1)
                x0, x1 = max(0, xs.min() - margin_vox), min(lab.shape[2], xs.max() + margin_vox + 1)
                sub_img = img[z0:z1, y0:y1, x0:x1].copy()
                sub_msk = mask[z0:z1, y0:y1, x0:x1].astype(np.uint8)
                crops[organ].append(OrganCrop(sub_img, sub_msk, organ, vid))
            n_scanned += 1
            if n_scanned % 50 == 0:
                counts = {o: len(c) for o, c in crops.items()}
                print(f"[copy_paste] scanned {n_scanned}  crops={counts}")

        bank = cls(crops)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"crops_by_organ": crops}, str(cache_path))
            print(f"[copy_paste] bank saved to {cache_path}")
        return bank

    # ------------------------------------------------------------------
    def maybe_paste(
        self,
        vol_patch: np.ndarray,       # (P, P, P) float32 HU-normalised
        lab_patch: np.ndarray,       # (P, P, P) int  {0..K}
        rng: np.random.Generator,
        p_paste: float,
        rare_organ_ids: Iterable[int] = DEFAULT_RARE_ORGAN_IDS,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Possibly paste one rare-organ crop into the current patch.

        Heuristics:
        - Only paste if the subject organ is NOT already present in the patch
          (avoids crushing the existing instance).
        - Paste at a random location; the crop is downsized if it exceeds
          the patch.
        - Image: crop intensities replace patch intensities where mask=1.
          Intensity is kept in the [0, 1] HU-normalised domain.
        - Label: organ id is written where mask=1 (overwriting any existing
          label; rare since subject organ is not present).
        """
        if p_paste <= 0:
            return vol_patch, lab_patch
        if rng.random() >= p_paste:
            return vol_patch, lab_patch

        rare_list = [o for o in rare_organ_ids if self.crops_by_organ.get(o)]
        if not rare_list:
            return vol_patch, lab_patch
        # pick a rare organ the patch doesn't already have
        candidates = [o for o in rare_list if (lab_patch == o).sum() < 5]
        if not candidates:
            return vol_patch, lab_patch
        organ = int(rng.choice(candidates))

        # pick a random crop; downsize if too large to fit
        crop = self.crops_by_organ[organ][int(rng.integers(0, len(self.crops_by_organ[organ])))]
        cimg = crop.image
        cmsk = crop.mask
        P = vol_patch.shape[0]
        cd, ch, cw = cimg.shape
        # clip size to at most patch / 2 so paste never dominates the whole patch
        max_dim = P // 2
        if max(cd, ch, cw) > max_dim:
            scale = max_dim / max(cd, ch, cw)
            new_shape = (max(2, int(round(cd * scale))),
                         max(2, int(round(ch * scale))),
                         max(2, int(round(cw * scale))))
            cimg = _zoom_vol(cimg, new_shape, mode="linear")
            cmsk = _zoom_vol(cmsk.astype(np.float32), new_shape, mode="nearest").astype(np.uint8)
            cd, ch, cw = new_shape

        # random paste location inside the patch
        z0 = int(rng.integers(0, max(1, P - cd + 1)))
        y0 = int(rng.integers(0, max(1, P - ch + 1)))
        x0 = int(rng.integers(0, max(1, P - cw + 1)))

        m = cmsk.astype(bool)
        # write image where mask is set
        patch_region_img = vol_patch[z0:z0 + cd, y0:y0 + ch, x0:x0 + cw]
        patch_region_lab = lab_patch[z0:z0 + cd, y0:y0 + ch, x0:x0 + cw]
        patch_region_img[m] = cimg[m].clip(0.0, 1.0)
        patch_region_lab[m] = organ
        return vol_patch, lab_patch


# ----------------------------------------------------------------------
def _zoom_vol(vol: np.ndarray, new_shape: Tuple[int, int, int], mode: str) -> np.ndarray:
    """Tiny zoom wrapper around F.interpolate to avoid a scipy dep at runtime."""
    t = torch.from_numpy(vol)[None, None].float()
    if mode == "linear":
        out = torch.nn.functional.interpolate(t, size=new_shape, mode="trilinear", align_corners=False)
    else:
        out = torch.nn.functional.interpolate(t, size=new_shape, mode="nearest")
    return out[0, 0].numpy()


# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--preproc-dir",
                    default="C:/Users/Raywa/Desktop/LiteSAM3D/data/amos22/preprocessed")
    ap.add_argument("--cache",
                    default="C:/Users/Raywa/Desktop/LiteSAM3D/data/amos22/copy_paste_bank.pt")
    ap.add_argument("--max-per-organ", type=int, default=300)
    args = ap.parse_args()

    pre = Path(args.preproc_dir)
    vol_ids = sorted(p.stem for p in pre.glob("amos_*.pt"))
    # train split only — last 10% is val (matches AMOS22V9Dataset)
    cut = int(round(len(vol_ids) * 0.9))
    train_ids = vol_ids[:cut]
    print(f"[copy_paste] building bank from {len(train_ids)} train volumes")

    bank = CopyPasteBank.build_from_preproc(
        preproc_dir=pre,
        volume_ids=train_ids,
        cache_path=Path(args.cache),
        max_crops_per_organ=args.max_per_organ,
    )
    for o, c in bank.crops_by_organ.items():
        print(f"  organ {o}: {len(c)} crops")
