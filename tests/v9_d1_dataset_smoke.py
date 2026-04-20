"""D1 smoke test - AMOS22 V9 dataset.

Uses a _SyntheticV9 stub that overrides disk loading so the test does
not require real AMOS22 data. Verifies slab-type ratios and output
shapes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.amos22_v9 import AMOS22V9Dataset, SlabType


class _SyntheticV9(AMOS22V9Dataset):
    def __init__(self) -> None:
        self.img_size = 64
        self.depth = 8
        self.volume_patch = 32
        self.n_organs = 15
        self.pos_frac = 0.5
        self.neg_frac = 0.3
        self.mix_frac = 0.2
        self.slabs_per_volume = 1
        self.hu_clip = (-200, 250)
        self._seed = 0
        self.length = 200
        self.volume_ids = ["fake"]

    def __len__(self) -> int:
        return self.length

    def _load_volume(self, idx: int):
        rng = np.random.default_rng(idx)
        vol = rng.normal(size=(64, 128, 128)).astype(np.float32)
        lab = np.zeros((64, 128, 128), dtype=np.int64)
        lab[10:20, 40:80, 40:80] = 4
        lab[30:40, 50:70, 50:70] = 7
        return vol, lab


def test_slab_type_ratios() -> None:
    ds = _SyntheticV9()
    counts = {SlabType.POSITIVE: 0, SlabType.NEGATIVE: 0, SlabType.MIXED: 0}
    n = 400
    rng = np.random.default_rng(0)
    for _ in range(n):
        t = ds._draw_slab_type(rng)
        counts[t] += 1
    pos = counts[SlabType.POSITIVE] / n
    neg = counts[SlabType.NEGATIVE] / n
    mix = counts[SlabType.MIXED] / n
    assert abs(pos - 0.5) < 0.1, f"pos ratio off: {pos:.2f}"
    assert abs(neg - 0.3) < 0.1, f"neg ratio off: {neg:.2f}"
    assert abs(mix - 0.2) < 0.1, f"mix ratio off: {mix:.2f}"
    print(f"[D1] slab ratios OK - pos={pos:.2f} neg={neg:.2f} mix={mix:.2f}")


def test_sample_shapes() -> None:
    ds = _SyntheticV9()
    s = ds[0]
    assert s["slab"].shape == (8, 3, 64, 64), f"slab shape wrong: {s['slab'].shape}"
    assert s["volume"].shape == (1, 32, 32, 32), f"volume shape wrong: {s['volume'].shape}"
    assert s["mask_slab"].shape == (15, 8, 64, 64)
    assert s["mask_volume"].shape == (15, 32, 32, 32)
    assert s["organ_id"].shape == ()
    assert s["slab_center_z"].shape == ()
    assert s["slab_type"].item() in (0, 1, 2)
    print("[D1] sample shapes OK")


def main() -> None:
    test_slab_type_ratios()
    test_sample_shapes()
    print("[D1] ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
