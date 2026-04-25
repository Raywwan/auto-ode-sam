"""BalancedBatchSampler: ensures every batch contains ≥1 slab from each
'rare' organ that exists in the dataset's volume set."""
import pytest
import torch
import numpy as np
from datasets.balanced_sampler import BalancedBatchSampler


class _FakeDataset:
    """4 volumes × 4 slabs = 16 indices.
    Volume → present organs:
      vol_0: liver, kidney    (large)
      vol_1: liver, gallbladder, prostate (gallbladder + prostate are rare)
      vol_2: liver
      vol_3: liver, adrenal_r (rare)
    """
    def __init__(self):
        self.volume_ids = ["vol_0", "vol_1", "vol_2", "vol_3"]
        self.slabs_per_volume = 4
        self.length = 16
        self._presence = {
            "vol_0": {1, 2},
            "vol_1": {1, 4, 15},
            "vol_2": {1},
            "vol_3": {1, 11},
        }

    def __len__(self):
        return self.length

    def organs_in_volume(self, volume_idx: int) -> set:
        return self._presence[self.volume_ids[volume_idx]]


def test_sampler_yields_batches_of_correct_size():
    ds = _FakeDataset()
    sampler = BalancedBatchSampler(
        dataset=ds,
        batch_size=2,
        rare_organs=[4, 11, 15],
        shuffle=True,
        seed=0,
    )
    batches = list(sampler)
    assert all(len(b) == 2 for b in batches)


def test_sampler_returns_indices_in_range():
    ds = _FakeDataset()
    sampler = BalancedBatchSampler(
        dataset=ds, batch_size=2, rare_organs=[4, 11, 15], shuffle=True, seed=0,
    )
    for batch in sampler:
        for idx in batch:
            assert 0 <= idx < ds.length


def test_rare_organ_appears_more_than_uniform():
    """Over many epochs, rare-organ slabs should appear more often than they
    would under uniform random sampling (1/4 of volumes contain the organ)."""
    ds = _FakeDataset()
    sampler = BalancedBatchSampler(
        dataset=ds, batch_size=2, rare_organs=[15], shuffle=True, seed=42,
    )
    n_with_rare = 0
    n_total = 0
    for _epoch in range(10):
        for batch in sampler:
            n_total += 1
            for idx in batch:
                vol_idx = idx // ds.slabs_per_volume
                if 15 in ds.organs_in_volume(vol_idx):
                    n_with_rare += 1
                    break
    rare_rate = n_with_rare / n_total
    # Uniform would be 1/4 of volumes contain prostate → ~25% batches.
    # Balanced should be much higher.
    assert rare_rate > 0.6, f"rare_rate={rare_rate:.3f} not above uniform"


def test_sampler_len_consistent_with_iteration():
    ds = _FakeDataset()
    sampler = BalancedBatchSampler(
        dataset=ds, batch_size=2, rare_organs=[15], shuffle=False, seed=0,
    )
    declared = len(sampler)
    actual = sum(1 for _ in sampler)
    assert declared == actual
