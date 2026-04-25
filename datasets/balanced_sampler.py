"""BalancedBatchSampler: per-batch rare-organ coverage for AMOS22 fine-tune.

Indexing model (matches AMOS22V9Dataset):
  - dataset.volume_ids: ordered list of volume IDs
  - dataset.slabs_per_volume: int (typically 4)
  - dataset.length = len(volume_ids) * slabs_per_volume
  - flat index `idx` decomposes as `(volume_idx, slab_idx)`
    via `volume_idx = idx // slabs_per_volume`
  - Dataset must implement `organs_in_volume(volume_idx) -> set[int]`

Sampling rule:
  - Each batch of size B picks 1 anchor index from a rare-organ-present pool
    (cycled deterministically across rare organs), and (B-1) indices from
    the general pool.
  - Anchor pool for organ `o` = all flat indices belonging to volumes where
    `o ∈ organs_in_volume(volume_idx)`.
  - Epoch length = ceil(dataset.length / batch_size) batches.
"""
from __future__ import annotations
from typing import Iterator, List, Sequence, Set
import math
import numpy as np
from torch.utils.data import Sampler


class BalancedBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        dataset,
        batch_size: int,
        rare_organs: Sequence[int],
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__(data_source=None)
        if not hasattr(dataset, "organs_in_volume"):
            raise AttributeError(
                "BalancedBatchSampler requires dataset.organs_in_volume(volume_idx)"
            )
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.rare_organs = list(rare_organs)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self._epoch = 0

        n_vols = len(dataset.volume_ids)
        spv = int(dataset.slabs_per_volume)
        self.length = n_vols * spv

        # Build per-organ index pool: flat indices whose volume contains the organ.
        self._pool: dict = {o: [] for o in self.rare_organs}
        for v in range(n_vols):
            organs = dataset.organs_in_volume(v)
            for o in self.rare_organs:
                if o in organs:
                    for s in range(spv):
                        self._pool[o].append(v * spv + s)
        # Drop empty pools (rare organ absent from this dataset split).
        self._pool = {o: p for o, p in self._pool.items() if len(p) > 0}
        self._rare_cycle = list(self._pool.keys())

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed + self._epoch)
        all_indices = np.arange(self.length)
        if self.shuffle:
            rng.shuffle(all_indices)
        n_batches = math.ceil(self.length / self.batch_size)

        # Per-organ shuffled queues for anchor sampling.
        organ_queues: dict = {}
        for o in self._rare_cycle:
            pool = list(self._pool[o])
            if self.shuffle:
                rng.shuffle(pool)
            organ_queues[o] = pool
        organ_cursor: dict = {o: 0 for o in self._rare_cycle}

        general_cursor = 0
        for b in range(n_batches):
            batch: List[int] = []
            # Anchor: pull next rare organ in round-robin if any rare organs exist.
            if self._rare_cycle:
                organ = self._rare_cycle[b % len(self._rare_cycle)]
                queue = organ_queues[organ]
                if organ_cursor[organ] >= len(queue):
                    if self.shuffle:
                        rng.shuffle(queue)
                    organ_cursor[organ] = 0
                batch.append(int(queue[organ_cursor[organ]]))
                organ_cursor[organ] += 1

            # Fill the rest from the general pool.
            while len(batch) < self.batch_size:
                if general_cursor >= len(all_indices):
                    if self.shuffle:
                        rng.shuffle(all_indices)
                    general_cursor = 0
                batch.append(int(all_indices[general_cursor]))
                general_cursor += 1
            yield batch

    def __len__(self) -> int:
        return math.ceil(self.length / self.batch_size)
