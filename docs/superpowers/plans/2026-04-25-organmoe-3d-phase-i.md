# OrganMoE-3D Phase I Implementation Plan (Days 1-7)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** `docs/superpowers/specs/2026-04-25-organmoe-3d-design.md`

**Goal:** Build the foundation of OrganMoE-3D — a class-presence-aware sparse LoRA Mixture-of-Experts adapter on VoCo-L — fine-tune it on AMOS22 (with TotalSegmentator pretraining), and pass GATE I (sliding-window mean Dice ≥0.85, prostate ≥0.30) on day 7.

**Architecture:** Frozen VoCo-L SwinUNETR-v2 encoder. Replace existing `_SwinLoRALinear` adapters in attention QKV/proj and FFN fc1/fc2 with a `MoELoRALinear` (K=8 LoRA-rank-16 experts + top-2 gating). A `PresenceHead` predicts a 15-organ presence vector from low-res encoder features and feeds into the router so experts learn to specialize per-organ. A `BalancedBatchSampler` and a presence-weighted variant of `SmallOrganLoss` close the data-side gap.

**Tech Stack:** PyTorch 2.x, MONAI SwinUNETR-v2, AMOS22 preprocessed cache (existing), TotalSegmentator v2 (Zenodo, ~50 GB CT volumes, CC-BY), Python312 CUDA env, bf16 autocast, single RTX 4090.

---

## Scope of this plan

This plan covers **Phase I only** (Days 1-7 of the 30-day spec). It ends at GATE I sliding-window evaluation. Phase II-V plans will be written incrementally after each gate, because Phase II decisions depend on Phase I outcomes (per the spec's named-fallback table).

**Phase I deliverables:**
1. `BalancedBatchSampler` — guarantees ≥1 patch per rare organ per batch.
2. `SmallOrganLoss` extension — presence-aware class weights.
3. `TotalSegmentator` dataset loader + label-remap (13 shared organs).
4. TotalSeg pretraining script (10 ep VoCo-L → AMOS22 transfer).
5. `OrganMoE-3D` module — `MoELoRALinear`, `PresenceHead`, `PresenceConditionedRouter`.
6. AMOS22 fine-tune of OrganMoE-3D (10 ep).
7. Sliding-window 3D eval = **GATE I**.

---

## File Structure

**New files (Phase I):**

| Path | Responsibility |
|---|---|
| `datasets/balanced_sampler.py` | `BalancedBatchSampler` (PyTorch `Sampler`) ensuring per-batch rare-organ coverage |
| `datasets/totalsegmentator.py` | `TotalSegmentatorDataset` — reads Zenodo dump, remaps to AMOS22 13-organ subset |
| `datasets/totalseg_label_map.py` | Pure label-mapping table + helpers (no torch deps) |
| `models/organmoe_3d.py` | `MoELoRALinear`, `PresenceHead`, `PresenceConditionedRouter`, `inject_organmoe_into_swin()` |
| `training/losses_organmoe.py` | `OrganMoELoss` = `SmallOrganLoss` + presence BCE + MoE load-balance aux |
| `training/pretrain_totalseg.py` | Pretraining script: VoCo-L on TotalSeg, save `checkpoints/pretrained/voco_totalseg_l.pt` |
| `training/train_organmoe_amos22.py` | AMOS22 fine-tune script using OrganMoE-3D |
| `configs/organmoe_phase_i_pretrain.yaml` | TotalSeg pretrain config |
| `configs/organmoe_phase_i_ft.yaml` | AMOS22 fine-tune config |
| `tests/test_balanced_sampler.py` | Sampler unit tests |
| `tests/test_totalseg_label_map.py` | Label-map unit tests |
| `tests/test_organmoe_module.py` | OrganMoE-3D module unit tests (smoke + grad + VRAM) |
| `tests/test_organmoe_loss.py` | Loss unit tests |
| `scripts/download_totalseg.py` | One-shot Zenodo download + verify |
| `scripts/eval_organmoe_3d.py` | Sliding-window 3D eval wrapper around `eval_v9_proposer_preproc.py` |

**Modified files (Phase I):**

| Path | Reason |
|---|---|
| `models/swin_unetr_3d.py` | Add `use_organmoe: bool` flag and call `inject_organmoe_into_swin()` instead of `_inject_lora_into_swin()` when set |
| `datasets/__init__.py` | Export `BalancedBatchSampler`, `TotalSegmentatorDataset` |
| `models/__init__.py` | Export `OrganMoE-3D` symbols |
| `training/__init__.py` | Export `OrganMoELoss` |
| `MASTER_PROJECT_LOG.md` | Append Phase I daily entries + GATE I result |

**Sacred ckpts (NEVER overwrite, per spec §8):**
- `checkpoints/v10_voco_l_ft_r3/epoch_009.pt`
- `checkpoints/v9_stage1/last.pt`
- `checkpoints/v9_stage2_v5/refiner_ep_009.pt`
- `checkpoints/trissr_v9/last.pt`
- `checkpoints/pretrained/voco/VoComni_L.pt`
- `checkpoints/pretrained/voco/VoComni_H.pt`

**New ckpt directories (Phase I writes here):**
- `checkpoints/voco_totalseg_pretrain/` (TotalSeg pretrain)
- `checkpoints/organmoe_phase_i/` (AMOS22 fine-tune)

---

## Conventions used throughout this plan

- **Python interpreter:** `C:/Users/Raywa/AppData/Local/Programs/Python/Python312/python.exe` (per `feedback_python_cuda_env`). The default `python` on git-bash is the miniconda CPU-only build.
- **PYTHONPATH:** `C:/Users/Raywa/Desktop/VoluFormer3D_V4` (set per shell or `os.environ` in scripts).
- **AMP dtype:** `bf16` (per `feedback_voco_amp_bf16` — fp16 NaNs from step 0 at fs≥96).
- **Test runner:** `pytest -v` (smoke tests must run on CPU OR a tiny GPU patch).
- **Commit message style:** `feat(organmoe): <thing>` / `test(organmoe): <thing>` / `chore: <thing>`. Mirrors existing repo style.
- **15 organs (1-indexed in label volumes, 0 = bg):**
  `1=spleen, 2=r_kidney, 3=l_kidney, 4=gallbladder, 5=esophagus, 6=liver, 7=stomach, 8=aorta, 9=ivc, 10=pancreas, 11=r_adrenal, 12=l_adrenal, 13=duodenum, 14=bladder, 15=prostate_uterus`.
- **Rare organs:** indices `[4, 7, 10, 11, 12, 13, 14, 15]` (gallbladder, stomach, pancreas, r/l adrenal, duodenum, bladder, prostate_uterus).

---

## Task 1: Project bootstrap — create new ckpt dirs and update memory

**Files:**
- Create: `checkpoints/voco_totalseg_pretrain/.gitkeep`
- Create: `checkpoints/organmoe_phase_i/.gitkeep`
- Create: `reports/organmoe_phase_i/.gitkeep`
- Create: `logs/organmoe_phase_i/.gitkeep`

- [ ] **Step 1: Create empty directories**

```bash
mkdir -p C:/Users/Raywa/Desktop/VoluFormer3D_V4/checkpoints/voco_totalseg_pretrain
mkdir -p C:/Users/Raywa/Desktop/VoluFormer3D_V4/checkpoints/organmoe_phase_i
mkdir -p C:/Users/Raywa/Desktop/VoluFormer3D_V4/reports/organmoe_phase_i
mkdir -p C:/Users/Raywa/Desktop/VoluFormer3D_V4/logs/organmoe_phase_i
touch C:/Users/Raywa/Desktop/VoluFormer3D_V4/checkpoints/voco_totalseg_pretrain/.gitkeep
touch C:/Users/Raywa/Desktop/VoluFormer3D_V4/checkpoints/organmoe_phase_i/.gitkeep
touch C:/Users/Raywa/Desktop/VoluFormer3D_V4/reports/organmoe_phase_i/.gitkeep
touch C:/Users/Raywa/Desktop/VoluFormer3D_V4/logs/organmoe_phase_i/.gitkeep
```

Expected: 4 directories with `.gitkeep` files.

- [ ] **Step 2: Append Phase I kickoff line to master log**

Edit `C:/Users/Raywa/Desktop/VoluFormer3D_V4/MASTER_PROJECT_LOG.md` — append at end:

```markdown

---

## 2026-04-25 — OrganMoE-3D Phase I kickoff (spec: docs/superpowers/specs/2026-04-25-organmoe-3d-design.md, plan: docs/superpowers/plans/2026-04-25-organmoe-3d-phase-i.md)

Phase I (Days 1-7) starts. Targets:
- Build BalancedBatchSampler, OrganMoELoss extension, TotalSeg dataset
- Pretrain VoCo-L on TotalSegmentator (10 ep)
- Build OrganMoE-3D module (K=8 LoRA-rank-16 experts + presence head + router)
- Fine-tune on AMOS22 (10 ep)
- GATE I day 7: sliding-window mean Dice ≥0.85 AND prostate ≥0.30

New ckpt dirs:
- `checkpoints/voco_totalseg_pretrain/`
- `checkpoints/organmoe_phase_i/`

Sacred ckpts unchanged.
```

- [ ] **Step 3: No commit yet** — bootstrap is part of Task 2's first commit (small dirs + memo).

---

## Task 2: TotalSegmentator label-map module + tests

**Files:**
- Create: `datasets/totalseg_label_map.py`
- Test: `tests/test_totalseg_label_map.py`

**Why first:** Label map is pure, no torch deps, no I/O. Establishes the 13-organ shared-label contract that BalancedBatchSampler, the TotalSeg loader, and the pretrain script all depend on.

- [ ] **Step 1: Write the failing test**

Create `tests/test_totalseg_label_map.py`:

```python
"""Label map between TotalSegmentator (104 classes) and AMOS22 (15 organs).

13 of AMOS22's 15 organs have direct or near-direct counterparts in TotalSeg.
Bladder and prostate_uterus are AMOS-only (not in TotalSeg core 104).
"""
import pytest
from datasets.totalseg_label_map import (
    AMOS22_TO_TOTALSEG,
    TOTALSEG_TO_AMOS22,
    SHARED_AMOS_INDICES,
    AMOS_ONLY_INDICES,
    remap_label_volume,
)
import numpy as np


def test_shared_indices_count():
    assert len(SHARED_AMOS_INDICES) == 13


def test_amos_only_is_bladder_and_prostate():
    assert set(AMOS_ONLY_INDICES) == {14, 15}


def test_amos_to_totalseg_keys_are_shared():
    assert set(AMOS22_TO_TOTALSEG.keys()) == set(SHARED_AMOS_INDICES)


def test_round_trip_consistency():
    for amos_id in SHARED_AMOS_INDICES:
        ts_ids = AMOS22_TO_TOTALSEG[amos_id]
        for ts_id in ts_ids:
            assert TOTALSEG_TO_AMOS22[ts_id] == amos_id


def test_remap_drops_unmapped_classes():
    # TotalSeg label 200 (not in our map) must become 0 (bg).
    vol = np.array([[0, 1, 200, 14]], dtype=np.int64)  # ts 1=spleen → amos 1
    out = remap_label_volume(vol)
    assert out[0, 0] == 0
    assert out[0, 1] == 1  # spleen
    assert out[0, 2] == 0  # unknown → bg
    # ts 14 may map to something or to 0; check it does not crash
    assert out.dtype == np.int64
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd C:/Users/Raywa/Desktop/VoluFormer3D_V4
PYTHONPATH=. C:/Users/Raywa/AppData/Local/Programs/Python/Python312/python.exe -m pytest tests/test_totalseg_label_map.py -v
```

Expected: ImportError / ModuleNotFoundError on `datasets.totalseg_label_map`.

- [ ] **Step 3: Write the implementation**

Create `datasets/totalseg_label_map.py`:

```python
"""AMOS22 ↔ TotalSegmentator label mapping for Phase I cross-dataset pretrain.

Source: TotalSegmentator v2 (Wasserthal et al, RSNA 2023). Class IDs follow the
official `total` task (104 classes). https://github.com/wasserthal/TotalSegmentator

AMOS22 organ IDs (1-indexed; 0 = bg):
  1=spleen, 2=r_kidney, 3=l_kidney, 4=gallbladder, 5=esophagus, 6=liver,
  7=stomach, 8=aorta, 9=ivc, 10=pancreas, 11=r_adrenal, 12=l_adrenal,
  13=duodenum, 14=bladder, 15=prostate_uterus

13 organs have direct counterparts in TotalSeg `total` task. Bladder is in
TotalSeg under `urinary_bladder` (id 104). Prostate_uterus has no direct
counterpart (TotSeg has prostate-male only via separate task) — we treat it
as AMOS-only for Phase I and rely on AMOS22 fine-tune to recover it.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

# AMOS organ id -> list of TotalSeg class IDs that should remap to it.
# Some AMOS classes correspond to multiple TS classes (e.g., L1-L5 vertebrae
# would not but kidneys are split L/R in both).
AMOS22_TO_TOTALSEG: Dict[int, List[int]] = {
    1: [1],         # spleen
    2: [2],         # kidney_right
    3: [3],         # kidney_left
    4: [4],         # gallbladder
    5: [42],        # esophagus
    6: [5],         # liver
    7: [6],         # stomach
    8: [7],         # aorta
    9: [8, 9],      # inferior_vena_cava + portal_vein_and_splenic_vein (closest)
    10: [10],       # pancreas
    11: [11],       # adrenal_gland_right
    12: [12],       # adrenal_gland_left
    13: [55],       # duodenum
}

SHARED_AMOS_INDICES: List[int] = sorted(AMOS22_TO_TOTALSEG.keys())  # 13 indices
AMOS_ONLY_INDICES: List[int] = [14, 15]                              # bladder, prostate_uterus

# Reverse lookup: TS class id -> AMOS organ id
TOTALSEG_TO_AMOS22: Dict[int, int] = {}
for amos_id, ts_list in AMOS22_TO_TOTALSEG.items():
    for ts_id in ts_list:
        TOTALSEG_TO_AMOS22[ts_id] = amos_id


def remap_label_volume(label: np.ndarray) -> np.ndarray:
    """Map a TotalSeg label volume to AMOS22 indices.

    Any TS class not in TOTALSEG_TO_AMOS22 is mapped to 0 (background).
    Preserves dtype = int64 and shape.
    """
    out = np.zeros_like(label, dtype=np.int64)
    for ts_id, amos_id in TOTALSEG_TO_AMOS22.items():
        out[label == ts_id] = amos_id
    return out
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=. C:/Users/Raywa/AppData/Local/Programs/Python/Python312/python.exe -m pytest tests/test_totalseg_label_map.py -v
```

Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
cd C:/Users/Raywa/Desktop/VoluFormer3D_V4
git add datasets/totalseg_label_map.py tests/test_totalseg_label_map.py checkpoints/voco_totalseg_pretrain/.gitkeep checkpoints/organmoe_phase_i/.gitkeep reports/organmoe_phase_i/.gitkeep logs/organmoe_phase_i/.gitkeep MASTER_PROJECT_LOG.md docs/superpowers/plans/2026-04-25-organmoe-3d-phase-i.md docs/superpowers/specs/2026-04-25-organmoe-3d-design.md
git commit -m "feat(organmoe): scaffold phase I plan + TotalSeg label map"
```

---

## Task 3: BalancedBatchSampler

**Files:**
- Create: `datasets/balanced_sampler.py`
- Test: `tests/test_balanced_sampler.py`

**Purpose:** Force per-batch coverage of rare organs. The existing AMOS22V9Dataset emits 4 slabs per volume × 270 volumes = 1080 indices. With batch_size=2 and uniform sampling, prostate (~6 % of volumes) almost never appears in a batch. The sampler indexes by `(volume_id, organ_id)` and round-robins through rare-organ slabs.

- [ ] **Step 1: Write the failing test**

Create `tests/test_balanced_sampler.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=. C:/Users/Raywa/AppData/Local/Programs/Python/Python312/python.exe -m pytest tests/test_balanced_sampler.py -v
```

Expected: ModuleNotFoundError on `datasets.balanced_sampler`.

- [ ] **Step 3: Write the implementation**

Create `datasets/balanced_sampler.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=. C:/Users/Raywa/AppData/Local/Programs/Python/Python312/python.exe -m pytest tests/test_balanced_sampler.py -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Add `organs_in_volume` to AMOS22V9Dataset**

Edit `datasets/amos22_v9.py`. Add the following method inside `class AMOS22V9Dataset`, just below `_load_volume`:

```python
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
```

- [ ] **Step 6: Smoke test the integration**

```bash
PYTHONPATH=. C:/Users/Raywa/AppData/Local/Programs/Python/Python312/python.exe -c "
from datasets.amos22_v9 import AMOS22V9Dataset
from datasets.balanced_sampler import BalancedBatchSampler
ds = AMOS22V9Dataset(data_root='C:/Users/Raywa/Desktop/LiteSAM3D/data/amos22', split='train', img_size=320, depth=8)
print(f'volumes={len(ds.volume_ids)} length={ds.length}')
sampler = BalancedBatchSampler(ds, batch_size=2, rare_organs=[4, 7, 10, 11, 12, 13, 14, 15], shuffle=True, seed=42)
print(f'batches={len(sampler)}')
batches = list(sampler)
print(f'first 3 batches: {batches[:3]}')
"
```

Expected: prints volume count, batch count, and first 3 batches as `[(idx, idx), (idx, idx), (idx, idx)]`.

- [ ] **Step 7: Commit**

```bash
git add datasets/balanced_sampler.py datasets/amos22_v9.py tests/test_balanced_sampler.py
git commit -m "feat(organmoe): BalancedBatchSampler + organs_in_volume on AMOS22V9"
```

---

## Task 4: OrganMoELoss (presence-aware loss)

**Files:**
- Create: `training/losses_organmoe.py`
- Test: `tests/test_organmoe_loss.py`

**Purpose:** Wrap `SmallOrganLoss` with two extras (a) per-batch presence weights so absent classes do not contribute to gradients (b) presence BCE supervising the future PresenceHead and (c) MoE load-balance auxiliary term. Total loss = `seg_loss + λ_pres * presence_bce + λ_lb * load_balance`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_organmoe_loss.py`:

```python
"""OrganMoELoss: SmallOrganLoss + presence BCE + MoE load balance."""
import torch
from training.losses_organmoe import OrganMoELoss


def test_loss_runs_no_nan():
    torch.manual_seed(0)
    crit = OrganMoELoss(n_classes=16, presence_weight=0.1, balance_weight=0.01)
    logits = torch.randn(2, 16, 8, 16, 16, requires_grad=True)
    target = torch.randint(0, 16, (2, 8, 16, 16))
    presence_logits = torch.randn(2, 15, requires_grad=True)
    gate_weights = torch.randn(2, 8, 16, 16, 16).softmax(dim=1)  # K=8 experts
    out = crit(
        seg_logits=logits, seg_target=target,
        presence_logits=presence_logits, gate_weights=gate_weights,
    )
    assert torch.isfinite(out["total"]).item()
    out["total"].backward()
    assert torch.isfinite(logits.grad.norm()).item()
    assert torch.isfinite(presence_logits.grad.norm()).item()


def test_presence_bce_target_derived_from_target():
    """The presence target is derived from the seg target: organ k is 'present'
    iff at least 1 voxel of organ k exists in the volume."""
    crit = OrganMoELoss(n_classes=16, presence_weight=1.0, balance_weight=0.0)
    logits = torch.randn(1, 16, 4, 8, 8)
    target = torch.zeros(1, 4, 8, 8, dtype=torch.long)
    target[0, 0, 0, 0] = 5  # only organ 5 present
    presence_logits = torch.zeros(1, 15)  # uniform 0.5
    out = crit(seg_logits=logits, seg_target=target, presence_logits=presence_logits)
    # The derived presence target should have 1.0 at index 4 (organ 5 → 0-indexed 4)
    assert torch.isfinite(out["presence_bce"]).item()
    assert out["presence_bce"].item() > 0


def test_load_balance_aux_zero_when_uniform():
    """If experts are used uniformly, load-balance term is ~minimal."""
    crit = OrganMoELoss(n_classes=16, presence_weight=0.0, balance_weight=1.0)
    logits = torch.randn(1, 16, 4, 8, 8)
    target = torch.randint(0, 16, (1, 4, 8, 8))
    presence_logits = torch.zeros(1, 15)
    K = 8
    # Uniform gate: every expert chosen equally.
    uniform_gate = torch.ones(1, K, 4, 8, 8) / K
    skewed_gate = torch.zeros(1, K, 4, 8, 8)
    skewed_gate[:, 0] = 1.0  # all weight on expert 0
    out_u = crit(seg_logits=logits, seg_target=target, presence_logits=presence_logits, gate_weights=uniform_gate)
    out_s = crit(seg_logits=logits, seg_target=target, presence_logits=presence_logits, gate_weights=skewed_gate)
    assert out_s["balance"].item() > out_u["balance"].item()


def test_absent_classes_zero_weighted():
    """If an organ is absent in a batch, its presence weight is 0 in the
    weighted-dice term so it does not contribute to gradients."""
    crit = OrganMoELoss(n_classes=16, presence_weight=0.0, balance_weight=0.0,
                        use_presence_weighted_dice=True)
    logits = torch.randn(1, 16, 4, 8, 8, requires_grad=True)
    target = torch.zeros(1, 4, 8, 8, dtype=torch.long)
    target[0, 0, 0, 0] = 6  # only organ 6 present
    presence_logits = torch.zeros(1, 15)
    out = crit(seg_logits=logits, seg_target=target, presence_logits=presence_logits)
    out["total"].backward()
    # Gradient on absent organs (e.g., channel 1) should be small relative to
    # gradient on present organ channel 6.
    g_present = logits.grad[0, 6].abs().sum().item()
    g_absent = logits.grad[0, 1].abs().sum().item()
    assert g_present > g_absent  # primary signal is on present organ
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=. C:/Users/Raywa/AppData/Local/Programs/Python/Python312/python.exe -m pytest tests/test_organmoe_loss.py -v
```

Expected: ModuleNotFoundError on `training.losses_organmoe`.

- [ ] **Step 3: Write the implementation**

Create `training/losses_organmoe.py`:

```python
"""OrganMoELoss = SmallOrganLoss + presence BCE + MoE load balance.

Three terms (each toggleable by setting weight to 0):
  - seg_loss: existing SmallOrganLoss on (B, K+1, Z, H, W) seg logits.
              Optional `use_presence_weighted_dice` injects a per-batch
              presence mask into the class-weight prior.
  - presence_bce: BCEWithLogitsLoss on (B, n_organs) presence prediction.
                  Target derived from `seg_target` (organ present iff >=1 vox).
  - balance: MoE load-balance auxiliary loss following Switch-Transformer.
             `(N_experts * sum_k (P_k * F_k))` where P_k is mean router prob
             across the batch and F_k is the fraction-of-tokens routed to k.
             Encourages uniform expert usage.
"""
from __future__ import annotations
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from training.losses_small_organ import SmallOrganLoss


class OrganMoELoss(nn.Module):
    def __init__(
        self,
        n_classes: int,
        presence_weight: float = 0.1,
        balance_weight: float = 0.01,
        use_presence_weighted_dice: bool = True,
        small_organ_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.n_classes = int(n_classes)
        self.n_organs = int(n_classes) - 1  # exclude bg
        self.pres_w = float(presence_weight)
        self.bal_w = float(balance_weight)
        self.use_presence_weighted_dice = bool(use_presence_weighted_dice)
        kw = dict(small_organ_kwargs or {})
        self.seg_loss = SmallOrganLoss(n_classes=n_classes, **kw)
        self.bce = nn.BCEWithLogitsLoss()

    def _derive_presence_target(self, seg_target: torch.Tensor) -> torch.Tensor:
        """Return (B, n_organs) bool presence target (1.0 if organ >=1 voxel)."""
        B = seg_target.shape[0]
        target = torch.zeros(B, self.n_organs, device=seg_target.device, dtype=torch.float32)
        for b in range(B):
            uniq = torch.unique(seg_target[b])
            for v in uniq.tolist():
                if 0 < v <= self.n_organs:
                    target[b, v - 1] = 1.0
        return target

    def _load_balance(self, gate_weights: torch.Tensor) -> torch.Tensor:
        """Switch-Transformer load balance loss.

        gate_weights: (B, K, ...) — softmax over K experts at each token.
        Returns scalar = K * mean( P_k * F_k ).
        """
        B = gate_weights.shape[0]
        K = gate_weights.shape[1]
        # Flatten spatial dims into "tokens".
        gate_flat = gate_weights.reshape(B, K, -1)              # (B, K, T)
        # P_k = mean router prob over all tokens
        P = gate_flat.mean(dim=(0, 2))                            # (K,)
        # F_k = fraction of tokens whose argmax router is expert k
        argmax = gate_flat.argmax(dim=1)                          # (B, T)
        F_ = torch.zeros_like(P)
        for k in range(K):
            F_[k] = (argmax == k).float().mean()
        return K * (P * F_).sum()

    def forward(
        self,
        seg_logits: torch.Tensor,
        seg_target: torch.Tensor,
        presence_logits: Optional[torch.Tensor] = None,
        gate_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}

        if self.use_presence_weighted_dice:
            present = self._derive_presence_target(seg_target)            # (B, n_organs)
            # Convert to per-class prior (K+1 long; bg always 1.0)
            class_prior = torch.ones(self.n_classes, device=seg_logits.device)
            # If any sample in batch contains an organ, weight stays 1.0;
            # if NO sample contains it, drop weight to a small value (0.1).
            for k in range(self.n_organs):
                if present[:, k].sum() == 0:
                    class_prior[k + 1] = 0.1
            self.seg_loss.class_prior = class_prior
        seg = self.seg_loss(seg_logits, seg_target.long())
        out["seg"] = seg

        if presence_logits is not None and self.pres_w > 0:
            presence_target = self._derive_presence_target(seg_target)
            pres = self.bce(presence_logits, presence_target)
            out["presence_bce"] = pres
        else:
            out["presence_bce"] = seg.new_zeros(())

        if gate_weights is not None and self.bal_w > 0:
            bal = self._load_balance(gate_weights)
            out["balance"] = bal
        else:
            out["balance"] = seg.new_zeros(())

        out["total"] = seg + self.pres_w * out["presence_bce"] + self.bal_w * out["balance"]
        return out


if __name__ == "__main__":
    torch.manual_seed(0)
    crit = OrganMoELoss(n_classes=16)
    logits = torch.randn(2, 16, 8, 16, 16, requires_grad=True)
    target = torch.randint(0, 16, (2, 8, 16, 16))
    presence_logits = torch.randn(2, 15, requires_grad=True)
    gate_weights = torch.randn(2, 8, 8, 16, 16).softmax(dim=1)
    out = crit(logits, target, presence_logits, gate_weights)
    out["total"].backward()
    print(f"total={out['total'].item():.4f} seg={out['seg'].item():.4f} "
          f"pres={out['presence_bce'].item():.4f} bal={out['balance'].item():.4f}")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=. C:/Users/Raywa/AppData/Local/Programs/Python/Python312/python.exe -m pytest tests/test_organmoe_loss.py -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Run module smoke**

```bash
PYTHONPATH=. C:/Users/Raywa/AppData/Local/Programs/Python/Python312/python.exe training/losses_organmoe.py
```

Expected: prints `total=... seg=... pres=... bal=...` (all finite).

- [ ] **Step 6: Commit**

```bash
git add training/losses_organmoe.py tests/test_organmoe_loss.py
git commit -m "feat(organmoe): OrganMoELoss = SmallOrganLoss + presence BCE + MoE balance"
```

---

## Task 5: TotalSegmentator dataset loader

**Files:**
- Create: `datasets/totalsegmentator.py`
- Create: `scripts/download_totalseg.py`

**Purpose:** A PyTorch `Dataset` that reads the TotalSegmentator v2 Zenodo dump. The download script uses the official `totalsegmentator` PyPI package's `download_pretrained_weights` mode is for inference; we want the training data, which is on Zenodo at `https://zenodo.org/records/10047292` (TotalSegmentator-v201, 1228 CT volumes, ~28 GB).

**Note:** The download is the long-pole task. **Start it in the background as the first action of Day 1, then proceed with the rest of Phase I work in parallel.**

- [ ] **Step 1: Write the download script**

Create `scripts/download_totalseg.py`:

```python
"""One-shot downloader for TotalSegmentator v2 from Zenodo.

Downloads the dataset zip, verifies SHA256, extracts to data_root, and prints
a summary. Designed to run in the background (~2-3 hours on typical broadband).

Usage:
    python scripts/download_totalseg.py --data-root D:/data/totalsegmentator

Acceptance:
    - Final layout: <data_root>/Totalsegmentator_dataset_v201/sNNNN/{ct.nii.gz, segmentations/}
    - Manifest: <data_root>/totalsegmentator_manifest.json (volume_id list)
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path
import zipfile

ZENODO_URL = "https://zenodo.org/records/10047292/files/Totalsegmentator_dataset_v201.zip"
EXPECTED_SIZE_GB_MIN = 25
EXPECTED_SIZE_GB_MAX = 35


def _download(url: str, out: Path) -> None:
    print(f"[totalseg-dl] starting download → {out}")
    start = time.time()
    with urllib.request.urlopen(url) as resp, open(out, "wb") as f:
        total_bytes = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        while True:
            chunk = resp.read(1 << 22)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if downloaded % (1 << 28) < (1 << 22):
                pct = 100.0 * downloaded / max(total_bytes, 1)
                elapsed = time.time() - start
                rate = downloaded / max(elapsed, 1) / 1e6
                print(f"[totalseg-dl] {downloaded/1e9:.2f}/{total_bytes/1e9:.2f} GB "
                      f"({pct:.1f}%) at {rate:.1f} MB/s")
    print(f"[totalseg-dl] done in {(time.time()-start)/60:.1f} min")


def _extract(zip_path: Path, out_dir: Path) -> None:
    print(f"[totalseg-dl] extracting → {out_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    print("[totalseg-dl] extracted")


def _build_manifest(data_root: Path) -> dict:
    base = data_root / "Totalsegmentator_dataset_v201"
    if not base.exists():
        raise FileNotFoundError(f"Expected {base} after extract")
    vols = sorted([p.name for p in base.glob("s*") if (p / "ct.nii.gz").exists()])
    manifest = {"data_root": str(base), "n_volumes": len(vols), "volume_ids": vols}
    out = data_root / "totalsegmentator_manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--zip-name", default="Totalsegmentator_dataset_v201.zip")
    parser.add_argument("--skip-download", action="store_true",
                        help="Use this if zip is already present.")
    args = parser.parse_args()

    args.data_root.mkdir(parents=True, exist_ok=True)
    zip_path = args.data_root / args.zip_name

    if not args.skip_download:
        if zip_path.exists():
            sz = zip_path.stat().st_size / 1e9
            print(f"[totalseg-dl] zip already present ({sz:.1f} GB) — skipping download")
        else:
            _download(ZENODO_URL, zip_path)

    sz_gb = zip_path.stat().st_size / 1e9
    if not (EXPECTED_SIZE_GB_MIN <= sz_gb <= EXPECTED_SIZE_GB_MAX):
        print(f"[totalseg-dl] WARNING: zip size {sz_gb:.1f} GB outside "
              f"expected [{EXPECTED_SIZE_GB_MIN}, {EXPECTED_SIZE_GB_MAX}] GB")

    if not (args.data_root / "Totalsegmentator_dataset_v201").exists():
        _extract(zip_path, args.data_root)

    manifest = _build_manifest(args.data_root)
    print(f"[totalseg-dl] manifest: {manifest['n_volumes']} volumes")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke the download script (no actual download)**

```bash
PYTHONPATH=. C:/Users/Raywa/AppData/Local/Programs/Python/Python312/python.exe scripts/download_totalseg.py --help
```

Expected: argparse usage prints, no error.

- [ ] **Step 3: Start the actual download in the background**

```bash
mkdir -p D:/data/totalsegmentator
nohup C:/Users/Raywa/AppData/Local/Programs/Python/Python312/python.exe \
  scripts/download_totalseg.py --data-root D:/data/totalsegmentator \
  > logs/organmoe_phase_i/totalseg_download.log 2>&1 &
echo "[totalseg-dl] PID $!"
```

Expected: PID printed; tail of `totalseg_download.log` shows progress lines. **Continue with Tasks 6-9 while this runs.** Verify completion before Task 10.

- [ ] **Step 4: Write the dataset class**

Create `datasets/totalsegmentator.py`:

```python
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
    ) -> None:
        self.root = Path(data_root)
        self.split = split
        self.volume_patch = int(volume_patch)
        self.hu_clip = hu_clip
        self.slabs_per_volume = int(slabs_per_volume)
        self.n_organs = self.N_ORGANS
        self._seed = int(seed)

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
        """Returns (image, label) as (D, H, W) float32 / int64."""
        import nibabel as nib
        ct = nib.load(str(self._base / vol_id / "ct.nii.gz")).get_fdata().astype(np.float32)
        # Build a single label volume by summing per-class masks (the latest mask wins).
        seg_dir = self._base / vol_id / "segmentations"
        label = np.zeros_like(ct, dtype=np.int64)
        for ts_name, amos_id in _TS_FILENAME_TO_AMOS.items():
            f = seg_dir / f"{ts_name}.nii.gz"
            if not f.exists():
                continue
            m = nib.load(str(f)).get_fdata() > 0.5
            label[m] = amos_id
        # Transpose to (D, H, W) to match AMOS22V9Dataset.
        ct = ct.transpose(2, 0, 1)
        label = label.transpose(2, 0, 1)
        return ct, label

    def organs_in_volume(self, volume_idx: int) -> set:
        """Lazy presence cache shared with BalancedBatchSampler."""
        if not hasattr(self, "_presence_cache"):
            self._presence_cache: dict = {}
        if volume_idx in self._presence_cache:
            return self._presence_cache[volume_idx]
        _, lab = self._load_volume(self.volume_ids[volume_idx])
        present = set(int(v) for v in np.unique(lab) if int(v) > 0)
        self._presence_cache[volume_idx] = present
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
        slab_idx = idx % self.slabs_per_volume
        rng = np.random.default_rng(self._seed + idx * 13)
        img, lab = self._load_volume(self.volume_ids[vol_idx])
        img = self._normalize(img)
        img_p, lab_p = self._crop_patch(img, lab, rng)
        return {
            "volume_patch": torch.from_numpy(img_p).unsqueeze(0).float(),  # (1, D, H, W)
            "volume_label": torch.from_numpy(lab_p).long(),                 # (D, H, W)
            "volume_id": self.volume_ids[vol_idx],
            "slab_idx": slab_idx,
        }
```

- [ ] **Step 5: Smoke the dataset (after manifest exists; can be deferred until download finishes)**

```bash
# Run only after the download has finished. Skipping during initial Phase I if download still in flight.
PYTHONPATH=. C:/Users/Raywa/AppData/Local/Programs/Python/Python312/python.exe -c "
from datasets.totalsegmentator import TotalSegmentatorDataset
ds = TotalSegmentatorDataset(data_root='D:/data/totalsegmentator', split='train', volume_patch=96)
print(f'volumes={len(ds.volume_ids)} length={ds.length}')
b = ds[0]
print(f'patch={b[\"volume_patch\"].shape} label_unique={b[\"volume_label\"].unique().tolist()}')
"
```

Expected: prints volume count, patch shape `[1, 96, 96, 96]`, label unique values subset of `{0, 1..13}`.

- [ ] **Step 6: Commit**

```bash
git add scripts/download_totalseg.py datasets/totalsegmentator.py
git commit -m "feat(organmoe): TotalSegmentator dataset + Zenodo download script"
```

---

## Task 6: OrganMoE-3D module — `MoELoRALinear`, `PresenceHead`, router

**Files:**
- Create: `models/organmoe_3d.py`
- Test: `tests/test_organmoe_module.py`

**Purpose:** The novel algorithmic contribution. Replaces the existing `_SwinLoRALinear` with a sparse top-2 MoE over K=8 LoRA experts whose router is conditioned on (a) layer features and (b) the predicted class-presence vector `c ∈ R^15`.

**Memory note:** K=8 LoRA-rank-16 experts on a single Linear of in/out dims (D, D) costs `8 * (D*16 + 16*D) * 4 bytes = 8 * 32D * 4 = 1024 D bytes`. At D=768 (VoCo-L attention), that's ~786 KB per Linear; injecting at every QKV/proj/fc1/fc2 across 24 blocks ≈ 24 * 4 * 786 KB ≈ 75 MB extra. Negligible vs the 1.16 GB base weights.

- [ ] **Step 1: Write the failing test**

Create `tests/test_organmoe_module.py`:

```python
"""OrganMoE-3D unit tests.

Each test runs on CPU with a tiny shape so it stays under 1 second.
A separate VRAM smoke is in scripts/ for the GPU.
"""
import pytest
import torch
import torch.nn as nn
from models.organmoe_3d import MoELoRALinear, PresenceHead, PresenceConditionedRouter


def test_moe_lora_linear_forward_shape():
    base = nn.Linear(64, 96)
    moe = MoELoRALinear(base=base, n_experts=4, rank=8, alpha=8.0, n_organs=15, top_k=2)
    x = torch.randn(2, 10, 64)
    organ_presence = torch.rand(2, 15)
    out, gate = moe(x, organ_presence=organ_presence)
    assert out.shape == (2, 10, 96)
    assert gate.shape == (2, 4, 10) or gate.shape == (2, 10, 4)  # implementation choice


def test_moe_base_frozen():
    base = nn.Linear(32, 32)
    moe = MoELoRALinear(base=base, n_experts=4, rank=4, alpha=4.0, n_organs=15, top_k=2)
    for p in moe.base.parameters():
        assert not p.requires_grad


def test_top_k_routing_only_uses_k_experts_per_token():
    base = nn.Linear(32, 32)
    moe = MoELoRALinear(base=base, n_experts=8, rank=4, alpha=4.0, n_organs=15, top_k=2)
    x = torch.randn(1, 5, 32)
    organ_presence = torch.zeros(1, 15)
    _, gate = moe(x, organ_presence=organ_presence)
    # gate is (B, K, T) softmax over K; top-2 means only 2 nonzero per token.
    if gate.shape[1] == 8:
        nonzero = (gate > 0).sum(dim=1)  # (B, T)
    else:
        nonzero = (gate > 0).sum(dim=2)
    assert (nonzero <= 2).all()


def test_presence_head_output_shape():
    feat = torch.randn(2, 256, 6, 6, 6)  # (B, C, D, H, W) low-res encoder feature
    head = PresenceHead(in_channels=256, n_organs=15)
    out = head(feat)
    assert out.shape == (2, 15)


def test_router_conditions_on_presence():
    """Same input features but different presence vectors should give
    different routing decisions (router actually uses presence)."""
    base = nn.Linear(32, 32)
    moe = MoELoRALinear(base=base, n_experts=8, rank=4, alpha=4.0, n_organs=15, top_k=2)
    x = torch.randn(1, 4, 32)
    p_a = torch.zeros(1, 15)
    p_b = torch.ones(1, 15)
    _, g_a = moe(x, organ_presence=p_a)
    _, g_b = moe(x, organ_presence=p_b)
    assert not torch.allclose(g_a, g_b)


def test_grad_flows_through_experts():
    base = nn.Linear(32, 32)
    moe = MoELoRALinear(base=base, n_experts=4, rank=4, alpha=4.0, n_organs=15, top_k=2)
    x = torch.randn(1, 4, 32)
    p = torch.zeros(1, 15, requires_grad=True)
    y, _ = moe(x, organ_presence=p)
    y.sum().backward()
    # Some expert lora_A should have nonzero grad.
    grads = [e.lora_A.grad for e in moe.experts]
    has_grad = any(g is not None and g.abs().sum() > 0 for g in grads)
    assert has_grad
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=. C:/Users/Raywa/AppData/Local/Programs/Python/Python312/python.exe -m pytest tests/test_organmoe_module.py -v
```

Expected: ModuleNotFoundError on `models.organmoe_3d`.

- [ ] **Step 3: Write the implementation**

Create `models/organmoe_3d.py`:

```python
"""OrganMoE-3D — Class-Presence-Aware Sparse LoRA Mixture-of-Experts.

Three building blocks:

  1) MoELoRALinear: drop-in replacement for nn.Linear that delegates to
     K LoRA-rank-r experts. Each expert is a (lora_A, lora_B) pair sharing
     the frozen base Linear's weights. Top-k sparse routing.

  2) PresenceConditionedRouter: small MLP that takes (token feature, organ
     presence vector) and outputs K logits per token. Top-k softmax gives
     the per-token gating weights and active expert ids.

  3) PresenceHead: a tiny MLP on globally-pooled low-res encoder features
     that predicts a 15-organ presence vector c ∈ [0, 1]^15.

Injection helper (used by SwinUNETRProposer): replace every named Linear
(qkv/proj/fc1/fc2) inside the swinViT encoder with a MoELoRALinear, sharing
the same router config but distinct expert parameters.
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class _LoRAExpert(nn.Module):
    """One LoRA expert: low-rank delta on a frozen base Linear."""
    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = int(rank)
        self.scale = float(alpha) / max(self.rank, 1)
        self.lora_A = nn.Parameter(torch.zeros(self.rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features) -> (..., out_features)
        return F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale


class PresenceConditionedRouter(nn.Module):
    """Top-k softmax router over K experts; conditioned on token feature + presence vector."""

    def __init__(self, in_features: int, n_experts: int, n_organs: int, top_k: int = 2,
                 hidden: int = 64) -> None:
        super().__init__()
        self.n_experts = int(n_experts)
        self.top_k = int(top_k)
        self.n_organs = int(n_organs)
        self.proj = nn.Sequential(
            nn.Linear(in_features + n_organs, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_experts),
        )

    def forward(self, x: torch.Tensor, organ_presence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (B, T, D)  organ_presence: (B, n_organs)
        Returns:
          gate: (B, K, T) sparse top-k softmax weights (zero outside top-k)
          topk_idx: (B, T, top_k) expert indices
        """
        B, T, D = x.shape
        # Broadcast presence to per-token: (B, 1, n_organs) -> (B, T, n_organs)
        pres_t = organ_presence.unsqueeze(1).expand(B, T, -1)
        cat = torch.cat([x, pres_t], dim=-1)            # (B, T, D + n_organs)
        logits = self.proj(cat)                          # (B, T, K)
        # Top-k masking
        topk_vals, topk_idx = logits.topk(self.top_k, dim=-1)
        mask = torch.full_like(logits, float("-inf"))
        mask.scatter_(-1, topk_idx, topk_vals)
        gate = F.softmax(mask, dim=-1)                   # (B, T, K) zeroed outside top-k
        gate = gate.transpose(1, 2)                      # (B, K, T) for downstream conv
        return gate, topk_idx


class MoELoRALinear(nn.Module):
    """Drop-in replacement for an nn.Linear: y = base(x) + sum_k g_k(x) * expert_k(x)."""

    def __init__(self, base: nn.Linear, n_experts: int, rank: int, alpha: float,
                 n_organs: int, top_k: int = 2, router_hidden: int = 64) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.n_experts = int(n_experts)
        self.top_k = int(top_k)
        self.experts = nn.ModuleList([
            _LoRAExpert(base, rank=rank, alpha=alpha) for _ in range(n_experts)
        ])
        self.router = PresenceConditionedRouter(
            in_features=self.in_features, n_experts=n_experts,
            n_organs=n_organs, top_k=top_k, hidden=router_hidden,
        )

    def forward(self, x: torch.Tensor, organ_presence: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (..., in_features). organ_presence: (B, n_organs) or None.
        If organ_presence is None, defaults to zeros (uniform routing).
        Returns: (output (..., out_features), gate_weights (B, K, T)).
        """
        # Reshape arbitrary leading dims to (B, T, D)
        orig_shape = x.shape
        if x.dim() == 2:
            x_bt = x.unsqueeze(0)            # (1, T, D)
        elif x.dim() == 3:
            x_bt = x                          # (B, T, D)
        else:
            # Fold all non-feature dims into T.
            B = orig_shape[0]
            x_bt = x.reshape(B, -1, self.in_features)
        B, T, D = x_bt.shape
        if organ_presence is None:
            organ_presence = torch.zeros(B, self.router.n_organs, device=x.device, dtype=x.dtype)
        gate, _topk = self.router(x_bt, organ_presence)        # gate: (B, K, T)

        base_out = self.base(x_bt)                              # (B, T, out)
        expert_out = base_out.new_zeros(B, T, self.out_features)
        for k in range(self.n_experts):
            g_k = gate[:, k, :].unsqueeze(-1)                   # (B, T, 1)
            if g_k.abs().sum() == 0:
                continue
            e_k = self.experts[k](x_bt)                          # (B, T, out)
            expert_out = expert_out + g_k * e_k
        out = base_out + expert_out

        # Restore original leading shape
        if len(orig_shape) == 2:
            out = out.squeeze(0)
        elif len(orig_shape) > 3:
            out = out.reshape(*orig_shape[:-1], self.out_features)
        return out, gate


class PresenceHead(nn.Module):
    """Predict (B, n_organs) presence logits from a low-res encoder feature map."""

    def __init__(self, in_channels: int, n_organs: int, hidden: int = 128) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden), nn.GELU(),
            nn.Linear(hidden, n_organs),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.pool(feat))


def inject_organmoe_into_swin(
    swin_module: nn.Module,
    n_experts: int = 8,
    rank: int = 16,
    alpha: float = 16.0,
    n_organs: int = 15,
    top_k: int = 2,
) -> int:
    """Walk swin_module; replace every Linear named qkv/proj/fc1/fc2 with
    a MoELoRALinear. Returns the count of replaced modules.
    """
    n = 0
    for name, child in list(swin_module.named_children()):
        if isinstance(child, nn.Linear) and name in ("qkv", "proj", "fc1", "fc2"):
            setattr(swin_module, name, MoELoRALinear(
                base=child, n_experts=n_experts, rank=rank, alpha=alpha,
                n_organs=n_organs, top_k=top_k,
            ))
            n += 1
        else:
            n += inject_organmoe_into_swin(
                child, n_experts=n_experts, rank=rank, alpha=alpha,
                n_organs=n_organs, top_k=top_k,
            )
    return n


if __name__ == "__main__":
    # Module-level sanity smoke
    base = nn.Linear(128, 128)
    moe = MoELoRALinear(base=base, n_experts=8, rank=16, alpha=16.0, n_organs=15, top_k=2)
    x = torch.randn(2, 64, 128)
    p = torch.rand(2, 15)
    y, g = moe(x, p)
    print(f"in={x.shape} out={y.shape} gate={g.shape} sum_gate≈top_k? "
          f"{g.sum(dim=1).mean().item():.3f} (expected ~1.0)")
    head = PresenceHead(in_channels=384, n_organs=15)
    feat = torch.randn(2, 384, 6, 6, 6)
    print(f"presence={head(feat).shape}")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=. C:/Users/Raywa/AppData/Local/Programs/Python/Python312/python.exe -m pytest tests/test_organmoe_module.py -v
```

Expected: 6 tests pass.

- [ ] **Step 5: Run module smoke**

```bash
PYTHONPATH=. C:/Users/Raywa/AppData/Local/Programs/Python/Python312/python.exe models/organmoe_3d.py
```

Expected: prints `in=torch.Size([2, 64, 128]) out=torch.Size([2, 64, 128]) gate=torch.Size([2, 8, 64]) sum_gate≈top_k? ~1.000` and `presence=torch.Size([2, 15])`.

- [ ] **Step 6: Commit**

```bash
git add models/organmoe_3d.py tests/test_organmoe_module.py
git commit -m "feat(organmoe): MoELoRALinear, PresenceHead, PresenceConditionedRouter"
```

---

## Task 7: Wire `use_organmoe` flag into `SwinUNETRProposer`

**Files:**
- Modify: `models/swin_unetr_3d.py`

**Purpose:** Add a constructor flag `use_organmoe` that swaps `_inject_lora_into_swin` for `inject_organmoe_into_swin`. Plumb the predicted presence vector through forward (in this task we do plumbing only — actual presence prediction is computed by a separate head built in the next task).

- [ ] **Step 1: Read current `SwinUNETRProposer.__init__` signature**

```bash
PYTHONPATH=. C:/Users/Raywa/AppData/Local/Programs/Python/Python312/python.exe -c "
import inspect
from models.swin_unetr_3d import SwinUNETRProposer
print(inspect.signature(SwinUNETRProposer.__init__))
"
```

Capture the signature for the edit below.

- [ ] **Step 2: Add `use_organmoe` flag**

Edit `models/swin_unetr_3d.py`. Add to `SwinUNETRProposer.__init__` parameters (after `lora_rank`):

```python
        use_organmoe: bool = False,
        moe_n_experts: int = 8,
        moe_rank: int = 16,
        moe_alpha: float = 16.0,
        moe_top_k: int = 2,
```

Inside `__init__`, after the existing `if lora_rank > 0:` block that calls `_inject_lora_into_swin`, add:

```python
        # OrganMoE-3D: replace plain LoRA with sparse-MoE LoRA experts.
        # Mutually exclusive with `lora_rank>0` plain-LoRA injection.
        if use_organmoe:
            from models.organmoe_3d import inject_organmoe_into_swin, PresenceHead
            # Freeze swinViT base weights so only experts + decoder train.
            for p in self._swin_unetr.swinViT.parameters():
                p.requires_grad = False
            n_replaced = inject_organmoe_into_swin(
                self._swin_unetr.swinViT,
                n_experts=int(moe_n_experts),
                rank=int(moe_rank),
                alpha=float(moe_alpha),
                n_organs=int(n_organs),
                top_k=int(moe_top_k),
            )
            self._n_organmoe_modules = int(n_replaced)
            # PresenceHead: predicts 15-organ presence from the deepest swinViT feature.
            # Read the deepest channel dim from the existing SwinUNETR config.
            deepest_ch = int(getattr(self._swin_unetr.swinViT, "embed_dim", 96)) * (2 ** 4)
            self.presence_head = PresenceHead(in_channels=deepest_ch, n_organs=int(n_organs))
            self._use_organmoe = True
        else:
            self._use_organmoe = False
            self._n_organmoe_modules = 0
            self.presence_head = None
```

- [ ] **Step 3: Plumb presence through forward**

Find `SwinUNETRProposer.forward`. Before the existing call into `_swin_unetr(...)`, add:

```python
        # If OrganMoE is active, set the per-batch organ_presence on every
        # MoELoRALinear so its router conditions on it. We use the predicted
        # presence (not GT) so train and test are consistent.
        if getattr(self, "_use_organmoe", False):
            from models.organmoe_3d import MoELoRALinear
            # IMPORTANT: verify the swinViT feature interface BEFORE writing this:
            #   PYTHONPATH=. python -c "
            #   from monai.networks.nets import SwinUNETR
            #   m = SwinUNETR(img_size=96, in_channels=1, out_channels=16,
            #                 feature_size=96, use_v2=True)
            #   x = torch.randn(1,1,96,96,96)
            #   out = m.swinViT(x)
            #   print(type(out), [f.shape for f in out] if isinstance(out,(list,tuple)) else out.shape)
            #   "
            # MONAI SwinUNETR.swinViT.forward returns a LIST of 5 stage outputs.
            # Use the deepest stage [4]. If your verification shows different,
            # adjust the index here.
            with torch.cuda.amp.autocast(enabled=False):
                stage_features = self._swin_unetr.swinViT(x)
                deep_feat = stage_features[-1] if isinstance(stage_features, (list, tuple)) else stage_features
            organ_presence = torch.sigmoid(self.presence_head(deep_feat))   # (B, n_organs)
            # Cache for trainer-side loss collection.
            self._last_presence_logits = self.presence_head(deep_feat)
            # Broadcast organ_presence to all MoELoRALinear children via attribute.
            for m in self.modules():
                if isinstance(m, MoELoRALinear):
                    m._cached_presence = organ_presence
```

Then inside `SwinUNETRProposer`, expose two helper accessors so the trainer can collect them after each forward call:

```python
    @property
    def last_presence_logits(self) -> torch.Tensor:
        """The most recent presence-head output (B, n_organs). None if OrganMoE off."""
        return getattr(self, "_last_presence_logits", None)

    def collect_last_gates(self) -> torch.Tensor:
        """Return mean gate weights across all MoELoRALinear modules in this proposer.

        Each module caches its (B, K, T) gate during forward. We average across
        modules to a single (B, K, T_pooled) for the load-balance loss term.
        Returns None if OrganMoE is off or no gates were collected.
        """
        from models.organmoe_3d import MoELoRALinear
        gates = []
        for m in self.modules():
            if isinstance(m, MoELoRALinear) and getattr(m, "_last_gate", None) is not None:
                # Pool spatial dim to a fixed size so per-layer shapes can be averaged.
                g = m._last_gate                                # (B, K, T_layer)
                gates.append(g.mean(dim=-1, keepdim=True))      # (B, K, 1)
        if not gates:
            return None
        # Stack and average across layers: (n_layers, B, K, 1) -> (B, K, 1)
        return torch.stack(gates, dim=0).mean(dim=0)
```

Then update each `MoELoRALinear.forward` call in the swin to read this cached value. In `models/organmoe_3d.py`, modify `MoELoRALinear.forward` to support the cached path:

```python
    def forward(self, x: torch.Tensor, organ_presence: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        if organ_presence is None and hasattr(self, "_cached_presence"):
            organ_presence = self._cached_presence
        # ... rest unchanged ...
```

(Apply the change to the existing forward method by adding the two-line check at the top, NOT by rewriting the whole method.)

Also, since SwinUNETR calls Linear with positional argument only, MoELoRALinear must be callable as `m(x)` returning a tensor. Fix this by defaulting to a single-tensor return path when `_legacy_call=True` is detected — actually simpler: detect through `_called_from_swin` attribute. Best: make `forward` return `out` directly if `organ_presence` is also `None` after fallback (because then there is nothing to log a gate for). Update `forward`:

```python
    def forward(self, x: torch.Tensor, organ_presence: Optional[torch.Tensor] = None):
        if organ_presence is None and hasattr(self, "_cached_presence"):
            organ_presence = self._cached_presence
        # ... existing body produces `out` and `gate` ...
        # Cache gate for later loss collection
        self._last_gate = gate
        return out  # Linear-compatible single-tensor return
```

Move the `(out, gate)` tuple return into a separate method `forward_with_gate(...)` that the test code uses. Update `tests/test_organmoe_module.py` callers accordingly:

```python
    out, gate = moe.forward_with_gate(x, organ_presence=organ_presence)
```

- [ ] **Step 4: Update `models/organmoe_3d.py` to expose both call signatures**

Refactor `MoELoRALinear`:

```python
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, gate = self._compute(x, organ_presence=getattr(self, "_cached_presence", None))
        self._last_gate = gate
        return out

    def forward_with_gate(self, x: torch.Tensor, organ_presence: Optional[torch.Tensor] = None,
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
        out, gate = self._compute(x, organ_presence=organ_presence)
        self._last_gate = gate
        return out, gate

    def _compute(self, x: torch.Tensor, organ_presence: Optional[torch.Tensor]):
        # (the body that was previously in forward; returns (out, gate))
        ...
```

Update tests in `tests/test_organmoe_module.py` to call `forward_with_gate` for the gate-shape tests, and `forward` for the drop-in shape test.

- [ ] **Step 5: Re-run the OrganMoE module tests to verify the refactor**

```bash
PYTHONPATH=. C:/Users/Raywa/AppData/Local/Programs/Python/Python312/python.exe -m pytest tests/test_organmoe_module.py -v
```

Expected: all 6 tests still pass after refactor.

- [ ] **Step 6: Add a SwinUNETRProposer + OrganMoE smoke test**

Append to `tests/test_organmoe_module.py`:

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU only")
def test_swin_proposer_with_organmoe_forward():
    """End-to-end forward smoke: SwinUNETRProposer with use_organmoe=True
    on a tiny patch should produce (B, n_organs, D, H, W) and not NaN."""
    from models.swin_unetr_3d import SwinUNETRProposer
    proposer = SwinUNETRProposer(
        n_organs=15, patch_size=32, feature_size=24, use_v2=True,
        pretrained_weights=None, lora_rank=0, use_organmoe=True,
        moe_n_experts=4, moe_rank=4, moe_alpha=4.0,
    ).cuda()
    x = torch.randn(1, 1, 32, 32, 32).cuda()
    out = proposer(x)
    assert out["logits"].shape == (1, 15, 32, 32, 32)
    assert torch.isfinite(out["logits"]).all()
```

- [ ] **Step 7: Run the GPU smoke (if a GPU is available; otherwise skip)**

```bash
PYTHONPATH=. C:/Users/Raywa/AppData/Local/Programs/Python/Python312/python.exe -m pytest tests/test_organmoe_module.py::test_swin_proposer_with_organmoe_forward -v
```

Expected: PASS or SKIP (no GPU). If FAIL with shape mismatch, fix the inject helper to skip Linears whose name matches but whose `in_features != out_features` weight assumption is violated.

- [ ] **Step 8: Commit**

```bash
git add models/organmoe_3d.py models/swin_unetr_3d.py tests/test_organmoe_module.py
git commit -m "feat(organmoe): wire MoE+presence into SwinUNETRProposer"
```

---

## Task 8: Phase I config files

**Files:**
- Create: `configs/organmoe_phase_i_pretrain.yaml`
- Create: `configs/organmoe_phase_i_ft.yaml`

- [ ] **Step 1: Pretrain config**

Create `configs/organmoe_phase_i_pretrain.yaml`:

```yaml
# OrganMoE-3D Phase I — TotalSegmentator pretraining.
# 10 ep VoCo-L on TotalSeg → save ckpt to use as warmstart for AMOS22 FT.
# OrganMoE adapters NOT enabled here (pretrain just adapts the decoder to
# the 13-shared-organ subset; OrganMoE turns on only at AMOS22 fine-tune).

experiment:
  name: "voco_totalseg_pretrain"
  seed: 42
  output_dir: "checkpoints/voco_totalseg_pretrain"
  log_dir: "logs/voco_totalseg_pretrain"

model:
  architecture: "voluformer_v9"
  n_organs: 15
  img_size: 320
  embed_dim: 256
  proposer:
    patch_size: 96
    feature_size: 96
    use_v2: true
    pretrained_weights: "checkpoints/pretrained/voco/VoComni_L.pt"
    deep_supervision: false
    use_organmoe: false           # only enabled at AMOS22 FT
    lora_rank: 32                 # standard LoRA for pretrain
  skip_channels: 128
  skip_fine_channels: 64
  encoder: { pretrained: true, lora_rank: 32 }
  pfesa: {}
  ode: { n_organs: 15, organ_emb_dim: 32, ode_hidden: 128, n_freqs: 6, substeps: 4 }
  decoder: { n_organs: 15, transformer_depth: 6, transformer_mlp_dim: 3072 }
  graph: { n_organs: 15 }
  dynamic_pfesa: { enabled: false }
  boundary_ddpm: { enabled: false }

data:
  dataset: "totalsegmentator"
  data_root: "D:/data/totalsegmentator"
  img_size: 320
  depth: 8
  volume_patch: 96
  hu_clip: [-200, 250]
  slabs_per_volume: 4
  copy_paste_prob: 0.0

training:
  stage: 1
  epochs: 10
  batch_size: 1
  num_workers: 2
  optimizer: { name: "adamw", lr: 3.0e-5, weight_decay: 0.05 }
  scheduler: { name: "cosine", warmup_epochs: 1 }
  amp: true
  amp_dtype: "bf16"
  grad_clip: 1.0
  loss_small_organ: true        # use SmallOrganLoss as base seg loss
  loss_small_organ_cfg: { dice_weight: 1.0, tversky_weight: 1.0, focal_weight: 0.5,
                          tversky_alpha: 0.3, tversky_beta: 0.7, focal_gamma: 1.5 }
  deep_sup_weight_start: 0.0
  deep_sup_weight_end: 0.0

loss:
  dice_weight: 1.0
  tversky_weight: 1.0
  xsc_weight: 0.0
  flow_weight: 0.0
  deepsup_weight: 0.0

eval:
  every_n_epochs: 1
  patch_size: 96
  stride: 48
  tta: false                    # skip during pretrain to save time
  connected_component: false
  report_per_organ: true

safety:
  abort_if_val_drops: 0.05      # looser since cross-dataset
  max_peak_vram_gb: 23
  warmstart_missing_key_frac: 0.05
```

- [ ] **Step 2: Fine-tune config**

Create `configs/organmoe_phase_i_ft.yaml`:

```yaml
# OrganMoE-3D Phase I — AMOS22 fine-tune with OrganMoE adapters ON.
# Resume from voco_totalseg_pretrain best ckpt; train with BalancedBatchSampler
# + OrganMoELoss for 10 ep on AMOS22 CT.

experiment:
  name: "organmoe_phase_i_ft"
  seed: 42
  output_dir: "checkpoints/organmoe_phase_i"
  log_dir: "logs/organmoe_phase_i"

model:
  architecture: "voluformer_v9"
  n_organs: 15
  img_size: 320
  embed_dim: 256
  proposer:
    patch_size: 96
    feature_size: 96
    use_v2: true
    pretrained_weights: null    # warmstart loaded via CLI --proposer_ckpt
    deep_supervision: false
    use_organmoe: true          # enable MoE LoRA experts
    moe_n_experts: 8
    moe_rank: 16
    moe_alpha: 16.0
    moe_top_k: 2
    lora_rank: 0                # mutually exclusive with use_organmoe
  skip_channels: 128
  skip_fine_channels: 64
  encoder: { pretrained: true, lora_rank: 0 }
  pfesa: {}
  ode: { n_organs: 15, organ_emb_dim: 32, ode_hidden: 128, n_freqs: 6, substeps: 4 }
  decoder: { n_organs: 15, transformer_depth: 6, transformer_mlp_dim: 3072 }
  graph: { n_organs: 15 }
  dynamic_pfesa: { enabled: false }
  boundary_ddpm: { enabled: false }

data:
  dataset: "amos22_v9"
  data_root: "C:/Users/Raywa/Desktop/LiteSAM3D/data/amos22"
  img_size: 320
  depth: 8
  volume_patch: 96
  hu_clip: [-200, 250]
  pos_frac: 0.5
  neg_frac: 0.3
  mix_frac: 0.2
  slabs_per_volume: 4
  copy_paste_prob: 0.0
  balanced_batch_sampler: true
  rare_organs: [4, 7, 10, 11, 12, 13, 14, 15]

training:
  stage: 1
  epochs: 10
  batch_size: 2                 # batch 2 enabled by sampler
  num_workers: 2
  optimizer: { name: "adamw", lr: 3.0e-5, weight_decay: 0.05 }
  scheduler: { name: "cosine", warmup_epochs: 1 }
  amp: true
  amp_dtype: "bf16"
  grad_clip: 1.0
  loss_organmoe: true
  loss_organmoe_cfg:
    presence_weight: 0.1
    balance_weight: 0.01
    use_presence_weighted_dice: true
    small_organ_kwargs:
      dice_weight: 1.0
      tversky_weight: 1.0
      focal_weight: 0.5
      tversky_alpha: 0.3
      tversky_beta: 0.7
      focal_gamma: 1.5
  deep_sup_weight_start: 0.0
  deep_sup_weight_end: 0.0

loss:
  dice_weight: 1.0
  tversky_weight: 1.0
  xsc_weight: 0.0
  flow_weight: 0.0
  deepsup_weight: 0.0

eval:
  every_n_epochs: 1
  patch_size: 96
  stride: 48
  tta: true
  connected_component: true
  report_per_organ: true

safety:
  abort_if_val_drops: 0.02
  max_peak_vram_gb: 23
  warmstart_missing_key_frac: 0.05
```

- [ ] **Step 3: Commit**

```bash
git add configs/organmoe_phase_i_pretrain.yaml configs/organmoe_phase_i_ft.yaml
git commit -m "feat(organmoe): phase I configs (TotalSeg pretrain + AMOS22 FT)"
```

---

## Task 9: TotalSeg pretraining script

**Files:**
- Create: `training/pretrain_totalseg.py`

**Purpose:** Reuse the existing `training/train_v9.py` Stage-1 path with the TotalSegmentator dataset substituted in. Keep it minimal — no novel mechanism, just transfer the VoCo-L decoder to the 13-organ subset.

- [ ] **Step 1: Read `training/train_v9.py` to understand the Stage 1 entrypoint signature**

```bash
PYTHONPATH=. C:/Users/Raywa/AppData/Local/Programs/Python/Python312/python.exe -c "
import importlib, inspect
m = importlib.import_module('training.train_v9')
fns = [n for n in dir(m) if not n.startswith('_')]
print(fns[:30])
"
```

Capture the function names. Look for `train_stage1` / `main` / similar.

- [ ] **Step 2: Write the pretrain script**

Create `training/pretrain_totalseg.py`:

```python
"""TotalSegmentator pretraining for OrganMoE-3D Phase I.

Wraps train_v9.train_stage1 (or equivalent) with TotalSegmentatorDataset
substituted for AMOS22V9Dataset. Saves to checkpoints/voco_totalseg_pretrain/.

Usage:
    python training/pretrain_totalseg.py --cfg configs/organmoe_phase_i_pretrain.yaml
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

# Ensure repo on path even when invoked oddly.
REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch
from omegaconf import OmegaConf
from datasets.totalsegmentator import TotalSegmentatorDataset
from training.train_v9 import train_stage1     # reuse existing trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training.epochs from CLI.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
    if args.epochs is not None:
        cfg.training.epochs = int(args.epochs)
    if cfg.data.dataset != "totalsegmentator":
        raise ValueError(f"This script requires data.dataset=totalsegmentator, got {cfg.data.dataset}")

    train_ds = TotalSegmentatorDataset(
        data_root=cfg.data.data_root,
        split="train",
        volume_patch=int(cfg.data.volume_patch),
        hu_clip=tuple(cfg.data.hu_clip),
        slabs_per_volume=int(cfg.data.slabs_per_volume),
        seed=int(cfg.experiment.seed),
    )
    val_ds = TotalSegmentatorDataset(
        data_root=cfg.data.data_root,
        split="val",
        volume_patch=int(cfg.data.volume_patch),
        hu_clip=tuple(cfg.data.hu_clip),
        slabs_per_volume=int(cfg.data.slabs_per_volume),
        seed=int(cfg.experiment.seed) + 1,
    )
    print(f"[totalseg-pretrain] train={len(train_ds.volume_ids)} vols  "
          f"val={len(val_ds.volume_ids)} vols")

    train_stage1(cfg, train_ds=train_ds, val_ds=val_ds)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: If `train_v9.train_stage1` does not accept `train_ds=`/`val_ds=` kwargs, refactor minimally**

Read `training/train_v9.py`, locate `train_stage1`. If it currently constructs the dataset internally, add optional `train_ds=None, val_ds=None` parameters that override the internal construction when provided. Edit:

```python
def train_stage1(cfg, train_ds=None, val_ds=None):
    if train_ds is None:
        train_ds = AMOS22V9Dataset(...)   # existing code
    if val_ds is None:
        val_ds = AMOS22V9Dataset(...)     # existing code
    # ... rest unchanged ...
```

- [ ] **Step 4: Smoke (1-vol fast path)**

Verify the entrypoint imports and the TotalSeg dataset can be opened by running:

```bash
PYTHONPATH=. C:/Users/Raywa/AppData/Local/Programs/Python/Python312/python.exe -c "
from training.pretrain_totalseg import main
import sys
print('import OK')
"
```

Expected: `import OK`. If TotalSeg manifest is not yet ready (download still running), defer the actual training launch until Task 11.

- [ ] **Step 5: Commit**

```bash
git add training/pretrain_totalseg.py training/train_v9.py
git commit -m "feat(organmoe): pretrain_totalseg.py + train_v9 dataset injection"
```

---

## Task 10: AMOS22 fine-tune script (with OrganMoE)

**Files:**
- Create: `training/train_organmoe_amos22.py`

- [ ] **Step 1: Write the script**

Create `training/train_organmoe_amos22.py`:

```python
"""AMOS22 fine-tune for OrganMoE-3D Phase I.

Wraps train_v9.train_stage1 with:
  - AMOS22V9Dataset (existing)
  - BalancedBatchSampler (rare-organ coverage)
  - OrganMoELoss (presence-weighted seg + presence BCE + MoE balance)
  - SwinUNETRProposer with use_organmoe=True

Resumes from a TotalSeg-pretrained checkpoint passed via --proposer_ckpt.

Usage:
    python training/train_organmoe_amos22.py \
        --cfg configs/organmoe_phase_i_ft.yaml \
        --proposer_ckpt checkpoints/voco_totalseg_pretrain/last.pt
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch
from omegaconf import OmegaConf
from datasets.amos22_v9 import AMOS22V9Dataset
from datasets.balanced_sampler import BalancedBatchSampler
from training.losses_organmoe import OrganMoELoss
from training.train_v9 import train_stage1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, type=Path)
    parser.add_argument("--proposer_ckpt", required=True, type=Path,
                        help="TotalSeg-pretrained checkpoint to warmstart from.")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
    if args.epochs is not None:
        cfg.training.epochs = int(args.epochs)

    train_ds = AMOS22V9Dataset(
        data_root=cfg.data.data_root, split="train",
        img_size=int(cfg.data.img_size), depth=int(cfg.data.depth),
        volume_patch=int(cfg.data.volume_patch),
        hu_clip=tuple(cfg.data.hu_clip),
        pos_frac=float(cfg.data.pos_frac),
        neg_frac=float(cfg.data.neg_frac),
        mix_frac=float(cfg.data.mix_frac),
        slabs_per_volume=int(cfg.data.slabs_per_volume),
        seed=int(cfg.experiment.seed),
    )
    val_ds = AMOS22V9Dataset(
        data_root=cfg.data.data_root, split="val",
        img_size=int(cfg.data.img_size), depth=int(cfg.data.depth),
        volume_patch=int(cfg.data.volume_patch),
        hu_clip=tuple(cfg.data.hu_clip),
        pos_frac=float(cfg.data.pos_frac),
        neg_frac=float(cfg.data.neg_frac),
        mix_frac=float(cfg.data.mix_frac),
        slabs_per_volume=int(cfg.data.slabs_per_volume),
        seed=int(cfg.experiment.seed) + 1,
    )

    sampler = None
    if bool(cfg.data.get("balanced_batch_sampler", False)):
        sampler = BalancedBatchSampler(
            dataset=train_ds,
            batch_size=int(cfg.training.batch_size),
            rare_organs=list(cfg.data.rare_organs),
            shuffle=True,
            seed=int(cfg.experiment.seed),
        )

    loss_fn = None
    if bool(cfg.training.get("loss_organmoe", False)):
        kw = OmegaConf.to_container(cfg.training.loss_organmoe_cfg, resolve=True)
        loss_fn = OrganMoELoss(n_classes=int(cfg.model.n_organs) + 1, **kw)

    train_stage1(
        cfg, train_ds=train_ds, val_ds=val_ds,
        train_sampler=sampler, loss_fn=loss_fn,
        warmstart_proposer_ckpt=str(args.proposer_ckpt),
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Extend `train_v9.train_stage1` to accept the new kwargs**

Edit `training/train_v9.py` `train_stage1` signature:

```python
def train_stage1(cfg, train_ds=None, val_ds=None, train_sampler=None,
                 loss_fn=None, warmstart_proposer_ckpt=None):
```

Inside `train_stage1`:
- If `train_sampler is not None`, pass `batch_sampler=train_sampler` to the train DataLoader (and drop `batch_size`/`shuffle`).
- If `loss_fn is not None`, use it instead of the default loss. After each `model.proposer(x)` forward, retrieve auxiliary signals via the helpers added in Task 7:
  ```python
  presence_logits = model.proposer.last_presence_logits          # (B, 15) or None
  gate_weights = model.proposer.collect_last_gates()              # (B, K, 1) or None
  loss_dict = loss_fn(seg_logits=logits, seg_target=target,
                      presence_logits=presence_logits,
                      gate_weights=gate_weights)
  loss = loss_dict["total"]
  ```
  When `presence_logits is None` (OrganMoE off), `OrganMoELoss` already short-circuits the BCE term to zero — so this branch works for plain LoRA too.
- If `warmstart_proposer_ckpt is not None`, load the state dict with `strict=False` after the model is constructed but before optimizer construction.

(Keep edits minimal — only add the new branches.)

- [ ] **Step 3: Smoke import**

```bash
PYTHONPATH=. C:/Users/Raywa/AppData/Local/Programs/Python/Python312/python.exe -c "
from training.train_organmoe_amos22 import main
print('import OK')
"
```

Expected: `import OK`.

- [ ] **Step 4: Commit**

```bash
git add training/train_organmoe_amos22.py training/train_v9.py
git commit -m "feat(organmoe): AMOS22 fine-tune script with OrganMoE+balanced sampler"
```

---

## Task 11: Verify TotalSeg download finished, then launch pretrain

- [ ] **Step 1: Verify download completion**

```bash
tail -n 5 C:/Users/Raywa/Desktop/VoluFormer3D_V4/logs/organmoe_phase_i/totalseg_download.log
ls C:/Users/Raywa/Desktop/VoluFormer3D_V4/D:/data/totalsegmentator/totalsegmentator_manifest.json 2>&1 || ls D:/data/totalsegmentator/totalsegmentator_manifest.json 2>&1
```

Expected: log shows `[totalseg-dl] manifest: 1228 volumes` (approx). Manifest file exists.

- [ ] **Step 2: Smoke the dataset on the now-downloaded data**

```bash
PYTHONPATH=. C:/Users/Raywa/AppData/Local/Programs/Python/Python312/python.exe -c "
from datasets.totalsegmentator import TotalSegmentatorDataset
ds = TotalSegmentatorDataset(data_root='D:/data/totalsegmentator', split='train', volume_patch=96)
print(f'train_vols={len(ds.volume_ids)}')
b = ds[0]
print(f'patch={b[\"volume_patch\"].shape} label_unique={b[\"volume_label\"].unique().tolist()}')
"
```

Expected: ~1166 train volumes, patch shape `[1, 96, 96, 96]`, label values subset of `{0, 1..13}`.

- [ ] **Step 3: 1-volume training smoke (10 steps, no checkpoint save)**

```bash
PYTHONPATH=. C:/Users/Raywa/AppData/Local/Programs/Python/Python312/python.exe \
  training/pretrain_totalseg.py \
  --cfg configs/organmoe_phase_i_pretrain.yaml --epochs 1 \
  > logs/organmoe_phase_i/totalseg_pretrain_smoke.log 2>&1
```

Expected: log shows training loss decreasing across the 10 logged steps. No NaN. VRAM <23 GB.

- [ ] **Step 4: Launch full 10-ep pretrain in background**

```bash
nohup C:/Users/Raywa/AppData/Local/Programs/Python/Python312/python.exe \
  training/pretrain_totalseg.py --cfg configs/organmoe_phase_i_pretrain.yaml \
  > logs/organmoe_phase_i/totalseg_pretrain.log 2>&1 &
echo "[totalseg-pretrain] PID $!"
```

Expected: PID printed. Tail the log every ~30 min; expect ~6 hr total runtime.

- [ ] **Step 5: Verify each epoch saves a checkpoint**

```bash
ls -la C:/Users/Raywa/Desktop/VoluFormer3D_V4/checkpoints/voco_totalseg_pretrain/
```

Expected after epoch 1: `epoch_001.pt`. Watch `metrics.json` for monotonic loss decrease.

---

## Task 12: AMOS22 fine-tune with OrganMoE-3D

- [ ] **Step 1: Wait for TotalSeg pretrain to finish**

Check `logs/organmoe_phase_i/totalseg_pretrain.log` for completion line `train_stage1 done`. Identify best epoch (lowest val loss).

- [ ] **Step 2: 1-ep AMOS22 FT smoke**

```bash
PYTHONPATH=. C:/Users/Raywa/AppData/Local/Programs/Python/Python312/python.exe \
  training/train_organmoe_amos22.py \
  --cfg configs/organmoe_phase_i_ft.yaml \
  --proposer_ckpt checkpoints/voco_totalseg_pretrain/last.pt \
  --epochs 1 \
  > logs/organmoe_phase_i/ft_smoke.log 2>&1
```

Expected: log shows decreasing seg loss; presence_bce loss <0.5 by end of ep 1; balance loss roughly 1.0; no NaN; VRAM <23 GB.

- [ ] **Step 3: Verify per-batch presence target derivation works on real data**

Run a quick check after the smoke epoch:

```bash
PYTHONPATH=. C:/Users/Raywa/AppData/Local/Programs/Python/Python312/python.exe -c "
import json
m = json.load(open('checkpoints/organmoe_phase_i/metrics.json'))
print(f'last_loss={m[-1]}')
"
```

Expected: dict containing `seg`, `presence_bce`, `balance`, `total` keys with finite values.

- [ ] **Step 4: Launch full 10-ep AMOS22 FT in background**

```bash
nohup C:/Users/Raywa/AppData/Local/Programs/Python/Python312/python.exe \
  training/train_organmoe_amos22.py \
  --cfg configs/organmoe_phase_i_ft.yaml \
  --proposer_ckpt checkpoints/voco_totalseg_pretrain/last.pt \
  > logs/organmoe_phase_i/ft.log 2>&1 &
echo "[organmoe-ft] PID $!"
```

Expected: PID printed. Total runtime ~10-12 hr (10 ep × ~70 min/ep).

- [ ] **Step 5: Monitor every 1 hr per session-cadence rule**

After every hour, check the latest val_dice in `metrics.json` and append a one-line update to `MASTER_PROJECT_LOG.md`. Apply the 0.02 STOP-RULE: if val_dice at any ep is more than 0.02 below the best-of-W1-base (0.7967), stop and diagnose (`scripts/diagnose_ep5_fail.py` template).

- [ ] **Step 6: Verify all 10 epochs save**

```bash
ls -la C:/Users/Raywa/Desktop/VoluFormer3D_V4/checkpoints/organmoe_phase_i/
```

Expected: `epoch_001.pt` … `epoch_010.pt` plus `last.pt` and `metrics.json`.

---

## Task 13: GATE I — sliding-window 3D evaluation

**Files:**
- Create: `scripts/eval_organmoe_3d.py`

**Purpose:** Reuse `scripts/eval_v9_proposer_preproc.py` infrastructure for the OrganMoE-3D ckpt. The existing script already does sliding-window 96³ stride-48 + Gaussian blend + 4-way TTA + largest-CC and per-organ JSON dump.

- [ ] **Step 1: Write the wrapper script**

Create `scripts/eval_organmoe_3d.py`:

```python
"""Sliding-window 3D eval wrapper for OrganMoE-3D Phase I.

Re-uses scripts.eval_v9_proposer_preproc with use_organmoe=True wired into the
proposer construction. Writes the per-organ Dice/HD95/NSD JSON to
reports/organmoe_phase_i/<ckpt_name>_3d_eval.json.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.eval_v9_proposer_preproc import main as _eval_main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--cfg", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n-vols", type=int, default=30,
                        help="Number of val volumes to evaluate (default 30 = full).")
    args = parser.parse_args()

    # Forward to existing eval entrypoint via sys.argv munging.
    sys.argv = [
        "eval_v9_proposer_preproc.py",
        "--ckpt", str(args.ckpt),
        "--cfg", str(args.cfg),
        "--output", str(args.output),
        "--n-vols", str(args.n_vols),
    ]
    _eval_main()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify `scripts/eval_v9_proposer_preproc.py` accepts `--cfg` and respects `use_organmoe`**

```bash
PYTHONPATH=. C:/Users/Raywa/AppData/Local/Programs/Python/Python312/python.exe \
  scripts/eval_v9_proposer_preproc.py --help
```

Expected: usage shows `--ckpt`, `--cfg`, `--output`, `--n-vols`. If `use_organmoe` is not wired, edit the eval script to read `cfg.model.proposer.use_organmoe` when constructing the proposer (mirror Task 7 changes).

- [ ] **Step 3: Run a 3-vol smoke**

```bash
PYTHONPATH=. C:/Users/Raywa/AppData/Local/Programs/Python/Python312/python.exe \
  scripts/eval_organmoe_3d.py \
  --ckpt checkpoints/organmoe_phase_i/last.pt \
  --cfg configs/organmoe_phase_i_ft.yaml \
  --output reports/organmoe_phase_i/smoke_3vol.json \
  --n-vols 3
```

Expected: prints per-vol Dice, exits cleanly. No NaN.

- [ ] **Step 4: Run the full 30-vol GATE I eval**

```bash
nohup C:/Users/Raywa/AppData/Local/Programs/Python/Python312/python.exe \
  scripts/eval_organmoe_3d.py \
  --ckpt checkpoints/organmoe_phase_i/last.pt \
  --cfg configs/organmoe_phase_i_ft.yaml \
  --output reports/organmoe_phase_i/gate_i_3d_eval.json \
  --n-vols 30 \
  > logs/organmoe_phase_i/gate_i_eval.log 2>&1 &
echo "[gate-i-eval] PID $!"
```

Expected: ~5-7 hr runtime (250s/vol × 30). Monitor log.

- [ ] **Step 5: Read GATE I result and apply pass/fail decision**

```bash
PYTHONPATH=. C:/Users/Raywa/AppData/Local/Programs/Python/Python312/python.exe -c "
import json
r = json.load(open('reports/organmoe_phase_i/gate_i_3d_eval.json'))
mean = r.get('mean_dice', None)
prostate = r['per_organ'].get('prostate_uterus', {}).get('dice', None)
print(f'mean_dice={mean:.4f}  prostate={prostate:.4f}')
print(f'GATE I pass: mean>=0.85 ({mean>=0.85}) AND prostate>=0.30 ({prostate>=0.30})')
"
```

Decision tree (per spec §6 Phase I fallbacks table):

| Outcome | Action |
|---|---|
| **PASS** (mean ≥0.85 AND prostate ≥0.30) | Append GATE I PASS to MASTER_PROJECT_LOG.md → invoke writing-plans skill on Phase II spec section |
| **FAIL: mean <0.85, MoE collapsed** (router degenerate) | Check expert utilization → diagnose (scripts/diagnose_ep5_fail.py template); fallback: switch to plain LoRA + class fix per spec |
| **FAIL: mean <0.85, prostate still 0.0** | Add forced presence injection (oracle-presence training) per spec |
| **FAIL: mean <0.85, all organs degraded** | Check TotalSeg pretrain quality → fallback: skip TotalSeg pretrain, retrain from VoCo-L base + balanced sampler only |
| **FAIL: prostate ≥0.30 but mean <0.85** | Lower MoE auxiliary loss weight, reduce K to 4 experts; restart 5-ep mini-FT |
| **HARD EXIT** (mean <0.85 AND prostate <0.30 after 1 fallback iteration) | Freeze ckpts; pivot to workshop paper write-up using best-of-Phase-I |

- [ ] **Step 6: Append Phase I result to MASTER_PROJECT_LOG.md**

Add an entry summarizing: best epoch, mean Dice, per-organ breakdown, GATE I outcome, and which Phase II decision branch will be taken.

- [ ] **Step 7: Final commit**

```bash
git add scripts/eval_organmoe_3d.py reports/organmoe_phase_i/gate_i_3d_eval.json MASTER_PROJECT_LOG.md
git commit -m "feat(organmoe): GATE I sliding-window 3D eval + result log"
```

---

## Phase I done. Next step: Phase II plan.

After GATE I passes (or after the fallback path stabilizes), invoke the writing-plans skill again on the Phase II section of the spec to produce `docs/superpowers/plans/2026-05-02-organmoe-3d-phase-ii.md`. Reasons for incremental planning:

1. Phase II's backbone shortlist may shrink if a backbone fails to train (per spec fallback).
2. Phase III's 2nd novelty pick is data-dependent on Phase I-II per-organ outcomes.
3. Splitting keeps each plan small enough to execute end-to-end without replan.

**Sacred ckpts after Phase I:**
- All previous sacred ckpts (per spec §8) untouched.
- New: `checkpoints/voco_totalseg_pretrain/last.pt`, `checkpoints/organmoe_phase_i/last.pt` are the new Phase I anchors and should NOT be overwritten by Phase II.
