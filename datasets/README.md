# `datasets/` — Dataset Loaders and Transforms

PyTorch `Dataset` implementations for AMOS22 and TotalSegmentator, plus
the shared transform stack, balanced sampler, and copy-paste augmentation
bank.

## Modules

| File | Purpose |
|------|---------|
| `__init__.py` | Package marker; re-exports the dataset classes. |
| `base_dataset.py` | Shared `Base3DDataset` superclass — handles split parsing, caching, transform composition, presence-cache loading. |
| `amos22.py` | Single-organ AMOS22 loader (liver-only headline). Reads `imagesTr/`/`labelsTr/` from the official AMOS22 release; produces 8-slice slabs at 256 px. |
| `amos22_multiorgan.py` | Multi-organ extension used by Path A1 (4 big solid organs). Maps 15-class AMOS22 labels to the K=4 reduced set. |
| `amos22_v9.py` | V9-era loader (legacy proposer–refiner cascade). Kept for backwards compatibility with V9 checkpoints. |
| `totalsegmentator.py` | Cross-dataset eval loader. Applies LAS reorientation, no spacing resample (V2 native), and maps TotalSeg labels to AMOS22 organ IDs via `totalseg_label_map.py`. |
| `totalseg_label_map.py` | Class-ID conversion dict between TotalSegmentator's per-organ filename schema and the AMOS22 15-class index. |
| `balanced_sampler.py` | Class-balanced batch sampler. Uses a precomputed `<dataset>_presence.json` cache (see `scripts/build_totalseg_presence.py`). |
| `copy_paste_bank.py` | Copy–paste augmentation bank (small-organ boost). Currently disabled for the thesis V2 path; see `memory/feedback_architecture_decisions.md`. |
| `transforms.py` | Shared transform composition: intensity clipping, normalisation, random affine/elastic, optional flip. |

## Datasets on disk

The loaders expect raw data at:

- **AMOS22** — `C:\Users\Raywa\Desktop\LiteSAM3D\data\amos22\` (`imagesTr`, `labelsTr`, `imagesVa`, `labelsVa`, `imagesTs`, `labelsTs`, plus `dataset.json`).
- **TotalSegmentator** — `C:\Users\Raywa\Desktop\LiteSAM3D\data\totalseg\` (per-volume folders with one NIfTI per organ).
- Path overrides go in the per-config `data.data_root` field under `configs/`.

## Cross-cutting rules

- **Presence cache is mandatory for class-balanced sampling on >300-vol datasets** — see `memory/feedback_presence_cache_for_balanced_sampler.md`.
- **TotalSeg eval pipeline requires LAS reorientation, no resampling, and largest-CC post-processing** — see `memory/feedback_totalseg_eval_lessons.md`.
- **CutMix is OFF for Auto-ODE-SAM V2** — copy-paste bank is loadable but not wired into the V2 training loop.
