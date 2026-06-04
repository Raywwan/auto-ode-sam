# Auto-ODE-SAM

**Continuous-Depth Cross-Slice Dynamics for SAM-Based Abdominal CT Segmentation: A Bidirectional Neural-ODE Approach**

> A SAM-style segmentation framework (TinyViT image encoder + SAM-v1 mask decoder) that models continuous anatomical dynamics across CT slices through a bidirectional Neural Ordinary Differential Equation applied to mid-resolution feature maps.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![Dataset: AMOS22](https://img.shields.io/badge/dataset-AMOS22-green.svg)](https://zenodo.org/record/7262581)

---

## Table of Contents

1. [Headline Results](#headline-results)
2. [What's in This Repository](#whats-in-this-repository)
3. [Quick Start](#quick-start)
4. [Reproduction Tiers](#reproduction-tiers)
   - [Tier 1 — Re-derive Every Cited Number on CPU](#tier-1--re-derive-every-cited-number-on-cpu)
   - [Tier 2 — Re-run Inference End-to-End on GPU](#tier-2--re-run-inference-end-to-end-on-gpu)
   - [Tier 3 — Retrain a Seed From Scratch](#tier-3--retrain-a-seed-from-scratch)
5. [Pre-trained Models](#pre-trained-models)
6. [Datasets](#datasets)
7. [Method at a Glance](#method-at-a-glance)
8. [Repository Layout](#repository-layout)
9. [Determinism and Provenance](#determinism-and-provenance)
10. [Citation](#citation)
11. [License](#license)
12. [Acknowledgments](#acknowledgments)

---

## Headline Results

All numbers evaluated on the AMOS22 validation split ($N{=}100$ volumes, liver).
Every figure can be re-derived in under 10 seconds on CPU from the shipped
per-volume CSVs — see [Tier 1](#tier-1--re-derive-every-cited-number-on-cpu).

| Claim | Value | Source |
|---|---|---|
| Single-seed DSC (seed = 42) | **0.9351 ± 0.016** | `thesis/results/amos22_liver_indomain/per_volume_metrics.csv` |
| 4-seed ensemble DSC at τ = 0.5 (**deployed**) | **0.9398 ± 0.016** | `thesis/results/ensemble_4seed_tau050/` |
| 4-seed ensemble DSC at τ = 0.40 (oracle ceiling) | 0.9402 | `thesis/results/ensemble_4seed_tau040/` |
| HD95 (4-seed, τ = 0.5) | 5.44 mm | same |
| NSD@1mm (4-seed, τ = 0.5) | 0.96775 | same |
| Inference latency (single ckpt, RTX 4090) | 2.75 ± 0.33 s/vol | `thesis/tab:inference_time` |
| Parameters | 18.0 M | architecture surgery — `models/` |
| Cross-dataset DSC on TotalSegmentator (N = 92 robust) | **0.9373** | `thesis/results/totalsegmentator_liver/` |
| Paired Wilcoxon vs. nnUNetv2 (HD95) | W = 490, p = 2.6 × 10⁻¹² | `thesis/scripts/wilcoxon_hd95_vs_nnunet.py` |

**Positioning.** Auto-ODE-SAM exceeds directly comparable bounding-box-prompted
SAM baselines (MA-SAM 0.917 DSC, MCP-MedSAM ~0.890) at a fraction of the
parameter count, and runs **~15× faster** than the in-house nnUNetv2 `3d_fullres`
baseline on the same RTX 4090. nnUNetv2 wins per-case on the three primary
metrics at the prompt-driven operating point, but Auto-ODE-SAM has lower mean
HD95 (5.52 vs 5.99 mm) and a far thinner outlier tail (max HD95 30.1 vs
133.9 mm; zero volumes above 50 mm vs nnUNetv2's three).

---

## What's in This Repository

```
Auto-ODE-SAM/
├── README.md              ← you are here
├── LICENSE                ← MIT (code) / CC-BY-4.0 (thesis PDF)
├── .gitignore             ← excludes weights, datasets, build artifacts
├── requirements.txt       ← Python dependencies (pip)
├── train.py               ← single training entry point
├── evaluate_3d.py         ← single 3D evaluation entry point
├── run_test_runs.sh       ← smoke harness
│
├── configs/               ← every YAML config for every experiment
├── models/                ← architectures: Auto-ODE-SAM, PFESA, PA-CODE, ...
├── training/              ← trainer + optimisation scaffolding
├── losses/                ← BD-DoU, PM-Dice, HD, focal, ...
├── datasets/              ← AMOS22, TotalSegmentator loaders
├── inference/             ← sliding-window inference + TTA
├── evaluation/            ← DSC, HD95, NSD with per-volume spacing
├── scripts/               ← preflight gates, statistical tests, postproc
├── tests/                 ← unit + smoke (BD-DoU, model surgery, ...)
├── utils/                 ← shared helpers
│
├── checkpoints/           ← (gitignored) trained weights — see checkpoints/README.md
├── nnUNet_workspace/      ← (gitignored) reproducible nnUNetv2 baseline
├── logs/                  ← (gitignored) training logs
│
├── thesis/                ← LaTeX source + compiled PDF + per-volume CSVs
│   ├── main.tex           ← LaTeX root
│   ├── main.pdf           ← compiled thesis
│   ├── references.bib     ← bibliography
│   ├── chapters/          ← 01_introduction.tex … A_appendix.tex
│   ├── figures/           ← all figures (PDF, PNG, TikZ sources)
│   ├── results/           ← per-volume CSVs / JSONs (cited by every table)
│   ├── scripts/           ← CPU-only stat scripts (every cited number)
│   └── repro/             ← reviewer-facing reproducibility bundle (verify_stats.py)
│
├── docs/
│   ├── internal/          ← project logs (MASTER_PROJECT_LOG, RESEARCH_LOG, ...)
│   └── ...                ← design specs
│
└── legacy_v3/             ← (gitignored) frozen V3 codebase + canonical seed=42 weights
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/raywandlawar/auto-ode-sam.git
cd auto-ode-sam
```

### 2. Install dependencies (Python 3.12 + CUDA 12.x recommended)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Unix:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Verify every cited number on CPU (~5 seconds)

```bash
cd thesis/repro
python verify_stats.py
```

Exit code `0` with `ALL CHECKS PASSED` means every statistic in the thesis is
reproducible to within `1e-3` of its printed value.

---

## Reproduction Tiers

| Tier | What it verifies | Cost | Needs GPU? | Needs dataset? |
|---|---|---|---|---|
| **1. Re-derive every cited number** | All stats from per-volume CSVs | ~5 s | No | No |
| **2. Re-run inference end-to-end** | The CSVs themselves | ~5 min/seed | Yes (≥16 GB VRAM) | Yes (AMOS22) |
| **3. Retrain a seed from scratch** | The checkpoint weights | ~30 GPU-h/seed | Yes (≥24 GB VRAM) | Yes (AMOS22) |

### Tier 1 — Re-derive Every Cited Number on CPU

```bash
cd thesis/repro
pip install scipy numpy pandas
python verify_stats.py
```

The verifier consumes the shipped per-volume CSVs at `thesis/results/`
(no model, no dataset) and reconstructs every statistic cited in the body.
Verified claims include:

- V2 single-seed (seed = 42) DSC = 0.9351 on AMOS22 liver (N = 100)
- 4-seed ensemble DSC = 0.9402 at τ = 0.40 (oracle) and 0.9398 at τ = 0.5 (deployed)
- Paired Wilcoxon (ensemble vs. single seed): W = 2, p = 4.14 × 10⁻¹⁸, 99/100 wins
- Paired Wilcoxon (Auto-ODE-SAM vs. nnUNetv2 on HD95): W = 490, p = 2.61 × 10⁻¹²
- Hodges–Lehmann pseudomedians (ΔDSC −0.041, ΔHD95 +2.50 mm)
- TotalSegmentator cross-dataset robust DSC = 0.9373 (N = 92)
- Mondrian split-CP foreground coverage 0.954 / 0.898 / 0.805 at α = 0.05 / 0.10 / 0.20

Each individual statistic also has a focused script under `thesis/scripts/cpu_add_*.py`.

### Tier 2 — Re-run Inference End-to-End on GPU

**2a. Obtain the AMOS22 dataset** (Zenodo record `7262581`):

```text
https://zenodo.org/record/7262581
```

Download `amos22.zip` and unpack so the layout is:

```
<ANYWHERE>/amos22/
├── dataset.json
├── imagesTr/  imagesVa/  imagesTs/
└── labelsTr/  labelsVa/  labelsTs/
```

Then point the configs at it, either by editing `configs/amos22.yaml`
(`data.data_root:`) or by creating an OS junction at the historical path
the configs were authored against (`C:/Users/Raywa/Desktop/LiteSAM3D/data/amos22`):

```cmd
mklink /J "C:\Users\Raywa\Desktop\LiteSAM3D\data\amos22" "D:\your\path\to\amos22"
```

**2b. Download the pre-trained checkpoints** — see [Pre-trained Models](#pre-trained-models)
below. Place them under `checkpoints/` matching the structure in
`checkpoints/README.md`.

**2c. Reproduce the headline single-seed DSC (seed = 42)**:

```bash
python evaluate_3d.py \
  --checkpoint checkpoints/phase3_odesam_v2_256px_liver/phase3_odesam_v2_256px_liver_best.pt \
  --split val \
  --output thesis/results/amos22_liver_indomain_repro \
  --per_organ \
  --batch_size 4
```

Expected: per-volume CSV with N = 100 rows, mean DSC ≈ 0.9351, wall-clock
≈ 4.6 min on RTX 4090 (= 2.75 s/vol).

**2d. Reproduce the 4-seed ensemble**: run `evaluate_3d.py` four times with
the four canonical checkpoints (seeds 42 – 45), then sigmoid-average the
probability maps. The ensemble driver lives under `scripts/` (search for
`*ensemble*.py`). Apply binarisation at τ = 0.5 (deployed, headline 0.9398)
or τ = 0.40 (oracle ceiling 0.9402).

### Tier 3 — Retrain a Seed From Scratch

```bash
python train.py --config configs/phase3_odesam_v2_seed43.yaml
```

Roughly 30 GPU-hours per seed on an RTX 4090 for the 40-epoch run. The trainer
writes per-epoch checkpoints into `checkpoints/<config_name>/`, follows the
BD-DoU + Pareto-promotion discipline documented in `thesis/chapters/05_ablations.tex`,
and runs preflight gates (`scripts/preflight_*.py`) before consuming any GPU.

The four canonical configs:

- `configs/phase3_odesam_v2_seed43.yaml`
- `configs/phase3_odesam_v2_seed44.yaml`
- `configs/phase3_odesam_v2_seed45.yaml`
- For seed = 42, the V3-era launch used the predecessor's config; the
  equivalent V4 config is `configs/phase3_odesam_v2.yaml`.

PA-CODE (negative result, documented in `thesis/chapters/07_discussion.tex` § 7.4):

- `configs/path_a1_pa_code.yaml`
- `configs/path_a1_pa_code_lr1e4.yaml`
- `configs/path_a1_pa_code_gbound.yaml`

---

## Pre-trained Models

All trained weights are **excluded from this git repository** (the four canonical
seed checkpoints alone are ~6 GB; the full ablation + comparator set is 22 GB).
They are published as a separate release. See
[`checkpoints/README.md`](checkpoints/README.md) for download links, SHA-256
hashes, and the expected directory layout.

The four checkpoints needed to reproduce the headline ensemble:

| Seed | Filename | SHA-256 (first 16 chars) |
|---|---|---|
| 42 (V3-era) | `phase3_odesam_v2_256px_liver_best.pt` | `dfec2350355e8bec…` |
| 43 | `phase3_odesam_v2_seed43_40ep_best.pt` | `5b54bc4e4c13a1bb…` |
| 44 | `phase3_odesam_v2_seed44_40ep_best.pt` | `f686287d35fdeef1…` |
| 45 | `phase3_odesam_v2_seed45_40ep_best.pt` | `643f57f5da2c41f5…` |

Full hashes are in `checkpoints/README.md`. All weights are saved by
`torch.save` in PyTorch 2.x `state_dict` format.

---

## Datasets

This repository **does not redistribute** any imaging data. Obtain the
datasets directly from their original hosts:

| Dataset | Source | License | Used for |
|---|---|---|---|
| **AMOS22** (`amos22.zip`) | [Zenodo 7262581](https://zenodo.org/record/7262581) | CC BY 4.0 | Training, in-domain evaluation |
| **TotalSegmentator v2** | [Zenodo 6802614](https://zenodo.org/record/6802614) | CC BY-SA 4.0 | Cross-dataset evaluation |

The thesis uses only the *liver* class from both datasets. Annotation-protocol
mismatch volumes are excluded from the TotalSegmentator robust mean (N = 92);
the list of excluded IDs is in `thesis/results/totalsegmentator_liver/excluded_volumes.txt`.

---

## Method at a Glance

Auto-ODE-SAM consists of four blocks:

1. **TinyViT image encoder** — pretrained, 5.0 M parameters, mid-resolution
   features at 1/16 stride.
2. **Bidirectional Neural-ODE feature dynamics** — a continuous-depth ODE
   block applied to the cross-slice axis of the feature map, integrated with
   `torchdiffeq.odeint_adjoint` for memory-efficient training. *This is the
   novel contribution.*
3. **Bounding-box prompt encoder** — SAM-v1 prompt encoder, frozen.
4. **SAM-v1 mask decoder** — fine-tuned end-to-end.

Total parameters: **18.0 M**. The ODE block adds ~700 K parameters but
accounts for the bulk of the DSC lift over a SAM-v1 + TinyViT baseline.

Loss: a curriculum mixing **BD-DoU** (boundary-aware DoU), **PM-Dice**, and
focal cross-entropy, with Pareto-promotion gating (a new metric is only
adopted when all three of DSC / HD95 / NSD improve or stay flat).

See `thesis/chapters/03_methodology.tex` for the full architectural derivation
and `thesis/chapters/05_ablations.tex` for the loss-function and
training-discipline ablations.

---

## Repository Layout

The repo is **flat**: every code module sits directly at the root. There is
no `src/` indirection.

```
configs/      models/       training/     losses/
datasets/     inference/    evaluation/   scripts/
tests/        utils/        docs/         thesis/
```

Heavy artifacts (`checkpoints/`, `nnUNet_workspace/`, `logs/`, `legacy_v3/`)
exist locally for development but are excluded from git via `.gitignore`.
Their contents are reproducible by following the relevant tier above.

For an exhaustive directory tour with file sizes, see the **What's in This Repository**
section above, or run `tree -L 2` after cloning.

---

## Determinism and Provenance

- The shipped per-volume CSVs are **bit-stable** across reruns (no stochastic
  resampling at eval, deterministic sliding-window inference at the shipped
  config, no AMP-induced reduction-order changes for the eval path).
- Bootstrap CIs and pseudomedian computations use `random.Random(seed=0)` and
  are exactly reproducible.
- Training is **not** bit-deterministic across GPUs (cuDNN nondeterminism).
  Across-seed variance σ ≈ 0.0013 DSC was the basis for the 4-seed
  reproducibility claim in `thesis/sec:repro-n4`.
- Per-checkpoint provenance (file size, tensor count, training-config hash)
  is documented in `thesis/chapters/A_appendix.tex` § `ap:provenance`.
- Every cited number traces back to a specific CSV under `thesis/results/`
  via `thesis/scripts/reproduce_headline.py`.

---

## Citation

If you find this work useful, please cite:

```bibtex
@mastersthesis{dlawar2026autoodesam,
  title  = {Continuous-Depth Cross-Slice Dynamics for {SAM}-Based Abdominal
            {CT} Segmentation: A Bidirectional Neural-{ODE} Approach},
  author = {Dlawar, Raywan},
  year   = {2026},
  school = {[Institution]},
  type   = {Master's Thesis},
  note   = {Auto-ODE-SAM; \url{https://github.com/raywandlawar/auto-ode-sam}}
}
```

(Workshop / journal version forthcoming — citation will be updated here when
the camera-ready is accepted.)

---

## License

- **Source code** (`*.py`, `*.sh`, `configs/*.yaml`, etc.): [MIT](LICENSE).
- **Thesis document** (`thesis/main.pdf` and its TeX sources): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Trained model weights**: inherit upstream backbone licenses
  (SAM-v1 Apache 2.0, TinyViT Apache 2.0, SAM-2 Hiera-Large BSD-3-Clause,
  VoCo Apache 2.0). See `checkpoints/README.md`.
- **Datasets**: not redistributed; obtain from original hosts under their
  own licenses (AMOS22 CC BY 4.0, TotalSegmentator CC BY-SA 4.0).

---

## Acknowledgments

- **AMOS22 challenge** organisers (Ji et al., 2022) for the public
  abdominal CT benchmark.
- **TotalSegmentator** authors (Wasserthal et al., 2023) for the
  cross-dataset evaluation corpus.
- **Meta AI** for SAM-v1 and SAM-2; **Microsoft** for TinyViT; **Bowang Lab**
  for MedSAM-2; the **VoCo** team for the pretrained backbones.
- **nnU-Net** team (Isensee et al.) for the nnUNetv2 reference implementation
  used as the in-house comparator.
- Thesis supervision and feedback from the supervisor team (acknowledged in
  the thesis front matter).

---

*For any questions, the thesis itself documents every assumption in
`thesis/chapters/09_data_code_availability.tex` and the appendix
reproducibility checklist at `thesis/chapters/A_appendix.tex`
§ `ap:repro-checklist`.*
