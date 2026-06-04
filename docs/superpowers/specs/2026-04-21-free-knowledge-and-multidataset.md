# V9 — Free-Knowledge, MRI, and Multi-Dataset Master Plan

**Date:** 2026-04-21
**Purpose:** The complete list of tricks that substitute for a 90 000-volume SegVol-style pretrain, the full MRI compatibility layer, the multi-dataset training plan, and a single prioritised table of what's left.

---

## 1. "90k-substitute" — all free-knowledge tricks available to V9

Each row is a real mechanism, ordered by expected gain on AMOS22 CT. Items marked ✅ are already wired; 🟡 means code exists but weight defaults to 0; ❌ means not yet built.

### 1A. Pretrained encoders (no training, just load weights)

| # | Trick | Source / scale | Expected gain | Status |
|---|---|---|:---:|:---:|
| 1 | MONAI SwinViT SSL (Tang et al.) | 5050 CT, SimMIM + rotation | **+3–5 pts** | ✅ loaded 126/134 keys |
| 2 | MedSAM2 encoder | 10 M medical images | **+2–4 pts** | ✅ in refiner, 42 LoRA layers |
| 3 | STU-Net pretrained (Huang 2023) | 1400 CT, AMOS-subsume | +1–2 pts (alt backbone) | ❌ optional swap of proposer |
| 4 | DINOv2-giant vision encoder | 142 M natural images | transfer weak for 3D | 🟡 used as teacher only |
| 5 | BiomedCLIP text+image | 15 M med pairs | +0.5–1 pt (text path) | 🟡 used in Novel #5 |
| 6 | MRSegmentator / MRISegmenter | 1200 MRI (for MRI mode) | +3–4 pts on MRI only | ❌ add in MRI branch |

### 1B. Self-supervised pretraining on *our own* data (1 GPU-week each)

| # | Trick | Method | Gain | Effort | Status |
|---|---|---|:---:|:---:|:---:|
| 7 | MIM pretrain on AMOS train | Mask 75 % of voxels, reconstruct | +1–2 pts | 3 GPU-days | ❌ |
| 8 | DINO-3D on AMOS | Two random crops, match class tokens | +1–2 pts | 5 GPU-days | ❌ |
| 9 | Volume Contrastive Learning (VoCo) | Positive = same volume, neg = other | +1 pt | 2 GPU-days | ❌ |
| 10 | Self-distillation EMA (SimSiam-style) | Student + EMA teacher | +0.5–1 pt | free (inline) | ❌ |
| 11 | Jigsaw / rotation prediction pretext | 24-way rotation classifier | +0.5 pt | 1 GPU-day | ❌ |

### 1C. Knowledge distillation from foundation models

| # | Trick | Teacher | Gain | Effort | Status |
|---|---|---|:---:|:---:|:---:|
| 12 | Multi-teacher feature distill | SAM2 + DINOv2 + BiomedCLIP | **+2–4 pts** | already coded | ✅ (Novel #3, λ=0 default) |
| 13 | Single-teacher TotalSegmentator distill | TotSeg-v2 outputs on unlabelled CT | **+1–2 pts** | 1 day (pseudo-label gen) | 🟡 (Novel #4, pending) |
| 14 | Logit distill from nnU-Net ensemble | 5 nnU-Net folds | +0.5–1 pt | 3 GPU-days | ❌ |
| 15 | Cross-modal text distill (CLIP-style) | BiomedCLIP text embeds | +0.5–1 pt | wired | ✅ (Novel #5) |
| 16 | Atlas-based prior distill | SPIDER / Visible Human atlas | +0.5 pt | 1 day | ❌ |

### 1D. Pseudo-labelling external data (label-free)

| # | Trick | Source | # volumes added | Gain | Status |
|---|---|---|:---:|:---:|:---:|
| 17 | TotSeg-v2 pseudo on AbdomenCT-1K | Public, partial labels | +1112 | **+1.5–2.5 pts** | 🟡 Novel #4 |
| 18 | TotSeg-v2 pseudo on FLARE22 unlab. | Public, 2000 unlabelled | +2000 | +2–3 pts | 🟡 |
| 19 | Noisy-Student iterative self-train | Student → teacher → student | incremental | +0.5–1 pt per round | ❌ |
| 20 | Confidence-gated pseudo-label (CAT) | Keep only conf > 0.9 | same pool | +0.3–0.8 pt | ❌ |

### 1E. Multi-dataset joint supervised training

| # | Trick | Datasets used | Total vols | Gain | Status |
|---|---|---|:---:|:---:|:---:|
| 21 | UniSeg-style dataset-ID embedding | AMOS + BTCV + FLARE + WORD | ~500 | **+1.5–2 pts** | ❌ (new loader needed) |
| 22 | Label-unified joint supervision | All 15-organ-superset mapping | ~500 | +1–2 pts | ❌ |
| 23 | Partial-label training (AbdomenCT-1K) | 1112 vols, partial labels | +1112 | +1–2 pts | ❌ |
| 24 | Cross-dataset continual learning | Train AMOS → FT BTCV → FT FLARE | same | +0.5–1 pt | ❌ |

### 1F. Augmentation & regularisation (zero extra data)

| # | Trick | Gain | Effort | Status |
|---|---|:---:|:---:|:---:|
| 25 | RandAugment-3D / TrivialAugment-3D | +0.5–1 pt | free | ❌ |
| 26 | CutMix-3D within batch | +0.5–1 pt | free | ❌ (disabled in V7, retry) |
| 27 | MixUp-3D intensity blend | +0.3–0.5 pt | free | ❌ |
| 28 | Multi-scale patch schedule (96→128→160) | +0.5 pt | training-time | ❌ |
| 29 | FFT / Fourier domain aug | +0.3–0.5 pt | free | ❌ |
| 30 | N4 bias-field correction (MRI) | +1–2 pts on MRI only | 1 day | ❌ |

### 1G. Test-time only

| # | Trick | Gain | Effort | Status |
|---|---|:---:|:---:|:---:|
| 31 | 8-way TTA (flip + transpose) | +0.5–1 pt | free | ✅ in `eval_v9_3d.py` |
| 32 | Multi-scale sliding-window | +0.3–0.5 pt | free | ✅ |
| 33 | Connected-component post-filter | +0.3–0.5 pt | free | ✅ |
| 34 | Gaussian-blend patch stitching | +0.2–0.5 pt | free | ✅ |
| 35 | Ensemble 5-fold CV | +0.5–1.5 pt | 5× training time | ❌ |
| 36 | Ensemble different architectures | +1–1.5 pt | 2× training time | ❌ |

### 1H. Synthetic data generation

| # | Trick | Method | Gain | Effort | Status |
|---|---|---|:---:|:---:|:---:|
| 37 | Diffusion-generated synthetic CT | MedSegDiff / HA-GAN | +0.5–1 pt | 10 GPU-days | ❌ (not worth it) |
| 38 | Organ-swap / morphing via registration | Deformable registration | +0.3–0.5 pt | 2 days | ❌ |
| 39 | Style-transfer CT↔MRI | CycleGAN | +0.5 pt MRI | 5 GPU-days | ❌ |

### Totals — *achievable* free-knowledge gain vs. SegVol's 90 k pretrain

Sum of realistic picks (✅ / 🟡 only, not the whole list):

| Source | Gain |
|---|:---:|
| SwinViT SSL loaded (already) | +4 |
| MedSAM2 encoder (already) | +3 |
| Teacher distill Novel #3 (on) | +3 |
| TotalSeg pseudo Novel #4 (planned) | +2 |
| BiomedCLIP Novel #5 (on) | +1 |
| TTA + CC + Gaussian (on) | +1 |
| MIM pretrain on AMOS (optional, +3 days) | +1–2 |
| UniSeg multi-dataset (optional, +5 days) | +1.5–2 |
| **Realistic total over V7 baseline (0.362)** | **+15–16 pts → 0.51–0.52** |
| **Realistic total with all 8 V9 novel losses on** | **+40–50 pts → 0.86–0.88** |

SegVol's 90 k pretrain is estimated to add about +5–7 pts over a strong baseline. We recover this via rows 1, 2, 12, 13, 15.

---

## 2. Full MRI compatibility — file-by-file spec

Everything needed so V9 trains on CT, MRI, or both with the same checkpoint.

### 2A. Config (new + modified)

| File | Change |
|---|---|
| `configs/v9_tierB.yaml` | Add `data.modality: "ct"` (default). |
| `configs/v9_tierB_mri.yaml` | **NEW** — `data.modality: "mri"`, `data.data_root: "…/amos22/mri"`, different SSL ckpt. |
| `configs/v9_tierB_multimodal.yaml` | **NEW** — `data.modality: "joint"`, CT/MRI sampler 80/20. |

### 2B. Dataset (`datasets/amos22_v9.py`)

```python
# New fields per sample:
{
    "modality": torch.tensor(0),   # 0 = CT, 1 = MRI
    "volume":   …,                  # normalised per-modality
    …
}

# Branch normalisation:
if modality == "ct":
    x = np.clip(x, -200, 250)
    x = (x + 200) / 450               # [-200,250] → [0,1]
elif modality == "mri":
    # Per-volume z-score — MRI has no absolute scale
    x = (x - x.mean()) / (x.std() + 1e-6)
    x = np.clip(x, -3, 3) / 3         # [-3σ,+3σ] → [-1,1]
```

Add `_discover_volumes()` branch for `*.nii.gz` MRI volumes under `amos22/imagesTr/` with id ≥ 500 (AMOS uses 5XX for MRI).

### 2C. Model (`models/voluformer_v9.py`)

```python
self.modality_emb = nn.Embedding(2, mcfg.embed_dim)   # 2 modalities

# In forward(), after patch-embed inside proposer + refiner:
mod_tok = self.modality_emb(batch["modality"])          # (B, C)
patch_tokens = patch_tokens + mod_tok[:, None, None, :]  # broadcast
```

This is a learnt additive bias that the model can use to route features per modality. Standard UniSeg / Omni-Seg pattern.

### 2D. Preprocessing (new file `scripts/preprocess_mri.py`)

| Step | Library | Why |
|---|---|---|
| N4 bias-field correction | SimpleITK | Removes scanner bias (+1–2 pts) |
| Resample to 1.5 mm iso | SimpleITK | Match CT grid |
| Histogram matching to atlas | SimpleITK | Intensity harmonisation (+0.5 pt) |
| Save as `.pt` like CT | NumPy | Match `AMOS22V9Dataset` loader |

### 2E. MRI-specific augmentation (`datasets/amos22_v9.py`)

MRI-only augs applied when `modality == 1`:
- Random bias field (TorchIO `RandomBiasField`)
- Random ghosting (TorchIO `RandomGhosting`)
- Random motion (TorchIO `RandomMotion`)
- Random gamma 0.7–1.3
- Random noise σ ∈ [0, 0.05]

Typical gain on MRI: +1–2 pts.

### 2F. Cross-modal loss (optional, Novel #11)

When both CT and MRI are in the batch for the same patient (AMOS22 doesn't pair, but we can simulate via registration):
- Cosine alignment between CT and MRI feature maps at the same anatomical location.
- Optional — adds modality-invariance without extra labels.

### 2G. MRI training recipes

| Recipe | Data | Expected MRI Dice | Effort |
|---|---|:---:|:---:|
| R1 zero-shot | CT-only train | 0.55–0.65 | free (today) |
| R2 joint CT+MRI | 300 CT + 40 MRI + modality token | **0.78–0.82** | 2 GPU-days |
| R3 MRI fine-tune | CT Stage-3 → FT 40 MRI | 0.75–0.80 | 1 GPU-day |
| R4 joint + N4 + TorchIO | R2 + MRI preproc | **0.80–0.84** | 3 GPU-days |

**Pick R4 for the final paper** — it's the standard AMOS-MRI protocol (MRISegmenter paper).

---

## 3. Multi-dataset training plan

### 3A. Datasets in scope

| Dataset | Modality | # volumes | # organs | Public? | Home |
|---|:---:|:---:|:---:|:---:|---|
| AMOS22 CT | CT | 300 | 15 | ✅ | downloaded |
| AMOS22 MRI | MRI | 100 | 15 | ✅ | downloadable |
| BTCV / Synapse | CT | 30 | 13 | ✅ | `synapse.org` |
| FLARE22 | CT | 50 lab + 2000 unlab | 13 | ✅ | Grand Challenge |
| FLARE23 | CT | 220 + 1200 unlab | 13 + cancer | ✅ | Grand Challenge |
| WORD | CT | 150 | 16 | ✅ | paper repo |
| AbdomenCT-1K | CT | 1112 | 4 (partial) | ✅ | paper repo |
| TotalSegmentator-v2 | CT | 1228 | 104 | ✅ | Zenodo |
| Pancreas-CT (NIH) | CT | 82 | 1 (pancreas) | ✅ | TCIA |
| **Total available** | — | **~5270 CT + 100 MRI** | 15-unified | | |

### 3B. Unified label schema

All datasets mapped to AMOS15:

```
0: background
1: spleen          → AMOS,BTCV,FLARE,WORD,TotalSeg
2: right kidney    → all
3: left kidney     → all
4: gallbladder     → AMOS,BTCV,FLARE,TotalSeg
5: esophagus       → AMOS,BTCV,WORD,TotalSeg
6: liver           → all
7: stomach         → all
8: aorta           → AMOS,BTCV,WORD,TotalSeg
9: inferior vena cava → AMOS,BTCV,WORD,TotalSeg
10: portal vein/splenic vein → AMOS,BTCV
11: pancreas       → all
12: right adrenal  → AMOS,BTCV,TotalSeg
13: left adrenal   → AMOS,BTCV,TotalSeg
14: duodenum       → AMOS,WORD,TotalSeg
15: bladder        → AMOS,WORD,TotalSeg
```

**Missing organs per dataset = masked out of loss**, not counted as negatives.

### 3C. Two-track training schedule

| Track | Use | Stage 1 budget | Stage 2 budget | Stage 3 budget |
|---|---|:---:|:---:|:---:|
| **A: AMOS-only** (headline result) | paper Table 2 "fair comparison" | 3 days | 5 days | 3 days |
| **B: All-dataset (unified)** (paper Table 3 "scale") | paper Table 3 "scaled V9" | 5 days | 7 days | 3 days |

Report both numbers. Track A is the fair-play number, Track B is the "upper bound with all free knowledge."

### 3D. Multi-dataset evaluation matrix (publishable)

| Train ↓ / Test → | AMOS-CT val | BTCV | FLARE23 val | WORD | AMOS-MRI |
|---|:---:|:---:|:---:|:---:|:---:|
| AMOS-CT only | primary | external | external | external | zero-shot |
| AMOS-CT+MRI joint | — | external | external | external | primary |
| All-unified | upper bound | upper bound | upper bound | upper bound | upper bound |

One paper figure = one heatmap of this 3 × 5 table. This is how A-Eval (2023) frames fair cross-dataset comparison.

### 3E. New files required

| File | Purpose | LOC est |
|---|---|:---:|
| `datasets/btcv.py` | BTCV loader with AMOS15 remapping | ~150 |
| `datasets/flare23.py` | FLARE23 + unlabelled pool | ~200 |
| `datasets/word.py` | WORD loader | ~120 |
| `datasets/totalseg.py` | TotalSeg-v2 loader + AMOS15 mapping | ~200 |
| `datasets/abdomenct1k.py` | AbdomenCT-1K partial-label loader | ~100 |
| `datasets/multi_dataset.py` | Unified wrapper (dataset-ID tokens, partial-label masking) | ~250 |
| `scripts/build_label_unified_cache.py` | Offline one-time remap + cache | ~150 |
| `evaluation/eval_cross.py` | 3×5 heatmap runner | ~200 |
| `configs/v9_unified.yaml` | All-dataset config | ~80 |

---

## 4. Single prioritised "what's left" table

Ordered by (expected gain × likelihood × inverse cost). Only the **first 10** need to happen before a publishable paper.

| # | Task | Category | Expected gain (pts) | Cost | GPU? | Status | Priority |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| 1 | Stage 1 training (SwinUNETR proposer) | core | +30–35 vs V7 | 3 days | ✅ | 🟢 running | **P0** |
| 2 | Stage 2 training (refiner + cascade) | core | +10–12 | 5 days | ✅ | ⏸ blocked on #1 | **P0** |
| 3 | Stage 3 joint fine-tune | core | +1–2 | 3 days | ✅ | ⏸ blocked on #2 | **P0** |
| 4 | Download SAM2, DINOv2, BiomedCLIP weights | novel #3 | +2–4 (enables) | 1 h | — | ❌ | **P0** |
| 5 | Download TotalSegmentator-v2 | novel #4 | +1.5–2.5 (enables) | 1 h | — | ❌ | **P0** |
| 6 | Build TotalSeg pseudo-labels on AbdomenCT-1K | novel #4 | +1.5–2.5 | 1 GPU-day | ✅ | ❌ | **P0** |
| 7 | Add MRI modality token + dataset branch | MRI | 0 on CT (enables MRI) | 1 h code | — | ❌ | **P1** |
| 8 | Preprocess AMOS22 MRI (N4 + resample) | MRI | enables +2 on MRI | 4 h | — | ❌ | **P1** |
| 9 | Joint CT+MRI Stage 3 retrain (R4) | MRI | +20 on MRI | 3 GPU-days | ✅ | ❌ | **P1** |
| 10 | BTCV + FLARE23 external eval | cross-dataset | 0 (reporting) | 1 GPU-day | ✅ | ❌ | **P1** |
| 11 | UniSeg multi-dataset joint (Track B) | scale | +1.5–2 | 5 GPU-days | ✅ | ❌ | P2 |
| 12 | MIM pretrain on AMOS train pool | SSL | +1–2 | 3 GPU-days | ✅ | ❌ | P2 |
| 13 | 5-fold CV ensemble | robustness | +0.5–1.5 | 5× training | ✅ | ❌ | P2 |
| 14 | Noisy-student iteration | SSL | +0.5–1 | 3 GPU-days | ✅ | ❌ | P3 |
| 15 | RandAugment-3D pipeline | aug | +0.5–1 | 1 day code | — | ❌ | P2 |
| 16 | CutMix-3D retry (disabled in V7) | aug | +0.3–0.8 | 2 h code | — | ❌ | P3 |
| 17 | Multi-scale patch schedule | schedule | +0.5 | 2 h code | — | ❌ | P3 |
| 18 | 8-way TTA already on | eval | already counted | — | — | ✅ | done |
| 19 | Connected-components post-filter | eval | already counted | — | — | ✅ | done |
| 20 | WORD external eval | cross-dataset | 0 (reporting) | 0.5 GPU-day | ✅ | ❌ | P2 |
| 21 | AbdomenCT-1K partial-label train | data | +1–2 | 3 GPU-days | ✅ | ❌ | P3 |
| 22 | Atlas shape prior distill | KD | +0.5 | 2 days | — | ❌ | P4 |
| 23 | Diffusion synthetic CT | synth | +0.5–1 | 10 GPU-days | ✅ | ❌ | P4 (skip) |
| 24 | CycleGAN CT↔MRI style transfer | synth | +0.5 MRI | 5 GPU-days | ✅ | ❌ | P4 (skip) |
| 25 | nnU-Net logit distill teacher | KD | +0.5–1 | 3 GPU-days | ✅ | ❌ | P3 |
| 26 | Multi-architecture ensemble | ensemble | +1–1.5 | 2× training | ✅ | ❌ | P3 |
| 27 | MRSegmenter MRI pretrained weights | MRI KD | +1–2 MRI | 1 h | — | ❌ | P1 (bundled with #8–9) |
| 28 | Cross-modal CT↔MRI feature alignment loss | novel #11 | +0.5–1 MRI | 1 day code | ✅ | ❌ | P2 |

### Minimum publishable set = rows 1-10. Total compute = **~20 GPU-days** on your 4090.

### Optional-strong set = rows 1-16. Total compute = **~50 GPU-days**.

---

## 5. Schedule (if GPU 24/7)

| Week | Days | Rows completed | End-of-week deliverable |
|---|---|---|---|
| 1 | 1–7 | 1, 4, 5, 6 | Stage 1 ckpt (CT) + TotalSeg pseudo ready |
| 2 | 8–14 | 2 | Stage 2 ckpt with Novel #3,4,5 on |
| 3 | 15–18 | 3 | Stage 3 ckpt = headline number |
| 3 | 19–21 | 7, 8, 9 | MRI added, R4 run starts |
| 4 | 22–25 | 9 finish, 10 | MRI Dice + BTCV/FLARE external eval |
| 4 | 26–28 | paper draft | V9 methodology + results Chapter |

**4 weeks from today = paper first draft.** 6 weeks with ablation grid (rows 11-16).

---

## 6. Hard stop conditions (don't sink time into these)

| Item | Reason to skip |
|---|---|
| 90 k-volume SegVol pretrain | Data doesn't exist publicly. |
| Training a new foundation model from scratch | Outside thesis scope. |
| Pure 3D transformer at full 512³ | VRAM infeasible on 4090. |
| Diffusion synthetic CT | 10 GPU-days for +0.5 pt. |
| CycleGAN CT↔MRI | 5 GPU-days for +0.5 pt. |
| Pure MRI-only training | Worse than joint CT+MRI. |

---

## 7. Bottom line

| Metric | Honest estimate |
|---|---|
| V9 Stage 3 final AMOS22 CT Dice | **0.86 – 0.88** |
| V9 Stage 3 final AMOS22 MRI Dice (R4) | 0.80 – 0.84 |
| V9 on BTCV (external) | 0.83 – 0.85 |
| V9 on FLARE23 (external) | 0.86 – 0.88 |
| Gap to SOTA (nnWNet ~0.92 CT) | –0.04 to –0.06 |
| Novelty budget | 8 wired + 3 MRI/multi-dataset extensions |
| Publishable venue | MICCAI / MedIA / IEEE TMI (all feasible) |
| Time to paper first draft | **4 weeks** with 24/7 GPU |
