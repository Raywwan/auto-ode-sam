# V9 Competitor Benchmark & Multi-Dataset Plan

**Date:** 2026-04-21
**Purpose:** Identify the papers V9 (OrganFlowSAM2 cascade) compares to most, the expected 3D-Dice we need to beat, whether each comparison is *fair*, how we would perform on sibling datasets (BTCV, FLARE22/23, WORD, TotalSegmentator), and which dataset is best for training vs. testing.

---

## 1. Direct Competitors — ranked by "same task, same dataset, same rules"

| Rank | Paper | Venue / Year | AMOS22 CT Dice | BTCV Dice | FLARE22/23 Dice | What's comparable | Fair vs V9? |
|:---:|---|:---:|:---:|:---:|:---:|---|:---:|
| **1** | **nnWNet** | CVPR 2025 | ~0.92 (SOTA) | ~0.87 | — | 3D transformer-conv hybrid, plain supervised | ✅ **Yes** — our headline benchmark |
| **2** | **nnU-Net** | Nature Methods 2021 | 0.924 | 0.863 | 0.921 | 3D CNN, self-configuring | ✅ Yes, but SWAP-style auto-config — we need to match recipe |
| **3** | **SwinUNETR-V2** | MICCAI 2023 | 0.837 | ~0.85 | ~0.90 (WORD) | 3D Swin encoder, same backbone as our proposer | ✅ Yes — this is the *floor* |
| **4** | **Auto3DSeg** (MONAI) | 2023 | 0.902 | — | — | 3D AutoML, ensemble 5-fold | ⚠ Only if we also run 5-fold ensemble |
| **5** | **TotalSegmentator v2** | Radiology AI 2023/24 | 0.801 on AMOS22 (zero-shot) | 0.932 | — | Trained on its own 1200-vol, tested cross-dataset | ⚠ Apples-to-oranges (different train data) |
| **6** | **MambaMIM** (Mamba-MIM) | 2024 | 0.845 | 0.802 | — | 3D Mamba SSM + MIM pretrain | ✅ Yes — newer-architecture peer |
| **7** | **UlikeMamba-3dMT** | 2024 | 0.8995 | — | — | Mamba-Transformer, 93 G FLOPs | ✅ Yes — efficiency peer |
| **8** | **EM-Net** | MICCAI 2024 | — | 0.83–0.84 | — | Frequency + channel Mamba | ✅ Yes |
| **9** | **TP-Mamba** | MICCAI 2024 | ~0.85 (few-shot) | +12% vs baseline (few-shot) | — | Tri-plane SAM adapter | ✅ *Too strong* (foundation-model transfer) |
| **10** | **SegVol** | NeurIPS 2024 | — (semantic + prompt) | strong (zoom-out-zoom-in) | — | 90 k-volume CT pretrain foundation | ❌ **Unfair** — we train on 300 vols, they train on 90 000 |
| **11** | **AutoProSAM** | WACV 2025 | — | ~0.86 | — | 3D SAM auto-prompting | ✅ Yes — closest peer in "SAM-for-3D" camp |
| **12** | **MCP-MedSAM** | arXiv 2024 | 0.79–0.82 (reported on our slice protocol) | 0.81 | — | 2D MedSAM + multi-class prompt | ✅ Yes — baseline the user originally cited |
| **13** | **BA-Net** | arXiv 2022 | 0.893 (AMOS CT) | — | — | Boundary-aware 3D net | ✅ Yes |

### Verdict — the "fair comparison" shortlist for the thesis chapter

**Must report these 6 (they cover the full architectural space):**
1. nnU-Net v2 ← *baseline every medical-seg paper reports*
2. SwinUNETR-V2 ← *same backbone family as our proposer*
3. nnWNet (CVPR 2025) ← *current SOTA*
4. MambaMIM / UlikeMamba ← *new-architecture peer (Mamba)*
5. AutoProSAM / MCP-MedSAM ← *SAM-based peer (same lineage as MedSAM2 encoder)*
6. TotalSegmentator v2 ← *foundation-model baseline (generalisation check)*

---

## 2. Expected V9 results per dataset

| Dataset | # vols / # organs | V7 baseline | V9 Stage 1 | V9 Stage 2 | V9 Stage 3 | SOTA to beat | Gap remaining |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **AMOS22 CT** | 300 / 15 | 0.362 | 0.78–0.80 | 0.84–0.86 | **0.86–0.88** | 0.91–0.92 | −0.03 to −0.06 |
| **BTCV** | 30 / 13 | n/a | 0.76 | 0.83 | **0.85** | 0.87 | −0.02 |
| **FLARE22** | 50 labelled + 2000 unlabel / 13 | n/a | 0.81 | 0.87 | **0.89** | 0.92 | −0.03 |
| **FLARE23** | 220 / 13+cancer | n/a | 0.79 | 0.85 | **0.88** | 0.90 | −0.02 |
| **WORD** | 150 / 16 | n/a | 0.80 | 0.85 | **0.87** | 0.91 (SwinUNETR-V2) | −0.04 |
| **AMOS22 MRI** | 60 / 15 | n/a | 0.65 | 0.73 | **0.78** | 0.83 (MRISegmenter) | −0.05 |
| **TotalSeg-104** | 1200 / 104 | n/a | n/a (scope mismatch) | 0.75 (organ subset) | **0.82** | 0.94 | −0.12 (not our target) |

**Interpretation**
- AMOS22 CT is our **headline dataset**. V9 aims 0.86-0.88 — within a credible publishing gap of nnWNet.
- BTCV is **small (30 vols)** — V9's cascade + ODE should *widen* the gap there vs. shallow 2D baselines.
- FLARE22/23 are **the cleanest win path** because of the unlabelled pool (2000 vols) — perfect fuel for Novel #3 (teacher distill) and Novel #4 (TotalSegmentator pseudo-labels).
- WORD is good external validation (16 organs, overlap with AMOS15) — no train.
- AMOS22 MRI is a bonus generalisation test — we should train on CT only and zero-shot evaluate on MRI.

---

## 3. Is the comparison fair? — per-paper audit

| Competitor | Train data size | Extra supervision | Same test protocol | **Verdict** |
|---|:---:|:---:|:---:|---|
| nnU-Net | same 200 CT | none | 5-fold CV | ✅ Fair |
| nnWNet | same 200 CT | none | 5-fold CV | ✅ Fair |
| SwinUNETR-V2 | same 200 CT | ImageNet pretrain | 5-fold CV | ✅ Fair (same) |
| TotalSegmentator | 1200 CT (different) | own labels | zero-shot | ⚠ Unfair (advantage) |
| SegVol | 90 000 CT | self-supervision | fine-tune | ❌ Unfair — we won't match |
| SAM-based (SAM2, MedSAM2) | 11M + 100K images | point prompts | prompt-driven | ⚠ Different task (interactive) |
| MCP-MedSAM | 200 CT + prompts | multi-class prompts | same | ✅ Fair |
| AutoProSAM | 200 CT + auto-prompts | auto-prompt supervision | same | ✅ Fair |

**What we need to do to keep comparisons fair:**
1. Train V9 with **only** AMOS22 labelled (no TotalSeg images, only its *pseudo-labels* for distillation).
2. Report 5-fold CV numbers (not just single run).
3. Report *without* any self-supervision on external data (to fairly compare against nnU-Net/nnWNet).
4. Report **separately** a "V9 + TotalSeg pseudo + BiomedCLIP" variant as the unfair-advantage number, labelled as such.

---

## 4. Which dataset is best to **train** vs **test** on?

### Best dataset to **train** on = AMOS22 CT (our current choice ✅)

**Why**
- 300 labelled CT volumes = enough for large models; larger than BTCV (30) and WORD (150).
- 15 organ classes = covers all other dataset's organ sets as a superset.
- Grand Challenge hosted leaderboard gives us an **official number**, not just a paper claim.
- No dataset-licence issues (already downloaded, preprocessed to 1.5mm).

**Why not FLARE22/23**
- FLARE is *unlabelled + few labels* by design — great for SSL, wrong protocol for supervised headline.
- FLARE labels only cover 13 organs, losing adrenal separation vs AMOS15.

### Best dataset to **test / generalise** on = 3-way external eval

| Use | Dataset | Why |
|---|---|---|
| **Primary** (headline) | AMOS22 CT val (60 vols) | Official leaderboard, matches train split |
| **External #1** (clinical cross-site) | BTCV (30 vols) | Zero-shot, different scanners, same 13 organs |
| **External #2** (pan-cancer) | FLARE23 val (50 vols) | Tests tumour-robustness & out-of-domain pathology |
| **External #3** (modality gap) | AMOS22 MRI (60 vols) | Tests flow-ODE's modality robustness — *nice-to-have* |
| **External #4** (scale-stress) | WORD (150 vols) | 16 organs incl. small structures (rectum, gallbladder) |

**Minimum publishable set = {AMOS22 CT val, BTCV, FLARE23} = 3 datasets → publishable as "strong generalisation."**

---

## 5. Action items added to project

| # | Action | File | Priority |
|---|---|---|:---:|
| A1 | Add BTCV loader (skeleton) | `datasets/btcv.py` | P2 (after Stage 1) |
| A2 | Add FLARE23 loader | `datasets/flare23.py` | P2 |
| A3 | Add 5-fold CV runner | `training/cv_runner.py` | P1 (fairness) |
| A4 | Add cross-dataset eval script | `evaluation/eval_cross.py` | P2 |
| A5 | Dual-report: "fair" vs "+pseudo-labels" | doc only | P1 |
| A6 | AMOS22 MRI zero-shot notebook | `notebooks/amos_mri.py` | P3 |

---

## 6. What "SOTA" means concretely for V9

| Claim level | Number needed | Condition | What we report |
|---|:---:|---|---|
| **SOTA on AMOS22 CT** | > 0.92 average Dice | beat nnWNet on held-out test | unlikely without TotalSeg pseudo + BiomedCLIP |
| **Competitive with SOTA** | 0.86-0.90 | within 2 pts of nnWNet, + novel contributions | ✅ our target |
| **Publishable at MICCAI** | 0.85+ with 4+ novel losses beating ablated baseline | ablation + generalisation | ✅ credible path |
| **Publishable at a journal (MedIA)** | 0.85+ on 3 datasets, 5-fold CV, 2+ novelties ablated | external validation needed | ✅ credible path (+ BTCV/FLARE eval) |
| **Thesis-acceptable** | +15 pts over V7 (0.362 → 0.51+) with 1+ novelty isolated in ablation | single-dataset ok | ✅ already almost certainly clears |

---

## 7. Final recommendation

**Strategy = Path 2: Competitive-with-SOTA + strong novelty story**

Don't chase raw SOTA (0.92+) — we lack the 90 000-volume pretrain budget of SegVol. Instead:
1. Land V9 at **0.86-0.88 on AMOS22 CT** via Stages 1→3.
2. Show **external generalisation** on BTCV + FLARE23 (2-3 pt drop = strong).
3. Publish **4-5 ablated novelties** (flow-ODE, cascade, teacher-distill, cross-modal, boundary-DDPM) — the novelty count is what makes a thesis defensible even if absolute SOTA is missed by 3-4 pts.
4. Add an optional "+TotalSeg-pseudo, +BiomedCLIP" row that can push to 0.89-0.90 and close the SOTA gap *honestly*.

This is the same strategy that MCP-MedSAM (0.79-0.82) and AutoProSAM (0.86) took — neither beat nnU-Net, both were published because their novelty story was clear.

---

## 8. MRI extension — adding AMOS22 MRI support

**Effort:** 4 small file changes, ~1 hour code + 2 days GPU.

| # | File | Change |
|---|---|---|
| 1 | `datasets/amos22_v9.py` | Add `modality` field; branch normalization (CT: HU-clip/450, MRI: per-vol z-score) |
| 2 | `models/voluformer_v9.py` | `nn.Embedding(2, embed_dim)` modality token added to patch embeds |
| 3 | `configs/v9_tierB_mri.yaml` | MRI-only config |
| 4 | `configs/v9_tierB_multimodal.yaml` | Joint 80% CT / 20% MRI sampler |

**Recommended recipe = Joint CT+MRI (R2).** Expected AMOS-MRI Dice 0.78-0.82 with a single checkpoint.
Adds optional Novelty #11: *modality-conditioned flow ODE*.

## 9. Why we're not doing a 90k-volume SegVol-style pretrain

| Blocker | Status |
|---|---|
| 90 k curated CT pool | Not released publicly (BAAI internal) |
| 27 TB storage | Not feasible |
| ~100 k GPU-hours on A100 | Our 4090 = 10-12 months continuous |
| Dataset-curation labour | 2 person-years |

**What we do instead (free-knowledge pipeline):**
- ✅ MONAI SwinViT SSL warm-start (5050 CT) → +3-5 pts
- ✅ Teacher distill from SAM2 + DINOv2 + BiomedCLIP → +2-4 pts
- 🟡 TotalSegmentator-v2 pseudo-labels on 1200 external CT → +1-2 pts
- ✅ BiomedCLIP text alignment → +0.5-1 pt

Total ≈ **+7-12 pts** from KD/SSL — substitutes for the 90 k pretrain. **This is the paper's scientific contribution.**

## Sources (what this plan is grounded in)

- [AMOS22 Grand Challenge Leaderboard](https://amos22.grand-challenge.org/)
- [AMOS benchmark paper, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/ee604e1bedbd069d9fc9328b7b9584be-Abstract-Datasets_and_Benchmarks.html)
- [nnWNet, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhou_nnWNet_Rethinking_the_Use_of_Transformers_in_Biomedical_Image_Segmentation_CVPR_2025_paper.pdf)
- [SwinUNETR-V2, MICCAI 2023](https://link.springer.com/chapter/10.1007/978-3-031-43901-8_40)
- [SegVol, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/c7c7cf10082e454b9662a686ce6f1b6f-Paper-Conference.pdf)
- [AutoProSAM, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/papers/Li_AutoProSAM_Automated_Prompting_SAM_for_3D_Multi-Organ_Segmentation_WACV_2025_paper.pdf)
- [TotalSegmentator, Radiology AI 2023](https://pubs.rsna.org/doi/full/10.1148/ryai.230024)
- [FLARE22, Lancet Digital Health 2024](https://pubmed.ncbi.nlm.nih.gov/39455194/)
- [MambaMIM, arXiv 2024](https://arxiv.org/html/2408.08070v2)
- [Tri-Plane Mamba, MICCAI 2024](https://papers.miccai.org/miccai-2024/804-Paper2184.html)
- [FlowSDF (flow matching for seg), arXiv 2024](https://arxiv.org/abs/2405.18087)
- [BA-Net boundary-aware AMOS, arXiv 2022](https://arxiv.org/pdf/2208.13774)
- [Cross-dataset eval benchmark A-Eval, arXiv 2023](https://arxiv.org/abs/2309.03906)
