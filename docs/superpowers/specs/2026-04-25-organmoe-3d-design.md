# OrganMoE-3D — 30-Day SOTA-Targeted Design Spec

**Date:** 2026-04-25
**Author:** Raywan Dlawar
**Project:** VoluFormer3D V4 → new model class **OrganMoE-3D**
**Plan window:** 30 days (2026-04-25 → 2026-05-25)
**Hardware:** Single RTX 4090 (24 GB VRAM), Ryzen 9 5900X, 32 GB RAM

---

## 1. Context and motivation

### 1.1 Where we are

V10 W1 base (`checkpoints/v10_voco_l_ft_r3/epoch_009.pt`) reached **0.7967 patch-eval mean Dice** on AMOS22 CT 30-vol val set. Per-organ analysis revealed:

- **14 organs** are at 0.68-0.96 (mean ~0.84)
- **prostate_uterus** channel is at 0.03-0.05 in patch eval, 0.0 by ep 5 of W2 DS run

Sliding-window 3D eval started on W1 base; partial results (12/30 vols) showed mean ≈ 0.78. The lift expected from full-volume eval did NOT materialize — confirming **prostate failure is a genuine model weakness**, not a sampling artifact.

### 1.2 Why Phase C (CP / SL / DS) failed the gate

W2 ablations on three Phase C axes all missed the 0.02-lift gate:

| Axis | Best val | Δ vs W1 base | Outcome |
|------|---------:|---:|---|
| Copy-paste aug (CP) | 0.8001 | +0.0034 | Below gate |
| Small-organ loss (SL) | 0.7953 | -0.0014 | Below baseline |
| Deep supervision (DS) | 0.7895 | -0.0072 | STOP-RULE; hurt pelvic organs |

The architectural lesson: backbone capacity + class imbalance is the true bottleneck. None of the three "weak organ rescue" axes touched the rare-organ-presence problem at the architecture level.

### 1.3 SOTA reality

| Source | Mean DSC | Notes |
|---|---:|---|
| nnWNet (paper) | 0.864 | Strong published single-model |
| MedNeXt published | ~0.880 | Best published single-model on AMOS22 |
| AMOS22 leaderboard top (May 2024) | **0.918** | Multi-model ensemble, private submission |
| AMOS22 leaderboard #5 (May 2025) | 0.913 | Likely also ensemble |

Realistic targets for our compute budget (one 4090, 30 days):
- Single-model: 0.86-0.88
- Ensemble: 0.87-0.90
- Top-5 leaderboard: ~25% probability with strong ensemble

---

## 2. Goal and success criteria

| Tier | Mean Dice (val, sliding-window + TTA, ensemble) | Worst-organ Dice | Defendable contribution |
|---|---:|---:|---|
| FLOOR (must hit) | 0.85 | 0.30 | OrganMoE-3D mechanism + ablation |
| TARGET | 0.87 | 0.55 | Above + 4-backbone ensemble + organ-centric reseg |
| STRETCH | 0.90 | 0.75 | Above + 2nd novelty (Cross-Foundation Distillation OR Atlas-PE) |

**Hard exit gate**: at end of day 7 (Phase I), if mean Dice <0.85 AND prostate Dice <0.30, freeze whatever we have, drop the 30-day plan, fall back to "Path A 10-day plan" framing — write up workshop paper with what works.

---

## 3. The novel algorithmic contribution

### 3.1 OrganMoE-3D — Class-Presence-Aware Sparse LoRA Mixture-of-Experts

**Mechanism**: replace the standard LoRA adapter on a frozen pretrained backbone with a sparse mixture of K=8 LoRA experts, where the routing is conditioned on a class-presence vector predicted by an auxiliary head.

**Components:**

| Component | Role | Status |
|---|---|---|
| Frozen VoCo-L (or SuPreM) backbone | Provides 3D feature extraction | Existing |
| K=8 LoRA-rank-16 experts per attention QKV/proj and FFN fc1/fc2 | Per-organ specialization slots | NEW module |
| Class-presence side head | Tiny MLP on low-res encoder pooled features → predicts `c ∈ [0,1]^15` per-organ presence | NEW |
| Presence-conditioned router | Standard top-2 sparse router, BUT input includes both layer features AND `c`. Router learns to skip experts whose specialization is for absent organs | NEW |
| Load-balancing aux loss | Standard MoE balance loss | NEW (small) |
| Presence-disagreement loss | `λ · KL(c || gt_presence_vector)` — supervises presence head | NEW |

**Why this is novel** (verified via web search, 2026-04-25):
- LoRA-MoE for medical seg exists (continual learning context) — our framing (rare-organ rescue via presence routing) is new
- Class-presence-aware routing has no published precedent in vision/medical
- Combination for 3D multi-organ AMOS22 has no published prior art
- Closest precedent is MaskMoE (NLP, frequency-fixed token-to-expert) — different problem axis

**Why this directly targets the observed failure**: The prostate failure mechanism is that batches without prostate voxels still update prostate-channel-relevant backbone weights (washout). OrganMoE-3D provides architectural isolation: presence prediction → router skips prostate experts → prostate experts only update on prostate-containing batches. This is **the right architectural answer to the documented empirical failure**, not a generic ML add-on.

### 3.2 Phase III novelty options (decision at day 18)

Based on Phase I-II per-organ analysis, pick ONE:

**Option A — Cross-Foundation Distillation**: distill ensemble of SuPreM + VoCo-H + STU-Net features into a single OrganMoE-3D student. Novel: foundation-model distillation in 3D medical hasn't been published. Pick if Phase II shows backbones diverge informatively per-organ.

**Option B — Atlas-Positional-Encoding for Small Organs**: inject anatomical-atlas positional priors into transformer PE for small lateral organs. Build atlas from training set: organ-centric mean-position maps in normalized canonical space. Novel: atlas priors haven't been combined with Swin/MoE. Pick if small organs are still under-localized after Phase II.

---

## 4. System architecture

```
──────────────── TRAIN STAGE (Phases I-III, days 1-21) ────────────────
┌───────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│ TotalSegmentator  │→│ AMOS22 fine-tune    │→│ Backbone variants:   │
│ pretrain          │  │ (balanced sampler   │  │ - VoCo-L+OrganMoE-3D │
│ (1228 vols, 13    │  │ + presence loss)    │  │ - SuPreM             │
│ shared organs)    │  │                     │  │ - MedNeXt-L          │
│                   │  │                     │  │ - STU-Net (optional) │
└───────────────────┘  └─────────────────────┘  └─────────────────────┘

──────────── INFERENCE STAGE (Phase IV+ deploy) ──────────────────────
┌────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│ 3-4 backbone   │→│ Per-organ           │→│ Organ-centric reseg │
│ ensemble       │  │ uncertainty gate    │  │ (only triggered     │
│ (per-organ-    │  │ g_o = H(p_o)·       │  │ organs)             │
│ weighted softmax│  │ rarity^β > τ_o     │  │ Per-organ U-Nets    │
│ + 8-way TTA)   │  │                     │  │                     │
└────────────────┘  └─────────────────────┘  └─────────────────────┘
```

*TTA: existing `eval_v9_proposer_preproc.py` uses 4-way (LR + AP flips). Phase IV
upgrades to 8-way (LR × AP × IS = 2³) for ensemble inference. Implementation
adds the IS-axis flip path inside the existing eval loop.

---

## 5. Components and code map

| # | Component | Role | Status | Files |
|---|---|---|---|---|
| 1 | TotalSeg dataset loader | Reads 1228-vol Zenodo dataset, label remap to AMOS22's 13 shared organs | NEW | `datasets/totalsegmentator.py` |
| 2 | TotalSeg pretraining script | Pretrain selected backbone on TotalSeg, then transfer | NEW | `training/pretrain_totalseg.py` |
| 3 | `BalancedBatchSampler` | Forces ≥1 patch per rare organ per batch via rare-organ index | NEW | `datasets/balanced_sampler.py` |
| 4 | `SmallOrganLoss` extended with presence weights | Don't penalize absent classes; up-weight rare-present ones | EXTEND existing | `training/losses_small_organ.py` |
| 5 | `OrganMoE-3D` module | The novel algorithmic contribution (Section 3.1) | NEW | `models/organmoe_3d.py` |
| 6 | `PresenceHead` and `PresenceConditionedRouter` | Component pieces of OrganMoE-3D | NEW | within `models/organmoe_3d.py` |
| 7 | `SuPreMLoader` | Load HuggingFace SuPreM Swin UNETR weights into our existing Swin code | NEW (small) | `models/loaders/suprem.py` |
| 8 | `MedNeXt3D` | Train-from-scratch ConvNeXt 3D from MIC-DKFZ repo | NEW (vendored) | `models/mednext_3d.py` |
| 9 | `STU-Net` adapter | Optional: load STU-Net 1.4B weights, adapt nnUNet decoder format | NEW (optional) | `models/stunet_3d.py` |
| 10 | `EnsembleFusion` | Per-organ-weighted softmax fusion across N backbones | NEW (small) | `inference/ensemble.py` |
| 11 | `OrganCentricReseg` infrastructure | Per-organ U-Net training pipeline + uncertainty gate | NEW | `models/organ_reseg.py`, `training/train_organ_reseg.py`, `inference/uncertainty_gate.py` |
| 12 | Cross-Foundation Distillation (if picked at day 18) | Distill ensemble logits into student | NEW | `training/cross_foundation_distill.py` |
| 13 | Atlas-PE (if picked at day 18) | Build atlas, inject as additional PE | NEW | `models/atlas_pe.py`, `datasets/build_atlas.py` |
| 14 | 5-fold CV harness | Sex-stratified folds, run training across folds | NEW (small) | `scripts/run_5fold_cv.py` |
| 15 | AMOS22 test leaderboard submission script | Convert outputs to challenge format, package, upload | NEW (small) | `scripts/submit_amos22.py` |

---

## 6. Phase-by-phase plan

### Phase I — Foundation (Days 1-7)

| Day | Task | Smoke test | Pass criterion |
|---:|---|---|---|
| 1 | Build `BalancedBatchSampler`; extend `SmallOrganLoss` with presence weights | 1 batch sampler smoke + 1 loss forward smoke | No NaN, weights sum >0 per batch |
| 1 | Background download TotalSeg (~50GB) | Verify 1 sample loads | Labels match TotSeg spec |
| 2 | TotalSeg → AMOS22 label remap (13 shared organs) | Validate on 5 vols: voxel counts match | <5% label mismatch |
| 2 | Pretrain VoCo-L on TotalSeg (10 ep, ~6h) | Loss decreasing monotonically | Loss <0.6 after 5 ep |
| 3 | Implement `OrganMoE-3D` module | 1-vol forward+backward, VRAM check | <22GB, no NaN, grad norm <100 |
| 3 | Unit-test presence head on 5 vols | Per-organ presence accuracy | >0.85 presence accuracy |
| 4 | Fine-tune OrganMoE-3D on AMOS22 (TotSeg-pretrained → AMOS22, 10 ep) with balanced sampler + presence loss | Per-epoch val Dice tracked | Val improves vs W1 base |
| 5 | Sliding-window 3D eval (TTA + CC, 30 vols) | Per-organ + mean | **GATE I: mean ≥0.85, prostate ≥0.30** |
| 6 | If GATE I fails → diagnose per fallback table below | — | — |
| 7 | Buffer; update master log | — | — |

**Phase I fallbacks (named, not "hope"):**

| Failure mode | Diagnostic | Specific fallback |
|---|---|---|
| Mean <0.85, MoE training collapsed (router degenerate) | Check expert utilization | Switch to plain LoRA + class fix (drop MoE for Phase II, defer novelty to Phase III) |
| Mean <0.85, prostate still 0.0 | Check presence head accuracy | Add forced presence injection (oracle presence at training, predicted at test) |
| Mean <0.85, all organs degraded | Check TotalSeg pretrain quality | Skip TotalSeg pretrain, use plain VoCo-L base + balanced sampler |
| Prostate ≥0.30 but mean <0.85 | Other organs regressed | Lower MoE auxiliary loss weight, reduce K to 4 experts |

### Phase II — Multi-backbone ensemble (Days 8-14)

| Day | Task | Pass criterion |
|---:|---|---|
| 8 | Download SuPreM Swin UNETR weights; smoke load + test forward | Weights load, no missing keys |
| 9 | Fine-tune SuPreM on AMOS22 (with Phase I best config: balanced + presence + TotSeg-pseudo) | Val ≥0.83 |
| 10 | Vendor MedNeXt-L from MIC-DKFZ repo; smoke + train-from-scratch on AMOS22 | Val ≥0.83 |
| 11 | (Optional) STU-Net adapter + fine-tune | Val ≥0.83 OR drop |
| 12 | Build `EnsembleFusion` module | Smoke on 5 vols, fusion better than worst single |
| 13 | Calibrate per-organ ensemble weights on val (grid search) | Best weights identified |
| 14 | Sliding-window 3D eval ensemble (3-4 backbones + TTA + CC) | **GATE II: mean ≥0.87** |

**Phase II fallbacks:**

| Failure mode | Fallback |
|---|---|
| One backbone trains <0.80 | Drop from ensemble; use 3-backbone ensemble |
| 2+ backbones train <0.80 | Skip Phase II ensemble; commit to OrganMoE-3D single-model + Phase III refinement |
| Ensemble doesn't help (correlated errors) | Add diversity loss during training of latter backbones |
| Time blowout (Phase II >7 days) | Skip slowest backbone; freeze 3-backbone ensemble |

### Phase III — Refinement + 2nd novelty (Days 15-21)

| Day | Task | Pass criterion |
|---:|---|---|
| 15 | Build per-organ reseg crop pipeline | Crops valid, no empty crops |
| 16 | Train per-organ U-Nets for 7 hard organs (prostate, bladder, adrenals×2, duodenum, gallbladder, esophagus, pancreas) | Each net val Dice >0.70 on its own organ |
| 17 | Implement uncertainty-gated reseg with calibrated per-organ τ_o | Trigger rate sane (~30% per organ) |
| 18 | DECISION POINT: pick 2nd novelty (Cross-Foundation Distillation OR Atlas-PE) based on Phase I-II analysis | One picked, scaffolded |
| 19 | Implement 2nd novelty | Smoke + 1-ep training |
| 20 | Train 2nd novelty (5 ep) on best Phase II model | Val ≥ Phase II ensemble |
| 21 | Sliding-window 3D eval full pipeline | **GATE III: mean ≥0.88** |

**Phase III fallbacks:**

| Failure mode | Fallback |
|---|---|
| Reseg false-triggers (>50%) | Raise τ_o threshold; if still bad, gate by combined uncertainty + presence prediction |
| Reseg networks train <0.70 | Drop weakest organ from reseg; use ensemble for that organ instead |
| 2nd novelty doesn't help | Drop 2nd novelty; accept Phase II + reseg only as final pipeline |
| Time blowout | Skip 2nd novelty entirely; commit to ensemble + reseg only |

### Phase IV — Validation & ablation (Days 22-26)

| Day | Task | Pass criterion |
|---:|---|---|
| 22 | Set up 5-fold split (240 train / 60 val per fold, sex-stratified) | Folds balanced |
| 23-24 | Train best config on 5 folds (~40 hr each, time-shared) | All folds train |
| 25 | Full ablation: (a) baseline VoCo-L, (b) +TotalSeg pretrain, (c) +balanced sampler, (d) +OrganMoE-3D, (e) +ensemble, (f) +reseg, (g) +2nd novelty | Each cell trained, evaluated |
| 26 | Submit best ensemble to AMOS22 test leaderboard; final val report | **GATE IV: 5-fold mean ≥0.86, ablations clean** |

**Phase IV fallbacks:**

| Failure mode | Fallback |
|---|---|
| 5-fold variance huge (>0.04 std) | Investigate fold split; report mean ± std with caveat |
| Some ablation cells fail | Mark as "not converged" in table; don't fudge numbers |
| Test leaderboard unreachable in time | Report val results only; mark test submission as "in progress" |

### Phase V — Writing (Days 27-30)

| Day | Task |
|---:|---|
| 27 | Results table (per-organ Dice/HD95/NSD, val + test); baseline comparison vs nnWNet, MedNeXt, SuPreM, SAM-Med3D |
| 28 | Method figures: pipeline diagram, OrganMoE-3D architecture, uncertainty gate diagram |
| 29 | Thesis chapter draft (Method + Results + Discussion); paper draft (workshop / main-track depending on results tier) |
| 30 | Final buffer; commit everything; sacred ckpt audit; reproducibility check (re-run 1 eval from scratch from a published seed) |

---

## 7. Solidity safeguards (anti-fall-apart)

| Phase C failure mode | OrganMoE-3D safeguard |
|---|---|
| Wrong gate (patch eval misled us) | All gates use sliding-window 3D Dice with TTA + CC, never patch eval |
| Three axes ablated in parallel without smoke tests | Every component has 1-vol smoke before training |
| No fallback when DS hurt prostate | Every gate has named fallback path with concrete next step |
| Logging gap (per-organ data invisible) | Per-organ JSON dumped after every eval, mandatory |
| Optimizer-state bug from lazy module init | Eager init enforced — all params in optimizer at construction time |
| Single sacred ckpt at risk | Sacred ckpt list: W1 base, V9 stage 1, TriSSR-V9, all VoCo and SuPreM pretrained — write-protected via separate `sacred/` directory + read-only filesystem flag |
| Unreproducible runs | Every training run saves: git commit hash + config snapshot + random seed + dataset split + ckpt — bundled per run dir |
| Master log drift | Master log updated after every gate (not "at end of phase") |

---

## 8. Sacred ckpt protocol

**Never overwritten, never deleted, write-protected:**

- `checkpoints/v10_voco_l_ft_r3/epoch_009.pt` (V10 W1 base, val 0.7967)
- `checkpoints/v9_stage1/last.pt` (V9 Stage 1, val 0.8433 patch)
- `checkpoints/v9_stage2_v5/refiner_ep_009.pt` (V9 Stage 2 v5)
- `checkpoints/trissr_v9/last.pt` (TriSSR-V9)
- `checkpoints/pretrained/voco/VoComni_L.pt` (1.17 GB pretrained)
- `checkpoints/pretrained/voco/VoComni_H.pt` (4.65 GB pretrained)
- `checkpoints/pretrained/medsam2_hiera_tiny.safetensors` (MedSAM2)
- (When downloaded) `checkpoints/pretrained/suprem/supervised_suprem_swinunetr_2100.pth`
- (When downloaded) `checkpoints/pretrained/stunet/stunet_h_totalseg.pth`

**Convention**: every new training run writes to its own NEW directory under `checkpoints/organmoe_*/`. Never reuse experiment names.

---

## 9. Reproducibility commitments

Each training run saves into its own directory:

| Artifact | Filename |
|---|---|
| Config snapshot (resolved YAML) | `config_used.yaml` |
| Git commit hash | `git_commit.txt` |
| Random seed | `seed.txt` |
| Dataset split (file IDs per fold) | `split.json` |
| Per-epoch metrics | `metrics/epoch_NNN.json` and rolling `metrics.json` |
| Per-epoch checkpoint | `epoch_NNN.pt` (save every epoch, no exception) |
| Final TensorBoard scalars | `tb/` |

A reproducibility check at end of Phase V re-runs 1 eval from scratch using only the bundled artifacts.

---

## 10. Risk register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| OrganMoE-3D training collapses (router degeneration) | Medium | High | Phase I fallback to plain LoRA + class fix |
| TotalSeg pretrain hurts (label noise / domain shift) | Medium | Medium | Phase I gate: must improve over plain VoCo-L base |
| VoCo-H LoRA OOM at fs=96 (deferred from V10 plan) | Low | Medium | Deferred to optional Phase II only if SuPreM/MedNeXt under-perform |
| MedNeXt-L training too slow (~24h+) | Medium | Medium | Run in time-share; if not done by day 14, drop from ensemble |
| Reseg gate over-triggers | Medium | Low | Per-organ τ_o calibration on val; ablate vs no-reseg |
| Ensemble doesn't help (correlated errors) | Medium | Medium | Diverse backbones (CNN/ConvNeXt/Swin/SSM) by design; diversity loss as backup |
| Total time blows past 30 days | Medium | High | Hard exit gate at day 7; phase-gate freeze at day 21 if behind; commit to writing day 27 regardless |
| 2nd novelty (Phase III) doesn't work | Medium | Low | Phase III fallback: skip 2nd novelty, report Phase II + reseg only |
| AMOS22 test leaderboard submission portal issue | Low | Low | Report val numbers only with caveat; pursue submission post-deadline |

---

## 11. Out-of-scope (explicitly NOT doing)

- MRI segmentation (AMOS22 has MRI data, but our val protocol is CT-only — adding MRI doubles eval time without clear thesis benefit)
- Multi-modal fusion (CT + clinical metadata)
- New backbone architecture invention beyond OrganMoE adapter (5-7 days isn't enough to credibly invent)
- VoCo-H training-from-scratch (using pretrained weights only)
- Active learning / human-in-the-loop annotation
- Real-time inference optimization (compile/TensorRT etc — research code is fine)

---

## 12. Deliverables

| Deliverable | Phase | Format |
|---|---|---|
| OrganMoE-3D code module + unit tests | I | Python modules + smoke tests in repo |
| 4-backbone ensemble inference pipeline | II | `inference/ensemble.py` + config |
| Organ-centric reseg infrastructure | III | Models + uncertainty gate + per-organ U-Nets |
| 2nd novelty implementation | III | One of: distillation OR atlas-PE |
| 5-fold CV results | IV | `reports/organmoe_5fold/*.json` |
| Full ablation table | IV | `reports/organmoe_ablation/table.md` |
| AMOS22 test submission | IV | Submitted via Grand Challenge portal |
| Per-organ Dice / HD95 / NSD final report | IV | `reports/organmoe_final.json` |
| Method figures | V | `figures/` (pipeline, architecture, gate, ablation barplot) |
| Thesis chapter draft | V | `thesis/ch_organmoe.md` |
| Workshop or main-track paper draft | V | `paper/organmoe_paper.md` |

---

## 13. Open questions (will resolve during execution)

1. **TotalSeg licensing for thesis use** — CC-BY allows redistribution but thesis-citation form needs verification (resolve day 1)
2. **Best K for MoE experts** — start at K=8, may sweep {4, 8, 12} during Phase I if time permits (resolve day 4)
3. **2nd novelty pick** — depends on Phase I-II per-organ data (resolve day 18)
4. **Fold-split strategy for sex stratification** — need to extract sex labels from AMOS22 metadata; if not available, stratify by prostate-presence as a proxy (resolve day 22)
5. **Workshop vs main-track venue** — depends on final ensemble Dice (resolve day 28-29)

---

*End of design spec. Implementation plan to follow via writing-plans skill.*
