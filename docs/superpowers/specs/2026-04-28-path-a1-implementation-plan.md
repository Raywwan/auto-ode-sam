# Path A1 — Implementation Plan (30-Day Execution)

> **Companion docs:**
> - Design: `2026-04-28-path-a1-design.md`
> - Research: `2026-04-28-research-findings.md`

**Mission**: Extend ODE-SAM V2 from single-organ liver (DSC 0.9358) to **K=4 big solid organs** (liver, spleen, R-kidney, L-kidney) on AMOS22, then writeup. Stretch: K=5 with stomach.

**Why A1 not A0 / B / C**: A1 has the highest expected value of any path:
- 96% probability of thesis defense
- 85% probability of workshop publication
- 70% probability of mean DSC ≥ 0.94 on the 4 organs
- Reuses already-working liver checkpoint → low risk
- Novel ODE contribution survives intact

---

## High-Level Calendar

| Phase | Days | Goal | Gate |
|-------|------|------|------|
| 0. Sanity reload | 1–2 | Reproduce V2 liver 0.9358 | ≥ 0.93 |
| 1. Multi-organ extend | 3–5 | Add spleen + L/R kidneys to head | Forward pass clean |
| 2. Train | 6–15 | 10–14 epochs FT on AMOS22 | mean DSC ≥ 0.91 by ep 10 |
| 3. Ablation | 16–22 | Each component contribution | clean numbers, no regressions |
| 4. Cross-dataset | 23–26 | BTCV / WORD / TotalSeg subset | mean DSC ≥ 0.90 transfer |
| 5. Writing | 27–30 | Workshop paper draft | submission-ready |

**Hard rule**: gate-fail at Phase 2 (ep 10 < 0.91) → **debug, don't pivot**.

---

## Phase 0 — Sanity Reload (Days 1–2)

### Day 1: Resurrect V2 environment

**Objective**: Confirm V2 liver checkpoint still loads and reproduces 0.9358.

#### Steps

1. **Locate the V2 liver checkpoint** (do NOT overwrite):
   - Expected path: `checkpoints/auto_ode_sam_v3/best.pt` or `v9_stage1/last.pt` family.
   - Verify with `git log` + `MASTER_PROJECT_LOG.md` what produced 0.9358.

2. **Sanity script** — `scripts/sanity_reload_a1.py`:
   - Load V2 liver checkpoint
   - Run on AMOS22 val (single-organ liver only)
   - Expect DSC 0.93 ± 0.01

3. **Schema audit** (per `feedback_preflight_long_runs.md` rule):
   - Print state_dict keys; compare against `models/auto_ode_sam.py` current `__init__`
   - If `>5%` keys missing, **STOP** and reconcile

4. **Reproducibility test**: re-run same eval twice, expect identical numbers.

#### Acceptance

- ✅ V2 liver-only mean DSC ≥ 0.93 (within ±0.01 of 0.9358)
- ✅ HD95 ≤ 0.6 mm
- ✅ All schema keys load with `<5%` missing

### Day 2: Add eval harness for K=4

**Objective**: Build the evaluation harness we'll use for the next 28 days.

#### Steps

1. **Patch eval driver** to handle K=4 organs:
   - `eval/eval_3d_sliding_window.py` — accept `--organs liver,spleen,r_kidney,l_kidney`
   - 96³ patches, stride 48, 4-way TTA, connected-component cleanup (per Phase I tooling)

2. **Build comparison table template**:
   - Per-organ DSC, HD95, NSD
   - Per-method comparison column (V2-liver-only baseline, A1 multi-organ, MaskSAM, nnU-Net, nnWNet)

3. **Sanity check**: run K=4 eval on V2 checkpoint with random init for spleen/kidney heads → expect liver ≈ 0.93, others ≈ 0.

#### Acceptance

- ✅ Eval finishes in < 3 hr on val set (30 vols)
- ✅ Liver still ≥ 0.93
- ✅ No tensor-shape errors in K=4 path

---

## Phase 1 — Multi-Organ Extension (Days 3–5)

### Day 3: Architecture surgery

**Objective**: Extend the model from K=1 → K=4 without breaking what worked.

#### File-level changes

1. **`models/auto_ode_sam.py`**:
   - Confirm `n_organs=15` config still supported (it should — V9 already used 15-class)
   - Verify `OrganQueryDecoder` initializes 15 query embeddings; just train 4 supervised
   - **Loss masking**: only compute DSC + Tversky + focal on classes [liver, spleen, r_kidney, l_kidney]; ignore others

2. **`models/ode_cross_slice.py`**:
   - Confirm organ-embedding tensor shape `[15, organ_emb_dim]` already supports K=4
   - No surgery required — bidirectional ODE is organ-agnostic by design

3. **`training/train_v9.py` (or new `train_a1.py`)**:
   - **Class subset**: filter labels to keep only {liver=6, spleen=1, r_kidney=2, l_kidney=3} (use AMOS22 official ID map)
   - **Loss**: DSC + Tversky (α=0.3, β=0.7 for kidney imbalance) + focal γ=2.0
   - **Sampler**: keep balanced batch sampler; rare-organ list = empty (all 4 are common)
   - **Deep supervision**: enable on decoder3/4/5 with weight 0.5 → 0.1

#### Acceptance

- ✅ Forward pass produces `[B, 4, D, H, W]` logits
- ✅ Loss converges to a finite number on 1 batch
- ✅ Backward pass updates `ode.f_theta` weights (verify with `param.grad.norm()`)

### Day 4: Config + dry-run

#### Config — `configs/path_a1_big_solid.yaml` (NEW)

```yaml
experiment:
  name: "path_a1_big_solid"
  seed: 42
  output_dir: "checkpoints/path_a1_big_solid"
  log_dir: "logs/path_a1_big_solid"

model:
  architecture: "auto_ode_sam_v3"
  n_organs: 4
  organs_supervised: [liver, spleen, right_kidney, left_kidney]
  encoder:
    type: "tinyvit_21m_stage2"
    pretrained: true
    freeze: false        # full FT — V2 trained encoder, we continue
  ode_cross_slice:
    enabled: true
    direction: "bidirectional"
    integrator: "heun"
    n_freqs: 6
    substeps: 4
    organ_emb_dim: 32
  pfesa:
    enabled: true
  organ_query_decoder:
    n_queries: 4
    transformer_depth: 4
  mask_decoder: { type: "sam_style" }

data:
  dataset: "amos22_v9"
  data_root: "C:/Users/Raywa/Desktop/LiteSAM3D/data/amos22"
  classes_supervised: [6, 1, 2, 3]   # liver, spleen, r_kidney, l_kidney AMOS22 IDs
  ignore_others: true
  img_size: 320
  depth: 8
  pos_frac: 0.7   # bias toward slabs containing target organs
  neg_frac: 0.2
  mix_frac: 0.1

training:
  epochs: 14
  batch_size: 2
  optimizer: { name: "adamw", lr: 2e-5, weight_decay: 0.05 }
  scheduler: { name: "cosine", warmup_epochs: 2 }
  amp: true
  amp_dtype: "bf16"
  grad_clip: 1.0
  loss:
    dice_weight: 1.0
    tversky_weight: 1.0
    tversky_alpha: 0.3
    tversky_beta: 0.7
    focal_weight: 0.5
    focal_gamma: 2.0
    deep_sup_weight_start: 0.5
    deep_sup_weight_end: 0.1
  warmstart:
    from: "checkpoints/auto_ode_sam_v3/best.pt"
    missing_key_frac_max: 0.05

eval:
  every_n_epochs: 1
  patch_size: 96
  stride: 48
  tta: true
  connected_component: true
  report_per_organ: true

safety:
  abort_if_val_drops: 0.03
  max_peak_vram_gb: 23
```

#### Dry-run checklist (per `feedback_preflight_long_runs.md`)

- [ ] Schema check — print all weight keys; compare against config
- [ ] Label-map audit — confirm AMOS22 IDs {1,2,3,6} = {spleen, r_kidney, l_kidney, liver}
- [ ] Config sanity — bf16, batch_size 2, fits in 23 GB
- [ ] 1-step forward+backward dry run
- [ ] First-epoch sentinel: log loss every 50 batches; abort if `nan` or > 10× ep1 mean

### Day 5: 1-epoch smoke test

**Objective**: Confirm training pipeline is sane before committing to the long run.

#### Steps

1. Run **1 epoch** with above config.
2. Verify:
   - Loss decreased monotonically
   - Liver DSC at val ≥ 0.85 (regression check)
   - Spleen DSC at val ≥ 0.20 (organ is starting to learn)
   - Kidney L/R DSC at val ≥ 0.30 (kidneys are anatomically large/easy)
   - VRAM peak < 23 GB
   - Epoch wall-clock ~ 13–15 min (matches Phase I timing)

#### Decision gate

- ✅ All checks pass → continue to Phase 2
- ❌ Any check fails → **diagnose, don't move on** (per `feedback_v9_alignment_lessons.md`)

---

## Phase 2 — Training (Days 6–15)

### Schedule

- 14 epochs total (incl. the smoke epoch)
- ~ 13 min/epoch × 14 = ~3 hr/epoch including val every epoch ≈ ~7–8 hr/epoch w/ 3D val
- **Wall-clock estimate**: ~5–6 days for 14 epochs with full 3D eval each
- **Strategy**: cheap patch-eval every epoch + full 3D eval at ep 4, 8, 12, 14

### Per-epoch ritual (per `feedback_session_cadence.md`)

Every 1 hour during a run:
1. Check `train_loss` is decreasing
2. Check VRAM peak (no OOM creep)
3. Update memory `project_path_a1_progress.md` with running mean DSC

### Mid-run gates

| Epoch | Mean DSC (4 organs, 3D eval) | Action if missed |
|------:|----------------------------:|------------------|
| 4 | ≥ 0.88 | continue, monitor |
| 8 | ≥ 0.91 | **GATE A1-1** → if missed, debug LR / loss / data |
| 12 | ≥ 0.93 | **GATE A1-2** → if missed, ablate one knob (LR↑, focal γ↑, or longer warmup) |
| 14 | ≥ 0.94 stretch / ≥ 0.91 floor | done |

### Acceptance Phase 2

- ✅ Mean 3D DSC ≥ 0.91 (floor for paper)
- ✅ Per-organ liver ≥ 0.94 (must not regress from V2)
- ✅ Per-organ kidneys ≥ 0.93
- ✅ Per-organ spleen ≥ 0.91
- ✅ Mean HD95 ≤ 1.5 mm
- ✅ MoE / OrganQuery active (verify via attention-map dump)

---

## Phase 3 — Ablation Study (Days 16–22)

> **Why ablate**: workshop reviewers will ask. Pre-empt.

### Ablation grid

| Variant | Drop / Replace | Train epochs | Expected Δ DSC | Why |
|---------|----------------|--------------|---------------:|-----|
| A1-base | (full A1) | 14 | reference | baseline |
| A1-no-ODE | drop bidirectional ODE; replace with 1×1 conv | 14 | **−1.5 to −2.5** | proves ODE contribution |
| A1-no-OQD | drop OrganQueryDecoder; use plain seg head | 14 | −0.5 to −1.0 | proves query design contribution |
| A1-no-PFESA | drop PFESA module | 14 | −0.3 to −0.7 | confirms residual contribution |
| A1-fwd-only | unidirectional ODE (no backward) | 14 | −0.3 to −0.6 | bidirectional vs unidir |
| A1-Euler | swap Heun → Euler | 14 | −0.1 to −0.3 | integrator choice |
| A1-no-deep-sup | turn off aux heads | 14 | −0.2 to −0.5 | confirms deep sup |

**Total compute**: 7 variants × 14 ep × ~13 min = **~21 hours of training** for ablation. Schedule overnight on Days 16–22.

### Acceptance

- ✅ Each ablation finishes (no nans / no OOM)
- ✅ A1-no-ODE shows ≥ 1.0 DSC drop (validates ODE contribution)
- ✅ All numbers logged in `results/a1_ablation_table.csv`

---

## Phase 4 — Cross-Dataset Transfer (Days 23–26)

> **Goal**: Demonstrate transfer to BTCV, WORD subset, TotalSegmentator subset (4-organ subset only).

### Steps

#### Day 23: Eval on BTCV (zero-shot)
- Use same A1 checkpoint
- BTCV has liver, spleen, kidneys at IDs {6,1,2,3 mapped from BTCV's {6,1,2,3}}
- Expect mean DSC ≥ 0.88 (cross-domain, zero-shot)

#### Day 24: Eval on WORD subset (zero-shot)
- WORD also has these 4 organs
- Expect ≥ 0.88

#### Day 25: Eval on TotalSegmentator subset (zero-shot)
- 100 vol subset, 4-organ subset only
- Expect ≥ 0.90

#### Day 26: Light FT on BTCV (10 epochs)
- Fine-tune for 10 epochs on BTCV train split
- Compare zero-shot vs fine-tuned
- Expect FT ≥ 0.92 (within-domain on BTCV)

### Acceptance

- ✅ Zero-shot transfer ≥ 0.88 on at least 2 of {BTCV, WORD, TotalSeg}
- ✅ FT on BTCV reaches ≥ 0.92

---

## Phase 5 — Writing (Days 27–30)

> **Format**: Workshop paper, 4–6 pages. Target venue: MICCAI 2026 workshop on multi-organ segmentation, or ISBI 2026.

### Day 27: Outline + abstract

- Title: *"Bidirectional Organ-Conditioned Neural ODE for Multi-Organ 3D Segmentation: A Liver-and-Friends Study"*
- Sections: Intro · Related Work · Method · Experiments · Discussion
- Abstract draft (200 words)

### Day 28: Method + Experiments

- Method: copy from `2026-04-28-path-a1-design.md` § Architecture
- Equations: dh/dt = f_θ(h, t) + MLP(organ_embed[k]); Heun integrator
- Tables: AMOS22 results, ablation, cross-dataset

### Day 29: Discussion + Limitations + Figures

- Discussion: compare against MaskSAM/Self-Prompt-SAM honestly (we don't beat mean SOTA on 15 organs; we present a **multi-organ subset SOTA** with a novel mechanism that's analyzable)
- Limitations: only big-solid organs; small organs not addressed; cross-domain not fully evaluated
- Figures:
  - Architecture diagram
  - ODE-trajectory visualization (organ embedding flow over slices)
  - Per-organ DSC bar chart
  - Ablation table

### Day 30: Final pass + submission

- Spell-check, citations, typos
- Re-run all numbers in tables once more for sanity
- Create supplementary: training curves, per-volume breakdowns

---

## Sacred Checkpoint Protection

**MUST NOT BE OVERWRITTEN** during A1:

- `checkpoints/v9_stage1/last.pt`
- `checkpoints/v9_stage1_ft/last.pt`
- `checkpoints/trissr_v9/last.pt`
- `checkpoints/v10_voco_l_ft_r3/epoch_009.pt`
- `checkpoints/voco_totalseg_pretrain/epoch_009.pt`
- `checkpoints/voco_totalseg_pretrain/best.pt`
- `checkpoints/organmoe_phase_i/last.pt`
- `checkpoints/auto_ode_sam_v3/best.pt` ← V2 liver — A1's warmstart source

A1 writes ONLY to `checkpoints/path_a1_big_solid/`.

---

## Daily Memory Update Protocol

> Per `feedback_session_cadence.md`: update memory and master log every ~1 hr during runs.

Each day:
1. Edit `project_path_a1_progress.md` with: today's date, what ran, mean DSC, any issues, tomorrow's plan
2. Append to `MASTER_PROJECT_LOG.md` with: per-epoch DSC numbers, decisions, file diffs
3. Don't batch — write as it happens

---

## Failure Modes & Pre-Planned Responses

| Failure | Trigger | Response |
|---------|---------|----------|
| Liver regresses from 0.93 → < 0.92 in early epochs | end of ep 2 | LR too high — drop to 1e-5, restart from V2 ckpt |
| Spleen stuck below 0.50 by ep 4 | bad sampling? | check pos_frac for spleen-containing slabs; raise pos_frac to 0.8 |
| OOM on 4090 | bf16 + batch_size 2 should fit | drop to batch_size 1 + grad accumulate ×2 |
| Bidirectional ODE training unstable | gradient explosion | reduce `n_freqs` 6 → 4; raise grad_clip to 0.5 |
| Mean DSC ep 14 < 0.91 | gate failure | **DEBUG, do not pivot.** Per the postmortem rule. Likely culprit: data-aug too aggressive or class imbalance. Rerun with simpler config. |

---

## Acceptance Criteria for "Path A1 = Done"

| Criterion | Threshold | Reported in |
|-----------|----------:|-------------|
| Mean DSC (4 organs, 3D eval, AMOS22 val) | ≥ 0.91 | Phase 2 |
| Liver DSC | ≥ 0.94 | Phase 2 |
| Mean HD95 | ≤ 1.5 mm | Phase 2 |
| Ablation: A1-no-ODE drop | ≥ 1.0 DSC | Phase 3 |
| Cross-dataset zero-shot | ≥ 0.88 on ≥ 2 datasets | Phase 4 |
| Workshop paper draft | submission-ready | Phase 5 |
| Thesis chapter draft | 1 chapter complete | Phase 5 |

When ALL above check, A1 is complete and ready for submission.

---

## Anti-Pivot Rule (LOCKED)

> Per `feedback_voluformer_experimentation.md` and the prior-versions postmortem:
>
> **A1 will not be abandoned mid-run for a "better idea".** If A1 fails Phase 2 gate, the response is to **debug A1**, not jump to Path B/C. We have already pivoted 6 times (V3→V4→V7→V8→V9→V10→OrganMoE) and each pivot cost ≥ 5 days of compute. A1 is the convergence path.
>
> Only acceptable pivot: A1 fails Phase 2 gate by ≥ 0.10 DSC after 2 debug attempts. Even then, fall back to **A0** (single-organ liver writeup with clean ablation) — that path has 99% thesis-defense probability.

---

**End of Path A1 implementation plan.**
