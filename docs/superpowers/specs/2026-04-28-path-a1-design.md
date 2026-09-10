# Path A1 — Big Solid Organs ODE-SAM Extension (2026-04-28)

> **Status:** APPROVED direction as of 2026-04-28. Replaces OrganMoE-3D Phase I as primary thesis path.
>
> **Tagline:** Lightweight Neural-ODE-augmented SAM for high-fidelity 3D big-solid-organ segmentation.

---

## 1. One-paragraph summary

Path A1 returns to the already-measured-publishable single-organ liver result (ODE-SAM V2: DSC 0.9358, HD95 0.56 mm, NSD 0.9678 at 21.2 M params) and **extends it to a curated set of big solid abdominal organs** (liver, spleen, right kidney, left kidney, optionally stomach). The architectural novelty — a bidirectional organ-conditioned Neural ODE for cross-slice feature evolution — is preserved unchanged. The thesis claim becomes "lightweight, ODE-conditioned segmentation for cross-slice-coherent solid organs," and the framing trades AMOS22-headline mean Dice for **HD95-grade boundary fidelity, parameter efficiency, and methodological clarity**.

This document captures every decision, table, research finding, and prior-version lesson that led to choosing Path A1 over Paths B (V9 + APAR) and C (OrganMoE Phase I + APAR).

---

## 2. The decision context

### 2.1 Honest probability table (the deciding factor)

| Outcome | Definition | Path A0 (liver only, current) | Path A1 (recommended) | Path B (V9 + APAR) | Path C (OrganMoE Phase I + APAR) |
|---|---|---:|---:|---:|---:|
| Mean DSC ≥ 0.94 | A1's headline | already 0.9358 | **70%** | n/a (different metric) | n/a |
| Mean DSC ≥ 0.91 | strong A1 outcome | already 0.9358 | **90%** | n/a | n/a |
| Mean 3D DSC ≥ 0.864 (AMOS22 SOTA) | the leaderboard win | n/a | n/a | **35%** | 30% |
| Mean 3D DSC ≥ 0.85 (MICCAI workshop floor) | conservative SOTA-adjacent | n/a | n/a | 58% | 50% |
| Workshop publication | MICCAI-W / MIDL / ML4H | 75-80% | **85%** | 70% | 60% |
| Main-track publication | MICCAI / IEEE TMI | 30-40% | **45%** | 35% | 30% |
| Valid masters thesis defended | committee accepts | 95% | **96%** | 92% | 80% |
| Compute risk (slip past 30 days) | schedule risk | low | low | medium | medium-high |

A1 is the **highest-expected-value path** when "valid thesis + ≥1 publication" is the actual goal.

### 2.2 What changed our mind

We went through 6+ multi-organ attempts (V3, V4, V7, V8, V9 cascade refiner, V10, OrganMoE-3D Phase I) since leaving the working ODE-SAM V2 liver result. Net measured 3D Dice today: **0.7203 mean** on 15 organs — *below* the original V2 single-organ liver number on the other 14 organs combined. The cause was always ambition, not architecture inadequacy. The math of cumulative pivot risk says we should now stop pivoting and convert the strongest measured result into a thesis.

---

## 3. Organ-suitability for the ODE mechanism (the math behind A1)

### 3.1 Why ODE helps where it does

The bidirectional organ-conditioned ODE evolves features `h(t)` along the slice axis with `dh/dt = f_θ(h, t) + MLP(organ_embed)`. Heun integration over D slices produces `D` interpolated features that fuse via a zero-init merge head. The mechanism is therefore a **cross-slice integrator**, and its empirical Dice gain is proportional to the number of slices over which it can integrate.

| Organ property | ODE behaviour | Dice gain |
|---|---|---|
| Many slices spanned (≥30) | many integration steps; signal compounds | **Large** |
| Smooth slice-to-slice shape change | Heun's local-linear assumption holds | Large |
| Solid contiguous mask | fwd + bwd trajectories converge | Large |
| Few slices (<10) | reduces to per-slice 2D | Marginal |
| Tortuous/branching topology | non-monotone trajectory; ODE struggles | Negative |
| Highly variable shape (gas, contents) | smoothness assumption violated | Marginal/negative |

### 3.2 AMOS22 organs ranked by ODE suitability

| Tier | Organ (AMOS id) | Slice span | ODE helps? | Realistic ceiling with ODE-SAM-V2 design |
|---|---|---:|:---:|---:|
| **Big-solid (A1's target)** | liver (6) | 60–120 | ✅ proven | 0.94–0.96 (already 0.9358) |
| | spleen (1) | 30–60 | ✅ | 0.94–0.96 |
| | right kidney (2) | 25–45 | ✅ | 0.94–0.95 |
| | left kidney (3) | 25–45 | ✅ | 0.94–0.95 |
| | stomach (7) | 30–60 | ✅ moderate (deformable) | 0.88–0.92 |
| | bladder (14) | 15–30 | ✅ moderate (variable filling) | 0.85–0.92 |
| **Tubular-continuous (A2 extension)** | aorta (8) | full FOV | ✅ + clDice | 0.93–0.95 |
| | IVC (9) | full FOV | ✅ + clDice | 0.85–0.90 |
| | esophagus (5) | 25–50 | 🟡 needs clDice + boundary | 0.75–0.85 |
| **Borderline** | pancreas (10) | 20–40 | 🟡 deformable + low contrast | 0.78–0.85 |
| | gallbladder (4) | 5–15 | ⚠️ too few slices | 0.75–0.85 |
| | duodenum (13) | 15–35 | 🟡 tortuous topology hurts ODE | 0.65–0.78 |
| **Small (out of scope for A1)** | right adrenal (11) | 4–10 | ❌ too few slices | 0.65–0.78 |
| | left adrenal (12) | 4–10 | ❌ too few slices | 0.65–0.78 |
| | prostate/uterus (15) | 8–20 | ❌ pelvic prior > ODE | 0.55–0.80 |

### 3.3 What's already in the repo, used vs unused

| Direction | Component | Path |
|---|---|---|
| ODE on big solid organs | bidirectional Neural ODE, Heun integrator, organ-conditioned bias | `models/ode_cross_slice.py`, `models/auto_ode_sam.py` |
| Stage-2 multiscale encoder | TinyViT-21M Stage-2 with skip from Stage-1 | `models/multiscale_encoder.py` |
| OrganQueryDecoder | 15 learned queries + HQ token + TwoWayTransformer | `models/organ_query_decoder.py` |
| Tubular extension (clDice) | soft skeletonisation + topology Dice | `models/topology.py` ✅ implemented, never used in published run |
| Boundary DDPM refiner | tiny UNet on uncertainty band, 5–10 steps | `models/boundary_ddpm.py` ✅ implemented, never used |
| Organ-conditioned ODE bias | bias_mlp(organ_embed) added to dh/dt | `models/ode_cross_slice.py` ✅ |
| Anatomy graph decoder (multi-scale FPN) | 3-scale deep-sup, anatomy graph attention | `models/anatomy_graph_decoder.py` ✅ available, A1 may not need it |
| FlowMatchedODE (org-conditioned + flow-matching + XSC pairs) | training-time pair generator | `models/flow_cross_slice.py` ✅ available for ablation |

---

## 4. Path A1 architecture (final)

### 4.1 Forward pass

```
3D volume input (B, D=8 to 16, 1, H=256, W=256)        [extend to D=24 if VRAM allows]
            │
   COMPONENT 1: TinyViT-21M Stage-2 encoder           ← 13.4 M params
       main: (B*D, 256, 16, 16)
       skip: (B*D, 64,  32, 32)
            │
   COMPONENT 2: PFESA spectral enhancement            ← 0 params (or PFESAPlus 3-param)
            │
   COMPONENT 3: Bidirectional Organ-Conditioned ODE   ← 225 K params
       dh/dt = f_θ(h, t) + MLP(organ_embed[k])
       Forward + Backward + zero-init merge
            │
       feat3d (B, D, 256, 16, 16)
            │
   COMPONENT 4: OrganQueryDecoder                     ← ~7.5 M params
       K=4 (or 5) learned queries × HQ token
       TwoWayTransformer (depth 2, heads 8)
       FiLM modality-style conditioning (cite MCP-MedSAM)
            │
   COMPONENT 5: MaskDecoder                           ← ~200 K params
       Stage-1 skip injection + 4× upsample
            │
   K mask logits per slice (B, K, H, W)               ← K=4 organs
            │
   3D sliding-window argmax over volume               ← inference time
```

### 4.2 K = 4 (or 5) organs, NOT 15

Path A1 trains on a **restricted label set**:
- A1-core: K=4 → liver, spleen, right kidney, left kidney
- A1-extended: K=5 → + stomach (only if first 4 hit ≥ 0.94 mean)

The remaining 11 organs are **left out of training**; they are mentioned in the discussion section as out-of-scope, with a documented argument that the ODE mechanism does not apply to organs with <15 slice spans.

### 4.3 Loss

```
L = Σ_k w_k · (Dice(pred_k, gt_k) + 0.5 · Focal(pred_k, gt_k))
  + 0.1 · L_deepsup           (deep-sup at ODE midpoint)
  + 0.5 · L_flow              (flow-matching, training only — see flow_cross_slice.py)
  + 0.0 · L_clDice            (kept at 0 for A1-core; enable if A2)
```

`w_k` per-organ weights initially uniform (1.0); reweight if any organ stalls at ep 5.

### 4.4 Why this is publishable

1. **Already measured baseline** for liver alone: DSC 0.9358, HD95 0.56 mm at 21.2 M params.
2. **Genuinely novel mechanism** confirmed by April 2026 web search: no prior work uses bidirectional organ-conditioned Neural ODEs for cross-slice feature evolution in medical segmentation.
3. **Clean ablation story**: ODE-on / fwd-only / bwd-only / no-organ-conditioning / no-Stage-2-tap. Each axis reports independent Dice + HD95 delta.
4. **Boundary fidelity is a publishable headline on its own** — most liver papers report HD95 = 1–3 mm; ODE-SAM V2 measured 0.56 mm.

---

## 5. Prior-version postmortem (lessons folded into A1)

### 5.1 Timeline of pivots

| Date | Decision | Stated reason | Honest reading |
|---|---|---|---|
| ~Feb 2026 | LiteSAM-3D V2 Phase 0 → 0.9358 / 0.56 mm liver | "Phase 0 complete, scale up" | Result was thesis-grade. Should have stopped. |
| ~Feb 2026 | Pivot to V3 multi-organ at 256 px | "Bigger thesis story" | Ambition, not technical necessity |
| Apr 2026 | V3 abandoned (val 0.1427) | "Architecture miswired" | Extending ODE design to 15 organs broke decoder |
| Apr 2026 | V4 OrganFlow-SAM2 designed | "Re-architect" | Spec only, never trained |
| Apr 2026 | V7 / V8 OrganFlow-SAM2 V8 | "Add DINOv2 + BiomedCLIP" | Refiner DSC 0.36; never beat V9 alone |
| Apr 2026 | V9 SwinUNETR proposer + cascade refiner | "Switch backbone" | Stage-1 0.84 patch (real 3D ~0.74-0.80); cascade refiner failed |
| Apr 2026 | V10 VoCo-L full-FT | "Stronger pretrained backbone" | queued, not yet run |
| Apr 2026 | OrganMoE-3D Phase I | "Add MoE for novelty" | Mean 3D Dice 0.7203 — net regression |

### 5.2 Forces that pushed us away from A and lessons

| Force | What it sounded like | What it actually meant | A1 mitigation |
|---|---|---|---|
| Ambition framing | "single-organ feels small" | Untested assumption | Frame around HD95 + efficiency + ODE mechanism, not organ count |
| Benchmark gravity | "AMOS22 is the benchmark to beat" | AMOS22 is *a* benchmark, not the only one | Use AMOS22 as one of several test sets (LiTS + MSD-Liver + AMOS22-subset) |
| Sunk cost | "we already invested 3 weeks in V3" | Forward decisions only | Mark all multi-organ work as bounded prior; don't try to salvage it into A1 |
| Architectural optimism | "next version will fix prior failure" | Six pivots, none recovered to original quality | Lock A1 direction for 30 days; no mid-flight pivots |
| Patch-eval vs 3D-eval confusion | "V9 hit 0.84" | Real 3D was ~0.74-0.80 | All A1 numbers reported at proper 3D sliding-window |

---

## 6. Research findings (AMOS22 SOTA + MCP-MedSAM lessons)

### 6.1 AMOS22 SOTA landscape (as of 2026-04-28)

| Method | AMOS22 mean DSC | Year | What carried the score |
|---|---:|:---:|---|
| nnU-Net (default 3D-fullres) | 0.889 | 2022 | Deep-sup + heavy aug + 5-fold ensemble |
| SwinUNETR-V2 | 0.864–0.882 | 2023 | Stagewise convs in Swin stages |
| MedNeXt | ~0.89 | 2023 | ConvNeXt + scale-up + UpKern |
| nnWNet | 0.864 | CVPR 2025 | W-shaped + LSB + GSB |
| AutoProSAM | 0.887 | WACV 2025 | 3D UNet auto-prompt + SAM |
| Self-Prompt SAM | 0.901 | Feb 2025 | Multi-scale prompt generator |
| MaskSAM | 0.905 | ICCV 2025 | SAM + end-to-end auto prompt + DETR |
| VoComni | ~0.886 | TPAMI 2025 | Large-scale CT pretraining (160K vols) |
| Primus-L | competitive with nnU-Net | Mar 2025 | Pure-Transformer + RoPE-3D + SwiGLU + LayerScale |

**Lesson:** SOTA above nnU-Net's 0.889 came from (a) more pretraining data or (b) better per-organ adaptive loss + boundary supervision. Architectural cleverness alone is a +0.01 to +0.02 lever.

### 6.2 MCP-MedSAM honest framing

MCP-MedSAM is **NOT AMOS22 SOTA**. It is the SOTA of the *CVPR 2024 "Segment Anything in Medical Images on Laptop"* challenge (1-day single-A100 training, multi-modality, edge deployment). Reported number: 87.50% mean across 9 modalities with box prompts, CT subset 90.02% averaged across many CT tasks.

#### How MCP-MedSAM achieved its laptop-track SOTA (mathematical decomposition)

| Component | Mechanism | Dice contribution (their ablation) |
|---|---|---:|
| Modality prompt | Learnable embedding + frozen PubMedCLIP text + MLP fusion | +2.1 |
| FiLM in decoder | γ, β = MLP(m_emb); feat ← γ ⊙ LN(feat) + β; identity init | +1.4 |
| Content prompt | sparse (frozen CLIP-img on RoI) + dense (small ResNet) + contrastive align | +1.8 |
| Modality-balanced sampler | P(x|m) = 1/(N_mod · |D_m|) | **+3.7 (largest)** |
| Modality classifier aux head | CE on pooled decoder feat, weight 0.05 | +0.4 |

**Lesson:** ~40% of the gain is architectural, ~50% is data engineering, ~10% is auxiliary regularisation. A1 inherits the FiLM design (already in `mask_decoder.py`) and the lesson that *balanced sampling matters* — relevant if A1 ever extends to multi-dataset (LiTS + MSD-Liver + AMOS22-subset cross-training).

#### What does NOT transfer to A1

- Box prompts: A1 is fully automatic.
- 2D pipeline: A1 is 3D.
- CLIP image content prompt: requires box.

---

## 7. Per-version factual scorecard

| Version | Architecture core | Best metric | Honest 3D AMOS22 Dice | Params | Status | Reproducible? |
|---|---|---|---:|---:|---|:---:|
| V1 (DA-ISA) | TinyViT + ISA | invalid | n/a | ~28M | killed | ❌ |
| V2 | TinyViT Stage-3 + ISA | 0.1074 (2D, 256px) | <0.5 | ~28M | killed | ⚠️ |
| V3 Auto-ODE-SAM | Stage-2 + ODE + 15 organs | 0.1427 peak (2D proxy) | <0.50 | 21.2M | abandoned | ✅ ckpts frozen |
| **ODE-SAM V2 (single-organ liver)** | **same backbone, liver-only** | **0.9358 / 0.56 mm / 0.9678** | n/a (1 organ) | **21.2M** | **frozen, A1's foundation** | ✅ |
| V4 OrganFlow-SAM2 | MedSAM2 + LoRA + ODE + AnatomyGraph | designed only | never trained | ~97M | spec | ✅ spec |
| V7/V8 OrganFlow-SAM2 V8 | + DINOv2 + BiomedCLIP | refiner <0.36 | unmeasured | ~110M | shelved | ⚠️ |
| V9 Stage-1 proposer | SwinUNETR-V2 fs=48 | 0.8433 patch-eval | likely 0.74–0.80 | ~64M | frozen | ✅ |
| V9 Stage-2 refiner | OrganFlow refiner | 0.12 alone, 0.70 fused | cascade hurts | +~15M | shelved | ✅ |
| V10 VoCo-L FT | VoCo-L fs=96 | unmeasured | not run | ~330M | queued | ✅ |
| OrganMoE-3D Phase I | VoCo-L + MoE-LoRA top-2 | mean 3D 0.7203 | 0.7203 measured | ~340M | last.pt ep10 | ✅ |

---

## 8. Decision rules locked for A1

| Item | Locked answer |
|---|---|
| Path | **A1** (liver + spleen + R-kid + L-kid; +stomach if A1-core ≥ 0.94 mean) |
| Lock direction for 30 days | **yes** |
| ODE architecture changes | **none** — preserve V2 design exactly |
| Number of organs in headline | **K=4** (K=5 if stretch) |
| Datasets for training | **AMOS22 train split, restricted to A1 organs** |
| Datasets for testing | **AMOS22 val + LiTS + MSD-Liver + MSD-Spleen** (cross-dataset generalisation table) |
| Headline metric | **HD95 + DSC + parameter count** (multi-axis, not just DSC) |
| Acceptable thesis floor | **mean DSC ≥ 0.91 across A1 organs** |
| Acceptable publication floor | **mean DSC ≥ 0.93 + HD95 ≤ 1.0 mm + ablation table** |
| Decision review cadence | **3 checkpoints — day 7 / 22 / 28** |
| Compute reservation | full 24/7 4090 wall-time |

---

## 9. Sacred-checkpoint protection (carried forward)

These ckpts must NEVER be overwritten. A1 writes only to NEW dirs.

```
v9_stage1/last.pt
v9_stage1_ft/last.pt
trissr_v9/last.pt
v10_voco_l_ft_r3/epoch_009.pt
voco_totalseg_pretrain/epoch_009.pt
organmoe_phase_i/last.pt
ODE-SAM V2 liver checkpoint (in C:\Users\Raywa\Desktop\VoluFormer3D\)
```

A1 output dirs:
```
checkpoints/path_a1_core/         (K=4, primary run)
checkpoints/path_a1_extended/     (K=5 if triggered)
checkpoints/path_a1_ablations/    (ODE-off, fwd-only, bwd-only, no-organ-cond)
logs/path_a1_*
reports/path_a1_*
configs/path_a1_*.yaml
```

---

## 10. Acceptance criteria

A1 is "done" (eligible for thesis writing + paper draft) when ALL of:

- [ ] ODE-SAM V2 liver result reproduced from frozen ckpt with proper 3D sliding-window eval (not patch-eval).
- [ ] A1-core trained on K=4 to convergence with measured 3D mean DSC ≥ 0.91.
- [ ] Ablation table populated (ODE-on / fwd-only / bwd-only / no-organ-cond / no-Stage-2-tap; each ≥ 1 ep).
- [ ] Cross-dataset generalisation table: liver on LiTS + MSD-Liver; spleen on MSD-Spleen; A1 mean on AMOS22 val.
- [ ] Per-organ HD95 ≤ 2 mm for big-solid; ≤ 5 mm for stomach.
- [ ] Full reproducibility bundle: config + seed + exact split + final ckpts + eval script.
- [ ] Discussion section outline drafted: covers V3 multi-organ failure as documented limitation.
- [ ] At least one ablation axis shows a measurable Dice delta (≥ 0.005) attributable to the ODE mechanism.

---

## See also

- Implementation plan: `2026-04-28-path-a1-implementation-plan.md`
- ODE novelty review: archived project notes outside this repository.
- Prior MoE phase notes: archived project notes outside this repository.
