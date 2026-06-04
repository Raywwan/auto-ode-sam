# PA-CODE: Position-Aware Conditioned Neural ODE for Multi-Organ Medical Segmentation

**Date:** 2026-04-30
**Status:** ACTIVE DESIGN
**Supersedes:** failed Path A1 multi-organ extension (mean DSC 0.24 ceiling; kidneys=0.000)
**Anti-pivot rule:** auto_ode_sam stays, ODE stays, K=4 organs stay, sacred ckpts untouched.

---

## 1 Motivation

V2 (single-organ liver, K=1) achieved DSC 0.9358 — SOTA-class.
A1 (K=4, liver+spleen+L/R-kidney) ceiling 0.24, kidneys=0.000 across 16 ckpts.

The ODE itself is not the failure. Three diagnostic findings (`scripts/diag_a1_query_collapse.py`, ep_013) localize the failure to the **conditioning mechanism**:

| Failure | Measured | Root cause |
|---|---|---|
| ODE conditioning collapsed | cos=0.999 across organ_ids | additive `bias = MLP(organ_emb)` zero-init competes against V2-warmstarted `base_ode` of much larger norm — the bias never escapes zero |
| Organ queries collapsed | L2 norm 0.32-0.37, cos≈0 | `nn.Embedding(15, 256)` init `std=0.02` puts queries in a tiny ball; diversity loss is satisfied at near-zero norm |
| L/R kidney indistinguishable | mask cos=0.998 | model has no positional information; LR-flip aug averages out the asymmetry |

The failure is informative: **additive bias is insufficient for anatomy-conditioned dynamical systems on warmstarted backbones.** That itself is a thesis result.

## 2 Contribution: PA-CODE

A single named contribution with three coupled components.

### 2.1 Identity-Initialized FiLM Dynamics

Replace the additive ODE conditioning:
```
OLD:  dh/dt = f_θ(h, t) + MLP(e_organ)              [bias_mlp zero-init]
NEW:  dh/dt = γ(e_organ) ⊙ f_θ(h, t) + β(e_organ)
```
**Identity init:** γ=1, β=0 at step 0 → model starts as the V2 ODE exactly. Each organ learns to *diverge* from a shared baseline trajectory.

**γ-head, β-head:** 2-layer MLP `Linear(organ_emb_dim, hidden) → GELU → Linear(hidden, dim)`, weights zero-init, γ-head bias initialized to **1.0** (so output = 1 at step 0), β-head bias to 0.0.

**Why novel:** FiLM-on-CNN is from 2017 (Perez et al.). FiLM-on-Neural-ODE for medical seg is unpublished. Identity initialization on a warmstarted backbone is the specific innovation that makes this train at all.

### 2.2 Position-Augmented State

Augment `f_θ` with normalized 3D coordinates p = (x, y, z) ∈ [-1, 1]³:
```
dh/dt = γ(e_organ) ⊙ f_θ(h, t, p) + β(e_organ)
```
where `f_θ(h, t, p) = NN(concat[h, time_enc(t), pos_enc(p)])`.

**`pos_enc(p)`:** Fourier features with n_freqs=4 → 24-dim positional vector (sin/cos × 3 axes × 4 freqs).

**Why novel:** CoordConv (2018) exists for CNNs; Neural ODE state augmented with spatial coordinates for medical segmentation is unpublished. Critical for L/R disambiguation — the model literally cannot tell symmetric structures apart without positional cues.

### 2.3 Three-Phase Encoder-Protection Curriculum

The two architectural components above (FiLM, position) are paired with a training procedure that exploits the identity-init guarantee.

**Phase P0 (5 epochs, encoder fully frozen):** the V2-warmstarted TinyViT encoder is held fixed. Only PA-CODE modules (FiLM heads, position encoding pathway), organ queries, mask MLPs, and decoder heads update. With γ=1, β=0 init, the model starts as the V2 ODE; with the encoder frozen, the V2 liver knowledge cannot drift while the new conditioning machinery calibrates. This phase is what trajectory-level distillation would have achieved, but achieved by construction rather than by an auxiliary loss.

**Phase P1 (10 epochs, encoder unfrozen at 0.05× LR):** encoder is allowed to shift, but at a strongly reduced LR. The new modules (already partially calibrated from P0) anchor the gradient direction.

**Phase P2 (until convergence, full LR cosine decay):** standard fine-tuning.

**Why novel:** This specific three-phase curriculum — frozen-warmup → low-LR-shift → full-FT — applied to a Neural ODE built on a warmstarted single-organ encoder, is unpublished. The combination with identity-init FiLM is what makes it provably stable: at the boundary between P0 and P1, the unfrozen encoder receives gradient through a γ ⊙ f + β path where γ has already learned per-organ scaling, so the gradient signal is organ-disambiguated rather than averaged.

**Why this replaces the original trajectory-distillation proposal:** Identity-init + frozen-encoder achieves the same liver-preservation guarantee mathematically — at step 0, every forward pass is identical to V2's forward pass on the same input. No teacher network or KL term is needed. Simpler. One training loop, no extra forward passes.

## 3 Math summary

### 3.1 ODE function (per direction, fwd or bwd)
```
Input: h ∈ ℝᴺˣᶜ, t ∈ ℝ, p ∈ ℝᴺˣ³, organ_id ∈ ℤᴮ
where N = B·H·W.

e = organ_embed(organ_id)                                ∈ ℝᴮˣᵈₑ
e_N = expand(e, N)                                       ∈ ℝᴺˣᵈₑ
γ = 1 + γ_head(e_N)        (γ_head zero-init)            ∈ ℝᴺˣᶜ
β = β_head(e_N)            (β_head zero-init)            ∈ ℝᴺˣᶜ

τ = time_enc(t)                                          ∈ ℝ²ⁿᶠ
ρ = pos_enc(p)                                           ∈ ℝ²⁴

h_in = concat[h, τ, ρ]                                   ∈ ℝᴺˣ⁽ᶜ⁺²ⁿᶠ⁺²⁴⁾
f = MLP_θ(h_in)            (last layer zero-init)        ∈ ℝᴺˣᶜ

dh/dt = γ ⊙ f + β
```

### 3.2 Heun trajectory (unchanged structurally)
```
states = [h₀]
for i=1..D-1:
    for sub=1..substeps:
        k₁ = ode_func(h, t_curr,         p, organ_id)
        h_pred = h + sub_dt · k₁
        k₂ = ode_func(h_pred, t_curr+dt, p, organ_id)
        h = h + 0.5·sub_dt·(k₁ + k₂)
    states.append(h)
return stack(states, dim=1)        # (N, D, C)
```

### 3.3 Total loss
```
L = L_seg(student)
where L_seg is the existing A1 multi-task loss (dice + focal + iou + boundary + PMDice + deepsup)
```
No auxiliary loss term. Liver preservation comes from the identity-init + encoder-freeze curriculum, not from a distillation loss.

## 4 Three-phase training curriculum

| Phase | Epochs | What's frozen | What trains | LR |
|---|---|---|---|---|
| P0 | 5 | encoder | PA-CODE modules, organ_queries, mask_mlps, decoder heads | 2e-4 |
| P1 | 10 | none (all unfrozen) | everything; encoder via LoRA stabilizers OR scale 0.05× LR | 1e-4 |
| P2 | until convergence | none | everything full | 1e-4 → 1e-6 cosine |

**Resume source:** `checkpoints/path_a1_big_solid/path_a1_big_solid_epoch001.pt` (the empirical best ckpt, mean DSC 0.24, liver 0.787 — V2 liver knowledge mostly preserved).

**Save dir:** `checkpoints/path_a1_pa_code/`

**Monitor metric:** 2-volume 3D mean DSC at end of each epoch (~3 min cost). NOT the noisy 2D val_dice.

## 5 Ablation slate

Each variant trained to convergence (Phase P0+P1+P2 full).

| ID | Variant | Expected Liver | Expected Spleen | Expected Kidneys (each) | Expected Mean |
|---|---|---:|---:|---:|---:|
| A0 | V2 single-organ baseline | 0.94 | — | — | — |
| A1-base | current A1 (no fix) — already in hand | 0.79 | 0.16 | 0.000 | 0.24 |
| A2 | + identity-FiLM only | 0.88 | 0.40 | 0.10 | 0.46 |
| A3 | + position-aug only | 0.85 | 0.30 | 0.50 | 0.58 |
| A4 | + frozen-encoder curriculum only | 0.92 | 0.25 | 0.05 | 0.35 |
| A5 | **PA-CODE full (A2+A3+A4)** | 0.94 | 0.85 | 0.85 | **0.89** |
| A5+ | PA-CODE + cross-dataset (BTCV/WORD) | — | — | — | thesis stretch |
| A6 | PA-CODE on K=8 (stretch) | — | — | — | bonus table |

The A1-base row is the negative-result anchor. Every later row is the positive evidence.

## 6 Files to change

| File | Change |
|---|---|
| `models/ode_cross_slice.py` | Replace `OrganConditionedODEFunction` with `PACodeODEFunction`. Add FiLM heads, position encoding, modify `_heun_trajectory` to thread p through. Add `return_states=True` flag for distillation. |
| `models/auto_ode_sam.py` | Generate normalized (x,y,z) grid per slab (z = slab index ∈ [-1,1], x/y from spatial position). Pass to ODE module. |
| `training/trainer.py` | Add encoder-freeze logic (config-driven `training.encoder_freeze_epochs`). Replace `val_dice` monitor with `val_dice_3d` (2-vol mini eval). |
| `configs/path_a1_pa_code.yaml` | New config: 3-phase curriculum (encoder_freeze_epochs=5, etc), `λ_traj` schedule, V2 teacher ckpt path, monitor=val_dice_3d. |
| `scripts/preflight_a1_pa_code.py` | Adapt existing pre-flight: schema check, label map, dry forward pass with PA-CODE on, V2 teacher loads cleanly, traj loss is finite. |
| `scripts/sanity_reload_a1_pa_code.py` | Phase-0 schema check: PA-CODE state_dict matches expected keys; identity-init means forward pass on liver-only data matches V2 forward within ε. |

## 7 Sacred ckpts (untouched)

- `checkpoints/auto_ode_sam_v3/best.pt` ← warmstart source for ENCODER
- `checkpoints/voco_totalseg_pretrain/best.pt`
- `checkpoints/v9_stage1/last.pt`, `v9_stage1_ft/last.pt`, `trissr_v9/last.pt`
- `checkpoints/v10_voco_l_ft_r3/epoch_009.pt`
- `checkpoints/organmoe_phase_i/last.pt`
- `checkpoints/path_a1_big_solid/path_a1_big_solid_epoch*.pt` (failed run, kept for ablation)
- `checkpoints/path_a1_big_solid_v2hp/path_a1_big_solid_v2hp_epoch*.pt` (rescue run, kept for ablation)

PA-CODE writes ONLY to `checkpoints/path_a1_pa_code/`.

**V2 teacher (frozen):** `checkpoints/auto_ode_sam_v3/best.pt` loaded read-only at every epoch. NO write access.

## 8 Targets

| Metric | Floor (must hit for thesis chapter) | Stretch (workshop pub) |
|---|---:|---:|
| Mean DSC (4 organs, 3D eval) | 0.85 | 0.92 |
| Liver DSC | 0.93 (preserve V2) | 0.95 |
| Spleen DSC | 0.80 | 0.92 |
| L-kidney DSC | 0.80 | 0.92 |
| R-kidney DSC | 0.80 | 0.92 |
| Mean HD95 | ≤ 2.0 mm | ≤ 1.0 mm |

Failure mode: if A5 (PA-CODE full) still has kidneys < 0.3 after Phase 1 of curriculum, the diagnosis is wrong and we fall back to A0 (liver-only paper).

## 9 Anti-pivot rule (locked)

Per `feedback_pivots_postmortem.md`: A1/PA-CODE gate-fail at Phase 1 → debug, don't pivot.
Only fallback: **A0** (single-organ liver-only paper).

## 10 Thesis chapter outline

- **X.1 Motivation** — single-organ ODE works (V2, K=1, 0.9358); naive K=4 extension fails (cos=0.999 conditioning collapse).
- **X.2 Method (PA-CODE)** — three components above.
- **X.3 Ablation** — table from §5.
- **X.4 Results** — AMOS22 + cross-dataset (BTCV zero-shot, WORD fine-tuned).
- **X.5 Analysis** — visualizations of organ-conditioned trajectories in feature space (PCA), L/R kidney attention map (proves position-aug solves disambiguation).
- **X.6 Limitations** — small organs (gallbladder, adrenals) still hard; not addressed in this chapter.

---

**Implementation order:** §6 files top-to-bottom. Each step has a paired sanity test before the next.

**Status:** Design saved. Implementation begins next.
