# V9 — What's Left (as of 2026-04-21 22:50 AST)

**Author:** Raywan Dlawar
**Context:** Stage 1 done (proposer val 0.8433 present / 0.6548 all). Stage 2 v5 running with alignment fix. See `docs/superpowers/plans/2026-04-20-voluformer-v9-remaining.md` for the original task breakdown; this file tracks what still needs to happen to reach thesis-ready state.

---

## Near-term (next 72 h)

| # | Task | Expected time | Gate |
|---|------|--------------:|------|
| N1 | Stage 2 v5 to finish 30 epochs (started 22:48 AST) | ~3 h | Fused Dice jumps to ≥0.7 at ep 1 (gate-bias-dominated), refiner climbs; stop-rule not triggered |
| N2 | Evaluate Stage 2 v5 best.pt with `scripts/eval_refiner_per_organ.py` and `scripts/eval_gate_sensitivity.py` | 30 min | Per-organ Dice table saved, gate-bias sweep shows fused ≥ proposer |
| N3 | 3D sliding-window eval on val (30 volumes) via `evaluation/eval_v9_3d.py` | 2 h (CPU-heavy) | Report fused / prop / ref 3D Dice + HD95 per organ; **this is the number that counts** |
| N4 | Decide: continue to Stage 3 joint fine-tune, or ablate before Stage 3 | user decision | — |

## Medium-term (3-7 days)

| # | Task | Time | Why |
|---|------|-----:|-----|
| M1 | Stage 3 joint fine-tune (10-20 epochs, all params trainable, cascade consistency λ up from 0) | 2-3 h | Novel #7, unifies proposer and refiner end-to-end |
| M2 | TotalSegmentator pseudo-labels for Novel #4 teacher distill | 1 h (offline inference) + 2 h training | Only if ablation shows Stage 2 plateau below 0.85 |
| M3 | Ablation study (proposer only / refiner only / cascade no-gate / cascade-with-gate / cascade-with-ODE) | 4-6 h total | **Required** for any publishable writeup — this is the evidence the novel combination helps |
| M4 | Hyperparameter sweep on gate temperature and ODE step count | 2 h | Small gains, nice-to-have |

## Long-term (1-3 weeks)

| # | Task | Time | Why |
|---|------|-----:|-----|
| L1 | **BTCV** evaluation (13 organs, classic Beyond-the-Cranial-Vault) | 1 day | Secondary benchmark for MICCAI-quality generalization claim |
| L2 | **TotalSegmentator benchmark** (104 structures — evaluate on those that overlap with AMOS22 15) | 1 day | Third benchmark; weights already local |
| L3 | MRI modality token activation + joint CT+MRI loader (already partly scaffolded, task #19) | 1-2 days | AMOS22 MM-track eligibility |
| L4 | Write-up: reframe thesis around "Anatomical cascade with organ-conditioned flow-ODE refinement over SAM2", 1 defensible system-level contribution (not 8) | 3-5 days | Workshop target MIDL / MICCAI workshop; stretch MICCAI main |

## Explicitly deferred / dropped

- **Novel #5 (Flow shape prior)** — kept as a loss with λ=0, no longer claimed as a contribution. FlowSDF (IJCV 2025) already publishes this pattern.
- **Novel #8 (Boundary DDPM)** — kept as a loss, not claimed. Diffusion-segmentation has become crowded by 2026.
- **Teacher ensemble distillation (Novel #4)** — deferred unless Stage 2 cascade stays below 0.85 3D Dice.

## Dependencies / critical path

```
Stage 1 ✓
   ↓
Stage 2 v5 (RUNNING, ETA 01:50 AST 2026-04-22)
   ↓
3D eval on val 30 vols (N3)  ← MUST be done before any SOTA claim
   ↓
Decision point
 ├→ If prop-fused ≥ 0.85 3D: go to Stage 3 + ablations (M1, M3)
 └→ If prop-fused <  0.85 3D: add teacher distill (M2) first
                                                   ↓
                                          BTCV + TotalSeg eval (L1, L2)
                                                   ↓
                                               Writeup (L4)
```

## Done so far

- V9 Phase A modules (`swin_unetr_3d.py`, `voluformer_v9.py`, `mamba_ode.py`)
- V9 Phase B losses (teacher_distill, cross_modal_contrastive, flow_shape_prior, cascade_consistency, boundary_ddpm)
- AMOS22 V9 dataset with pos/neg/mixed slab sampling + now aligned to volume patch
- 3-stage trainer (`trainer_v9.py`)
- `evaluation/eval_v9_3d.py` (Gaussian blend + CC + TTA)
- All smoke tests passing on CPU
- Stage 1: 50 epochs, val 0.8433 present
- Teacher weights downloaded: SAM2 (898 MB), DINOv2 (304 M), BiomedCLIP (196 M), TotalSeg package
- Four critical Stage 2 bugs found and fixed (BCE-only loss, slab_center_z semantics, slab-index vs patch-index confusion, slab↔patch H×W misalignment)
