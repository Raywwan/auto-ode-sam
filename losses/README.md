# `losses/` — Auxiliary Loss-Function Library

Five auxiliary losses that complement the thesis primary stack. The
*primary* losses for Auto-ODE-SAM V2 (PMDice, focal, optional clDice)
live in `training/losses.py` — this folder hosts the auxiliary heads
that were trialled in the V4-era exploratory runs.

## Modules

| File | Purpose |
|------|---------|
| `__init__.py` | Package marker. |
| `cascade_consistency.py` | Consistency loss between a coarse-stage proposer and a fine-stage refiner (V9 cascade). Not used in the thesis headline. |
| `cross_modal_contrastive.py` | InfoNCE-style contrastive loss between organ text embeddings and image features (V9-era). Frozen. |
| `flow_shape_prior.py` | Flow-matched ODE shape prior for V4 OrganFlow-SAM2 (Wasserstein-style penalty on the ODE trajectory). Not in V2 headline. |
| `teacher_distill.py` | Knowledge distillation from a frozen teacher (VoCo / TotalSeg pretrained). Used in the V10 moonshot plan; not in thesis headline. |

## What lives elsewhere (the thesis stack)

The **active** thesis losses are in `training/`:

- `training/losses.py` — primary V2 stack: Dice + focal + (optional) clDice.
- `training/losses_v4.py` — V4 OrganFlow-SAM2 stack (frozen pilot).
- `training/losses_small_organ.py` — per-class weighting (adrenal ×3, gallbladder ×2.5) for the multi-organ Path A1 experiment.
- `training/losses_organmoe.py` — OrganMoE auxiliary stack (frozen).

See `memory/feedback_loss_functions.md` for the V2 loss-design rationale
(why PMDice replaces DiceTopK, why clDice is OFF for solid organs).

## Standing rules

- **Do not enable cross-modal contrastive on the thesis V2 path** — V2 has no text encoder.
- **Cascade consistency is for the V9 proposer–refiner architecture only** — it has no meaning under V2 single-stage inference.
