# `training/` — Training Loop, Loss Stacks, Pretraining Entry Points

The core training loop (`trainer.py`), the active loss stacks for each
model generation, and the standalone pretraining entry points.

## Modules

| File | Purpose |
|------|---------|
| `__init__.py` | Package marker. |
| `trainer.py` | **Active V2 trainer.** Handles config resolution, dataset construction, optimiser, AMP, validation, checkpointing, TensorBoard logging, gate checks, and resume. |
| `trainer_v9.py` | V9 trainer with proposer–refiner cascade scheduling (frozen). |
| `losses.py` | **V2 primary loss stack:** PMDiceLoss + focal + (optional) clDice. The V2 default is Dice 0.5 + focal 0.5. |
| `losses_v4.py` | V4 OrganFlow-SAM2 losses (flow-matched ODE + reconstruction). Frozen pilot. |
| `losses_small_organ.py` | Per-class weighting for small organs (adrenal ×3, gallbladder ×2.5). Used by Path A1. |
| `losses_organmoe.py` | OrganMoE auxiliary stack (frozen 2026-04-28). |
| `pretrain_totalseg.py` | TotalSegmentator pretraining entry point. Currently disabled in the V2 path. |
| `train_organmoe_amos22.py` | OrganMoE-3D training entry (frozen). |
| `train_trissr_v9.py` | TriSSR-V9 training entry (frozen). |
| `train_v9.py` | V9 generic training entry (frozen). |

## Active training command (V2 headline)

```bash
"C:/Users/Raywa/AppData/Local/Programs/Python/Python312/python.exe" \
  D:/Project/train.py \
  --config configs/phase3_odesam_v2.yaml
```

For re-seeded runs swap to `phase3_odesam_v2_seed{43,44,45}.yaml`.

## Phase / experiment expectations

| Phase | Config | Wall-clock (4090) | Expected DSC |
|-------|--------|-------------------|--------------|
| V2 single-seed 40 ep | `phase3_odesam_v2.yaml` | ~9–10 days | 0.9351 ± 0.016 |
| V2 re-seed 40 ep | `phase3_odesam_v2_seed{43,44,45}.yaml` | ~9–10 days each | 0.9356 / 0.9379 / 0.9373 |
| V2 fwd-only ablation | `phase3_odesam_v2_fwdonly.yaml` | ~9–10 days | within seed noise of bidi |
| Boundary-DoU FT 5 ep | `phase3_odesam_v2_seed43_bdou_ft5ep.yaml` | ~1 day | + or − ~0.001 (ns) |

## Loss-design rules (from `memory/feedback_loss_functions.md`)

- **PMDice replaces DiceTopK** in V2.
- **Per-sample weights inside the loss**, not at batch mean.
- **clDice OFF for solid organs** (liver, kidneys, spleen). Only useful for vessels / pancreas tail.
- **CutMix OFF for Auto-ODE-SAM** — the discrete pasting interacts badly with the cross-slice ODE prior.

## Standing rule

**Never reuse an experiment name across runs** — the trainer will refuse
to overwrite a non-empty checkpoint dir to prevent silent loss of work
(see `memory/feedback_checkpoints.md`).
