# `configs/` — Training and Evaluation YAML Configs

Hydra-style configs consumed by `train.py`, the evaluation scripts under
`scripts/`, and the model factories under `models/`. Every config
inherits from `base.yaml` via `defaults: [base]`.

## Conventions

- `base.yaml` — global defaults: optimiser, schedule, AMP dtype, logging.
- `amos22.yaml` — AMOS22 dataset block reused by most experiments.
- Filenames encode the experiment lineage: `phase3_*` = thesis V2 line;
  `phase3a_*` / `phase3b_*` = transitional 2-organ tests; `path_a1_*` =
  Path A1 multi-organ extension; `phase2_*` = older ablation series;
  `v4_*`, `v5_*` … `v10_*` = exploratory model generations;
  `organmoe_*` = OrganMoE-3D experiments; `pa_code_*` = PA-CODE
  negative-result runs.

## Thesis-relevant configs (active)

| Config | Purpose |
|--------|---------|
| `base.yaml` | Global defaults (optimiser, AMP, schedule, logging). |
| `amos22.yaml` | AMOS22 dataset block (data root, target organs, spacing, splits). |
| `phase3_odesam_v2.yaml` | **Headline V2 config — DSC 0.9351 (seed 42).** |
| `phase3_odesam_v2_seed43.yaml` | Re-seed for reproducibility (DSC 0.9356). |
| `phase3_odesam_v2_seed44.yaml` | Re-seed for reproducibility (DSC 0.9379). |
| `phase3_odesam_v2_seed45.yaml` | Re-seed for reproducibility (DSC 0.9373). |
| `phase3_odesam_v2_fwdonly.yaml` | Unidirectional ODE ablation (forward only). |
| `phase3_odesam_v2_hdloss_rw.yaml` | HD-loss + reweighted Dice ablation. |
| `phase3_odesam_v2_seed43_bdou_ft5ep.yaml` | Boundary-DoU fine-tune for 5 epochs from seed 43. |

## Multi-organ / exploratory (frozen)

| Family | Configs | Status |
|--------|---------|--------|
| Path A1 big-solid 4-organ | `path_a1_big_solid*.yaml` | Paused; spec at `docs/superpowers/specs/2026-04-28-*.md`. |
| PA-CODE negative result | `path_a1_pa_code*.yaml` | Terminated 2026-05-03; lives in §7.4 as negative result. |
| OrganMoE-3D | `organmoe_phase_i_*.yaml` | Superseded by A1. |
| Ablation suite (phase 2) | `phase2_*.yaml` | Older ablation series; some numbers used in §5. |
| Test architectures | `test_*.yaml` | One-shot model sanity smokes. |
| V4–V10 exploratory | `v4_*.yaml` … `v10_*.yaml` | Pre-V2 generations; kept for reference. |

## Reading a config

Each file starts with `defaults: [base]`, then overrides keys under
`experiment`, `data`, `model`, `training`, `optimiser`. Anchor numbers
(e.g. checkpoint paths, presence-cache paths) are absolute Windows paths
because the repo runs on a single workstation.

## Standing rule

When adding a new config: copy an existing thesis-relevant file, change
the `experiment.name` field first (so log/checkpoint dirs don't collide
with prior runs), and **never reuse an experiment name across runs** —
see `memory/feedback_checkpoints.md` for the safety rationale.
