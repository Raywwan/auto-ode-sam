# `inference/` — Post-Processing, TTA, and Zoom-Refinement Utilities

Inference-time helpers applied **after** the forward pass. The training
loop does not touch these; only the `scripts/eval_*.py` entry points and
the live inference pipeline do.

## Modules

| File | Purpose |
|------|---------|
| `__init__.py` | Package marker. |
| `postproc.py` | Largest-connected-component filter and small-component removal. The largest-CC step is mandatory for the headline number (see `memory/feedback_totalseg_eval_lessons.md`). |
| `tta.py` | Flip-safe intensity test-time augmentation. Currently the **only** TTA flavour validated for V2 — gamma-only TTA at γ ∈ {0.8, 1.0, 1.2} lifts DSC 0.9352 → 0.9366 on AMOS22 (see `project_a0_t3_tta_intensity_2026-05-22`). Flip-TTA is intentionally disabled because V2 was not trained with `RandomFlip` (see `memory/feedback_tta_requires_flip_training.md`). |
| `zoom_refine.py` | Region-of-interest zoom and re-inference pass. Crops a tight bounding box around the first-pass prediction and re-runs the model at full resolution. Used by the cross-dataset eval to recover from extreme acquisition shift. |

## Pipeline order at eval time

1. Forward pass (with intensity TTA if enabled).
2. Threshold logits at τ (default 0.5; 4-seed ensemble headline at 0.5; oracle ceiling at 0.40 — see thesis §1.5).
3. Largest-connected-component post-processing.
4. Optional zoom-refine pass (off by default).

## Standing rules

- **Do not enable flip-TTA on V2 weights.** Code is correct, but the model wasn't trained with random flips, so flip-TTA collapses DSC 0.93 → 0.64.
- All postproc must clear the **all-three-metrics** gate (DSC, HD95, NSD) before promotion to the headline.
