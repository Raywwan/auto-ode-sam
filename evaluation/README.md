# `evaluation/` — 3D Metric Computation and Volume Reconstruction

Core evaluation utilities used by every `scripts/eval_*.py` entry point.
Computes per-volume DSC, HD95, NSD, ASSD, IoU, sensitivity, precision,
specificity, and volume-similarity from predicted vs ground-truth NIfTI
volumes.

## Modules

| File | Purpose |
|------|---------|
| `__init__.py` | Package marker. |
| `metrics.py` | 2D / per-slice metric primitives (used by training-loop validation). |
| `metrics_3d.py` | **3D volumetric metrics**: `dsc_3d`, `hd95_mm`, `nsd_at_tolerance`, `assd_mm`, plus the headline `compute_all_metrics_3d` aggregator. Uses MedPy and SciPy under the hood. |
| `sliding_window_3d.py` | Sliding-window inference for 3D volumes: tiles a volume into overlapping slabs, runs the 2D-stack model on each, and stitches predictions with Gaussian importance weighting. |
| `volume_reconstructor.py` | Re-assembles per-slice 2D predictions into a 3D NIfTI volume with the original spacing/affine preserved. Handles LAS reorientation, padding, and resample-back if the dataset was resampled. |
| `eval_v9_3d.py` | V9-era evaluation entry point (proposer–refiner cascade). Legacy; kept for V9 checkpoint compatibility. |

## Subfolders

| Folder | Purpose |
|--------|---------|
| `reports/` | Auto-generated evaluation reports (markdown summaries of headline runs). |

## Metric definitions used in the thesis

The thesis follows the *Metrics Reloaded* recommendation: three
complementary 3D metrics rather than a single Dice number.

- **DSC** (Dice Similarity Coefficient): overlap metric, 0..1, bigger better.
- **HD95**: 95th-percentile symmetric Hausdorff distance in mm. Bounded below by slice spacing (5 mm on AMOS22). See `memory/feedback_metric_axis_blindness_2026-05-22.md`.
- **NSD** (Normalised Surface Dice): surface-overlap with a tolerance (1 mm in the headline). Lower-bounded by the inference grid voxel size.

## Cross-cutting rule

A postproc/TTA/loss change can **only** be promoted if all three metrics
(DSC, HD95, NSD) improve or stay flat. See
`memory/feedback_metric_axis_blindness_2026-05-22.md`.
