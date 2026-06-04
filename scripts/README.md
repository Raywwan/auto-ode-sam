# `scripts/` — One-Shot Tools: Eval, Diagnostics, Queue Runners

58 entries — every CLI entry-point that isn't `train.py`. Grouped below
by purpose. Most are Python; a few `.sh` wrappers orchestrate multi-step
pipelines.

## Headline evaluation scripts (thesis-relevant)

| Script | Purpose |
|--------|---------|
| `eval_amos22_liver_indomain.py` | Single-seed AMOS22 liver headline eval. Produces `headline_single_seed_0.9351/summary.json`. |
| `eval_4seed_ensemble.py` | 4-seed ensemble at configurable τ. Produces `ensemble_4seed_tau050_*` and `ensemble_4seed_tau040_*`. |
| `eval_with_threshold_sweep.py` | Threshold sweep τ ∈ [0.30, 0.70] step 0.025. Produces `threshold_sweep_v2/per_threshold_summary.csv`. |
| `eval_totalseg_liver.py` | TotalSegmentator cross-dataset eval (V2 native preproc: LAS reorient, no resample, largest-CC). |
| `eval_tta_intensity.py` | Intensity-gamma TTA at γ ∈ {0.8, 1.0, 1.2}. Produces the §5.7 TTA numbers. |
| `eval_3d_amos22.py` | Generic 3D AMOS22 eval (multi-organ variant). |
| `eval_gate_sensitivity.py` | Sensitivity sweep around the gate threshold for the §5 sensitivity ablation. |
| `eval_save_predictions_seed42.py` | Save per-volume predictions for the baseline seed (for downstream conformal / fairness work). |
| `eval_robustness_battery_seed42.py` | Robustness battery: noise, contrast, slice-drop perturbations. |

## Baseline scripts

| Script | Purpose |
|--------|---------|
| `build_nnunet_amos22_liver.py` | Sets up the nnU-Net `Dataset511` raw layout. |
| `run_nnunet_baseline.sh` | Preprocess + train nnU-Net fold 0. |
| `rerun_nnunet_predict_eval.sh` | Predict + compute metrics from a trained nnU-Net fold. |
| `eval_nnunet_predictions.py` | Compute DSC/HD95/NSD on nnU-Net predictions. |
| `eval_medsam2_zeroshot_amos22.py` | MedSAM-2 zero-shot baseline (DSC 0.85, HD95 24.5). |
| `download_medsam2.py`, `download_totalseg.py` | Dataset/model download helpers. |

## Conformal / fairness / TTA

| Script | Purpose |
|--------|---------|
| `conformal_sets_amos22_liver.py` | Mondrian split-CP + CRC conformal calibration. Produces `conformal_amos22_liver/summary.json`. |
| `fairness_audit.py` | Sex / age / site / manufacturer fairness audit. Produces `fairness/fairness_audit.json`. |

## SWA / ensembling

| Script | Purpose |
|--------|---------|
| `build_swa_seed43.py` | Build SWA-top-5 weight average from the seed-43 trajectory. |
| `aggregate_curve.py` | Aggregate epoch-curve JSONs into a single plot. |

## PA-CODE diagnostics (negative-result chapter)

| Script | Purpose |
|--------|---------|
| `diag_pa_code_filmhead_drift.py` | Diagnose FiLM-head γ drift on saved PA-CODE checkpoints. |
| `diag_pa_code_gbound_drift.py` | Same, for the bounded-γ variant. |
| `figure_pa_code_drift.py` | Generate the γ-drift / collapse figures for §7.4. |
| `verify_gamma_bound.py` | Sanity check the bounded-γ formula numerically. |

## Path A1 / OrganMoE (frozen)

`eval_a1_all_epochs.py`, `eval_a1_per_organ.py`, `diag_a1_query_collapse.py`,
`preflight_a1.py`, `preflight_a1_pa_code.py`, `prepare_a1_warmstart.py`,
`prepare_a1_pa_code_warmstart.py`, `sanity_pa_code_identity.py`,
`sanity_reload_a1.py` — all frozen with the Path A1 / OrganMoE pause.

## Sanity / smoke / preflight

`smoke_dataset_aug.py`, `smoke_test_v4.py`, `smoke_voco_l_loss.py`,
`preflight_bdou_ft.py`, `preflight_totalseg_pretrain.py`,
`vram_preflight_voco_l.py`, `diag_keys.py`, `diagnose_ep5_fail.py`,
`inspect_refiner_probs.py`.

## Queue runners

| Script | Purpose |
|--------|---------|
| `run_gpu_queue.sh` | Generic GPU-queue runner. Writes per-job exit codes to `logs/queue_status/`. |
| `run_phaseA_sweep_and_swa.sh` | Phase-A sweep + SWA build pipeline. |
| `run_phaseB_seed4445_eval_and_ensemble.sh` | Phase-B seed-44/45 train+eval+ensemble pipeline. |
| `run_seed44_train.sh`, `run_seed45_train.sh` | Per-seed launchers. |
| `run_curve_queue.sh`, `eval_ckpt_curve.sh` | Curve-evaluation queue. |
| `rescue_seed43_evals.sh` | One-shot rescue script for the seed-43 monitor-cron bug. |

## Build / cache

`build_totalseg_cache.py`, `build_totalseg_presence.py` — build the
balanced-sampler presence cache (mandatory before any class-balanced
training on TotalSeg). See `memory/feedback_presence_cache_for_balanced_sampler.md`.

## Misc

`fetch_pretrained.py`, `fetch_teachers.py`, `warmstart_v8_from_v7.py`,
`eval_refiner_per_organ.py`, `eval_v4_ckpt.py`, `eval_v9_proposer_preproc.py`.

## Standing rule

Argparse flags with hyphens vs underscores: `--n-slices` is registered
in some scripts and `--n_slices` in others. **Audit launchers against
`add_argument` lines before queueing.** See
`memory/feedback_argparse_hyphen_flags.md`.
