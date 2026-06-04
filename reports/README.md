# `reports/` — Auto-Generated Experiment Reports

JSON / Markdown / PDF summaries produced by the evaluation scripts and
training-loop callbacks. Most files here are V9-era; thesis-relevant
reports live under `D:\Project\thesis\results\` and
`D:\Project\Dr_Aram\results\`.

## Top-level files (V9 / V10 era)

| File | Purpose |
|------|---------|
| `v9_combined_3d_eval.json` | Combined 3D evaluation of the V9 proposer + refiner cascade. |
| `v9_proposer_3d_eval_ft.json`, `v9_proposer_3d_eval_ft.log` | V9 proposer-only 3D eval after fine-tune. Used to bound the proposer ceiling at 0.8433. |
| `v9_proposer_3d_eval_ttafix.json`, `v9_proposer_3d_eval_ttafix.log` | V9 proposer after the TTA-correctness fix. |
| `v9_proposer_smoke.json`, `v9_proposer_smoke_argmax.json`, `v9_proposer_smoke_ttafix.json` | V9 proposer smoke evaluations. |
| `v9_stage1_ft.log` | V9 Stage-1 fine-tune log. |
| `organmoe_phase_i_3d_eval.json` | OrganMoE Phase-I 3D evaluation (mean 0.7203, gate fail). |
| `trissr_v9_train.log` | TriSSR-V9 training log. |

## Subfolders

| Folder | Purpose |
|--------|---------|
| `organmoe_phase_i/` | OrganMoE Phase-I per-epoch reports (frozen 2026-04-28). |
| `v10_w2/` | V10 Week-2 moonshot reports (VoCo-L / VoCo-H tracks). |

## Where to find thesis-relevant reports

- Per-volume metrics: `D:\Project\thesis\results\<exp>\per_volume_metrics.csv`
- Summary JSON: `D:\Project\thesis\results\<exp>\summary.json`
- Conformal calibration: `D:\Project\thesis\results\conformal_amos22_liver\summary.json`
- Fairness audit: `D:\Project\thesis\results\fairness\fairness_audit.json`

## Standing rule

This folder is *write-mostly* — new training runs append, old runs are
not pruned. If you need the latest authoritative number for a thesis
claim, look in `D:\Project\thesis\results\` first, not here.
