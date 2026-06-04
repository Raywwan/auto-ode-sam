# `tests/` — Unit Tests and CUDA Smoke Tests

Two test families:

1. **Module-level unit tests** (`test_*.py`) — pytest-compatible, run on
   CPU, exercise individual components in isolation.
2. **End-to-end smoke tests** (`v8_smoke.py`, `v9_*_smoke.py`) — small
   GPU runs that load a real config, take a few training steps, and
   verify gradients flow and the loss stays finite. Used as preflight
   before queueing a full multi-day training job.

## Unit tests (CPU, pytest)

| File | Tests |
|------|-------|
| `test_amos22_multiorgan.py` | AMOS22 multi-organ loader: label-map correctness, presence-cache consistency. |
| `test_anatomy_graph_decoder.py` | Anatomy-graph attention decoder forward pass and gradient flow. |
| `test_balanced_sampler.py` | Class-balanced batch sampler: per-class frequencies under different temperatures. |
| `test_boundary_dou.py` | Boundary-DoU loss numerics on synthetic masks. |
| `test_flow_cross_slice.py` | Flow-matched cross-slice module forward / backward. |
| `test_losses_v4.py` | V4 loss stack (PMDice, focal, optional clDice). |
| `test_medsam2_encoder.py` | MedSAM-2 video-memory encoder loading + forward. |
| `test_organflow_sam2.py` | OrganFlow-SAM2 full-model forward. |
| `test_organmoe_loss.py` | OrganMoE auxiliary loss. |
| `test_organmoe_module.py` | OrganMoE soft-MoE routing. |
| `test_pfesa_plus.py` | PFESA++ spectral enhancement numerics. |
| `test_totalseg_label_map.py` | TotalSeg → AMOS22 label-ID conversion (caught a near-miss for classes 14/15; see `memory/feedback_label_map_audit.md`). |

## End-to-end smoke tests (GPU)

| File | Purpose |
|------|---------|
| `v8_smoke.py` | V8 MCP-killer architecture smoke. |
| `v9_cuda_smoke.py` | V9 CUDA build + AMP sanity. |
| `v9_d1_dataset_smoke.py` | V9 dataset-loader smoke. |
| `v9_full_pipeline_smoke.py` | V9 end-to-end (proposer + refiner) forward+backward. |
| `v9_m4_teacher_distill_smoke.py` | Teacher distillation loss smoke. |
| `v9_m5_infonce_smoke.py` | InfoNCE contrastive loss smoke. |
| `v9_m6_flow_shape_prior_smoke.py` | Flow shape prior smoke. |
| `v9_m7_dynamic_pfesa_smoke.py` | Dynamic PFESA smoke. |
| `v9_m8_cascade_consistency_smoke.py` | Cascade consistency loss smoke. |
| `v9_m9_boundary_ddpm_smoke.py` | Boundary DDPM head smoke. |

## How to run

```bash
# CPU unit tests
"C:/Users/Raywa/AppData/Local/Programs/Python/Python312/python.exe" -m pytest tests/test_*.py -x

# GPU smoke (example)
"C:/Users/Raywa/AppData/Local/Programs/Python/Python312/python.exe" tests/v9_full_pipeline_smoke.py
```

## Standing rule

**Run the smoke test before queueing any >30 min training job.** See
`memory/feedback_preflight_long_runs.md` — the pre-flight discipline
caught two label-map and four cascade-alignment bugs before they
consumed multi-day training budgets.
