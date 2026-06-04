# `models/` — Network Architectures

Every architecture trialled in the project, including the thesis V2
headline (`auto_ode_sam.py`) and the various exploratory variants kept
around for ablation reproducibility.

## Thesis-relevant (active)

| File | Purpose |
|------|---------|
| `auto_ode_sam.py` | **Thesis V2 headline architecture (18.0 M params).** TinyViT image encoder + SAM-v1 mask decoder + bidirectional Neural-ODE block on Stage-2 features (4× spatial resolution vs Stage-3). RK4 fixed-step integrator. |
| `ode_cross_slice.py` | The Neural-ODE block itself — `dh/dz = f_θ(h, c, z)` parameterised by a small CNN. ~0.5 M params. Used inside `auto_ode_sam.py`. |
| `ode_sam.py` | Older V1 ODE-SAM (Stage-3 tap). Kept for the Stage-2-vs-Stage-3 ablation in §5. |
| `mask_decoder.py` | SAM-v1 mask decoder (reused verbatim from the SAM codebase). |
| `prompt_encoder.py` | SAM-v1 bounding-box prompt encoder. |

## Multi-organ / exploratory (frozen)

| File | Purpose |
|------|---------|
| `organflow_sam2.py`, `organflow_sam2_v8.py` | V4 OrganFlow-SAM2 architecture (flow-matched ODE training). |
| `organmoe_3d.py` | OrganMoE-3D (frozen 2026-04-28 in favour of Path A1). |
| `organ_query_decoder.py` | V3 per-organ query decoder (frozen). |
| `organ_text_encoder.py` | V3 organ-text embedding encoder. |
| `anatomy_graph_decoder.py` | V4 anatomy-graph attention decoder. |
| `voluformer3d.py`, `voluformer_v9.py` | V9-era full architectures. |
| `trissr_v9.py` | V9 TriSSR proposer–refiner cascade. |

## Building blocks

| File | Purpose |
|------|---------|
| `dynamic_pfesa.py`, `pfesa_plus.py` | Parameter-free spectral enhancement (PFESA / PFESA++). Used in some V4-V9 variants. |
| `boundary_ddpm.py` | Boundary-aware DDPM head (V9-era). |
| `depth_aware_isa.py` | Depth-aware inter-slice attention (V1; superseded by ODE). |
| `flow_cross_slice.py` | Flow-matched cross-slice module (V4 alternative to the ODE). |
| `mamba_ode.py`, `mamba_pure.py` | State-space (Mamba) cross-slice variants. |
| `multiscale_encoder.py`, `dual_encoder.py` | Encoder variants. |
| `medsam2_encoder.py` | MedSAM-2 video-memory encoder (for zero-shot baseline). |
| `acm_sam.py`, `fca_sam.py`, `trimamba_sam.py`, `swin_unetr_3d.py` | Competitor / ablation architectures. |
| `soft_moe.py`, `topology.py` | Auxiliary modules. |

## Standing rules

- **`auto_ode_sam.py` is the headline.** Any changes there require a fresh ablation against the existing checkpoint.
- **Stage-2 tap (4× resolution) is the contribution.** Don't move the tap to Stage-3 without a deliberate ablation.
- **Pre-norm ODE** is mandatory; post-norm collapses the integrator. See `memory/feedback_architecture_decisions.md`.
- **Identity-init FiLM heads** must zero-init the **last** layer only (not both); see `memory/feedback_film_init_gradient_trap.md`.
- **Bounded-γ FiLM**: `γ = 1 + α · tanh(γ_raw − 1)` to prevent unbounded drift (the PA-CODE fix from C3).
