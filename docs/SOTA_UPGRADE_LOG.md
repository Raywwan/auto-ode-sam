# V4 → V5 SOTA Upgrade Log

Started: 2026-04-19 12:40 (after V4 ep45 plateau at val_dice 0.3399)
Goal: competitive/SOTA on AMOS22 abdominal 15-organ segmentation.

## Baseline we are beating

V4 OrganFlow-SAM2 @ ep45 (Tversky + full-res multi-organ soft Dice):

| Epoch | val_dice | Δ |
|-------|----------|---|
| 0 | 0.0205 | — |
| 5 | 0.0662 | +0.046 |
| 10 | 0.0991 | +0.033 |
| 15 | 0.1599 | +0.061 |
| 20 | 0.2277 | +0.068 |
| 25 | 0.2599 | +0.032 |
| 30 | 0.3036 | +0.044 |
| 35 | 0.3280 | +0.024 |
| 40 | 0.3317 | +0.004 |
| **45** | **0.3399** | +0.008 |

Plateau at 0.34 — config saturated.

## Target

AMOS22 SOTA reference:
- nnU-Net V2 residual: ~0.87 DSC (full 3D, high-res)
- TotalSegmentator: ~0.85
- MedFormer/SwinUNETR: ~0.83
- MedSAM2 fine-tuned: ~0.80

Fair-compare target on our subsampled 256² × D=8 regime:
- Short-term (a few hundred epochs): **0.55–0.65** mean soft Dice
- SOTA claim if we can hit: **0.70+** with novel flow-ODE + graph contributions

## Why V4 plateaued (diagnosis)

1. **Resolution too low** — 256² × D=8 loses small-organ detail (adrenals, vessels)
2. **Shallow decoder** — transformer_depth=4, mlp_dim=2048 under-parameterised for 15 classes
3. **LoRA rank=16** — too conservative for the adaptation needed
4. **No deep supervision on decoder** — only a single-scale head
5. **No data augmentation** — 360 train volumes is small, needs flips/rotations/intensity
6. **Metric definition** — soft-Dice-threshold-free is aligned with loss but caps ~0.35 at this config

## Contribution map for V5

Keep from V4:
- ✅ Flow-matched cross-slice ODE (novel, works, measurable)
- ✅ Anatomy-Graph DETR decoder (novel, works)
- ✅ PFESA++ learnable FFT amplifier
- ✅ MedSAM2 + LoRA encoder

Upgrade:
- **LoRA rank** 16 → 32 (more adaptation capacity, stays efficient)
- **Decoder transformer** depth 4 → 6, mlp_dim 2048 → 3072
- **Add deep-supervision heads at 3 scales** (64, 128, 256) with scale-weighted Tversky — nnU-Net staple
- **Multi-scale skip fusion** from encoder stages (not just final feature)
- **Data augmentation** — random flip (H, axial), rotation ±10°, intensity jitter

New-novel contribution candidates:
- **Organ-Anatomical-Prior (OAP)** queries: initialise DETR queries from learned per-organ centroid + scale prior (organs have known typical CT positions) — faster convergence for small organs
- **Boundary Distance Penalty (BDP)** loss: L1 on signed distance transform of sigmoid probability field — sharpens boundaries (common problem on small organs)
- **Organ-Query Self-Consistency** regulariser: in cross-attention, penalise two different organ queries attending to identical regions at high weight (encourages disjoint organ predictions)

## Plan

1. Run per-organ breakdown on ep45 to identify which organs are the bottleneck
2. Design final architecture (possibly via Plan subagent)
3. Implement changes to models/decoder/loss/aug
4. Smoke test (forward+backward, VRAM check)
5. Run 10-epoch pilot to validate gain rate
6. If pilot > ep45 baseline at matched compute → full run
7. Full run to convergence, monitor

## Per-organ breakdown at ep45 plateau (n=100 val volumes)

| Organ | Soft Dice | Note |
|-------|-----------|------|
| Liver | 0.7294 | Large, easy |
| Spleen | 0.6109 | Good |
| L kidney | 0.5394 | Good |
| R kidney | 0.4819 | OK |
| Pancreas | 0.3296 | Mid |
| Aorta | 0.2970 | Mid |
| Stomach | 0.2778 | Mid |
| IVC | 0.2338 | Mid |
| Gallbladder | 0.2315 | Mid |
| Duodenum | 0.2209 | Mid |
| R adrenal | 0.0253 | Near-zero |
| L adrenal | 0.0001 | Near-zero |

Binary@0.5 = 0.352, @0.3 = 0.356 — threshold-insensitive now. Adrenals are 1–2 pixels at 64×64 decoder res — physical pixel limit, not learnable at this config.

## Opus architect plan (received 2026-04-19 12:45)

Diagnosis (file-cited):
- D1 (biggest): **zero augmentation** + deterministic center-slice pick → dataset memorised by ep30
- D2: supervision is center-slice only; 7/8 of model compute unsupervised
- D3: decoder output 64×64, GT downsampled with nearest → small organs vanish
- D4: mask MLP bottleneck (32-dim)
- D5: flat skip topology (only stride-8 tap, gate=0.05)
- D6: LoRA rank=16, alpha=16 → scale=1.0, conservative
- D7: loss has no boundary or focal signal
- D8: DETR queries from random init, not anatomical prior
- D9: flow MSE in fp16
- D10: anatomy reg pulls A back to init — no learning

Pilot (STEPS 1–4, 9, 10):
- STEP 1: augmentation (flip/rot/intensity + z-jitter)
- STEP 2: decoder 64→128, wider mask MLP
- STEP 3: compute loss at GT res (upsample preds, not downsample GT)
- STEP 4: all-slice supervision (5 of 8 slices, quadratic weight falloff from center)
- STEP 9: Organ-Anatomical-Prior (OAP) query init
- STEP 10: **NEW NOVEL — Cross-Slice ODE Consistency (XSC) loss** — integrated-consistency twin of flow-matching; closes the loop on the flow-matched ODE thesis claim

Pilot gate (10 epochs): val Dice ≥ 0.20 at ep5, ≥ 0.28 at ep10, flow + xsc both decreasing. If met → full run with STEPS 5–8 added.

## Decisions log

- 2026-04-19 12:45 — V4 stopped at ep45 val=0.3399, checkpoint preserved
- 2026-04-19 12:45 — Pilot approach approved: STEPS 1–4, 9, 10 before STEPS 5–8
- 2026-04-19 12:45 — Starting implementation: STEP 1 (augmentation) first, biggest ROI

## Implementation progress

- 2026-04-19 12:50 — STEP 1 landed: paired H-flip / 90°-rot / ±10° rot / intensity jitter + ±3 z-jitter in `AMOS22MultiOrgan3D_Dataset`. Smoke test aligned img/mask.
- 2026-04-19 13:05 — STEP 9 landed: `OrganAnatomicalPriorQuery` (learnable pos+content+pos_proj, `weight` property back-compat) wired into `AnatomyGraphDecoder`.
- 2026-04-19 13:10 — STEP 2 landed: 3-stage decoder upsample 16→32→64→128, mask MLP output widened to `embed_dim//4` (64). Skip fuses at 32×32 stage. Smoke test masks.shape=(2,15,128,128).
- 2026-04-19 13:20 — STEP 3 landed: loss now upsamples preds 128→256 bilinear to GT res instead of downsampling GT (restores small-organ supervision).
- 2026-04-19 13:30 — STEP 10 landed: `FlowMatchedOrganConditionedODE._consistency_pairs` emits (h_{i→i+1}^pred, h_{i+1}^real·stop_grad) for both fwd and bwd ODE. L_xsc = 0.25 · MSE. XSC pairs attached to `flow_targets["xsc_pairs"]`.
- 2026-04-19 13:40 — STEP 4 landed: deep-sup now covers 5 slices (mid±2), logits (B, 5, 15, 64, 64). Loss applies quadratic weight falloff centered at mid slice.
- 2026-04-19 13:45 — Config upgrades: LoRA 16→32, transformer_depth 4→6, mlp_dim 2048→3072, added `lambda_xsc: 0.25`.

**Full-pipeline smoke test (B=4, D=8, fp16, 256²):**
- Trainable params: 15.4M (V4 was ~10M)
- VRAM peak: 4.00 GB (plenty of headroom on RTX 4090 24GB)
- Loss components at random init: mask=0.80, flow=0.10, deepsup=0.71, anatomy=0.0, xsc=0.002, total=0.92
- All 8 V4 unit tests pass after update.

## Pilot run (V5) — launched 2026-04-19

### Pilot attempt 1 — STOPPED ep8 (90° rotation instability)

| Epoch | Train Loss | Val Dice |
|-------|-----------|----------|
| 0 | 0.88 | 0.0206 |
| 5 | 0.52 | 0.0432 |
| 8 | 0.50 | **0.0000** (collapse) |

Diagnosis: the 90° random rotation aug destabilises the OAP anatomical priors.
Abdominal CT has a canonical upright orientation; rotating 3/4 of training samples
by 90°/180°/270° teaches the model orientation-invariance, but the OAP queries
encode a fixed (y, x, scale) prior for canonical orientation. The learned content
drifts to match rotated inputs → at val time (no aug) the OAP-positioned queries
fire in wrong locations → IOU collapses to zero.

**Fix applied**: Removed `k * 90°` rotation. Kept H-flip, ±10°, intensity jitter,
z-jitter. Restarting pilot.

### Pilot attempt 2


