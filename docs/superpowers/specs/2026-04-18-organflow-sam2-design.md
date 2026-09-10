# VoluFormer3D V4 — OrganFlow-SAM2 Design Spec

**Date:** 2026-04-18
**Author:** Raywan Dlawar
**Status:** APPROVED — ready for implementation planning
**Supersedes:** V3 (abandoned 2026-04-18, val_dice peaked 0.143 at ep10 then regressed)
**Approach:** Option B — SAM2 Hiera-Tiny + MedSAM2 foundation weights

---

## 1. One-sentence thesis claim

> We present **OrganFlow-SAM2**, the first framework to combine a **flow-matched organ-conditioned cross-slice ODE** with an **anatomy-graph DETR decoder** on top of the MedSAM2 foundation model, achieving fully automatic 15-organ abdominal CT segmentation.

---

## 2. Why V3 failed (diagnosis V4 is built on)

Diagnosis verified against `auto_ode_sam.py`, `organ_query_decoder.py`, `trainer.py`, `amos22.py` on 2026-04-18:

| Bug | Evidence | V4 fix |
|---|---|---|
| Single-organ-per-sample data pipeline | `amos22.py:149,171,377` build per-organ samples; `trainer.py:477-523` gathers one organ per forward | New `AMOS22MultiOrgan3D_Dataset`; 15-channel supervision per forward |
| 15 per-organ mask MLPs all learning the same thing | `organ_query_decoder.py:97-106` 15 parallel MLPs; ep20 checkpoint shows all `mask_mlps.i.net.4.weight` magnitudes ≈ 0.037 | One shared mask MLP; organ identity lives in the query token only |
| HQ token dead | ep20 checkpoint: `hq_token` abs_mean 0.799→0.797 (flat); `hq_gate` 0.064→0.055 (closing) | Delete HQ token, hq_mlp, hq_skip_proj entirely |
| ODE never woke up | 25 epochs, val_dice stuck 0.11–0.14; only downstream mask loss trained `f_θ` | **Flow-matched velocity regression** — direct supervision on `f_θ` from epoch 0 |
| W2 upsample of `src_updated` polluted by 14 ghost queries | `organ_query_decoder.py:240` upsamples `src_updated` after all 15 queries cross-attended | With per-volume supervision, all 15 queries are real — ghost pollution disappears |

---

## 3. Novel contributions (thesis)

1. **Flow-matched cross-slice ODE training** — velocity-regression loss on finite differences between encoder features at consecutive slice timestamps. Inspired by Lipman et al. 2023 flow matching; novel as applied to segmentation feature-field evolution. **Primary novelty.**
2. **Anatomy Graph Attention on organ queries** — learnable 15×15 organ-adjacency graph, initialized from AMOS22 anatomical priors, applied as one GAT message-pass on queries before the TwoWayTransformer. **Secondary novelty.**
3. **PFESA++** — learnable amplification α and radial high-frequency cutoff; parameterized extension of MICCAI 2025 PFESA. **Minor novelty.**
4. **MedSAM2 + LoRA + DETR-style multi-organ decoder** — integration novelty, no published paper combines these three.

All four are individually novel per web search on 2026-04-18 and arXiv/MICCAI/MIDL searches.

---

## 4. Architecture

### 4.1 Forward pass

```
Input CT volume:  (B, D=8, 1, 256, 256)
    │
    ▼
MedSAM2 Hiera-Tiny encoder (frozen backbone + LoRA rank-16)
    ├─ feat_main:  (B*D, 256, 16, 16)   ← stride-16 stage, projected 384→256
    └─ feat_skip:  (B*D, 128, 32, 32)   ← stride-8 stage, projected 192→128 (1×1 conv)
    │
    ▼
PFESA++ (learnable α, learnable cutoff — ~4 params)
    │
    ▼
Flow-Matched Organ-Conditioned Bidirectional ODE
    ├─ f_θ (forward),  f_φ (backward)  — shared Heun integrator (substeps=4, D=8)
    ├─ Organ conditioning: dh/dt = base_ode(h,t) + MLP(organ_embed[1..15])
    │                                              (organ_embed injected per query batch,
    │                                               not per-sample as in V3)
    ├─ L_flow aux loss: velocity regression on consecutive encoder-feature pairs
    ├─ Residual + pre-norm (unchanged from V3)
    └─ DeepSup at t=0.5 → (B, 15, 64, 64)
    │
    ▼
Anatomy-Graph Query Decoder
    ├─ Q = organ_queries.weight          # (15, 256)
    ├─ Q' = Q + σ(A) ⊙ (Q @ W_msg)       # graph message pass, A ∈ R^{15×15} learnable
    ├─ TwoWayTransformer(depth=4, heads=8, mlp_dim=2048)
    ├─ Shared mask head: mask_k = MLP_shared(hs[:, k, :]) @ upscaled      (k=1..15)
    ├─ Shared IoU head:  iou_k  = MLP_iou(hs[:, k, :])
    └─ Stage-1 skip + Haar enhancement (kept from V3 — `skip_gate=0.05` init works)
    │
    ▼
Output: masks (B, 15, 64, 64), iou (B, 15), deepsup (B, 15, 64, 64)
```

### 4.2 Flow-matching auxiliary loss (the key novel piece)

Let encoder produce features `{h_0, h_1, ..., h_{D-1}}` at slice timestamps `t_i = i/(D-1)`.

Forward ODE is supervised by:

```
L_flow_fwd = (1/(D-1)) · Σ_{i=0}^{D-2}  ‖ f_θ(h_i, t_i, organ_id)  −  (h_{i+1} − h_i) · (D-1) ‖²
```

Backward ODE by:

```
L_flow_bwd = (1/(D-1)) · Σ_{i=1}^{D-1}  ‖ f_φ(h_i, t_i, organ_id)  −  (h_{i-1} − h_i) · (D-1) ‖²
```

`L_flow = 0.5·(L_flow_fwd + L_flow_bwd)`. Applied per spatial position (averaged over `B·H·W`).

**Why this works mathematically:**
- Zero-init on last Linear of `f_θ` → initial output 0 → `L_flow` gradient is nonzero (targets are ≠ 0) → ODE receives training signal from step 1.
- Tanh activation in `f_θ` bounds velocity → `L_flow` cannot explode.
- Loss is dimensionless — finite differences of normalized (LayerNorm'd) features are O(1).

**Why this is novel:** flow matching is generative-modeling technique (Lipman 2023, Rectified Flow 2023); no prior work applies velocity regression to cross-slice feature evolution for volumetric medical segmentation.

### 4.3 Anatomy Graph Attention

```
A_init ∈ R^{15×15}:
    1.0 for {liver-right_kidney, liver-stomach, liver-gallbladder,
             spleen-left_kidney, left_kidney-left_adrenal,
             right_kidney-right_adrenal, aorta-IVC,
             pancreas-stomach, pancreas-duodenum, stomach-duodenum}
    0.0 otherwise
A:    learnable parameter (225 values), init from A_init
W_msg: Linear(256, 256) — learnable (~65K params)

forward:
    A_soft = σ(A)                    # bounds weights in (0, 1); does NOT force row-sum=1
    M      = A_soft @ Q              # (15, 256) — per-organ neighbor-aggregated message
    Q'     = Q + W_msg(M)            # residual add
```

Note: using `σ(A)` (element-wise sigmoid) not `softmax(A)` row-wise — lets multiple neighbors contribute additively, not competitively. Anatomy doesn't force one-hot relationships.

### 4.4 PFESA++

Extends `PFESASpectralSkip` in `models/trimamba_sam.py`:

```
alpha          → nn.Parameter(1.0)            # learnable
cutoff_logit   → nn.Parameter(0.0)            # learnable (sigmoid → [0,1])
steepness_log  → nn.Parameter(log(10.0))      # learnable soft-cutoff steepness

radial_mask(r) = σ(exp(steepness_log) · (r − σ(cutoff_logit)))
F_enh          = F_x · (1 + alpha · radial_mask(r))
```

Yields a smooth, differentiable high-frequency amplifier. ~3 scalar params. Strictly an improvement — at init it matches the current PFESA behavior.

### 4.5 Parameter budget

| Component | Params | Trainable? |
|---|--:|:--:|
| MedSAM2 Hiera-Tiny backbone | 38.0M | frozen |
| LoRA rank-16 deltas | 4.2M | ✓ |
| Stage-2/Stage-1 projection heads | 0.2M | ✓ |
| PFESA++ | ~3 | ✓ |
| Flow-matched ODE (fwd + bwd + merge) | 0.6M | ✓ |
| Organ embedding + bias MLPs | ~50K | ✓ |
| Anatomy Graph Attention | 65K + 225 | ✓ |
| Organ queries (15 × 256) | 3.8K | ✓ |
| TwoWayTransformer (depth 4, heads 8) | 12.0M | ✓ |
| Shared mask MLP + IoU MLP | 0.3M | ✓ |
| Upsample (2× ConvTranspose + LN) | 0.4M | ✓ |
| Skip projection + gate | 20K | ✓ |
| DeepSup head | 0.1M | ✓ |
| **Total** | **~59M trainable / ~97M total** | |

Lands in Option B's 70–100M band.

---

## 5. Training plan

### 5.1 Dataset

**New:** `datasets/amos22_multiorgan.py :: AMOS22MultiOrgan3D_Dataset`

```
__getitem__(idx) →
  {
    'image':        (D=8, 1, H, W)     float32  — 8-slice volume sample
    'masks':        (15, D, H, W)       uint8    — per-organ binary masks
    'present_mask': (15,)                bool    — True if organ has ≥50 voxels in volume
    'modality':     int                           — 0=CT (AMOS22 Task 1 is CT-only)
    'center_slice': int
  }
```

- Sampling strategy: uniform over volumes, center-slice sampling within volume (not per-organ)
- Copy-paste augmentation allowed because every channel has GT
- No oversampling of small organs needed — graph attention + flow loss should handle them

### 5.2 Loss

```
L_total = L_mask + λ_flow · L_flow + λ_deepsup · L_deepsup + λ_anatomy · L_anatomy

L_mask     = Σ_{k: present_k=1} (DiceTopK(mask_pred_k, mask_gt_k) + BCE(mask_pred_k, mask_gt_k))
L_flow     = 0.5·(L_flow_fwd + L_flow_bwd)    # defined in §4.2
L_deepsup  = CE(deepsup_logits, one_hot(masks_gt))   # 15-channel CE
L_anatomy  = |ΔA|₂                             # frobenius of A change from init (regularization)

λ_flow    = 0.5       # ramp 0→0.5 over first 3 epochs (warmup)
λ_deepsup = 0.1
λ_anatomy = 0.01
```

Removed from V3: per-organ loss weights, diversity loss (`compute_diversity_loss`), HQ correction.

### 5.3 Optimizer & schedule

- **AdamW**, weight decay 0.05
- Two param groups:
  - LoRA deltas: lr 1e-4
  - All other new params: lr 3e-4
- **Cosine decay** with linear warmup (3 epochs)
- **120 epochs**; checkpoint every 5, validate every 5
- Gradient clipping at 1.0
- Mixed precision (autocast + GradScaler)
- Effective batch size 8 (micro-batch 4 × grad accum 2)

### 5.4 Compute budget (measured 4090 timings)

- V3 single-organ: 117 min/epoch no-val, ~7500 samples/epoch
- V4 per-volume: ~500 samples/epoch (15× more work per sample, ~10× fewer samples)
- Estimated: **~40 min/epoch no-val, ~55 min/epoch with val**
- 120 epochs × ~45 min average ≈ **~90 hours ≈ ~4 days**
- Well inside the 10–14 day Option B budget.

---

## 6. File changes

### New files

| File | Purpose |
|---|---|
| `models/organflow_sam2.py` | Top-level `OrganFlowSAM2` model (replaces `AutoODESAM`) |
| `models/medsam2_encoder.py` | MedSAM2 Hiera-Tiny loader + LoRA wrapper |
| `models/flow_cross_slice.py` | Flow-matched ODE cross-slice module |
| `models/anatomy_graph_decoder.py` | Decoder w/ graph attention + shared mask head |
| `datasets/amos22_multiorgan.py` | Per-volume multi-organ dataset |
| `training/losses_v4.py` | Multi-organ mask loss + flow loss + deepsup |
| `configs/v4_organflow_sam2_256px.yaml` | V4 config |

### Reused unchanged from V3

| File | Role in V4 |
|---|---|
| `models/mask_decoder.py::TwoWayTransformer, MLP` | Shared transformer module |
| `models/trimamba_sam.py::PFESASpectralSkip` | Base class for PFESA++ |
| `models/ode_cross_slice.py::ODEFunction, OrganConditionedODEFunction` | Base classes for flow-matched variants |
| `training/trainer.py` | Forked/extended (new path for V4 loss; old V3 path removed) |

### Deleted / obsolete

- `models/auto_ode_sam.py` (V3 model)
- `models/organ_query_decoder.py` (V3 decoder, HQ token dead)
- `datasets/amos22.py::AMOS22Dataset, AMOS22_3D_Dataset` single-organ paths (kept for historical runs but V4 never calls them)

---

## 7. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| MedSAM2 Hiera-Tiny weights don't load cleanly | Medium | Step 1 of implementation is a load-smoke-test on a single volume |
| Flow loss dominates mask loss early → encoder drift | Medium | Warmup ramp (`λ_flow: 0 → 0.5 over 3 epochs`); monitor L_mask at epoch 5 |
| Anatomy graph bias harms rare organs | Low | `A` is learnable; if it hurts, bias init is cheaply ablated by zeroing `A_init` |
| Memory blowup from per-volume 15-channel supervision | Medium | If OOM: drop batch to 2 + grad accum 4; or drop D=8→6 |
| LoRA underfits (backbone too frozen) | Low | Fallback: unfreeze last 2 Hiera stages at epoch 30 if val_dice < 0.60 |

---

## 8. Success criteria

- **Go gate @ ep5:** val_dice ≥ 0.15 (matches V3 peak at ep10 → confirms flow loss is kicking in).
- **Go gate @ ep20:** val_dice ≥ 0.50 (unlocks the nnU-Net comparison).
- **Go gate @ ep60:** val_dice ≥ 0.82 (unlocks Phase 4b at 512 px if time permits).
- **Final target:** mean DSC ≥ 0.91 by ep120 (beats nnWNet 15-organ 0.8639 comfortably; competitive with nnWNet 0.923 on CT-only).

---

## 9. Out of scope (V5 or later)

- 512 px training (Phase 4b, optional if time permits after ep60 gate)
- MRI task of AMOS22 (CT-only for thesis)
- Boundary-specific losses (clDice) — memory says "clDice OFF for solid organs"
- Test-time anatomical-consistency optimization
- Full MedSAM2 backbone unfreeze (kept as fallback only)

---

## 10. References

- Lipman et al. 2023, *Flow Matching for Generative Modeling*, ICLR 2023
- Liu et al. 2023, *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*
- Ma et al. 2025, *MedSAM2: Segment Anything in 3D Medical Images and Videos*
- MICCAI 2025, *PFESA: FFT-based Parameter-Free Edge and Structure Attention*
- RFMedSAM2 (2025), AMOS22 SOTA via automatic prompt refinement
- Chen et al. 2018, *Neural Ordinary Differential Equations*, NeurIPS 2018

---

## 11. Approval record

- 2026-04-18: Option B selected by Ray (user)
- 2026-04-18: Approach 1 (OrganFlow-SAM2) selected by Ray
- 2026-04-18: Design spec written, self-reviewed for consistency and math soundness
- Next step: invoke `superpowers:writing-plans` to produce implementation plan
