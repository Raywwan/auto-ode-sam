# MASTER PROJECT LOG — VoluFormer3D (LiteSAM-3D V3)

**Author:** Ray (Masters researcher)
**Project:** VoluFormer3D = LiteSAM-3D V3 → **V4 (OrganFlow-SAM2)** as of 2026-04-18
**Document created:** 2026-04-15
**Last updated:** 2026-04-18
**Status:** V3 ABANDONED 2026-04-18. V4 design approved, implementation pending. New code at `C:\Users\Raywa\Desktop\VoluFormer3D_V4\`.

---

## Document Purpose & Update Protocol

This document is the **single source of truth** for the VoluFormer3D masters thesis project. It captures every architectural decision, experimental result, code change, training fix, novelty claim, competitor comparison, and roadmap milestone. Anyone reading only this file should be able to fully understand the project state, history, and trajectory.

### Update Protocol

When a new training run completes, a new architectural change lands, or a strategic decision is made, update this file as follows:

1. **Update `Quick Reference`** at the top with the latest status, best result, and resume checkpoint.
2. **Append** new epoch-by-epoch tables under the appropriate `Phase` section. **Never overwrite** prior tables — they form the experimental record.
3. **Add a dated entry** under `Change Log` (bottom of file) with a one-line summary of what changed.
4. **Update `Complete Results Ranking`** if a new run beats an existing entry or adds a new model.
5. **Update `Phase Roadmap`** statuses (RUNNING / COMPLETE / BLOCKED).
6. **Never delete numbers.** If a result is superseded, mark it `[superseded by X]` but keep it.
7. **Keep all dates in ISO-8601 (YYYY-MM-DD).**
8. **Never rewrite old Phase 3a (v1) logs.** The v1 run remains as historical record; v2 is appended below.

---

## Quick Reference

| Field | Value |
|-------|-------|
| **Active track** | **V4 OrganFlow-SAM2** (design approved 2026-04-18, implementation pending) |
| **Project best (liver, 256 px)** | ODE-SAM V2 — DSC **0.9358**, HD95 **0.56 mm**, NSD **0.9678** (ep15) |
| **Project best (15-organ)** | N/A yet — V3 abandoned at val_dice 0.1427 (ep10 peak), V4 targets ≥ 0.91 |
| **V3 status** | **ABANDONED 2026-04-18** — diagnosed as fundamentally miswired (single-organ data pipeline starves multi-organ decoder); see `docs/superpowers/specs/2026-04-18-organflow-sam2-design.md` in V4 folder for full root cause. |
| **V4 status** | Design spec approved; folder scaffolded at `C:\Users\Raywa\Desktop\VoluFormer3D_V4\`; implementation plan pending (`superpowers:writing-plans`). |
| **V4 target** | ≥ 0.91 mean DSC by ep120 — beats nnWNet on AMOS22 CT-only. |
| **V4 backbone** | MedSAM2 Hiera-Tiny (frozen) + LoRA rank-16, ~59M trainable / ~97M total. |
| **V4 novel contributions** | (1) Flow-matched cross-slice ODE training, (2) Anatomy Graph Attention decoder, (3) PFESA++ learnable, (4) MedSAM2+LoRA+DETR integration. |
| **V3 run command (historical)** | `C:\Users\Raywa\Desktop\LiteSAM3D\.venv\Scripts\python train.py --config configs/phase3a_autoodesam_256px.yaml` |
| **V3 latest checkpoint (preserved)** | `C:\Users\Raywa\Desktop\VoluFormer3D\checkpoints\phase3a_autoodesam_256px_15org_v3\...ep20.pt` |
| **V4 run command (future)** | `cd VoluFormer3D_V4 && ...python train.py --config configs/v4_organflow_sam2_256px.yaml` |
| **Hardware** | RTX 4090 (24 GB), Ryzen 9 5900X, 32 GB RAM, Windows 11 |
| **Dataset** | AMOS22 — 500 CT volumes, 15 abdominal organs |
| **V3 code root (frozen, for reference)** | `C:\Users\Raywa\Desktop\VoluFormer3D\` |
| **V4 code root (active)** | `C:\Users\Raywa\Desktop\VoluFormer3D_V4\` |
| **Data root** | `C:\Users\Raywa\Desktop\LiteSAM3D\data\amos22\` |
| **venv** | `C:\Users\Raywa\Desktop\LiteSAM3D\.venv` |

### Phase 3a v3 — Checkpoint Inventory (fresh start — not yet run)

| Checkpoint | Epoch | val_dice | val_HD95 | val_NSD | val_loss | train_dice | Saved |
|------------|------:|---------:|---------:|--------:|---------:|-----------:|-------|
| (none yet) | — | — | — | — | — | — | — |

### Phase 3a v2 — Checkpoint Inventory (STOPPED, historical)

| Checkpoint | Epoch | val_dice | val_HD95 | val_NSD | val_loss | train_dice | Saved |
|------------|------:|---------:|---------:|--------:|---------:|-----------:|-------|
| epoch000.pt | 0 | 0.0067 | 20.12 mm | 0.0180 | 0.8406 | 0.0068 | 2026-04-14 17:53 |
| epoch005.pt / latest.pt / best.pt | **5** | **0.1074** | **11.92 mm** | **0.1858** | **0.7193** | **0.1048** | 2026-04-15 05:12 |

### Critical Watch Items (v3 — Epoch 5)

- **v3 not yet started — run the command to begin fresh start.**
- **ep5 val_dice must be ≥ 0.10** — comparable to v2 (0.1074). If below, the W2 fix may need a learning rate warmup adjustment.
- **ep5 val_dice > 0.12** would suggest the W2 fix (transformer-updated features) is delivering measurable gain vs v2.
- ep10 val_dice: compare against v2 ep10 (expected ~0.20–0.35 based on convergence estimate). If v3 tracks above v2, the architectural fix is confirmed working.
- skip_gate at v3 ep0 = 0.05 (not 0.0 as in v2) — Stage 1 boundary features should contribute from the first step. Watch HD95 / NSD — these benefit most from the skip path being open.

---

## 1. Project Identity & Goal

VoluFormer3D (formerly LiteSAM-3D V3) is a **lightweight, fully automatic SAM-based 3D CT segmentation framework** targeting the AMOS22 abdominal organ benchmark (15 organs, 500 CT volumes). The project is intended for publication at MICCAI / MIDL workshop or main-track level, and forms the basis of Ray's masters thesis.

The thesis claim is one sentence:

> *We present Auto-ODE-SAM V3, the first framework to model continuous anatomical dynamics across CT slices using a bidirectional organ-conditioned Neural ODE within a SAM-based segmentation architecture, achieving fully automatic 15-organ abdominal CT segmentation at 21.2 M parameters on AMOS22.*

---

## 2. Project Evolution: V1 → V2 → V3

### V1 — Failed
Wrong dataset class — DA-ISA never actually ran. The ablation study reported in V1 was **invalid**; numbers cannot be trusted. Discarded.

### V2 — Diagnosed Failure
- Used **TinyViT-21M Stage 3** at 256 px → 8 × 8 = **64 tokens**.
- Too coarse for cross-slice attention: DA-ISA had nothing meaningful to attend to.
- Critical bug in `multiscale_encoder.py`: `feature_info[-1]` always returned Stage 3 channels (576) regardless of `stage_index`. Caused `RuntimeError` when Stage 2 was selected (projection layer built for 576 ch but Stage 2 outputs 384 ch). **Fixed** to `feature_info[stage_index]`.

### V3 — The Fix (current project)
Move from **Stage 3 to Stage 2** — yielding 4× richer spatial resolution and 16× more tokens at 256 px.

| Stage | Spatial @512 px | Spatial @256 px | Tokens @256 px | Channels |
|-------|----------------|----------------|----------------|----------|
| 0 | 128 × 128 | 64 × 64 | 16,384 | 96 |
| 1 | 64 × 64 | 32 × 32 | 4,096 | 192 |
| **2 (V3)** | **32 × 32** | **16 × 16** | **1,024** | **384 → proj 256** |
| 3 (V2) | 16 × 16 | 8 × 8 | 256 | 576 → proj 256 |

V3 also uses **Stage 1 as a skip connection** to the decoder.

---

## 3. Architecture: Auto-ODE-SAM V3 (Phase 3 Model)

**Total parameters: 21,196,379 (≈ 21.2 M)**

### 3.1 End-to-End Forward Pass

```
CT Volume Input (B, D=8, 1, H, W)
│
▼
COMPONENT 1: MultiScaleEncoder (TinyViT-21M Stage 2)
  13.4 M params, pretrained ImageNet-21k
  main: (B*D, 256, 16, 16)  — Stage 2
  skip: (B*D, 64,  32, 32)  — Stage 1
│
▼
COMPONENT 2: PFESA Spectral Enhancement (zero params)
  F_enh = F_x * (1 + alpha * hf_mask)
  hf_mask = 1 where freq_radius > highfreq_cutoff
│
▼
COMPONENT 3: Organ-Conditioned Bidirectional Neural ODE
              (PRIMARY NOVEL CONTRIBUTION)
  dh/dt = f_θ(h, t) + MLP(organ_embed[organ_id])
  Heun's method (RK2), substeps = 4
  Forward ODE:  h(0) = feat[:, 0]     (t: 0 → 1)
  Backward ODE: h(0) = feat[:, D-1]   (flipped back post-hoc)
  merge = Linear(2C → C, bias=False), zero-init weight
  Residual + PRE-norm LayerNorm on ode_context
│
├─────────────────────────────┐
▼                             ▼
COMPONENT 4: DeepSup head     COMPONENT 5: OrganQueryDecoder
(cached at ODE midpoint)      15 learned queries + HQ token
Loss weight: 0.1              TwoWayTransformer (depth=2, heads=8)
                              Per-organ mask MLPs + IoU heads
                              Diversity regularisation (|cos|, w=0.01)
│                             │
▼                             ▼
COMPONENT 6: MaskDecoder (SAM-modified)
  HQ-SAM output token (fine boundary recovery)
  Stage 1 skip injection at 32 × 32
  FiLM conditioning (from MCP-MedSAM — NOT novel, cite it)
  4× upsampling → 15 mask logits (B, 15, 256, 256)
│
▼
INFERENCE ONLY: Two-Stage Zoom-In
  For organs {4 gallbladder, 5 esophagus, 11/12 adrenals, 13 duodenum}
  Pass 1: coarse 256 px → extract bbox
  Pass 2: 4× effective resolution
```

### 3.2 Parameter Breakdown

| Component | Parameters |
|-----------|-----------:|
| TinyViT-21M encoder | 13,358,684 |
| Stage 1 skip projection | ~12,000 |
| PFESA spectral enhancement | 0 |
| ODE module (organ-conditioned BiDir ODE) | 225,024 |
| Organ conditioning (embeddings + bias MLP) | ~2,500 |
| OrganQueryDecoder | ~7,586,064 |
| Deep supervision head | 26,607 |
| MaskDecoder (HQ-SAM + FiLM) | ~200,000 |
| **Total** | **21,196,379** |

### 3.3 ODEFunction (`models/ode_cross_slice.py`)

Architecture (verified from source):

```
Time encoding: t → [sin(2π t · k), cos(2π t · k)]   for k = 1 .. n_freqs
Linear(dim + 2·n_freqs → hidden)     → Tanh
Linear(hidden → hidden)              → Tanh
Linear(hidden → dim)                 ← ZERO-INITIALIZED (weight AND bias)
```

Phase 3a v2 instantiation: `dim=256, hidden=128, n_freqs=6`.

The last `Linear`'s **weight AND bias** are both zero-initialised (`nn.init.zeros_(self.net[-1].weight)` and `nn.init.zeros_(self.net[-1].bias)`). Consequence:

- `dh/dt = 0` for all (h, t) at epoch 0.
- ODE is the **identity function** at initialisation.
- Model starts exactly at the Stage2-noISA baseline — the safety property that lets the ODE learn productively rather than damaging the encoder.

### 3.4 Bidirectional Neural ODE (unconditioned variant, used by ODE-SAM)

`BidirectionalNeuralODECrossSlice` runs two ODE instances in parallel:

- Forward ODE: `h(0) = feat[:, 0]`, evolve t ∈ [0, 1] (top → bottom).
- Backward ODE: `h(0) = feat[:, D-1]`, evolve t ∈ [0, 1]; trajectory is then flipped along the depth axis to realign with the forward one.

**Heun's method (RK2) integration** (from source):

```
for each interval i in [1..D-1]:
    sub_dt = (t_end - t_start) / substeps
    for j in [0..substeps-1]:
        t_curr = t_start + j * sub_dt
        k1 = f(h, t_curr)
        h_pred = h + sub_dt * k1
        k2 = f(h_pred, t_curr + sub_dt)
        h = h + 0.5 * sub_dt * (k1 + k2)
```

- Local truncation error O(dt³) vs O(dt²) for forward Euler → halved per step.
- With `substeps = 4` and `D = 8`: 7 intervals × 4 sub-steps = **28 ODE evaluations per direction** (56 total for bidirectional).
- Extra cost is small because D is small.

**Merge** of forward and backward trajectories:

```
ode_context = Linear(dim*2 → dim, bias=False)([traj_fwd ; traj_bwd])
```

- Weight **zero-initialised** (`nn.init.zeros_(self.merge.weight)`).
- Result: `ode_context = 0` at init → residual add is identity.

**PRE-norm residual**: LayerNorm is applied to `ode_context` **before** the residual add:

```
return features + norm(ode_context)
```

This is important: when `merge.weight = 0` → `ode_context = 0` → `norm(0) = 0` → output = features **exactly**. Post-norm (`norm(features + ode_context)`) could not satisfy this property because LayerNorm would still rescale the pure `features` input.

### 3.5 OrganConditionedODEFunction

```
dh/dt = base_ode(h, t) + bias_mlp(organ_embed[organ_id])
```

- `base_ode`: a full `ODEFunction` (dim, hidden, n_freqs).
- `organ_embed = nn.Embedding(n_organs + 1, organ_emb_dim)` — size is `n_organs + 1 = 16` because `organ_id` is **1-indexed** (1..15) and used directly as an embedding index. Phase 3a v2 uses `organ_emb_dim = 64`.
- `bias_mlp = nn.Sequential(nn.Linear(organ_emb_dim, dim))` — **both weight and bias are zero-initialised** (`nn.init.zeros_(self.bias_mlp[0].weight)` and `nn.init.zeros_(self.bias_mlp[0].bias)`). At init, the conditioning contributes zero bias → model degrades to the base ODE.
- **`organ_id` expand strategy** inside `forward()`:
  ```
  N = h.shape[0]   # = B * H * W
  HW = N // B
  organ_id_expanded = organ_id.repeat_interleave(HW)   # (B,) → (B*H*W,)
  organ_emb = self.organ_embed(organ_id_expanded)      # (N, organ_emb_dim)
  bias = self.bias_mlp(organ_emb)                      # (N, dim)
  return self.base_ode(h, t) + bias
  ```

### 3.6 OrganConditionedBidirectionalNeuralODE

Wraps two `OrganConditionedODEFunction` instances (forward + backward). Input validation is asserted once up front:

```
assert organ_id.min() >= 0 and organ_id.max() <= self.ode_fwd.n_organs
assert organ_id.shape[0] == features.shape[0]
```

Returns `features + norm(ode_context)` (same residual + PRE-norm structure as the unconditioned variant).

### 3.7 AutoODESAM Class (`models/auto_ode_sam.py`)

- **Sinusoidal dense positional encoding** uses the SAM-convention temperature scaling: `1 / (10000^(2i / half))`. This is **NOT** `1 / 2**i`; the latter would alias for indices > 14 at feature size 16 × 16.
- `_deepsup_cache`: set as a side effect of `encode_image()` (populated at the ODE midpoint, pre-PFESA), consumed in `forward()`. Trains the cross-slice trajectory independently of spectral post-processing.
- PE resizing: if the feature shape is not the default 16 × 16, the dense_pe buffer is bilinearly interpolated to match.
- `forward()` returns a dict:
  ```
  {
    "masks":          (B, K, H, W),
    "iou_pred":       (B, K),
    "organ_ids":      list,
    "deepsup_logits": (B, n_organs, H, W),
  }
  ```

#### DeepSupervisionHead

```
Conv2d(embed_dim, embed_dim//4, k=1)                 → GELU
ConvTranspose2d(embed_dim//4, embed_dim//8, k=2, s=2) → GELU
ConvTranspose2d(embed_dim//8, n_organs,    k=2, s=2)
Output: (B, n_organs, H_feat*4, W_feat*4)
```

Applied at the **ODE midpoint** (D // 2 slice), pre-PFESA. Auxiliary loss with weight 0.1.

### 3.8 OrganQueryDecoder (`models/organ_query_decoder.py`)

Module-level constants:

```
N_ORGANS = 15
SMALL_ORGAN_IDS = [4, 5, 11, 12, 13]
```

Learned parameters:

- `organ_queries = nn.Embedding(15, embed_dim)` with `nn.init.normal_(std=0.02)`
- `hq_token = nn.Embedding(1, embed_dim)` — fine-boundary recovery.
- `skip_gate = nn.Parameter(torch.zeros(1))` — Stage 1 skip starts **closed**.
- `hq_gate = nn.Parameter(torch.zeros(1))` — HQ correction starts **off**.

Upsampling chain (total 4×):

```
ConvTranspose2d(embed_dim → embed_dim//4, k=2, s=2)  → LayerNorm → GELU
ConvTranspose2d(embed_dim//4 → embed_dim//8, k=2, s=2)
```

#### HQ `skip_proj` zero-init strategy (asymmetric, deliberate)

```python
nn.init.zeros_(self.hq_skip_proj[0].weight)   # weight = 0
# bias left at default PyTorch init (NOT zero)
```

Rationale (from code comment): if **both** weight and bias were zero, then `hq_gate.grad` would be identically zero forever — the HQ head would be permanently dead. Leaving the bias at its default init allows a non-zero forward signal through the HQ pathway, which gives `hq_gate` a non-zero gradient and bootstraps learning.

#### `_haar_edge_enhance()` (static method, zero-param)

Exact formula (from source):

```
x00 = x[:, :, 0::2, 0::2]
x01 = x[:, :, 0::2, 1::2]
x10 = x[:, :, 1::2, 0::2]
x11 = x[:, :, 1::2, 1::2]

LH = (x00 - x01 + x10 - x11) * 0.25
HL = (x00 + x01 - x10 - x11) * 0.25
HH = (x00 - x01 - x10 + x11) * 0.25

hf = |LH| + |HL| + |HH|
result = x + 0.5 * upsample(hf, orig_size)
```

Handles odd-dimension padding (`pad_h = orig_h % 2`, `pad_w = orig_w % 2`) and trims the output back to original size after the 2× upsample.

#### `compute_diversity_loss()`

**Uses ABSOLUTE cosine similarity** — critical distinction:

```
sim_matrix = cos_sim(queries, queries)
diversity_loss = (sim_matrix.abs() * off_diagonal_mask).sum() / (n_organs * (n_organs - 1))
loss = diversity_loss_weight * diversity_loss
```

Without `abs()`, tokens could collapse to **antipodal pairs**: cos_sim = −1 would register as a "diverse" pair (large negative similarity), and the sum would be minimised by pushing pairs to be antiparallel — a degenerate solution. Absolute value penalises both positive AND negative correlation, ensuring true orthogonality.

#### `forward()` flow

1. Gather organ queries for target organ IDs (1-indexed → 0-indexed mapping).
2. Append HQ token → `all_queries` shape `(K+1, embed_dim)`.
3. `query_pe = zeros_like(queries)` — no positional encoding on the query side.
4. Upsample image features 4× via two ConvTranspose2d steps with LayerNorm + GELU.
5. Stage 1 skip injection (if provided): align size, `skip_ln`, Haar edge enhance, then `upscaled += skip_gate * sf`.
6. Flatten image embeddings to `(B, H*W, C)` for transformer input.
7. `TwoWayTransformer` produces `hs (B, K+1, embed_dim)`.
8. Runtime assertion: `hs.shape[1] == K + 1`.
9. Per-organ: `weights = mask_mlps[organ_idx](token_out)` → dot with upscaled feature map → `(B, 1, H_out, W_out)`.
10. HQ: `hq_feat = hq_gate * hq_skip_proj(sf_hq)`; `hq_mask = hq_weights @ hq_feat` is added to all organ masks.

### 3.9 MaskDecoder (`models/mask_decoder.py`)

#### FiLM layer — CRITICAL BUG FIX (documented in code)

Previous (broken):
```python
nn.init.ones_(self.gamma_proj.weight)   # WRONG
# At init, gamma(c) = sum(c_i) — not 1
```

Current (fixed):
```python
nn.init.zeros_(self.gamma_proj.weight)
nn.init.ones_(self.gamma_proj.bias)   # gamma(c) = 1 exactly at init
nn.init.zeros_(self.beta_proj.weight)
nn.init.zeros_(self.beta_proj.bias)   # beta(c) = 0 at init
```

FiLM formula: `y = gamma(condition) * x + beta(condition)`, where `gamma`, `beta` are linear projections of the modality embedding. At init → `y = 1 * x + 0 = x` (identity).

#### Tokens

- `iou_token = nn.Embedding(1, embed_dim)`
- `mask_tokens = nn.Embedding(3, embed_dim)` — 3 candidates for multimask output.
- `hq_token = nn.Embedding(1, embed_dim)`
- All three prepended to `sparse_prompt_embeddings` before the transformer.

#### Skip connection in MaskDecoder

- Stage 1 features: 64 channels, 32 × 32 at 256 px input.
- Injected **after** `upsample_conv1` (at 32 × 32 resolution).
- `skip_gate` parameter starts at 0 (closed); `skip_ln = LayerNorm(64)`.
- Haar edge enhancement applied to skip features **before** gate scaling.

#### TwoWayAttentionBlock (4 steps per block)

1. Self-attention on queries (prompt tokens attend to each other).
2. Cross-attention: queries → keys (image tokens).
3. FFN on queries.
4. Cross-attention: keys → queries (image-to-prompt feedback).

Each with LayerNorm + residual.

#### Attention module

- `internal_dim = embedding_dim // downsample_rate` (default `downsample_rate = 2` → `internal_dim = 128`).
- Q, K, V projected to `internal_dim`, then expanded back to `embedding_dim` at output.

### 3.10 MultiScaleEncoder (`models/multiscale_encoder.py`)

- `out_indices = [1, stage_index]` when `stage_index > 1` — returns both Stage 1 skip and Stage 2 main.
- Stage 1 skip: TinyViT Stage 1 output (192 ch) projected to `embed_dim // 4 = 64` ch.
- Stage 2 main: TinyViT Stage 2 output (384 ch) projected to `embed_dim = 256` ch.
- Projection block: `Conv1×1 → BatchNorm → GELU`.
- Neck on Stage 2: 2× `Conv3×3` residual blocks.
- `skip_channels` attribute: `0` if `stage_index == 1` (no skip), else `embed_dim // 4`.

### 3.11 PFESASpectralSkip (`models/trimamba_sam.py`)

Exact formula:

```
F_enh = F_x * (1 + alpha * hf_mask)
hf_mask = 1  where freq_radius > highfreq_cutoff else 0
```

- Zero parameters — the math is fully deterministic from the spatial-frequency grid.
- Returns the same dtype as the input (fp16-safe).
- Imported from `trimamba_sam.py` and used in both `TriMambaSAM` and `AutoODESAM`.

### 3.12 TriDirectionalMambaModule

- Three scans:
  - **Axial:**    reshape to `(B*H*W, D, C)`
  - **Coronal:**  reshape to `(B*D*W, H, C)`
  - **Sagittal:** reshape to `(B*D*H, W, C)`
- Merge: `direction_gate` parameter of shape `(3,)`, softmax → weighted sum of the three scan outputs.
- Each scan uses `MambaBidirectional1D` (forward + backward Mamba, concat + linear).

### 3.13 All 7 Models + 1 Auto Model Implemented

| # | Model | Cross-Slice Method | Module Params | Total Params | Novelty | File |
|---|-------|--------------------|--------------:|-------------:|---------|------|
| 1 | V2 DA-ISA (baseline) | DA-ISA at Stage 3 | 1.71 M | 19 M | Prior art | `depth_aware_isa.py` |
| 2 | MultiScaleISA | DA-ISA at Stage 2 | 1.71 M | 19.5 M | Stage 2 tap = ours | `voluformer3d.py` |
| 3 | FCA-SAM | FFT along depth axis | 536 | 17.8 M | Novel (FFT depth) | `fca_sam.py` |
| 4 | ACM-SAM | Anatomical memory bank | 280 K | 18.1 M | Incremental | `acm_sam.py` |
| 5 | Stage2-noISA | None (ablation) | 0 | 17.8 M | Ablation | `voluformer3d.py` (isa=off) |
| 6 | TriMamba-SAM | Tri-dir Mamba + SoftMoE + PFESA | 668 K | 18.4 M | Risky — TP-Mamba overlap | `trimamba_sam.py` |
| 7 | **ODE-SAM** | **Bidirectional Neural ODE** | **207 K** | **18.0 M** | **Confirmed novel** | `ode_sam.py` |
| 8 | **Auto-ODE-SAM V3** | **Organ-conditioned BiDir ODE + 15 organ queries** | **225 K** | **21.2 M** | **Novel — first of kind** | `auto_ode_sam.py` |

### 3.14 build_model() Dispatch (`models/__init__.py`)

```
"multiscale_isa" → VoluFormer3D
"fca_sam"        → FCASAM
"acm_sam"        → ACMSAM
"trimamba_sam"   → TriMambaSAM
"ode_sam"        → ODESAM
"auto_ode_sam"   → AutoODESAM   ← Phase 3a
```

### 3.15 Supporting Component Files

- `models/multiscale_encoder.py` — TinyViT Stage 2 tap + Stage 1 skip.
- `models/ode_cross_slice.py` — `ODEFunction`, `BidirectionalNeuralODECrossSlice`, `OrganConditionedODEFunction`, `OrganConditionedBidirectionalNeuralODE`.
- `models/organ_query_decoder.py` — 15 organ query tokens + HQ token + TwoWayTransformer + diversity loss + Haar edge enhance.
- `models/prompt_encoder.py` — Box PE + modality embed + content CNN (from MCP-MedSAM; FiLM NOT novel).
- `models/mask_decoder.py` — Two-way transformer + FiLM (bug-fixed) + HQ-SAM token + 4× upsample.
- `models/mamba_pure.py` — Pure-PyTorch bidirectional Mamba (no `torchdiffeq`).
- `models/soft_moe.py` — Soft MoE organ router (2 experts).
- `inference/zoom_refine.py` — Two-stage zoom-in for small organs.
- `training/losses.py` — `CombinedLoss`: Dice + PMDiceLoss + Boundary + DeepSup + IoU.
- `datasets/amos22.py` — `AMOS22_3D_Dataset` with oversampling + copy-paste aug.

---

## 4. Training Pipeline

### 4.1 CombinedLoss (`training/losses.py`)

#### PMDiceLoss (Pixel-wise Modulated Dice)

Per-voxel modulation weights:

- Foreground: `m_i = (1 - p_i)^gamma` (penalise missed FG — high weight where model is wrong).
- Background: `m_i = p_i^gamma` (penalise false positive).
- `gamma = 2.0` (default).

Formula:

```
PMDice = 1 - (2·Σ(m · p · g) + ε) / (Σ(m · p) + Σ(m · g) + ε)
```

Continuous modulation vs DiceTopK's hard binary top-K mask — smoother gradients. Reference: arXiv 2506.15744.

#### BoundaryLoss

- `scipy.ndimage.distance_transform_edt` on CPU (GT → distance map).
- `loss = mean(pred_prob * dt_map)` — pushes boundary predictions toward the GT surface.
- Computed per-sample, averaged over batch.

#### `sample_weight` in DiceLoss / DiceTopK / CombinedLoss

```
loss_per_sample = loss_per_sample * sample_weight   # (B,) element-wise
return loss_per_sample.mean()
```

Applied BEFORE `.mean()` — correct per-sample weighting (never batch-mean scaling).

#### Phase 3a loss weights

```yaml
w_dice:      0.5   # DiceLoss (primary segmentation objective)
w_focal:     0.0   # disabled
w_cldice:    0.0   # disabled — hurts solid organs
w_iou:       0.1   # IoUPredictionLoss
w_boundary:  0.1   # BoundaryLoss (ramps at epoch ≥ 50)
w_dice_topk: 0.5   # PMDiceLoss (replaces DiceTopK)
w_deepsup:   0.1   # auxiliary ODE-midpoint supervision
```

Boundary ramp schedule:
```
w_eff = w * min(1.0, (epoch - ramp_epoch + 1) / 10)   for epoch ≥ ramp_epoch
```

DeepSup: gathers the per-item organ channel from `deepsup_logits` using `organ_idxs`.

### 4.2 Trainer (`training/trainer.py`)

#### CutMix explicitly disabled for AutoODESAM (preserved code comment)

> "DISABLED for AutoODESAM: CutMix mixes items with different organ_ids. The pasted region carries item-B's mask (e.g. adrenal) into item-A's training (e.g. liver), producing wrong ground truth in the pasted area."

Detection: `is_auto_ode = hasattr(model, 'decoder') and not hasattr(model, 'prompt_encoder')`.

#### Layer-wise LR — 4 optimizer groups

- `enc_decay`: encoder params with weight_decay.
- `enc_nodecay`: encoder bias / norm params, no weight_decay.
- `dec_decay`: decoder (non-encoder) params with weight_decay.
- `dec_nodecay`: decoder bias / norm params, no weight_decay.
- `encoder_lr = base_lr * encoder_lr_scale` (default 0.1) → 1e-5 vs decoder 1e-4.

#### AutoODESAM training path

```
outputs = model(images, organ_id, is_3d=is_3d)
pred_masks_all  # (B, n_organs, H, W)
pred_masks = pred_masks_all.gather(1, organ_idxs.view(-1,1,1,1).expand(-1,1,H,W))
diversity_loss = decoder.compute_diversity_loss()
total_loss = combined_loss + diversity_loss
loss_components["total"] = total_loss   # logged loss matches backpropagated loss
```

#### Validation

- The trainer computes **2D slice-level Dice for monitoring only**.
- **Final reported metrics come from `evaluate_3d.py`** (volumetric 3D Dice + HD95 + NSD). Trainer `val_dice` is a monitoring proxy, not the headline number.

#### WarmupCosineScheduler

- Warmup: linear from `warmup_lr_init` to `base_lr` over `warmup_epochs`.
- Cosine: `lr = lr_min + 0.5 * (base_lr - lr_min) * (1 + cos(π * t / T))`.

### 4.3 Dataset (`datasets/amos22.py`)

Two distinct classes:

- `AMOS22Dataset` — 2D slice-level (single slice per sample).
- `AMOS22_3D_Dataset` — 3D slice-stack (D consecutive slices per sample).

`AMOS22_3D_Dataset`:

- `n_slices` default = 16 (configurable; Phase 3a uses `slices_per_volume: 8`).
- Valid center slices: `center ∈ [half, n_total - half]` where `half = n_slices // 2`.
- **Copy-paste augmentation handles spatial resizing**: bilinear for images, nearest for labels (donor volumes may have different spatial resolution).
- `get_oversampled_indices()` returns an index list where small-organ samples appear at `small_organ_fraction = 0.33` frequency.

`_donor_index` is built for organs `{4, 5, 11, 12, 13}`. `_copypaste_aug()` pastes the same-organ donor across all D slices; anatomically valid because AMOS22 volumes are resampled to the same 1.5 mm³ spacing.

### 4.4 Training Improvements Stack (all active in Phase 3a v2)

| Fix | Implementation | Expected gain | Status |
|-----|---------------|---------------|:------:|
| All-slice supervision | Supervise all D = 8 slices, not just center | +1–3 % DSC | ✓ |
| CutMix augmentation | `prob = 0.5` for non-AutoODESAM models | +4.9 % DSC | ✓ (other models) |
| **CutMix DISABLED for AutoODESAM** | Mixes different organ_ids → wrong GT in pasted area | — | ✓ (correctness) |
| PMDiceLoss | Replaces DiceTopK — smoother gradient | +1.85–2.66 % hard organs | ✓ |
| clDice OFF | Designed for tubular, hurts solid organs | +0.3–0.8 % | ✓ |
| Deep supervision | Auxiliary head at ODE midpoint, w = 0.1 | +0.3–1.0 % | ✓ |
| Boundary loss | DT-surface loss, ramps ep50, w = 0.1 | −0.1–0.3 mm HD95 | ✓ |
| Foreground oversampling | 33 % of batches from small-organ slices | +0.5 % small organs | ✓ |
| Layer-wise LR | Encoder 0.1× (1e-5), decoder/ODE at 1e-4 | +0.5–1.5 % | ✓ |
| Heun's method (RK2) | substeps = 4, halves ODE truncation error | +0.2–0.8 % | ✓ |
| Per-organ loss weighting | Per-sample inside loss (adrenal 3×, gallbladder 2.5×) | +0.5–1.5 % | ✓ |
| Copy-paste aug | Full D-slice paste, prob = 0.1, small organs only | +0.5–2.0 % small organs | ✓ |

---

## 5. Experimental Results

> **CRITICAL CAVEAT:** All Phase 0–2b results are **liver-only** (organ 6), 256 px, ~20 epochs. They CANNOT be compared directly to published 15-organ averages. **Phase 3a is the first apples-to-apples comparison** with the AMOS22 leaderboard.

### 5.1 Phase 0 — Cross-Slice Method Ablations (256 px, liver, ~20 epochs) — COMPLETE

| Model | DSC | HD95 (mm) | NSD | Module Params | Key Finding |
|-------|-----:|---------:|----:|--------------:|-------------|
| V2 DA-ISA (baseline) | 0.8921 | 2.22 | — | 1.71 M | Stage 3 too coarse |
| MultiScaleISA | 0.9031 | 1.31 | 0.9547 | 1.71 M | +1.1 % from Stage 2 fix |
| FCA-SAM | 0.9015 | 1.37 | 0.9515 | 536 | Novel FFT, 536 params |
| ACM-SAM | 0.8999 | 1.35 | 0.9520 | 280 K | Memory bank — below noISA |
| **Stage2-noISA** | **0.9068** | **—** | **—** | **0** | **Best Phase 0 — no cross-slice wins** |

**Headline finding (publishable negative result):** the Stage 2 tap *alone* beats every cross-slice method tested. Cross-slice modules **hurt** Stage 2 features. Defensible thesis contribution.

### 5.2 Phase 1 — Novel Architecture Tests — COMPLETE

| Model | DSC | HD95 (mm) | NSD | vs Stage2-noISA |
|-------|----:|---------:|----:|----------------:|
| TriMamba-SAM | 0.9044 | 1.33 | 0.9560 | −0.0024 (−0.26 %) |
| ODE-SAM | 0.9000 | 1.34 | 0.9504 | −0.0068 (−0.75 %) |

**Decision: chose ODE-SAM for Phase 2** because:
1. Loss at ep20 = 0.0718 < TriMamba 0.0729 → **still learning**.
2. DSC gain was **accelerating** (+0.0036 → +0.0041 per 5-epoch window).
3. ODE-SAM is **genuinely novel**; TriMamba overlaps with TP-Mamba (MICCAI 2024).

#### TriMamba-SAM epoch-by-epoch

| Epoch | DSC | HD95 (mm) | NSD |
|------:|----:|---------:|----:|
| 0 | 0.1295 | 38.96 | 0.3252 |
| 5 | 0.8609 | 2.20 | 0.9094 |
| 10 | 0.8912 | 1.55 | 0.9474 |
| 15 | 0.9007 | 1.40 | 0.9538 |
| 20 | 0.9044 | 1.33 | 0.9560 |

#### ODE-SAM Phase 1 epoch-by-epoch

| Epoch | DSC | HD95 (mm) | NSD | Loss |
|------:|----:|---------:|----:|----:|
| 0 | 0.1896 | 31.14 | 0.4177 | 0.5633 |
| 5 | 0.8586 | 2.02 | 0.9085 | 0.1106 |
| 10 | 0.8923 | 1.45 | 0.9502 | 0.0900 |
| 15 | 0.8959 | 1.41 | 0.9481 | 0.0771 |
| 20 | 0.9000 | 1.34 | 0.9504 | 0.0718 |

### 5.3 Phase 2 — Training Fixes on ODE-SAM — COMPLETE

**Config:** `phase2_ode_full.yaml`. Applied: all-slice supervision, CutMix (prob = 0.5), DiceTopK (replaced Focal), clDice OFF.

| Epoch | DSC | HD95 (mm) | NSD | Loss |
|------:|----:|---------:|----:|----:|
| 0 | 0.1929 | 51.88 | 0.5046 | 1.1772 |
| 5 | 0.9105 | 0.94 | 0.9607 | 0.0912 |
| 10 | 0.9142 | 0.94 | 0.9560 | 0.0726 |
| **15** | **0.9183** | **0.76** | **0.9589** | 0.0617 |
| 20 | 0.9162 | 0.72 | 0.9570 | 0.0561 |

- **Best checkpoint: epoch 15.**
- Gain vs Phase 1: **+0.0183 DSC (+2.0 %)**, HD95 1.34 → 0.76 mm.
- At ep5: Phase 1 = 0.8586 vs Phase 2 = 0.9105 (+5.5 %) — **CutMix dominates early**.

**Companion ablation (Stage2-noISA + Phase 2):** DSC = 0.9156, HD95 = 0.88 mm, NSD = 0.9553.

### 5.4 Phase 2b — Architecture Upgrades — COMPLETE

Applied: Stage 1 skip connections, PFESA, Haar edge enhancement.
**Config:** `phase3_odesam_v2_256px_liver` (named `phase3_*` for historical reasons — conceptually Phase 2b).

| Epoch | DSC | HD95 (mm) | NSD | Loss |
|------:|----:|---------:|----:|----:|
| 0 | 0.1524 | 54.79 | 0.5229 | 1.1762 |
| 5 | 0.9240 | 0.87 | 0.9606 | 0.0792 |
| 10 | 0.9261 | 0.68 | 0.9618 | 0.0618 |
| **15** | **0.9358** | **0.56** | **0.9678** | 0.0524 |
| 20 | 0.9346 | 0.58 | 0.9674 | 0.0502 |

**ODE-SAM V2 = 0.9358 DSC, 0.56 mm HD95, 0.9678 NSD — PROJECT BEST (liver, 256 px)**

#### Decomposition of gains

- Stage 1 baseline → Phase 2 (training fixes): **+0.0183 DSC**
- Phase 2 → Phase 2b (arch upgrades): **+0.0175 DSC, −0.20 mm HD95**
- Phase 1 → Phase 2b total: **+0.0358 DSC (+3.97 %), −0.78 mm HD95**

**Findings:**
- **Skip connections = single largest architectural gain.**
- **PFESA primarily drove HD95** (0.76 → 0.56 mm).
- **Loss at ep5 was 0.0792** (Phase 2b) vs **0.0912** (Phase 2) — skips accelerate early convergence.

### 5.5 Complete Results Ranking

| Rank | Model | DSC | HD95 (mm) | NSD | Notes |
|----:|-------|----:|---------:|----:|------|
| 1 | ODE-SAM V2 (skip + PFESA + Haar) | **0.9358** | **0.56** | **0.9678** | Project best (liver) |
| 2 | ODE-SAM + Phase 2 | 0.9183 | 0.76 | 0.9589 | Training fixes only |
| 3 | Stage2-noISA + Phase 2 | 0.9156 | 0.88 | 0.9553 | Ablation |
| 4 | Stage2-noISA | 0.9068 | — | — | No cross-slice |
| 5 | TriMamba-SAM | 0.9044 | 1.33 | 0.9560 | |
| 6 | MultiScaleISA | 0.9031 | 1.31 | 0.9547 | |
| 7 | FCA-SAM | 0.9015 | 1.37 | 0.9515 | |
| 8 | ODE-SAM Phase 1 | 0.9000 | 1.34 | 0.9504 | |
| 9 | ACM-SAM | 0.8999 | 1.35 | 0.9520 | |
| 10 | V2 DA-ISA baseline | 0.8921 | 2.22 | — | |
| 11 | **Auto-ODE-SAM V3** | **RUNNING** | **RUNNING** | **RUNNING** | 15-organ, Phase 3a v2 |

---

## 6. Phase 3a — Current Status (15-Organ AMOS22)

Phase 3a exists as **two distinct runs**. The v1 run is retained as a historical record; the v2 run is the currently active one.

### 6.1 Phase 3a v1 (OLD run — STOPPED)

- **Experiment name:** `phase3a_autoodesam_256px_15org`
- **Start:** 2026-04-12 02:29
- **Status:** STOPPED (copy-paste regression at ep15)

#### v1 Validation Log

| Epoch | DSC | HD95 (mm) | NSD | Loss | LR | Time/epoch |
|------:|----:|---------:|----:|-----:|---:|-----------:|
| 0 | 0.1368 | 26.32 | 0.4594 | 1.0393 | 1.0 e-6 | 117 min |
| 5 | 0.3199 | 21.68 | 0.6081 | 0.9471 | 1.0 e-4 | 137 min |
| 10 | 0.3599 | 21.02 | 0.6216 | 0.9320 | 9.93 e-5 | 134 min |
| ⚠ 15 | 0.3370 | 22.13 | 0.5997 | 0.9248 | 9.73 e-5 | 116 min |

#### v1 Incident

- **ep0–10:** Healthy progression — DSC rose 0.1368 → 0.3599, loss 1.039 → 0.932.
- **ep10–15:** Regression — DSC dropped to 0.3370.
- **Root cause:** `copypaste_prob = 0.3` was too aggressive **before ep30**. Model hadn't learned basic organ shapes yet; aggressive paste disrupted feature learning.
- **Backup:** `checkpoints/phase3a_autoodesam_256px_15org/phase3a_autoodesam_256px_15org_epoch010_backup.pt` preserved for reference.

### 6.2 Phase 3a v2 (CURRENT run — FRESH START, RUNNING)

- **Experiment name:** `phase3a_autoodesam_256px_15org_v2`
- **Start:** 2026-04-14 ~15:02 (epoch000 saved 17:53)
- **Status:** RUNNING as of 2026-04-15 (TF events last written 10:20)
- **Launched as fresh start**: `resume_from: null` at first launch.
- Config `resume_from` has since been updated to `checkpoints/phase3a_autoodesam_256px_15org_v2/phase3a_autoodesam_256px_15org_v2_latest.pt` so re-running the same command now resumes from ep5.

#### v2 Architecture Change (vs v1)

Key ODE hyperparameter scaling — all increased before v2 launch:

| Parameter | v1 value | v2 value |
|-----------|---------:|---------:|
| `ode_hidden` | 64 | **128** |
| `n_freqs` | 4 | **6** |
| `substeps` | 2 | **4** |
| `organ_emb_dim` | 32 | **64** |

Consequence: the ODE is substantially more expressive, but also has more parameters to ramp up from zero-init. The lower ep5 val_dice in v2 (0.107 vs v1 0.320) is consistent with this — the ODE module takes longer to move away from the identity function.

#### v2 Validation Log

| Epoch | val_dice | val_HD95 (mm) | val_NSD | val_loss | train_dice | train_loss | Saved |
|------:|---------:|--------------:|--------:|---------:|-----------:|-----------:|-------|
| 0 | 0.0067 | 20.12 | 0.0180 | 0.8406 | 0.0068 | — | 2026-04-14 17:53 |
| 5 | **0.1074** | **11.92** | **0.1858** | **0.7193** | **0.1048** | 1.2339 | 2026-04-15 05:12 |
| 6 | *(no val)* | — | — | — | 0.1140 | 1.2200 | *(no ckpt — saves every 10)* |
| 7 | *(no val)* | — | — | — | 0.1200 | 1.2087 | **STOPPED HERE** (2026-04-15 09:23) |

`epoch005.pt`, `latest.pt`, and `best.pt` all point to the epoch-5 state.

### 6.3 Dataset Sizes

- **Training set:** 78,407 slice stacks (D = 8), 15 organs
- **Validation set:** 47,123 slice stacks, 15 organs

### 6.4 Expected Convergence (Phase 3a v2)

| By epoch | Revised DSC estimate |
|---------:|---------------------|
| 10 | 0.20 – 0.35 (need to exceed 0.15 clearly) |
| 20 | 0.40 – 0.55 |
| 40 | 0.65 – 0.75 |
| 80 | 0.80 – 0.88 |
| 110 | **≥ 0.87 – 0.905** |

### 6.5 Decision Rule After Phase 3a v2

- **≥ 0.875 DSC** → proceed to Phase 3b (512 px)
- **< 0.875 DSC** → investigate convergence before Phase 3b

---

## 7. All Code Changes Applied

All changes are **live** in the current codebase. Phase 3a v2 runs with all of them from ep0.

### 7.1 Architecture Built (before v1 run)

- `models/auto_ode_sam.py` — `AutoODESAM` class with `_deepsup_cache` side-effect pattern.
- `models/ode_cross_slice.py` — organ-conditioned variants (`OrganConditionedODEFunction`, `OrganConditionedBidirectionalNeuralODE`).
- `models/organ_query_decoder.py` — 15 organ queries + HQ token + Haar edge enhance + diversity loss (with `.abs()` guard — see Section 8).
- Sinusoidal dense PE converted from `1/2**i` to SAM-convention `1/10000^(2i/half)` to avoid aliasing at feat_size = 16.
- Heun's method (RK2) integration chosen over forward Euler.

### 7.2 Changes on 2026-04-13 / 2026-04-14 (loss + trainer refactor)

1. **`training/losses.py`** — `PMDiceLoss` class added (replaces `DiceTopKLoss` inside `CombinedLoss`). Smoother gradient signal (arXiv 2506.15744).
2. **`training/losses.py`** — `DiceLoss`, `DiceTopKLoss`, `CombinedLoss.forward()` accept optional `sample_weight: Optional[torch.Tensor]`, applied per-sample before `.mean()`.
3. **`training/trainer.py`** — `_ORGAN_LOSS_WEIGHTS` constant updated with calibrated weights:
   ```python
   _ORGAN_LOSS_WEIGHTS: dict = {
       4: 2.5,   # gallbladder   (~25 ml — variable shape/absence)
       5: 2.0,   # esophagus     (thin tubular, hard boundaries)
       10: 2.0,  # pancreas      (~80 ml — irregular, hard)
       11: 3.0,  # right adrenal (~4 ml — AMOS22 SOTA 0.81 DSC)
       12: 3.0,  # left adrenal  (~4 ml — AMOS22 SOTA 0.84 DSC)
       13: 2.0,  # duodenum      (thin loop, hard to delineate)
   }
   ```
   Adrenal raised 2.0 → 3.0; gallbladder raised 2.0 → 2.5.
4. **`training/trainer.py`** — `sample_weight` computed per-sample inside `autocast`, passed to `self.criterion(..., sample_weight=sample_weight)`.
5. **`training/trainer.py`** — Old `batch_w` block removed (was computing `mean(weights)` and scaling the already-averaged loss — diluted hard organs, over-weighted easy organs in mixed batches).
6. **`datasets/amos22.py`** — `copypaste_prob` parameter added. `_donor_index` for `{4, 5, 11, 12, 13}`. `_copypaste_aug()` pastes same-organ donor across all D slices with bilinear (images) / nearest (labels) resizing.
7. **`training/trainer.py`** — **CutMix DISABLED for AutoODESAM** with detection `hasattr(model, 'decoder') and not hasattr(model, 'prompt_encoder')`. Rationale: CutMix mixes items with different `organ_id`s, pasting item-B's mask into item-A's training target.
8. **`training/trainer.py`** — Diversity loss added to `total_loss`; `loss_components["total"]` reflects the post-diversity value (logged loss = backpropagated loss).

### 7.3 Changes Before v2 Run (ODE config scaling + fresh start)

- `configs/phase3a_autoodesam_256px.yaml`:
  - Experiment name → `phase3a_autoodesam_256px_15org_v2`
  - `resume_from: null` (fresh start; later updated to v2 latest.pt once ep5 saved)
  - `copypaste_prob: 0.1`
  - `epochs: 110`
  - ODE config: `ode_hidden: 128` (was 64), `n_freqs: 6` (was 4), `substeps: 4` (was 2), `organ_emb_dim: 64` (was 32).

### 7.4 FiLM Initialisation Bug Fix

See Section 8.

### 7.5 Diversity Loss Antipodal Collapse Fix

See Section 8.

---

## 8. Known Bugs Fixed (all in current codebase)

| Bug | File | Fix |
|-----|------|-----|
| `feature_info[-1]` always returns Stage 3 | `multiscale_encoder.py` | Changed to `feature_info[stage_index]` |
| FCA `ComplexHalf` (no fp16 FFT) | `fca_sam.py` | Cast to float32 before `torch.fft.rfft` |
| FCA inplace op on autograd tensor | `fca_sam.py` | Use `torch.cat([mixed_kept, zeros], dim=1)` |
| `vis_every_n_epochs` `KeyError` | `configs/base.yaml` | Added `vis_every_n_epochs: 10` |
| `test_multiscale_isa_a1.yaml` broken defaults | config | Replaced with fully self-contained version |
| `models/__init__.py` dead import `litesam3d_v2` | `__init__.py` | Replaced with `raise NotImplementedError` |
| Old `batch_w` averaging bug in trainer | `trainer.py` | Removed; replaced with per-sample `sample_weight` |
| `copypaste_prob = 0.3` causing ep10–15 regression (v1 only) | config | Reduced to 0.1; v2 started fresh |
| **FiLM `gamma_proj` init = ones_** | `mask_decoder.py` | **`zeros_(weight) + ones_(bias)` → gamma(c) = 1 exactly at init** |
| **Diversity loss: plain cosine (antipodal collapse)** | `organ_query_decoder.py` | **Use `sim_matrix.abs()` — penalise both positive AND negative correlation** |
| **AutoODESAM + CutMix contamination** | `trainer.py` | Detect AutoODESAM and skip CutMix entirely |
| **HQ `skip_proj` double-zero → dead head** | `organ_query_decoder.py` | Zero weight only; leave bias at default for gradient bootstrap |

---

## 9. Locked Architecture & Methodology Decisions

These rules govern all current and future runs. Departures must be explicitly justified.

- Never mix 256 / 512 channel numbers.
- No auto-prompt in Phase 0–3 for ablation models (pure box prompt).
- ODE-SAM leads over TriMamba (zero overlap risk).
- **Stage 2 tap is the primary architectural contribution.**
- **FiLM is NOT novel** — cite MCP-MedSAM (CVPR 2024).
- ISA hurts Stage 2 — confirmed negative result, publishable.
- PMDiceLoss replaces DiceTopK (smoother gradient).
- Per-sample weights computed **inside the loss**, not by batch-mean scaling.
- clDice OFF for solid organs.
- Adrenal 3×, gallbladder 2.5×, others 2.0× or 1.0×.
- CutMix OFF for AutoODESAM (mixes organ_ids → wrong GT).
- Diversity loss uses `|cos_sim|`, not plain cos_sim.
- HQ pathway init: zero weight, non-zero bias (avoid gradient dead head).
- ODE PRE-norm on `ode_context`, not post-norm on residual sum.
- All checkpoints stored **inside** `VoluFormer3D/checkpoints/` (never outside).
- Never reuse experiment names — add `_v2`, `_v3` suffix.
- Copy critical checkpoints **immediately** with `_backup` suffix.

---

## 10. Novelty Assessment (Online-verified 2026-04-12)

### 10.1 Neural ODE × SAM × Cross-Slice — **CONFIRMED NOVEL**

- Search terms: *"Neural ODE segmentation SAM"*, *"ODE cross-slice CT"*, *"neural ODE medical segmentation 2024 2025"*.
- "Neural ODEs for Colon Gland Segmentation" (2019): 2D, no SAM, no cross-slice.
- "Attention Guided Neural ODE for Breast Tumor" (CBM 2023): 2D, no SAM.
- AI Review 2024 survey: zero SAM × cross-slice × ODE papers.
- **Verdict: UNAMBIGUOUSLY NOVEL.**

### 10.2 Organ Query Tokens Replacing SAM PromptEncoder — **NOVEL in this context**

- Surgical-DeSAM (IJCARS 2024): DETR generates box prompts → feeds PromptEncoder. Still uses PromptEncoder.
- Semantic-SAM (ECCV 2024): General vision, not medical CT.
- Our claim: *"We **replace, not augment**, the PromptEncoder — eliminating all prompt engineering at inference."*

### 10.3 Stage 2 Feature Tap — **Partially Novel**

- MCP-MedSAM (CVPR 2024) uses Stage 3.
- **No paper uses TinyViT Stage 2 for cross-slice CT.**
- Empirical finding *"cross-slice hurts Stage 2"* = publishable negative result.

### 10.4 TriMamba-SAM — **High Risk**

TP-Mamba (MICCAI 2024) uses tri-plane Mamba inside SAM encoder. Different injection point but close mechanism. **This is precisely why ODE-SAM is the lead contribution.**

### 10.5 FCA-SAM — **Novel**

No paper applies FFT along the depth axis specifically for SAM CT cross-slice modelling.

---

## 11. Competitor Comparison (Verified)

| Model | Mean DSC | HD95 (mm) | Params | Venue | SAM? | Auto? |
|-------|---------:|---------:|-------:|-------|:----:|:----:|
| AMOS22 Challenge #1 | **0.924** | — | — | Grand Challenge 2025 | — | — |
| nnU-Net | 0.889–0.924 | 4.26 | 56 M | MICCAI 2021 | ✗ | ✓ |
| MM-UNet (Mamba) | ~0.910 | — | — | arXiv 2025 | ✗ | ✓ |
| RFMedSAM 2 | ~0.905 | — | — | arXiv 2025 | ✓ | ✓ |
| TP-Mamba | ~0.900 | — | — | MICCAI 2024 | ✓ | ✗ |
| MCP-MedSAM | ~0.890 | — | ~18 M | CVPR 2024 | ✓ | ✗ |
| nnWNet | 0.8639 | 3.71 | 56 M | CVPR 2025 | ✗ | ✓ |
| **Auto-ODE-SAM V3** | **TBD** | **TBD** | **21.2 M** | **Thesis 2026** | **✓** | **✓** |

### Required Scores to Publish

| Venue | Required DSC | Probability |
|-------|-------------:|------------:|
| Masters thesis | > 0.8921 (beats V2) | **95 %** |
| MICCAI workshop | > 0.905 | **55 %** |
| MICCAI main track | > 0.910 | **30 %** |
| TMI / top venue | > 0.913 + analysis | **10–15 %** |

---

## 12. Phase Roadmap

| Phase | Goal | Status |
|-------|------|:------:|
| Phase 0 | Cross-slice ablations (7 models, liver, 256 px) | ✅ COMPLETE |
| Phase 1 | Novel architectures (TriMamba, ODE-SAM) | ✅ COMPLETE |
| Phase 2 | Training fixes on ODE-SAM | ✅ COMPLETE |
| Phase 2b | Architecture upgrades (skip + PFESA + Haar) | ✅ COMPLETE — project best @ 0.9358 |
| **Phase 3a v1** | 15-organ AMOS22 @ 256 px (first attempt) | ⚠ STOPPED (copy-paste regression) |
| **Phase 3a v2** | **15-organ AMOS22 @ 256 px (fresh start, expanded ODE)** | 🔄 **RUNNING** (ep5 / 110) |
| Phase 3b | 15-organ AMOS22 @ 512 px (progressive from 3a v2) | ⏳ PENDING — gated on 3a v2 ≥ 0.875 |
| Phase 4 | Per-organ analysis, ODE viz, zoom-in eval, runtime, thesis writing | ⏳ PENDING |

### Phase 3b Config Template (when Phase 3a v2 completes ≥ 0.875)

```yaml
experiment:
  name: "phase3b_autoodesam_512px_15org"
model:
  img_size: 512
training:
  batch_size: 2              # VRAM: 512 px needs ~21–23 GB at B=2
  grad_accumulation_steps: 8 # Effective B=16
  epochs: 50                 # Progressive from Phase 3a v2 checkpoint
  warmup_epochs: 3
checkpoint:
  resume_from: "checkpoints/phase3a_autoodesam_256px_15org_v2/best.pt"
```

#### Expected Phase 3b Gains over Phase 3a v2

- Adrenal: **+7.0 %** DSC (4 px → 8 px wide, biggest beneficiary)
- Gallbladder: **+6.0 %** DSC
- Pancreas: +3.5 %
- Duodenum: +4.0 %
- **Mean: +3.3 % → estimated ~0.911 mean DSC (assuming 3a v2 lands ~0.88)**
- Timing: ~2 days (progressive from 3a v2, 50 epochs)
- VRAM: ~21–23 GB at B = 2, D = 8 — fits 4090.

### Phase 4 — Next After Phase 3

- Per-organ breakdown (15 organs) — full ablation table
- ODE trajectory visualisation (`dh/dt` magnitude per organ, per slice)
- Two-stage zoom-in evaluation for small organs
- Runtime profiling
- Thesis writing

---

## 13. Configs Available

```
configs/
├── base.yaml
├── amos22.yaml
├── test_multiscale_isa_a1.yaml          ✅ DSC 0.9031
├── test_fca_sam_a1.yaml                 ✅ DSC 0.9015
├── test_acm_sam_a1.yaml                 ✅ DSC 0.8999
├── test_stage2_noisa_a1.yaml            ✅ DSC 0.9068
├── test_trimamba_a1.yaml                ✅ DSC 0.9044
├── test_ode_sam_a1.yaml                 ✅ DSC 0.9000
├── phase2_ode_f1_allslice.yaml          ✅ (intermediate)
├── phase2_ode_f2_cutmix.yaml            ✅ (intermediate)
├── phase2_ode_f3_topk.yaml              ✅ (intermediate)
├── phase2_ode_full.yaml                 ✅ DSC 0.9183
├── phase2_noisa_full.yaml               ✅ DSC 0.9156 (ablation)
├── phase3_odesam_v2.yaml                ✅ DSC 0.9358 (ODE-SAM V2, project best)
├── phase3a_autoodesam_256px.yaml        🔄 RUNNING (Phase 3a v2, 15 organs)
├── phase3a_odesam_v2_full.yaml          (alternative, not used)
└── phase3b_autoodesam_256px_smoke.yaml  (smoke test config)
```

---

## 14. Checkpoint Safety Rules

- All checkpoints stored **inside** `VoluFormer3D/checkpoints/` (never outside).
- **Never reuse experiment names** — always add `_v2`, `_v3` suffix on restart.
- **Copy critical checkpoints immediately** with `_backup` suffix.
- Preserved critical checkpoints:
  - v1 backup: `checkpoints/phase3a_autoodesam_256px_15org/phase3a_autoodesam_256px_15org_epoch010_backup.pt`
  - v2 latest: `checkpoints/phase3a_autoodesam_256px_15org_v2/phase3a_autoodesam_256px_15org_v2_latest.pt` (= ep5)
  - v2 best: `checkpoints/phase3a_autoodesam_256px_15org_v2/phase3a_autoodesam_256px_15org_v2_best.pt` (= ep5)

---

## 15. Thesis Contribution Summary (Defensible Claims)

1. **Stage 2 feature tap** — diagnoses and fixes the V2 root cause empirically.
2. **ODE-SAM** — first Neural ODE for cross-slice SAM CT segmentation. Confirmed novel.
3. **FCA-SAM** — first FFT along depth axis for SAM CT. Novel.
4. **Negative result** — cross-slice attention hurts Stage 2 features (Stage2-noISA > all cross-slice methods at Phase 0).
5. **Systematic evaluation** — 7 cross-slice methods on AMOS22, first study of this kind.
6. **Auto-ODE-SAM V3** — fully automatic, 21.2 M parameters, competitive with 56 M nnU-Net.
7. **Engineering rigor** — FiLM init bug fix, diversity-loss antipodal fix, HQ gradient-bootstrap init, CutMix/AutoODESAM incompatibility documented: all non-trivial findings worth a methodology subsection.

### One-Sentence Thesis Contribution

> *We present Auto-ODE-SAM V3, the first framework to model continuous anatomical dynamics across CT slices using a bidirectional organ-conditioned Neural ODE within a SAM-based segmentation architecture, achieving fully automatic 15-organ abdominal CT segmentation at 21.2 M parameters on AMOS22.*

---

## 16. Operational Reference

### Run Commands

```bash
# Phase 3a v2 (current)
C:\Users\Raywa\Desktop\LiteSAM3D\.venv\Scripts\python train.py \
    --config configs/phase3a_autoodesam_256px.yaml

# Generic
C:\Users\Raywa\Desktop\LiteSAM3D\.venv\Scripts\python train.py \
    --config configs/<experiment>.yaml

# 3D evaluation (headline metrics)
C:\Users\Raywa\Desktop\LiteSAM3D\.venv\Scripts\python evaluate_3d.py \
    --config configs/<experiment>.yaml \
    --checkpoint checkpoints/<exp>/best.pt
```

### Key Paths

| Item | Path |
|------|------|
| Code root | `C:\Users\Raywa\Desktop\VoluFormer3D\` |
| Data root | `C:\Users\Raywa\Desktop\LiteSAM3D\data\amos22\` |
| venv | `C:\Users\Raywa\Desktop\LiteSAM3D\.venv` |
| Checkpoints | `C:\Users\Raywa\Desktop\VoluFormer3D\checkpoints\` |
| Configs | `C:\Users\Raywa\Desktop\VoluFormer3D\configs\` |
| Logs | `C:\Users\Raywa\Desktop\VoluFormer3D\logs\` |

### Hardware

- GPU: **NVIDIA GeForce RTX 4090 (24 GB VRAM)** — upgraded from RTX 3090
- CPU: AMD Ryzen 9 5900X
- RAM: 32 GB
- OS: Windows 11

---

## 17. Change Log

| Date | Change |
|------|--------|
| 2026-04-12 | Phase 3a v1 launched (Auto-ODE-SAM V3, 15-organ, 256 px, 100 epochs, copypaste_prob = 0.3) |
| 2026-04-13 | Loss / trainer refactor: PMDiceLoss replaces DiceTopK; per-sample `sample_weight` inside loss; old `batch_w` removed |
| 2026-04-14 | Calibrated organ weights (adrenal 3×, gallbladder 2.5×); copy-paste aug added for small organs |
| 2026-04-14 | v1 ep10 → ep15 regression observed; root-caused to copypaste_prob = 0.3 |
| 2026-04-14 | **Phase 3a v2 launched as FRESH START** (`resume_from: null`) with expanded ODE (ode_hidden 64→128, substeps 2→4, n_freqs 4→6, organ_emb_dim 32→64); copypaste_prob = 0.1; epochs = 110; experiment name `phase3a_autoodesam_256px_15org_v2` |
| 2026-04-14 | FiLM init bug fixed (ones_ → zeros_+ones_) in `mask_decoder.py` |
| 2026-04-14 | Diversity loss antipodal collapse fixed (plain cos → `|cos|`) in `organ_query_decoder.py` |
| 2026-04-14 | CutMix disabled for AutoODESAM in `trainer.py` (mixes organ_ids → wrong GT) |
| 2026-04-14 | HQ skip_proj init: zero weight, non-zero bias (prevents dead head) |
| 2026-04-15 | v2 ep5 checkpoint saved: val_dice = 0.1074, val_HD95 = 11.92 mm, val_NSD = 0.1858 |
| 2026-04-15 | Config `resume_from` updated to v2 latest.pt — re-running command now resumes from ep5 |
| 2026-04-15 | MASTER_PROJECT_LOG.md rewritten with full code-level architecture detail and dual v1/v2 Phase 3a history |
| 2026-04-15 | Training stopped at ep7 (09:23) — no crash log, likely killed/machine went idle. ep6 train_dice=0.1140 loss=1.2200; ep7 train_dice=0.1200 loss=1.2087. Trajectory healthy. Must resume from latest.pt (ep5). |
| 2026-04-15 | **Phase 3a v3 LAUNCHED** — full code review found 5 bugs/weaknesses. Applied fixes: W2 (OrganQueryDecoder now upsamples transformer-updated features, not raw — estimated +1–3 DSC); W3 (skip_gate + hq_gate 0→0.05); B1 (PFESA no longer computed on D-1 wasted slices); B5 (dense_pe dtype cast); B3 (iou_pred.flatten() vs squeeze). Experiment name bumped to v3. Fresh start (resume_from: null) required for skip_gate fix to take effect. |
| 2026-04-18 | **V3 ABANDONED.** ep20 checkpoint shows val_dice peaked at ep10 (0.1427) and regressed to 0.1159 at ep20; train_dice 0.135 indicated under-fitting. Code-level diagnosis: (1) single-organ-per-sample dataset starves 14 of 15 decoder queries every forward; (2) per-organ `mask_mlps` all had identical weight magnitudes (~0.037) — proof of zero specialization; (3) `hq_token` dead (abs_mean 0.799→0.797) and `hq_gate` closing (0.064→0.055); (4) `q_proj` magnitude decreasing (0.0316→0.0306) — not learning. V3 kept frozen at `C:\Users\Raywa\Desktop\VoluFormer3D\` for reference. |
| 2026-04-18 | **V4 (OrganFlow-SAM2) DESIGN APPROVED.** Option B backbone (MedSAM2 Hiera-Tiny + LoRA rank-16). Four novel contributions: (1) **flow-matched cross-slice ODE training** (velocity regression on inter-slice finite differences, inspired by Lipman 2023 flow matching — primary novelty); (2) **Anatomy Graph Attention** on 15 organ queries (learnable 15×15 organ-adjacency prior); (3) **PFESA++** with learnable amplification and soft high-freq cutoff; (4) MedSAM2 + LoRA + DETR-style multi-organ decoder integration. Target ≥0.91 mean DSC by ep120. Estimated training time ~4 days on 4090. Spec at `VoluFormer3D_V4\docs\superpowers\specs\2026-04-18-organflow-sam2-design.md`. |
| 2026-04-18 | **V4 folder scaffolded** at `C:\Users\Raywa\Desktop\VoluFormer3D_V4\` (verbatim copy of V3 minus `checkpoints/` and `logs/`). V4 README and design spec written. Next step: invoke `superpowers:writing-plans` to produce implementation plan. |

---

## 18. Phase 4 — OrganFlow-SAM2 (V4) — ACTIVE

### Status

- **Design:** approved 2026-04-18
- **Spec document:** `C:\Users\Raywa\Desktop\VoluFormer3D_V4\docs\superpowers\specs\2026-04-18-organflow-sam2-design.md`
- **Code folder:** `C:\Users\Raywa\Desktop\VoluFormer3D_V4\`
- **Implementation plan:** pending (to be produced by `superpowers:writing-plans`)
- **First training run:** blocked on implementation plan

### One-line claim

> We present **OrganFlow-SAM2**, the first framework to combine a flow-matched organ-conditioned cross-slice ODE with an anatomy-graph DETR decoder on top of the MedSAM2 foundation model, achieving fully automatic 15-organ abdominal CT segmentation.

### Architecture summary

```
MedSAM2 Hiera-Tiny encoder (frozen) + LoRA rank-16
    ↓
PFESA++ (learnable α, soft HF cutoff)
    ↓
Flow-Matched Organ-Conditioned Bidirectional ODE
    (aux loss: velocity regression on consecutive encoder-slice pairs)
    ↓
Anatomy-Graph Query Decoder
    (15 organ queries + learnable 15×15 adjacency prior, then TwoWayTransformer
     depth=4 heads=8, shared mask head)
    ↓
(B, 15, 64, 64) masks + (B, 15) IoU + deepsup logits
```

**Params:** ~59M trainable / ~97M total. Fits Option B band (70–100M).

### Loss

`L_total = L_mask + 0.5·L_flow + 0.1·L_deepsup + 0.01·L_anatomy`

- `L_mask` = per-channel DiceTopK + BCE over 15 present-masked organs per volume
- `L_flow` = velocity regression on inter-slice finite differences (the novel contribution)
- `L_deepsup` = 15-channel CE at ODE midpoint
- `L_anatomy` = Frobenius of `A - A_init` (keep graph near anatomical prior)

### Why V4 fixes what V3 broke

| V3 bug | V4 fix |
|---|---|
| 1 organ per sample, 14 queries unsupervised | Per-volume multi-organ data: 15 supervised channels per forward |
| 15 per-organ mask MLPs collapsed to identical weights | Single shared mask MLP; organ identity lives in query token only |
| HQ token + hq_skip_proj dead | Deleted entirely |
| ODE never woke up through ep25 (downstream-only supervision) | `L_flow` gives direct gradient on `f_θ` from epoch 0 |
| Diversity loss (antipodal) was load-bearing for organ specialization | Removed — specialization emerges from per-organ GT supervision |

### Go gates

| Epoch | Required val_dice | Meaning |
|------:|-----------------:|---------|
| 5 | ≥ 0.15 | Flow loss is working — ODE getting direct supervision |
| 20 | ≥ 0.50 | Multi-organ supervision converging — unlocks nnU-Net comparison |
| 60 | ≥ 0.82 | Competitive with published baselines — unlocks Phase 4b (512 px, optional) |
| 120 | ≥ 0.91 | Final thesis result — beats nnWNet on AMOS22 CT-only |

### Risks watched

- MedSAM2 checkpoint load — smoke test is the first implementation step
- Flow loss dominating early → encoder drift. Mitigated by `λ_flow` warmup 0→0.5 over 3 epochs
- Memory blow-up from 15-channel supervision — batch 4 + grad accum 2 as default; batch 2 + accum 4 as fallback

---

## 19. V7 Rescue + V8 MCP-Killer Plan (2026-04-19 → 2026-04-20)

### V7 breakthrough (config: `v7_rescue_320px.yaml`)

V6 plateaued at val_dice = 0.091 at ep24 (vs V4's 0.28 at ep24). V6 failure
diagnosed as too-many-loss interference (focal + boundary + multi-scale DS
+ nnU-Net elastic aug) on 200 training volumes with 24M trainable params.

V7 rescue (launched 2026-04-19 22:14, PID 21012):
- img_size 384 → 320, D=8 (was 12), batch 2 → 4, grad_accum 4 → 2
- LoRA rank 16 → 32 retained, decoder depth 6 + mlp_dim 3072 retained
- Losses: dropped focal + boundary + reduced ds_scale to 0.15, deepsup 0.05
- Aug: light_aug = H-flip + intensity only (no elastic/gamma/noise/rotation)
- **Multi-slab dataset: 4 distinct Z-offsets per volume → effective train data 200 → 800**
- lr 3e-4 → 1.5e-4, warmup 1 ep, cosine over 35 ep

Live validation table (partial):

| Epoch | V7 val_dice | V4 reference |
|------:|------------:|-------------:|
|   0   | 0.029       | —            |
|   2   | 0.083       | —            |
|   4   | 0.088       | —            |
|   6   | 0.313       | —            |
|   8   | 0.436       | —            |
|  10   | **0.489**   | V4 ep10=0.099, **V4 peak ep45=0.34** |

**Result at ep10: V7 already beats V4's entire 45-epoch peak by +44%.** Gate
criterion (ep10 ≥ 0.20) hit 2.4×. The architecture + multi-slab plan is
validated.

### V8 MCP-Killer build (2026-04-20)

While V7 trains, V8 modules built + smoke-tested for warm-start launch once
V7 converges:

- **Free wins (Phase A)**: `evaluation/sliding_window_3d.py` (Gaussian-blended 3D
  inference), `inference/tta.py` (hflip/vflip/rot90 ensemble), `inference/postproc.py`
  (largest-CC + hole-fill per organ). All smoke-tested together.
- **Novel modules (Phase A cont.)**:
  - `models/organ_text_encoder.py`: BiomedCLIP text prompts → (K, 512) organ
    features. Xavier fallback if open_clip not installed.
  - `models/dual_encoder.py`: DINOv2 ViT-S/14 (84 MB, downloaded + cached) +
    gated-channel fusion with MedSAM2. Init bias toward MedSAM2 (sigmoid(+1.5)).
  - `models/organflow_sam2_v8.py`: inherits V7; adds text query_bias into
    AnatomyGraphDecoder (backwards-compatible kwarg) + DINOv2 fusion in encoder path.
  - `configs/v8_mcp_killer.yaml`: warm-start V7, lr 7.5e-5, 25 epochs, batch 3.
  - Smoke: 52M total params (16M trainable), all output shapes correct, grad
    coverage 412/430, DINOv2 ACTIVE in fusion.

- **Evaluation**: `scripts/eval_3d_amos22.py` — standalone AMOS22-val 3D Dice
  report per organ + mean. Ready to run on V7's best.pt after training ends.

### Next steps (chronological)

1. Let V7 finish 35 epochs (ETA ~8 h remaining at ~1580 s/epoch).
2. Run `eval_3d_amos22.py --config configs/v7_rescue_320px.yaml
   --checkpoint checkpoints/v7_rescue_320/v7_rescue_320_best.pt --tta --cc`
   — gives MCP-comparable 3D Dice per organ.
3. If V7 3D Dice ≥ 0.55: warm-start V8 from V7 best.pt and launch `v8_mcp_killer.yaml`.
4. V8 expected uplift (rough prior): +0.04 from DINOv2 features, +0.02 from
   BiomedCLIP text, +0.06 from proper 3D eval + TTA + CC → target ~0.70-0.75
   on AMOS22 CT (MCP-MedSAM reports 0.79-0.82).

---

## 20. V7 3D Eval Result + Training-Distribution Bug (2026-04-20)

### V7 training finished
- 35 epochs, final 2D center-slice val_dice **0.6978** at ep34.
- LR decayed to 3.84e-6 by ep32; last 3 epochs flat (0.697 → 0.6976 → 0.6978).
- `checkpoints/v7_rescue_320/v7_rescue_320_best.pt` (247 MB).

### First real 3D AMOS22 eval
Before claiming anything vs MCP-MedSAM (0.79-0.82 CT), ran the proper 3D volume
Dice via `scripts/eval_3d_amos22.py`. Two bugs found + fixed:

1. **`torch.load` weights_only=True default (torch 2.6)** — ckpt has numpy scalar
   in metrics → `UnpicklingError`. Fixed: pass `weights_only=False`.
2. **Broadcast bug in `evaluation/sliding_window_3d.py`** — the old impl took the
   center-slice mask from each D-slab window and broadcast it across ALL D slices
   (`np.broadcast_to(..., (n_organs, depth, H, W))`). Adjacent windows overwrote
   each other's thin-organ predictions. Fixed: stride=1, reflect-pad volume ±D/2,
   one inference per target slice — each slice becomes the center of exactly one
   window. ~4× slower but correct.

### Results (20 val volumes, CC postproc, no TTA)

| Metric                     | Value   | Notes |
|----------------------------|---------|-------|
| 2D center-slice val (train)| 0.6978  | Misleading — trained-distribution only |
| 3D buggy (broadcast)       | 0.322   | Baseline, known-wrong |
| 3D fixed (stride=1)        | **0.362** | stable across 5/20-vol runs |
| MCP-MedSAM reference       | 0.79-0.82 | Prompt-based baseline |

Per-organ fixed 3D Dice (20 vols):
```
liver          0.83
aorta          0.74
ivc            0.55
l_kidney       0.47
spleen         0.47
stomach        0.47
r_kidney       0.44
r_adrenal      0.37
gallbladder    0.30
l_adrenal      0.28
duodenum       0.19
prostate_ut    0.14
pancreas       0.12
bladder        0.07
esophagus      0.003   ← effectively dead
MEAN           0.362
```

### Root cause — training-distribution shift

`datasets/amos22.py:370-380` only adds (volume, center_slice, organ) samples
where `organ_mask[:, :, center].sum() >= 50`. Every training slab is guaranteed
to contain the target organ at its center slice. At 3D eval with stride=1 the
model is asked to predict on slabs where the organ is absent (upper chest for
liver, pelvis for lung-adjacent organs, volume edges for everything) — it has
never seen that distribution and hallucinates positives.

Consequences:
- Large-extent organs (liver, aorta) survive because most slabs contain them.
- Thin/small organs (esophagus, bladder, pancreas, duodenum) die because a
  random slab rarely contains them, so false positives swamp true positives.
- `all_slice_supervision: false` in `v7_rescue_320px.yaml` means even the
  center-slice-only training signal has no penalty on absent-organ predictions.

### Decision point — pending user input

Gap to MCP-MedSAM is 0.43 Dice. V8's text + DINOv2 + TTA + CC cannot close that
alone. Four options surfaced (2026-04-20 ~13:40):

| # | Option | Cost | Risk | Expected 3D Dice |
|--:|--------|-----:|------|------------------|
| 1 | Retrain V7 from scratch with negative-slab sampling (30-40% empty slabs, per-organ FP penalty) | +15 h GPU | Low — clean one-axis change | 0.55-0.65 |
| 2 | Fine-tune V7 for 5-10 ep with negative slabs | +4-8 h | Mixes experiments — violates "one axis at a time" rule | 0.45-0.55 |
| 3 | Launch V8 as-planned from current V7 best.pt (no retrain) | +10 h | High — inherits sampling bug | 0.40-0.50 |
| 4 | Eval-side hack: use TotalSegmentator-like localizer to mask out-of-ROI predictions | +2 h | Not SOTA-legit; external model in pipeline | 0.50-0.60 |

No option launched yet. Autonomous loop has:
- Filed this section.
- Launched full 100-vol V7 3D eval (background) to lock in the final V7 number
  on complete AMOS22 CT val.
- Paused until user chooses.

---

## Section 21 — V9 Tier-B scaffolding complete (2026-04-21)

User selected **Tier-B cascade** with full architectural latitude ("change the
architecture as much as you need"). All 9 scaffolding tasks from the plan at
`docs/superpowers/plans/2026-04-20-voluformer-v9-remaining.md` landed across 9
commits. The module is ready for Stage 1 training on real AMOS22 once GPU is
available.

### Architecture (final)

```
SwinUNETRProposer (3D, 96³ patch, feat=48)
    │
    │   prob: (B, 15, D_v, H_v, W_v)
    ▼
[slab-z gather + bilinear resize to refiner H×W]
    │
OrganFlowSAM2 (2D-slab refiner, MedSAM2 Hiera-Tiny + LoRA r=32)
    │
    │   prob: (B, 15, H_r, W_r)
    ▼
AnatomicalCascadeFusion (per-organ learned gate, organ embedding 32-d)
    │
    ▼
fused prob + gate weights
```

- Proposer parameters fed through frozen in Stage 2 (bias-init -1.0 → gate
  favors refiner until organ-specific trust is learned).
- Trainable param count at Stage 2: **29.3 M** (refiner + fusion only).

### Landed components

| Commit  | Component  | Novel # | What it does |
|---------|-----------|--------:|--------------|
| 7d1666b | `models/dynamic_pfesa.py` | #6 | HyperNetwork generates organ-conditioned PFESA kernels via grouped conv; zero-init + residual = identity at init. |
| e67942b | `losses/cascade_consistency.py` | #7 | Symmetric KL + squared-sum soft-Dice between proposer prob and refiner prob. |
| cca46e9 | `models/boundary_ddpm.py` | #8 | Tiny UNet DDPM; 5-10 denoising steps restricted to uncertainty band; soft `4·p·(1-p)` weighting for differentiable training. |
| b02c6c7 | `datasets/amos22_v9.py` | (D1) | Positive / Negative / Mixed slab sampling (50/30/20%); returns volume + slab + both masks + organ_id + slab_center_z. |
| c4f6231 | `configs/v9_tierB.yaml` | (cfg) | img_size=320, depth=8, volume_patch=96, all novel-loss λ=0.0 (loss-preservation clause §7b). |
| 22d947e | `evaluation/eval_v9_3d.py` | (E1) | Gaussian-blended 96³ sliding window + 4-way flip TTA + per-organ CC-keeplargest + 1.5 mm isotropic resample. |
| fc158e7 | `training/trainer_v9.py` | (T1) | 3-stage freeze controller, `step_losses()` dispatches novel losses by λ, `stop_rule_update()` 2-drop abort (spec S7c). |
| 370f5e1 | `tests/v9_cuda_smoke.py` | (T2) | Peak VRAM check (<22 GB, spec S7c gate 1). Skips if no CUDA. |
| 94a3740 | `tests/v9_full_pipeline_smoke.py` | (T3) | End-to-end: cascade forward + trainer losses + backward + optimizer step + stop-rule. **PASSES** on CPU. |

### Deliberate deviations from plan

User granted full architectural latitude; deviated where the plan spec was
mathematically broken:

| Plan said | Actually shipped | Reason |
|-----------|------------------|--------|
| `soft-Dice = 2·<p,q> / (sum(p) + sum(q))` | `2·<p,q> / (sum(p²) + sum(q²))` | Linear-sum fails self-identity: `dice(p, p) ≠ 1` for non-binary probs. Squared-sum restores the property. |
| `DDPM loss uses hard threshold on uncertainty band` | Training path uses soft `w = 4·p·(1-p)`; inference retains hard threshold | Hard threshold blocks gradient; `coarse_mask.grad = None` in M9 smoke. Soft weight preserves the uncertainty-restriction semantics while keeping grad flow. |
| Trainer wires TeacherDistill + CrossModalInfoNCE losses | Only `FlowShapePriorLoss` + `CascadeConsistencyLoss` + `BoundaryDDPM` | Class signatures for the two distillation losses don't match plan's assumed API; they stay standalone for Stage 3 when pseudo-labels are staged. |
| Trainer recomputes z-gather for cascade consistency | `VoluFormerV9.forward` exposes `out["_prop_slice_for_consistency"]` | Avoids a second `prob.gather(dim=2)` on a 5-D tensor per step. |

### Smoke tests — all passing (CPU)

```
v9_m4_teacher_distill_smoke.py        PASS
v9_m5_infonce_smoke.py                PASS
v9_m6_flow_shape_prior_smoke.py       PASS
v9_m7_dynamic_pfesa_smoke.py          PASS
v9_m8_cascade_consistency_smoke.py    PASS
v9_m9_boundary_ddpm_smoke.py          PASS
v9_d1_dataset_smoke.py                PASS
v9_full_pipeline_smoke.py             PASS (stage=2 trainable=29.3M, extras=[l_shape, l_cascade])
v9_cuda_smoke.py                      SKIPPED (no CUDA on dev box)
```

Full-pipeline smoke also verifies stop-rule: fires correctly after 2
consecutive drops below (best − 0.02).

### Known gaps before Stage 1 can launch

1. **No GPU on dev box** — CUDA VRAM gate (S7c gate 1, <22 GB) still unverified.
   Must re-run `tests/v9_cuda_smoke.py` on the training machine before training.
2. **Real AMOS22 volumes** — `AMOS22V9Dataset` currently validated on synthetic
   volumes only; `_load_volume` needs path wiring to AMOS22 CT NIfTI directory.
3. **Teacher checkpoint for Novel #4** — `TeacherDistillLoss` needs nnWNet
   pseudo-label paths. Task 12 (optional) from the plan. Deferred pending user
   decision on whether to run TotalSegmentator.

### Next up

- **Task 11** (parallel track): V8 finalization — CPU smoke, warm-start from
  V7 best.pt, launch V8 background training, 3D eval, ablate novel axes.
- **V9 Stage 1 training**: SwinUNETR proposer alone on real AMOS22 CT, once
  GPU available.

---

## 2026-04-21 — V9 Stage 1 complete, Stage 2 bug-hunt, alignment fix

### Timeline (this session)

| Time (AST) | Event |
|------------|-------|
| 02:57      | V9 Stage 1 ep 1 saved (`checkpoints/v9_stage1/epoch_001.pt`) |
| 13:04      | V9 Stage 1 ep 2 saved |
| ~20:00     | V9 Stage 1 completed 50 epochs. `last.pt` (epoch 50) val_dice_3d_present=**0.8433**, all-channel **0.6548**. Massive jump over V7's 0.362. Proposer alone already hits SOTA-adjacent. |
| 20:30      | Stage 2 v3 running — refiner Dice plateau at 0.11–0.14 for 7 epochs. Suspicion: loss, alignment, or eval protocol bug. |
| 21:10      | Diagnosed OrganFlowSAM2 is organ-conditioned → only the channel matching `organ_id` is faithful. Wrote `scripts/eval_refiner_per_organ.py` — plateau still ~0.15. |
| 21:40      | Wrote `scripts/eval_gate_sensitivity.py`, swept gate bias {-2, -1, 0, +1, +2, +4}. **Proposer 2D Dice reported 0.043** at every bias — uncovered `slab_center_z` bug (was 4 instead of 48). |
| 21:55      | **Bug fix #1** (`datasets/amos22_v9.py`): `slab_center_z = volume_patch // 2 = 48` (was `depth // 2 = 4`). The value indexes into the 3D proposer prob (96 slices deep), not the slab. |
| 22:00      | **Bug fix #2** (`models/voluformer_v9.py`): gate bias init `+2.0` (was `-1.0`). With proposer at 0.84 Dice and refiner near random, the gate must default to trusting the proposer. |
| 22:10      | Stage 2 v4 launched, crashed immediately: `IndexError: index 48 is out of bounds for dimension 2 with size 8` — trainer used `slab_center_z` as a slab index. |
| 22:20      | **Bug fix #3** (5 sites): `training/train_v9.py`, `training/trainer_v9.py`, `scripts/*`. Slab midplane = `mask_slab.shape[2] // 2` (slab-local). `slab_center_z` is only for the model's proposer-prob gather. |
| 22:25      | Stage 2 v4 #2 launched. Ran 4 epochs clean. Printed `ep 1 prop=0.8433 ref=0.1185 fus=0.0294` — **fused worse than refiner**, impossible if alignment is correct given gate bias +2.0. |
| 22:45      | **Bug fix #4** (`datasets/amos22_v9.py`): slab now crops to **same `(y0, x0)` window as the volume patch**. Previously slab was full H×W, volume patch was center-96×96 — so the proposer's interpolated slice was pasted onto unrelated anatomy. Verified with alignment test: volume-patch midslice ↔ slab midslice pixel-diff = 0.00. |
| 22:48      | Stage 2 v5 launched (`b8sr01fhb`), 30 epochs, warm-started from Stage 1 ep 50. Result TBD (wake-up at 23:27). |

### Stage 1 final table

| Stage | Model | Val Dice 3D (present) | Val Dice 3D (all ch) | Notes |
|-------|-------|----------------------:|---------------------:|-------|
| 1 ep 50 | SwinUNETRProposer (48M params) | **0.8433** | **0.6548** | Frozen for Stage 2+. Loaded from `checkpoints/v9_stage1/last.pt`. |

### Stage 2 false-starts (all archived under `checkpoints/v9_stage2_buggy_*`)

| Version | Bug | Outcome |
|---------|-----|---------|
| v2 (was v1) | BCE-only loss → refiner all-zero | Refiner Dice ≈ 0.0000 |
| v3 | +BCE+Dice+Tversky, 7 epochs | Refiner ~0.13, **fused = 0.03 (broken)** |
| v4 | Bug fixes #1+#2 applied | Instant crash on line 367 (bug #3 uncovered) |
| v4 #2 | Bug fix #3 added (5 sites) | 4 epochs clean but **fused still 0.03** (bug #4 uncovered) |
| **v5 (RUNNING)** | Bug fix #4 — aligned slab↔patch crop | Started 22:48, 30 epochs |

### Novelty audit (honest reframe)

Searched 2024-2026 literature against the 8 original "novel contributions" claim.
**Honest finding: 1 defensible system-level contribution, not 8.** Most pieces are
individually derivative (gated fusion, flow-matching segmentation, multi-teacher KD,
consistency losses — all in literature). The defensible combination is:

> **"Anatomical cascade with per-organ gated 3D→2D fusion and organ-conditioned
> flow-ODE refinement over SAM2 features."**

That framing is workshop-publishable (MIDL / MICCAI workshop tracks). For MICCAI
main track we need AMOS22 + at least one other benchmark (BTCV or TotalSegmentator
are natural fits; TotalSeg weights are already downloaded). Dropping Novel #5
(flow shape prior) and #8 (boundary DDPM) from the "novelty" list — they stay as
*losses* but won't be claimed as contributions.

### Current running process

`bash task b8sr01fhb` — Stage 2 v5 training, started 22:48 AST 2026-04-21.
30 epochs × ~6 min/epoch = ~3 hours to finish. Full val-metrics dict logged to
TensorBoard at `logs/v9_tierB/stage2/`.

---

## 2026-04-22 — Stage 2 v5 refiner plateau observed (user logging out)

### Val trajectory through ep 9

| ep | train_loss | prop   | ref    | fus    |
|---:|-----------:|-------:|-------:|-------:|
| 1  | 1.8961     | 0.8433 | 0.1116 | 0.7005 |
| 2  | 1.8309     | 0.8433 | 0.1020 | 0.6995 |
| 3  | 1.8029     | 0.8433 | 0.1129 | 0.7006 |
| 4  | 1.9186     | 0.8433 | 0.1194 | 0.7005 |
| 5  | 1.7587     | 0.8433 | 0.0868 | 0.6997 |
| 6  | 1.7407     | 0.8433 | 0.1208 | 0.7002 |
| 7  | 1.7101     | 0.8433 | 0.1267 | 0.7007 |
| 8  | 1.7005     | 0.8433 | 0.1175 | 0.6992 |
| 9  | 1.6921     | 0.8433 | 0.1199 | 0.7003 |

### Interpretation

- **Alignment fix worked**: fused=0.70 vs 0.03 pre-fix. The four 2026-04-21 bugs are all cured.
- **Refiner is plateaued, not slow-learning**: 9 epochs, refiner oscillating 0.09-0.13, no monotonic climb. Train loss IS dropping (1.90 → 1.69) so the model learns something — just not something that transfers to Dice.
- **Cascade currently HURTS**: fused 0.70 < proposer 0.84. Gate bias +2.0 keeps proposer dominant, but the 12% weight on a near-random refiner still pulls borderline pixels below threshold.
- **Root-cause hypothesis**: The ODE is organ-conditioned on a single `organ_id` per sample, but the decoder outputs K=15 channels trained against multi-organ GT. Conflicting signal across channels. Needs ablation to confirm (see `memory/feedback_v9_refiner_plateau.md`).

### Action taken before logout

- Saved Stage 2 v5 checkpoints at epochs 2, 4, 6, 8, 10 (every 2 epochs).
- Wrote `memory/project_v9_status.md` (updated), `memory/project_v9_sota_plan.md` (new), `memory/feedback_v9_refiner_plateau.md` (new), `docs/superpowers/plans/2026-04-22-v9-resume-plan.md` (new).
- Updated `memory/MEMORY.md` index.
- Training task `b8sr01fhb` will terminate with the CC session. `epoch_010.pt` is resumable but no value — refiner won't unstick without architecture change.

### On resume (next session)

1. Read `memory/project_v9_status.md`, `memory/project_v9_sota_plan.md`, `docs/superpowers/plans/2026-04-22-v9-resume-plan.md`.
2. Run Tier 0: 3D sliding-window eval on `checkpoints/v9_stage1/last.pt` (proposer-only). This is the paper number.
3. Only then decide whether to debug refiner or accept proposer-only as the contribution.

---

## 2026-04-22 (cont.) — TTA channel-swap bug fixed; training-data spatial bug found

### Key findings during Tier 0 diagnosis

1. **The 0.8433 Stage 1 number is patch-level, not volume-level.** Training's
   `_val_metrics` uses 96³ patches sampled with `pos_frac=1.0` (every val patch
   centered on a random present organ). With `(probs > 0.5)` per-channel and
   NaN-mask on empty channels, the metric measures "given the patch is on an
   organ, can you segment it" — much easier than full-volume sliding window.

2. **TTA L-R channel-swap bug found and fixed** in `scripts/eval_v9_proposer_preproc.py`.
   - After `Orientationd(RAS)` preprocessing, axis 0 of the stored tensor = R (left-right).
   - Eval transposes `(H,W,D) -> (D,H,W)`, so axis 1 of the working tensor (axis 2 of `preds`) is the L-R body axis.
   - The TTA list `flip_axes = [[1],[2],[1,2]]` flips this L-R axis on 2 of 4 passes.
   - Old code un-flipped spatially but did NOT swap `r_kidney <-> l_kidney` and `r_adrenal <-> l_adrenal` channels — the un-flipped predictions ended up in the WRONG channels.
   - Fix: when `2 in flips`, swap channels (2,3) and (11,12) before accumulating.
   - Smoke on amos_0374 (single val volume): r_kidney 0.007 → 0.051, l_kidney 0.139 → 0.204; mean present-only essentially unchanged (small organ-set rebalancing).

3. **Training-data spatial bug discovered (not yet fixed).** `datasets/amos22_v9.py:213-215`:
   ```python
   y0 = max(0, min(H - p, H // 2 - p // 2))
   x0 = max(0, min(W - p, W // 2 - p // 2))
   ```
   The (y0, x0) crop is **always the volume's H,W center**. Only z varies between samples. The model has only ever trained on the central column of every volume. SwinUNETR is mostly translation-equivariant so it should generalize, but this clamp-to-center fundamentally limits the data distribution the model sees.

4. **HU normalization is consistent.** The preprocessed `[-175, 250]` -> `[0,1]` round-trips cleanly through the training-time `_load_volume` un-scale + `_normalize` re-scale (the [-175,250] interval is fully contained in the config `hu_clip=[-200,250]`, so the clip is a no-op). Eval reading the preprocessed tensor directly matches training input. Not the bug.

### First-volume Tier 0 eval (amos_0374, with TTA channel-swap fix)

| organ           | dice |
|-----------------|-----:|
| spleen          | 0.0266 |
| r_kidney        | 0.0513 |
| l_kidney        | 0.2042 |
| gallbladder     | 0.0000 |
| esophagus       | 0.0000 |
| liver           | 0.5078 |
| stomach         | 0.0000 |
| aorta           | 0.8164 |
| ivc             | 0.4573 |
| pancreas        | 0.0146 |
| r_adrenal       | 0.0071 |
| l_adrenal       | 0.0000 |
| duodenum        | 0.0000 |
| bladder         | 0.1583 |
| prostate_uterus | 0.7376 |

mean (organs with dice > 0) = 0.298, mean (all 15 organs) = 0.199.

Pattern: **central, large, compact organs work** (aorta 0.82, prostate 0.74, liver 0.51, IVC 0.46). **Lateralized, small, or variable-shape organs fail** (spleen 0.03, kidneys low, gallbladder 0, stomach 0, pancreas 0.01, adrenals 0, esophagus 0, duodenum 0).

### In flight at log-write time

- Background task: full 30-volume val 3D eval with TTA channel-swap fix → `reports/v9_proposer_3d_eval_ttafix.json`. ETA ~42 min.
- Will publish proper per-organ AMOS22 numbers once it lands.

### Next decisions (after full eval lands)

- If full-set numbers track this single-volume snapshot (~0.30 mean), the cause is structural (training data distribution), not eval bug.
- Options: (a) retrain Stage 1 with random (y0,x0) crops + spatial augmentation, ~6-7h on 4090; (b) bbox-prior eval on full volumes; (c) reframe paper around the structural-bug discovery + a more honest patch-wise metric.

---

## 2026-04-22 (cont. 2) — structural training-data fixes deployed

### Both data-distribution fixes implemented in `datasets/amos22_v9.py`

**Fix A — `_pick_yx` random/jittered crop (replaces center-clamped (y0, x0)):**
- Train POSITIVE slabs: jitter ±max(p//4, 8) voxels around the chosen organ's centroid in (H,W). Falls back to volume centroid if the chosen z-slice doesn't contain the organ, then to volume center if the organ is absent altogether.
- Train NEGATIVE/MIXED slabs: uniform random (yc, xc) in `[p//2, dim - p//2]`.
- Val: deterministic center crop (preserves comparability with the historical 0.8433 patch-level number).
- Slab and 3D volume patch share the same (y0, x0) → cascade alignment intact.

**Fix B — `_maybe_flip_lr` train-only L-R flip + paired-organ remap:**
- p=0.5 random flip along axis 1 (= R axis after RAS + (D,H,W) transpose).
- Both slab and volume_patch flipped in lockstep.
- Label remap when flipped: 2↔3 (r_kidney↔l_kidney), 11↔12 (r_adrenal↔l_adrenal). The `organ_id` query is also remapped so the refiner's organ conditioning stays consistent with the GT mask channel after the swap.
- Implemented as a per-class `_LR_LABEL_REMAP` lookup table built lazily and cached on the class.

**Smoke test (`scripts/smoke_dataset_aug.py`)** — pulled 16 train samples and confirmed shapes `slab=(8,3,320,320)`, `mask_slab=(15,8,320,320)`, `volume=(1,96,96,96)`, `mask_volume=(15,96,96,96)`, all label values within `[0, 15]`, all image values finite, mask values in `[0,1]`. Also pulled 64 samples and confirmed paired-organ queries land at ~25 % (16/64), matching the 4/15 expected rate.

### Why these are the right fixes (not a guess)

The model has a near-deterministic view of every volume:
1. Center-only (y0, x0) → only the central column of the abdomen ever appears under the patch.
2. No flip aug → the model has never seen a flipped torso.

Both axes of the `(y0, x0)` plane correspond to (R, A): axis 1 = R (left-right body axis), axis 2 = A (anterior-posterior). With the fix, lateralized organs like kidneys, adrenals, gallbladder, and stomach now appear at varying patch positions, in both left and right body hemispheres, during training.

SwinUNETR's window-attention is *approximately* translation-equivariant within a window but not across windows (windows are fixed in the patch coordinate frame). So the center-clamp is more harmful for SwinUNETR than for a fully-conv UNet. This makes the fix particularly load-bearing for our architecture.

### Plan once full eval lands

- **If mean dice ~0.30 (predicted)**: launch Stage 1 fine-tune from `last.pt` with the new dataset for ~10 epochs (~24 h on 4090, vs ~6 days from scratch). Use the existing `proposer_ckpt` mechanism in `training/train_v9.py` to preload all proposer weights, then just resume training. Expected uplift: lateralized organs (kidneys, adrenals, gallbladder, stomach) should climb from near-0 to >0.5; midline organs should hold or improve slightly.
- **If mean dice >0.45**: hold off on retrain; first run a sigma sweep on TTA Gaussian blend and a stride-32 sliding-window comparison. Cheap Tier-1 wins before committing to retrain time.

### Files changed

- `datasets/amos22_v9.py` — added `_pick_yx`, `_lr_label_remap`, `_maybe_flip_lr`; rewired `__getitem__` to use them; pre-flip extract of `vol_patch_pre` so the lockstep flip works.
- `scripts/smoke_dataset_aug.py` — new smoke test (verifies shapes / ranges / paired-organ rate).
- `scripts/eval_v9_proposer_preproc.py` — already had `LR_SWAP_CHANNELS` swap from the previous session.

---

## 2026-04-23 — Full 30-volume 3D eval landed: structural failure confirmed

`reports/v9_proposer_3d_eval_ttafix.json` — `checkpoints/v9_stage1/last.pt`, 30 val volumes, TTA ON (with the L-R channel-swap fix from 2026-04-22), 4-way flips, CC post-proc, sliding window 96³ stride 48.

**Mean 3D Dice (present-only across val volumes): 0.1655**
Mean 3D Dice (all-channel): nan (two volumes had empty/empty per-organ pairs → NaN → any/nan contaminates the mean)

### Per-organ 3D Dice — 30 val volumes

| organ            | dice   | vols |
|------------------|-------:|-----:|
| aorta            | 0.8611 | 30   |
| ivc              | 0.3158 | 30   |
| liver            | 0.2877 | 30   |
| bladder          | 0.2840 | 30   |
| prostate_uterus  | 0.2066 | 30   |
| esophagus        | 0.1402 | 29   |
| l_kidney         | 0.0880 | 30   |
| r_kidney         | 0.0844 | 30   |
| duodenum         | 0.0194 | 30   |
| spleen           | 0.0110 | 30   |
| stomach          | 0.0069 | 30   |
| r_adrenal        | 0.0063 | 30   |
| pancreas         | 0.0051 | 30   |
| l_adrenal        | 0.0001 | 30   |
| gallbladder      | 0.0000 | 29   |

### The pattern (unambiguous)

- **Aorta (0.86)** is the ONLY high-scoring organ. It's the one organ that sits on the centerline of almost every abdominal CT. The center-clamped (y0, x0) training put it right where the model always looked.
- **Liver (0.29) and bladder (0.28)** — large, mostly-central but with off-center lobes/extent → partial credit only.
- **Kidneys (0.08, 0.09)** — lateralized, small compared to liver → mostly missed.
- **Spleen / stomach / pancreas / adrenals / duodenum / gallbladder (~0)** — lateralized and/or variable-shape → near-complete failure.

The 0.84 "paper ceiling" was an artifact of `pos_frac=0.5` organ-centered 96³ patches + per-channel `(probs > 0.5)` thresholding with NaN-mask on empty channels. The model was scored only on patches that happened to contain the organ in the center; it learned to detect "something organ-like in the middle of my field of view" without ever learning WHERE each organ lives in the abdomen.

### Decision gate triggered

0.1655 << 0.40 → **launch Stage 1 fine-tune** with the deployed (y0, x0) + L-R flip fixes. Config: `configs/v9_stage1_finetune.yaml` (10 epochs, lr=5e-5, proposer_ckpt=checkpoints/v9_stage1/last.pt, output_dir=checkpoints/v9_stage1_ft — segregated so original `last.pt` is preserved). ETA ~24 h on the 4090.

### What this doesn't change

- Aorta result (0.86) is genuinely strong and should survive fine-tuning — central organs still benefit from the diversified sampling.
- Cascade refiner assessment (plateaued at 0.12) is independent of this bug; even with a fixed proposer, the refiner conditioning problem diagnosed 2026-04-22 remains.
- The paper framing shifts from "SOTA-adjacent proposer" to "a faithful reconstruction of a common failure mode in patch-wise training + a structural fix with before/after numbers."

---

## 2026-04-23 (evening) — Stage 1 fine-tune complete; TriSSR module shipped

### Fine-tune run: `checkpoints/v9_stage1_ft/last.pt`

Launched 20:14, done 21:05 — **51 min wall**, 10 epochs × ~4.5 min/ep on the 4090. Warm-started from `checkpoints/v9_stage1/last.pt` (67.8M trainable), running with the structural fixes: random/jittered `(y0, x0)` crop + L-R flip + paired-organ label & organ_id remap.

Patch-level val Dice (same metric family as the historical 0.8433 "paper ceiling"):

| epoch | train_loss | val (patch) |
|------:|-----------:|------------:|
|  1    | 0.7669     | 0.8186      |
|  2    | 0.5160     | 0.8273      |
|  3    | 0.4346     | 0.8300      |
|  4    | 0.3923     | 0.8424      |
|  5    | 0.3576     | 0.8451      |
|  6    | 0.3349     | 0.8499      |
|  7    | 0.3133     | 0.8527      |
|  8    | 0.2999     | **0.8541**  | best
|  9    | 0.2920     | 0.8536      |
| 10    | 0.2868     | 0.8538      |

Train loss monotonically decreased. Val plateaued at ep 8 (0.8541/0.8536/0.8538). Log: `reports/v9_stage1_ft.log`. All even-numbered checkpoints retained (ep 2/4/6/8/10 + `last.pt`).

### CAVEAT: the 0.8541 is still patch-level

Same pos-frac=0.5 organ-centered 96³ patches + per-channel `(probs > 0.5)` + NaN-mask on empty channels that inflated the 0.8433 number. The **real question** — does the structural-fix fine-tune actually lift the 0.1655 full-volume 3D Dice — was not answered by this run.

### 3D full-volume eval launched

`reports/v9_proposer_3d_eval_ft.json` (in flight at log-write time) — `checkpoints/v9_stage1_ft/last.pt`, same 30 val volumes, same TTA-with-LR-channel-swap + CC + argmax protocol, apples-to-apples with the 0.1655 baseline. ETA ~45 min. Decision gate on the 3D number:
- ≥ 0.70 → fine-tune worked, launch TriSSR-V9 training
- 0.40–0.70 → partial rescue, diagnose remaining failure modes before TriSSR
- < 0.40 → structural fix is not the main bug, deeper investigation needed

### New module: TriSSR-V9 (tri-scan SSM logit refiner)

`models/trissr_v9.py` + `training/train_trissr_v9.py` — 0.18M params, inspired by SegMamba (MICCAI 2024). A 3-block tri-direction Mamba bottleneck at 24³ operating on `(proposer_logits ⊕ CT_image)`, residual-zero-initialized (identity at epoch 0 → cannot regress). Provides global 3D context at the logit level — precisely the capability SwinUNETR's windowed attention lacks.

Novelty framing for the thesis: **parallel dual-refiner cascade**. Proposer → (OrganFlow 2D-slab ODE refiner ∥ TriSSR 3D SSM refiner) → per-organ gated fusion. Two refiners with orthogonal inductive biases (temporal-continuous vs. spatial-continuous) feeding one fusion — distinct from single-branch cascades in SegMamba, MCP-MedSAM, and nnWNet.

CPU smoke passed: `identity_init_diff = 0.00e+00`. Training script loads frozen fine-tuned proposer, trains only TriSSR with Dice + BCE vs. volume-patch GT for 8 epochs × ~4 min ≈ 30 min on the 4090.

### Files changed or added today
- `checkpoints/v9_stage1_ft/{epoch_002.pt,epoch_004.pt,epoch_006.pt,epoch_008.pt,epoch_010.pt,last.pt}` — fine-tune outputs
- `models/trissr_v9.py` — new, tri-scan SSM refiner head
- `training/train_trissr_v9.py` — new, Stage-2.5 trainer for TriSSR on top of frozen fine-tuned proposer
- `reports/v9_stage1_ft.log`, `reports/v9_proposer_3d_eval_ft.log` — run outputs

---

## 2026-04-23 (late evening) — FT proposer 3D eval: 0.5091 (3.08× lift)

30 val volumes, TTA+CC, argmax post-softmax, 38.6 min total. Report: `reports/v9_proposer_3d_eval_ft.json`.

**Mean 3D Dice (present-only): 0.5091** (up from 0.1655 baseline — **3.08× lift**). Every organ improved.

| organ | pre-FT | post-FT | Δ | bucket |
|-------|-------:|--------:|--:|--------|
| liver           | 0.288  | **0.9163** | +0.628 | strong |
| aorta           | 0.861  | **0.9194** | +0.058 | strong |
| ivc             | 0.316  | **0.8478** | +0.532 | strong |
| r_kidney        | 0.084  | 0.7549     | +0.671 | strong |
| l_kidney        | 0.088  | 0.7246     | +0.637 | strong |
| spleen          | 0.011  | 0.6619     | +0.651 | mid |
| esophagus       | 0.140  | 0.6372     | +0.497 | mid |
| bladder         | 0.284  | 0.4279     | +0.144 | mid |
| prostate_uterus | 0.207  | 0.4042     | +0.197 | mid |
| gallbladder     | 0.000  | 0.3734     | +0.373 | weak |
| r_adrenal       | 0.006  | 0.3109     | +0.305 | weak |
| duodenum        | 0.019  | 0.2033     | +0.184 | weak |
| l_adrenal       | 0.0001 | 0.1610     | +0.161 | weak |
| pancreas        | 0.005  | 0.1561     | +0.151 | weak |
| stomach         | 0.007  | 0.1376     | +0.131 | weak |

**Diagnosis**: the structural fix (random/jittered `(y0, x0)` + L-R flip + paired-organ remap) worked. Liver jumped +0.63, kidneys +0.64/+0.67, spleen +0.65 — all previously broken by the patch-center clamp, now recovered. Small/tubular organs (stomach, pancreas, adrenals, duodenum) are the remaining weak spots — this is where TriSSR's global 3D context and downstream Phase C gains (small-organ loss + copy-paste aug) should help most.

**Per-volume variance**: individual volume means ranged 0.35-0.64, std around 0.08. Stable signal.

### Decision gate (from earlier): 0.40-0.70 = partial rescue → diagnose before TriSSR
Diagnosis done (above table). TriSSR is motivated (global context for weak small/tubular organs). **TriSSR training launched** at 22:12 via `training/train_trissr_v9.py` — 8 epochs × ~4 min on 4090, proposer frozen, Dice + BCE against volume-patch GT. Output: `checkpoints/trissr_v9/last.pt`.

### Next steps (committed two-month plan)
- Combined proposer + TriSSR 3D eval (~40 min after TriSSR trains)
- Phase B validation rigor (lock 30-vol val, 5-vol dev carve)
- Phase C: deep-sup in SwinUNETR decoder (MONAI exposes decoder5..1 — tappable); small-organ loss (inverse-vol Dice + Tversky β=0.7 + focal γ=1.5); copy-paste small-organ aug; organ-centric re-seg at 0.75 mm; 2-model ensemble.
- Phase D: scale TriSSR or pivot to small-organ expert head
- Phase E: writeup + ablation cells per contribution

### Files changed this session
- `training/train_trissr_v9.py` — bug fix: use `full_logits` (K+1) not `logits` (K) to match TriSSR's 16-channel residual input
- `scripts/eval_v9_proposer_preproc.py` — extended with `--trissr-ckpt` flag for combined-cascade eval
- `reports/v9_proposer_3d_eval_ft.{log,json}` — FT proposer 3D eval outputs
- `reports/trissr_v9_train.log` — TriSSR training output (in flight)

## 2026-04-24 (early morning) — Phase C infrastructure landed (all behind flags)

Built the three Phase-C levers targeting the weak organs (pancreas, adrenals, duodenum, gallbladder, stomach) identified in the 04-23 eval. All new code is gated behind config flags that default to OFF, so existing runs are unaffected. Ablations can be run one-axis-at-a-time per the V8+ experimentation discipline.

### 1. Copy-paste small-organ augmentation (novel for 3D CT multi-organ)
- New module: `datasets/copy_paste_bank.py` (`OrganCrop`, `CopyPasteBank.build_from_preproc`, `maybe_paste`, `_zoom_vol`)
- Bank built once from 270 train volumes (cache: `data/amos22/copy_paste_bank.pt`); stores up to 300 bbox crops per rare organ with 4-voxel margin.
- At train time, with prob `p_paste` per sample, pastes a rare-organ crop at a random location if the subject organ isn't already present. Crop downsized to <= P//2 so it can't dominate the patch.
- Wired into `datasets/amos22_v9.py` (`__init__` kwargs + `__getitem__` post-normalise); `training/train_v9.py` loads bank from cache when `cfg.data.copy_paste_prob > 0`.

### 2. Small-organ weighted loss
- New module: `training/losses_small_organ.py` — `SmallOrganLoss` combining:
  - Inverse-sqrt-volume weighted soft Dice (bg excluded)
  - Tversky (alpha=0.3, beta=0.7) — penalises FN 2.3x more than FP
  - Focal CE (gamma=1.5) — down-weights easy background
- Wired into `training/train_v9.py stage1_loss` via optional `small_organ` module; activated by `cfg.training.loss_small_organ=true`.

### 3. Deep supervision on SwinUNETR decoder taps
- `models/swin_unetr_3d.py` — added `deep_supervision`, `deep_sup_layers` kwargs. Forward hooks capture decoder5/4/3 features; lazy 1x1x1 conv aux heads emit (B, K+1, D', H', W') logits at 4x / 8x / 16x downsample.
- Aux heads are `nn.ModuleDict` with `__` as dot-replacement in key names (ModuleDict forbids `.`).
- Wired into `models/voluformer_v9.py` (plumbing) + `training/train_v9.py stage1_loss` (aux CE with area-nearest downsampled labels) + training loop (linear weight schedule `deep_sup_weight_start -> deep_sup_weight_end`).
- Smoke: 3 aux heads on a 64^3 patch produce 4^3/8^3/16^3 logits; 5.4k aux-head params; gradients flow through full stack.

### Config surface (configs/v9_tierB.yaml)
- `data.copy_paste_prob` (default 0.0) + `data.copy_paste_cache`
- `training.loss_small_organ` (default false) + `training.loss_small_organ_cfg` sub-block
- `training.deep_sup_weight_start` / `training.deep_sup_weight_end` (default 0.0 / 0.0)
- `model.proposer.deep_supervision` (default false) + `model.proposer.deep_sup_layers`

### Also
- `training/train_trissr_v9.py`: added `--resume` flag that loads the latest `epoch_NNN.pt` and advances the LR scheduler. TriSSR-V9 training now survives session interruptions: the dataset edit in this session killed the running process, `--resume` brought it back from epoch 2/8.

### Files changed this session (2026-04-24)
- NEW: `datasets/copy_paste_bank.py`
- NEW: `training/losses_small_organ.py`
- MOD: `datasets/amos22_v9.py` (copy-paste hook in `__getitem__`)
- MOD: `training/train_v9.py` (small-organ loss + deep-sup schedule + copy-paste bank load)
- MOD: `models/swin_unetr_3d.py` (deep-sup hooks + aux heads)
- MOD: `models/voluformer_v9.py` (proposer kwargs plumbing)
- MOD: `configs/v9_tierB.yaml` (Phase C flags)
- MOD: `training/train_trissr_v9.py` (--resume flag)

### TriSSR-V9 training finished (2026-04-24 03:59)

8 / 8 epochs complete on proposer-frozen TriSSR-V9. Per-epoch patch-level dice:

| epoch | loss   | dice   | bce    | dt (min) |
|-------|--------|--------|--------|----------|
| 1     | 0.5786 | 0.5085 | 0.0871 | 7.6      |
| 2     | 0.5747 | 0.5113 | 0.0859 | 7.7      |
| 3     | 0.5846 | 0.5001 | 0.0847 | 7.9      |
| 4     | 0.5728 | 0.5108 | 0.0837 | 8.7      |
| 5     | 0.5628 | 0.5196 | 0.0824 | 9.4      |
| 6     | 0.5666 | 0.5151 | 0.0816 | 9.4      |
| 7     | 0.5618 | 0.5193 | 0.0811 | 9.3      |
| 8     | 0.5550 | 0.5257 | 0.0807 | 9.3      |

Total run: ~71 min (with a crash / resume mid-way). Patch-level dice lifted from 0.5085 → 0.5257 (+0.017). Checkpoint: `checkpoints/trissr_v9/last.pt` (770 KB; 0.18 M params). **Combined 3D eval launched** immediately after.

### Copy-paste bank built (2026-04-24 03:17)

Scanned 270 train volumes; crops/organ: gallbladder 253, stomach 265, pancreas 270, r_adrenal 269, l_adrenal 270, duodenum 269. Cache: `data/amos22/copy_paste_bank.pt` (2.3 GB). Ready for use via `cfg.data.copy_paste_prob > 0`.

---

## 2026-04-24 V10 moonshot plan (month-long, single 4090)

### Context

V9 baseline on 30-vol AMOS22 CT val:
- FT proposer alone (3D full-volume Dice, present-only): **0.5091**
- FT proposer + TriSSR + TTA cascade: **0.5353** (+0.026, passed 0.02 gate)

Per-organ (cascade): liver 0.925, aorta 0.921, ivc 0.850, r_kidney 0.80, l_kidney 0.78, spleen 0.693, gallbladder 0.473, bladder 0.461, prostate 0.415, r_adrenal 0.345, duodenum 0.206, stomach 0.180, pancreas 0.166, l_adrenal 0.164, esophagus 0.02.

V9 limit = from-near-scratch SwinUNETR. Paper-reported SOTA uses large 3D CT pretraining. The ceiling-lifter is VoCo (PreCT-160K, 160K CT volumes, TPAMI 2025) — drop-in SwinUNETR-v2 weights.

### V10 = distinct model generation

Sacred V9 checkpoints (`v9_stage1/last.pt`, `v9_stage1_ft/last.pt`, `trissr_v9/last.pt`) remain untouched. All new V10 artifacts go under `v10_*` paths.

### Base-model ladder (4090 feasibility)

| variant | params | fs  | patch | full FT 4090     | expected AMOS22 Dice |
|---------|-------:|----:|------:|------------------|---------------------:|
| VoCo-B  |    72M |  48 |  96³  | fits, batch 2    | ~0.82                |
| VoCo-L  |   290M |  96 |  96³  | fits, batch 1    | ~0.86                |
| VoCo-H  |  1.2B  | 192 |  96³  | does NOT fit     | ~0.88-0.89 via LoRA  |

VoCo-L is **primary**. VoCo-H is Week 3/4 stretch via LoRA (rank 16 on qkv + MLP + out-proj → ~25-50M trainable, ~7-8 GB VRAM).

### Tiers

- 0.83 committed: VoCo-L baseline FT
- 0.87 stretch: VoCo-L + best Phase C axis + MedNeXt ensemble
- 0.89 aspirational: above + text-prompted weak-organ rescue + 8-way TTA
- 0.90+ moonshot: above + VoCo-H LoRA on weakest organs + uncertainty-guided re-seg

### Week-by-week

| week | main task                                              | gate                           |
|-----:|--------------------------------------------------------|--------------------------------|
|    1 | VoCo-L full FT (50 ep, 60-100 h)                       | A: mean 3D Dice ≥ 0.82         |
|    2 | Phase C single-axis ablations (cp / sl / ds) from W1   | B: ≥+0.01 on winning axis      |
|    3 | MedNeXt-L FT + text-rescue head; start VoCo-H LoRA eval| C: MedNeXt ≥0.84, rescue ≥+0.02 weak organs |
|    4 | TriSSR-on-VoCo + organ-centric re-seg + TTA + ensemble | D: ≥ 0.87 (moonshot 0.90+)    |

### Configs landed this session

- `configs/v10_voco_b.yaml` — fallback base (fs=48, batch 2, 50 ep)
- `configs/v10_voco_l.yaml` — primary base (fs=96, batch 1, 50 ep)
- `configs/v10_voco_l_cp.yaml` — W2 copy-paste axis (prob=0.3, 15 ep resume from W1)
- `configs/v10_voco_l_sl.yaml` — W2 small-organ loss axis (tversky β=0.7)
- `configs/v10_voco_l_ds.yaml` — W2 deep-sup axis (weight 0.4→0.1)
- `configs/v10_voco_l_full.yaml` — W2 full stack (only if ≥2 axes pass gate)

All W2 configs currently point `pretrained_weights` at raw VoCo-L; swap to `checkpoints/v10_voco_l_ft/last.pt` after W1 finishes.

### Pretrained weights

- `checkpoints/pretrained/voco/VoComni_B.pt` — 299 MB, 72.8M params, 21-class out
- `checkpoints/pretrained/voco/VoComni_L.pt` — 1.17 GB, 290.4M params, 21-class out
- `checkpoints/pretrained/voco/VoComni_H.pt` — 4.65 GB, ~1.2B params, 21-class out

Loading via existing `SwinUNETRProposer.load_pretrained()` — handles VoCo keys out of the box; 21→16 class head is shape-dropped at load time. VoCo-L smoke test: 134/134 encoder keys filled, forward shape OK, 290.4M params.

### VRAM pre-flight (VoCo-L, 96³, batch 1, AMP)

- peak_allocated: 5.10 GB
- peak_reserved: 7.22 GB
- RESULT: PASS (<22 GB safety ceiling). Huge headroom for full V9 wrapper.

### Launched runs (2026-04-24)

- **v10_voco_l_ft Week-1 baseline** — `python -m training.train_v9 --config configs/v10_voco_l.yaml --stage 1`
  - Env: `C:/Users/Raywa/AppData/Local/Programs/Python/Python312/python.exe` (torch 2.8.0+cu126)
  - Log: `logs/v10_voco_l_ft/train.log`
  - Ckpt dir: `checkpoints/v10_voco_l_ft/`
  - Steady-state GPU: ~11.4 GB / 24 GB VRAM, 95% util, 68-71 °C
  - Throughput: 3.49 steps/s ⇒ 5.3 min per epoch ⇒ 50-epoch run completes in ~4.4 h (not the 60-100 h worst-case estimate)

### NaN incident and bf16 fix (2026-04-24 20:24 → 20:34)

First launch produced NaN train_step/loss for 844 straight steps. Root cause: `F.cross_entropy` on fp16 `full_logits` inside autocast — VoCo-L's fs=96 activations overflow fp16, logits become Inf, CE → NaN, `GradScaler` cannot recover. Fixed by switching autocast dtype to bf16 (same exponent range as fp32; 4090 supports it natively). Added `training.amp_dtype` config knob (`fp16` or `bf16`); `GradScaler` auto-disabled for bf16. Smoke test `scripts/smoke_voco_l_loss.py` confirms finite loss over 3 real-batch steps.

### W1 Run 1 complete — STOP-RULE abort (2026-04-24 20:39 → 21:15)

| ep | train_loss | val prop Dice | dt (min) |
|---:|-----------:|--------------:|---------:|
|  1 |    1.2917  |       0.5786  |    5.3   |
|  2 |    0.7854  |       0.7484  |    5.1   |
|  3 |    0.7309  |       0.7649  |    5.1   |
|  4 |    0.7080  |       0.7917  |    5.2   |
|  5 |    0.6936  |  **0.8085**   |    5.2   |
|  6 |    0.6836  |       0.7973  |    5.2   |
|  7 |    0.6685  |       0.7875  |    5.2   |
|  8 |    0.4359  |       0.7672  |    5.2   |

STOP-RULE tripped at ep 8 (val fell 0.04 from best). Pattern: train loss 0.67→0.44 in one epoch = overfit. lr=1e-4 was too aggressive; cosine schedule over 50 ep kept the LR high. Ep 5 was global peak but `every_n_epochs=2` meant the ckpt was not saved — best saved = ep 6 (0.7973).

Run 1 artifacts archived: `checkpoints/v10_voco_l_ft_run1/`.

### W1 Run 2 launched (2026-04-24 21:17) — lr=3e-5, 15 ep, save-every-ep

Config changes from Run 1:
- `optimizer.lr`: 1e-4 → 3e-5 (3× lower)
- `optimizer.weight_decay`: 0.01 → 0.05 (5× stronger regularization)
- `epochs`: 50 → 15
- `scheduler.warmup_epochs`: 2 → 1
- `eval.every_n_epochs`: 2 → 1 (save every epoch; do not lose the peak)

Expected: smoother curve, peak ~0.82-0.84 around ep 8-12, total wall-time ~80 min.

### W1 Run 2 complete (2026-04-25 ~04:24) — plateau at 0.7667, BELOW Run 1 peak

| ep | train_loss | val proposer 3D Dice |
|----|-----------:|---------------------:|
|  1 | 2.0458 | 0.2887 |
|  2 | 1.0153 | 0.4933 |
|  3 | 0.8852 | 0.5813 |
|  4 | 0.8199 | 0.6660 |
|  5 | 0.7753 | 0.7311 |
|  6 | 0.7516 | 0.7495 |
|  7 | 0.7371 | 0.7488 |
|  8 | 0.7274 | 0.7574 |
|  9 | 0.7201 | 0.7635 |
| 10 | 0.7149 | 0.7638 |
| 11 | 0.7112 | 0.7661 |
| 12 | 0.7086 | 0.7657 |
| 13 | 0.7069 | 0.7665 |
| 14 | 0.7059 | **0.7667** ← peak |
| 15 | 0.7054 | 0.7667 |

**Diagnosis:** under-trained, not over-regularised. lr=3e-5 + wd=0.05 + 15 ep all
pulled the recipe toward "safe" simultaneously, breaking the V8+ one-axis rule.
Result is well below Run 1's ep-5 peak of 0.8085 and below Gate A (0.82). All
15 ep ckpts preserved at `checkpoints/v10_voco_l_ft/epoch_NNN.pt`. Best is
`epoch_014.pt` at 0.7667.

### W1 Run 3 launching (2026-04-25) — lr=1e-4 reverted, 10 ep, wd=0.05

Recipe = Run 1's aggressive LR (proven to hit 0.8085) + Run 2's stronger wd
+ short cosine schedule + save-every-ep. Cosine over 10 ep means LR halves by
ep 5, so the post-peak overfit Run 1 saw is bounded. Expected peak 0.81-0.83
around ep 5-7.

- `configs/v10_voco_l_r3.yaml` (NEW)
- `output_dir`: `checkpoints/v10_voco_l_ft_r3/` (Run 2 ckpts preserved untouched)
- `optimizer.lr`: 1.0e-4 (back to Run 1)
- `optimizer.weight_decay`: 0.05 (kept from Run 2)
- `epochs`: 10
- `eval.every_n_epochs`: 1
- `amp_dtype`: bf16

### W1 Run 3 complete (2026-04-25) — peak 0.7967 at ep 9 (V10 W1 base ckpt)

| ep | train_loss | val proposer 3D Dice |
|----|-----------:|---------------------:|
|  1 | 1.2906 | 0.5801 |
|  2 | 0.7860 | 0.7492 |
|  3 | 0.7313 | 0.7640 |
|  4 | 0.7088 | 0.7871 |
|  5 | 0.6948 | 0.7958 |
|  6 | 0.6853 | 0.7960 |
|  7 | 0.6789 | 0.7907 |
|  8 | 0.6746 | 0.7964 |
|  9 | 0.6717 | **0.7967** ← peak |
| 10 | 0.6702 | 0.7961 |

**Result vs alternatives:** Run 3 (0.7967) is +0.030 over Run 2 (0.7667) but
−0.012 below Run 1's unsaved ep-5 peak (0.8085). The "save-every-ep" fix did
its job — peak is preserved at `checkpoints/v10_voco_l_ft_r3/epoch_009.pt`.

**Gate A status:** 0.82 target → 0.7967 actual = MISS by 0.023. Per V8+
discipline, **not burning more compute on W1 hyperparam sweeps** — Phase C
small-organ knobs (cp/sl/ds) are the real lever for the missing ~0.025. They
each resume from `epoch_009.pt`.

### W2 Phase C ablation kicking off (2026-04-25)

Each ablation = 15 ep, lr=5e-5 (lower than W1 since base is already trained),
wd=0.05, bf16, save every ep. Gate per ablation = beat 0.7967 by ≥ 0.005 (a
relaxed gate vs the original 0.02 to allow stacking). Best result feeds W2
combined config or W3 LoRA-encoder swap.

- `v10_voco_l_cp.yaml` — copy_paste_prob=0.3 only
- `v10_voco_l_sl.yaml` — small-organ weighted loss only
- `v10_voco_l_ds.yaml` — deep supervision only
- `v10_voco_l_full.yaml` — all three (only if ≥2 axes pass individual gate)

All configs updated to `pretrained_weights: null`; W1 base loaded via CLI
`--proposer_ckpt checkpoints/v10_voco_l_ft_r3/epoch_009.pt` (cleanly: 167
proposer keys loaded, 0 missing/unexpected).

### W2 CP ablation complete (2026-04-25) — peak 0.8001 at ep 5, MISS gate

| ep | train_loss | val proposer 3D Dice |
|----|-----------:|---------------------:|
|  1 | 0.6785 | 0.7971 |
|  2 | 0.6735 | 0.7983 |
|  3 | 0.6696 | 0.7953 |
|  4 | 0.6665 | 0.7994 |
|  5 | 0.6640 | **0.8001** ← peak |
|  6 | 0.6616 | 0.7992 |
|  7 | 0.6584 | 0.7974 |
|  8 | 0.6437 | 0.7945 |
|  9 | 0.6266 | 0.7934 |
| 10 | 0.6197 | 0.7932 |

Lift on mean 3D Dice: +0.0034 vs W1 base 0.7967 (gate 0.005 miss). Train loss
collapse from 0.66→0.62 over ep 7-10 → classic overfit; cosine LR didn't decay
fast enough at lr=3e-5. Best ckpt `checkpoints/v10_voco_l_cp/epoch_005.pt`.

**Pickle fix during CP launch:** the cached copy_paste_bank.pt stored
`OrganCrop` under module `__main__` (cache built via `python -m datasets.copy_paste_bank`).
When loaded inside `python -m training.train_v9`, unpickler couldn't resolve
`__main__.OrganCrop`. Fix: `from datasets.copy_paste_bank import OrganCrop` at
top of `training/train_v9.py` so the class is reachable as `train_v9.OrganCrop`
(=current `__main__.OrganCrop`).

**Interpretation:** mean Dice barely moves with copy-paste alone — large organs
dominate the average. The aug targets weak organs (gallbladder, pancreas,
adrenals, duodenum); per-organ eval after all 3 ablations is required to
verify the actual goal.

### W2 SL ablation complete (2026-04-25 ~16:24) — peak 0.7953 ep 2, FAIL on mean

| ep | train_loss | val proposer 3D Dice |
|----|-----------:|---------------------:|
|  1 | 2.0103 | 0.7917 |
|  2 | 2.0092 | **0.7953** ← peak |
|  3 | 2.0087 | 0.7886 |
|  4 | 2.0084 | 0.7886 |
|  5 | 2.0081 | 0.7952 |
|  6 | 2.0079 | 0.7935 |
|  7 | 2.0078 | 0.7916 |
|  8 | 2.0076 | 0.7931 |
|  9 | 2.0076 | 0.7926 |
| 10 | 2.0075 | 0.7920 |

Mean lift: **−0.0014 vs base 0.7967** (FAIL gate). Train loss flat at 2.01
because SmallOrganLoss = Dice (already saturated) + Tversky (FN-weighted) +
focal CE; the new components anchor the loss high. Tversky β=0.7 explicitly
recalls small-organ FNs at the cost of large-organ over-segmentation —
expected to hurt mean Dice. Best ckpt `checkpoints/v10_voco_l_sl/epoch_002.pt`.

**Per-organ analysis required.** Mean Dice is the wrong metric for SL by
design. After all 3 ablations finish, run per-organ comparison across base /
CP / SL / DS to see whether weak organs (gallbladder, pancreas, adrenals,
duodenum, stomach) lifted under SL or CP — the moonshot path stacks them.

### Logging upgrade landed mid-W2 (2026-04-25)

User asked for "full evaluation metrics saved when doing long epochs and TB":
- TB already had per-organ scalars under `val/proposer_dice/<organ>` ✓
- ADDED: per-epoch per-organ summary line to stdout/train.log
- ADDED: `output_dir/metrics/epoch_NNN.json` per epoch + rolling
  `output_dir/metrics.json` list — survives crashes, no TB events parsing
  required for offline analysis

Effective from DS launch onward (SL still on old logging).

### W2 DS ablation launched (2026-04-25 ~16:25)

Same recipe shape but `proposer.deep_supervision: true`,
`deep_sup_weight_start: 0.4`, `deep_sup_weight_end: 0.1` (linearly decayed
aux loss on decoder3/decoder4/decoder5).

### W2 DS ablation STOP-RULE abort (2026-04-25 ~17:25) — peak 0.7895 ep 2, regressing

Trajectory (patch eval val):

| ep | train_loss | val_dice | bladder | prostate_uterus |
|---:|-----------:|---------:|--------:|----------------:|
| 1 | 1.1341 | 0.7888 | — | 0.046 |
| 2 | 0.8719 | **0.7895** | 0.730 | 0.032 |
| 3 | 0.8006 | 0.7811 | 0.680 | 0.046 |
| 4 | 0.6777 | 0.7733 | 0.604 | 0.000 |
| 5 | 0.5494 | 0.7699 | 0.562 | 0.000 |

Stopped at ep 5/10. STOP-RULE threshold (0.7895 - 0.02 = 0.7695); ep 5 = 0.7699,
0.0004 above the line — but bladder is in active monotonic regression
(-0.04/epoch) and prostate has fully collapsed. Trend is unmistakable.

**Verdict: NO-GO at gate.** Best DS = ep 2 (0.7895), still 0.0072 below W1 base
(0.7967). Hypothesis: aux heads on decoder3/4/5 (1/8, 1/16, 1/32 spatial res)
cannot resolve sub-2cm pelvic structures (bladder, prostate) and the aux loss
drags the main head off them. Opposite of intended effect. Best ckpt at
`checkpoints/v10_voco_l_ds/epoch_002.pt` retained for sliding-window comparison.

### W2 patch-eval ablation summary (2026-04-25 17:25)

| axis | best ep | val_dice | Δ vs W1 | gate (≥0.02) | STOP-RULE? |
|------|--------:|---------:|--------:|:------------:|:----------:|
| W1 base | 9 (R3) | 0.7967 | — | — | — |
| CP (copy-paste) | 5 | 0.8001 | +0.0034 | FAIL | no |
| SL (small-organ loss) | 2 | 0.7953 | -0.0014 | FAIL | no |
| DS (deep supervision) | 2 | 0.7895 | -0.0072 | FAIL | yes (ep 5) |

**All three Phase C axes fail the patch-eval gate.** But every per-organ log
shows `prostate_uterus` channel scoring 0.03-0.05 in patch eval (small lateral
pelvic organ rarely captured by random 96³ val patches), tanking the mean.
Mean of 14 organs (excluding prostate) ≈ 0.84 across all four runs — already
within reach of Gate B. Patch eval is the bottleneck on the metric, not the
model. **Need sliding-window 3D eval to settle real gate decisions.**

### W2 sliding-window 3D eval launched (2026-04-25 17:29)

Sequential bash chain via `scripts/eval_v9_proposer_preproc.py` (sliding 96³,
stride 48, Gaussian blend, 4-way TTA, largest-CC postproc) on 30-vol AMOS22
CT val split, all four checkpoints:

- W1 base: `checkpoints/v10_voco_l_ft_r3/epoch_009.pt` + `configs/v10_voco_l_r3.yaml`
- CP best: `checkpoints/v10_voco_l_cp/epoch_005.pt` + `configs/v10_voco_l_cp.yaml`
- SL best: `checkpoints/v10_voco_l_sl/epoch_002.pt` + `configs/v10_voco_l_sl.yaml`
- DS best: `checkpoints/v10_voco_l_ds/epoch_002.pt` + `configs/v10_voco_l_ds.yaml`

Outputs to `reports/v10_w2/eval_<axis>.json`. Reference Run-3 patch val for
W1 base = 0.7967; expecting sliding-window mean 0.84+ as the prostate channel
will be captured properly across full volumes. Total wall-time estimate ~1.5 hr.

### Files added / modified this V10 session

- NEW: `configs/v10_voco_b.yaml`, `v10_voco_l.yaml`, `v10_voco_l_cp.yaml`, `v10_voco_l_sl.yaml`, `v10_voco_l_ds.yaml`, `v10_voco_l_full.yaml`, `v10_voco_h_lora.yaml`
- NEW: `checkpoints/pretrained/voco/VoComni_B.pt` (299 MB), `VoComni_L.pt` (1.17 GB), `VoComni_H.pt` (4.65 GB)
- NEW: `scripts/vram_preflight_voco_l.py`, `scripts/smoke_voco_l_loss.py`
- MOD: `training/train_v9.py` — `training.amp_dtype` config support, conditional `GradScaler`
- MOD: `models/swin_unetr_3d.py` — `_SwinLoRALinear`, `_inject_lora_into_swin`, new `lora_rank`/`lora_alpha` kwargs on `SwinUNETRProposer`
- MOD: `models/voluformer_v9.py` — propagate `lora_rank`/`lora_alpha` from config to proposer

### Python CUDA env note

Git-bash default `python` is miniconda with torch CPU-only. For ALL V10 training runs use `C:/Users/Raywa/AppData/Local/Programs/Python/Python312/python.exe`. Deps installed this session into that env: torch 2.8.0+cu126, monai 1.5.2, omegaconf 2.3.0, einops, tensorboard, timm, transformers, scipy, scikit-image, tqdm, nibabel, pyyaml, safetensors.

### Publishable novelties targeted

1. Copy-paste 3D small-organ aug on pretrained 3D CT base (MICCAI)
2. Text-prompted small-organ rescue on pretrained base (MICCAI/TMI)
3. TriSSR residual refiner — framed as 0.18M-param residual head, not a backbone (workshop)
4. Patch-center-clamp + cascade alignment diagnostics (methodology note)
5. Uncertainty-guided organ-centric re-seg trigger (main paper)
6. LoRA FT of 1B-scale pretrained SwinUNETR on abdominal CT (memory-efficient adaptation study)

---

## 2026-04-25 — OrganMoE-3D Phase I kickoff

Spec: `docs/superpowers/specs/2026-04-25-organmoe-3d-design.md`
Plan: `docs/superpowers/plans/2026-04-25-organmoe-3d-phase-i.md`

V10 Phase C (CP/SL/DS) all FAILED the 0.02 lift gate (best CP +0.0034, SL -0.0014, DS NO-GO). Root cause analysis: backbone capacity + class imbalance, not augmentation. Pivoting to a new model class `OrganMoE-3D` = Class-Presence-Aware Sparse LoRA Mixture-of-Experts (K=8 LoRA-rank-16 experts + class-presence side head + presence-conditioned router). Targets the documented prostate-washout failure architecturally.

**30-day plan**: Floor 0.85 / Target 0.87 / Stretch 0.90 mean Dice. Phase I (Days 1-7): TotalSeg pretrain + balanced sampler + OrganMoE-3D module + AMOS22 FT → **GATE I** (sliding-window mean >=0.85 AND prostate >=0.30). Hard exit if FAIL.

New ckpt dirs (Phase I writes here, sacred ckpts unchanged):
- `checkpoints/voco_totalseg_pretrain/`
- `checkpoints/organmoe_phase_i/`
- `reports/organmoe_phase_i/`
- `logs/organmoe_phase_i/`

Execution mode: subagent-driven development. Implementer + spec reviewer + code-quality reviewer per task.

---

*End of MASTER_PROJECT_LOG.md — update via the protocol in "Document Purpose & Update Protocol".*
