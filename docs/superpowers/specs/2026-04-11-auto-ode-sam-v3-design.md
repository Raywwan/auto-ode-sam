# Auto-ODE-SAM V3 Design Spec
*Date: 2026-04-11 | Project: VoluFormer3D / LiteSAM-3D V3 Masters Thesis*

---

## Goal

Transform ODE-SAM from a box-prompted single-organ model into a fully automatic 15-organ CT segmentation model that is competitive with the AMOS22 leaderboard top (0.918 mean DSC) while remaining lightweight (~25–30M params) and novel.

**Target**: 0.90–0.92 mean DSC across all 15 AMOS22 organs at 256px resolution.

---

## Context

### Current State (ODE-SAM + Phase 2)
- Architecture: TinyViT-21M Stage 2 → Bidirectional Neural ODE → SAM MaskDecoder
- Best result: **0.9183 DSC, 0.76mm HD95** on liver (organ 6), 256px, 21 epochs
- Weakness: box-prompted (requires user input), organ-agnostic ODE, no zoom for small organs
- Confirmed novel: Bidirectional Neural ODE cross-slice — zero prior art

### Why Small Organs Fail
| Organ | Size at 256px | Root Cause |
|-------|--------------|-----------|
| Adrenal glands | ~10px | Below reliable detection threshold at 256px |
| Esophagus | ~3–5px wide | Thin tubular structure, decoder loses topology |
| Duodenum | Variable | Complex shape, decoder can't recover from 16×16 bottleneck |
| Gallbladder | Variable | May be absent (collapsed/removed) |

### What Doesn't Change
- TinyViT-21M Stage 2 encoder (keeps pretrained weights)
- Bidirectional Neural ODE cross-slice module (core novel contribution)
- Stage 1 hierarchical skip connections (already implemented)
- PFESA spectral enhancement (already implemented)
- All Phase 2 training fixes (all-slice supervision, CutMix, DiceTopK, clDice)

---

## Architecture: Five Stacked Upgrades

### Upgrade 1: Organ Query Decoder (replaces box prompt)

**What**: Replace `PromptEncoder` with 15 learned organ query tokens — one per AMOS22 organ. Each token is a learnable embedding of dimension `embed_dim=256`. They attend to ODE-enriched image features through the existing TwoWayTransformer and each decode one binary mask.

**How it works**:
```
organ_queries: nn.Embedding(15, embed_dim)          # 15 × 256 = 3,840 params
     ↓
TwoWayTransformer(queries=organ_queries, keys=image_features)
     ↓
15 parallel mask predictions (one per organ)
```

**Why it's automatic**: No box required. The model processes a CT volume and outputs all 15 organ masks in a single forward pass.

**Training note**: Each training sample specifies which organs are present via a `target_organs` mask. Loss is computed only for present organs. Absent organs contribute zero loss.

**Novel framing**: "Organ Query Transformer with Neural ODE cross-slice dynamics" — combining learned organ queries with continuous trajectory modeling is unpublished.

**Files**: New `models/organ_query_decoder.py`, modify `models/ode_sam.py`

---

### Upgrade 2: Organ-Conditioned ODE

**What**: Add a small organ bias term to the ODE function so each organ gets a different integration trajectory.

**Mechanism**:
```python
# Current: dh/dt = f_θ(h, t)
# New:     dh/dt = f_θ(h, t) + organ_bias(organ_id)
# where:   organ_bias = nn.Linear(organ_embed_dim=32, ode_hidden=64)(organ_embed[organ_id])
```

**Why it matters**: Liver spans 100+ CT slices — its ODE trajectory should be wide and slow. Adrenal glands span ~10 slices — tight and sharp. Currently the same ODE function handles both, which is a mismatch. With organ conditioning, the ODE learns organ-specific dynamics from data.

**Params added**: `nn.Embedding(15, 32)` + `nn.Linear(32, ode_hidden=64)` = 480 + 2,048 = ~2.5K params.

**Files**: Modify `models/ode_cross_slice.py`, `models/ode_sam.py`

---

### Upgrade 3: HQ-SAM Output Token

**What**: Add one learnable HQ (High-Quality) output token to the mask decoder. It fuses early encoder features (Stage 1, 32×32) with final decoder features to recover fine boundary detail lost in the 16×16 transformer bottleneck.

**Mechanism** (from HQ-SAM, NeurIPS 2023, arXiv:2306.01567):
```
HQ_token: nn.Embedding(1, embed_dim)   # 1 × 256 = 256 params
     ↓
TwoWayTransformer (same as existing, HQ token appended to queries)
     ↓
HQ_MLP: 3-layer MLP → embed_dim//8  (same as existing mask token MLPs)
     ↓
HQ_mask_logits = HQ_MLP(hq_token_out) @ upscaled_features
final_mask = base_mask + HQ_mask_logits     # element-wise sum
```

Additionally, the HQ token receives a direct skip from Stage 1 encoder features (already extracted by our MultiScaleEncoder), fused before the dot-product step via element-wise addition after projection.

**Expected gain**: +1–3% DSC on fine structures, especially adrenal glands and esophagus boundaries.

**Params added**: ~2K (HQ token + its MLP + feature fusion projection).

**Files**: Modify `models/mask_decoder.py`

---

### Upgrade 4: Two-Stage Zoom-In for Small Organs

**What**: After the first full-volume pass at 256px produces coarse masks, extract ROI crops around predicted small organ locations, resize to full 256px input (4× effective resolution), and run a second refined pass.

**Small organs that get zoom-in treatment**: adrenal glands (L/R), esophagus, duodenum, gallbladder.

**Mechanism** (inference):
```
Pass 1: Full volume @ 256px → 15 coarse masks + bounding boxes
  For each small organ o in [adrenal_L, adrenal_R, esophagus, duodenum, gallbladder]:
    if coarse_mask[o].sum() > 0:  # organ detected
      bbox = extract_bbox(coarse_mask[o], padding=32px)
      crop = volume[:, :, bbox]           # spatial crop all D slices
      crop_resized = F.interpolate(crop, (256, 256))  # 4× effective resolution
      mask_refined[o] = Pass2(crop_resized)           # second inference pass
      mask_final[o] = paste_back(mask_refined[o], bbox, original_shape)
```

**At training time**: For small organ training samples, randomly apply zoom-in crops with probability 0.5, treating them as high-resolution training examples. This teaches the model to segment at both scales.

**Expected gain**: +3–8% DSC on small organs (PRNet 2025, RAPS-3D). Adrenal glands go from ~10px to ~40px effective width.

**Files**: New `inference/zoom_refine.py`, modify `training/trainer.py`

---

### Upgrade 5: Training Improvements (nnU-Net Tricks)

**5a. Foreground Oversampling**
Guarantee that 33% of training slices per batch contain at least one small organ foreground voxel. Implement via weighted slice sampling in `datasets/amos22.py` — maintain a precomputed organ-presence index and oversample slices containing adrenal, esophagus, duodenum.

**5b. Boundary Loss with Schedule**
Add distance-transform-based boundary loss (Kervadec et al. 2019). Ramp in linearly from weight 0.0 to 0.3 after epoch 50 of Phase 3 training. Requires precomputing distance transform maps of ground truth masks (one-time, stored in npy_cache).

**5c. Deep Supervision at ODE Steps**
Add auxiliary segmentation heads at intermediate ODE integration timesteps (t=0.5). Auxiliary loss weight = 0.5 × main loss. Forces the ODE to produce meaningful intermediate representations.

**5d. Batch-Level Dice**
Compute Dice loss across the batch dimension (not per-sample then averaged). Prevents large-organ batches from overwhelming the gradient for small-organ batches.

**Files**: Modify `datasets/amos22.py`, `training/losses.py`, `training/trainer.py`

---

## Complete Architecture Summary

```
CT Volume (B, D=8, 1, H, W)
    ↓
MultiScaleEncoder [TinyViT-21M Stage 2 + Stage 1 skip]
    ↓ (B*D, 256, 16, 16) main features
    ↓ (B*D, 64, 32, 32) skip features
    ↓
PFESA Spectral Enhancement [zero params]
    ↓
Organ-Conditioned BidirectionalNeuralODE
    ↓ organ_id → organ_embed → ODE bias
    ↓
Deep Supervision head at t=0.5 [auxiliary loss only]
    ↓
Organ Query Decoder
    ├── 15 organ query tokens (learned)
    ├── HQ-SAM output token (learned)
    ├── TwoWayTransformer (shared)
    ├── Stage 1 skip injection at 32×32
    ├── HQ feature fusion
    └── 15 binary mask outputs
    ↓
Pass 1 output: 15 coarse masks (256×256 each)
    ↓
[Inference only] Two-Stage Zoom-In
    └── For small organs: crop → 4× resolution → Pass 2 → paste back
    ↓
Final output: 15 organ masks
```

---

## Parameter Budget

| Component | Params | Notes |
|-----------|--------|-------|
| TinyViT-21M encoder | ~13.4M | Pretrained, trainable |
| Stage 1 skip projection | ~12K | New (already implemented) |
| Bidirectional Neural ODE | ~207K | Novel contribution |
| Organ conditioning (ODE bias) | ~2.5K | New |
| Organ query tokens (15) | ~3.8K | New — replaces PromptEncoder |
| HQ-SAM output token + MLP | ~2K | New |
| Mask decoder (15 parallel heads) | ~4.5M | Modified from SAM decoder |
| PFESA | 0 | Zero params |
| Deep supervision head | ~33K | New |
| **Total** | **~18.2M** | Within 24GB VRAM at B=4 |

The PromptEncoder (~205K params) is removed. Net change: −195K params.

---

## Training Plan

### Phase 3a: Single-organ validation (liver, 21 epochs, 256px)
- Confirm V3 architecture works before committing to 100-epoch multi-organ run
- Expected: ≥ 0.920 DSC on liver (vs 0.9183 baseline)
- Config: `configs/phase3a_autoodesam_liver.yaml`

### Phase 3b: Full 15-organ training (100 epochs, 256px)
- All 15 AMOS22 organs, foreground oversampling enabled
- Boundary loss ramps in at epoch 50
- Two-stage zoom-in applied to small organs during training
- Config: `configs/phase3b_autoodesam_full.yaml`
- Estimated time: ~3–4 days GPU

### Phase 3c (optional): 512px fine-tuning
- Fine-tune V3 from Phase 3b checkpoint at 512px, 20 epochs
- Primarily benefits small organs (adrenal, esophagus)
- Estimated time: ~1 day GPU

---

## Expected Results

| Organ | AMOS22 #1 | Auto-ODE-SAM V3 Target |
|-------|-----------|----------------------|
| Liver | 0.982 | 0.95–0.97 |
| Spleen | 0.977 | 0.94–0.96 |
| Kidneys (L/R) | 0.971–0.973 | 0.93–0.96 |
| Aorta | 0.961 | 0.90–0.93 |
| Stomach | 0.950 | 0.88–0.92 |
| Postcava | 0.929 | 0.88–0.92 |
| Bladder | 0.931 | 0.88–0.92 |
| Pancreas | 0.909 | 0.83–0.88 |
| Esophagus | 0.894 | 0.85–0.89 (clDice + zoom) |
| Gallbladder | 0.885 | 0.80–0.86 |
| Duodenum | 0.881 | 0.80–0.85 |
| Adrenal L | 0.836 | 0.82–0.86 (zoom-in) |
| Adrenal R | 0.809 | 0.80–0.84 (zoom-in) |
| **15-organ Mean** | **0.918** | **0.88–0.92** |

---

## Novelty Claims

1. **Primary**: First application of Bidirectional Neural ODE to cross-slice feature dynamics in SAM-based CT segmentation — confirmed no prior art.
2. **Secondary**: Organ-conditioned ODE — each organ gets a learned trajectory bias. Novel combination.
3. **Supporting**: Organ Query Transformer with Neural ODE backbone — automatic multi-organ without prompts.
4. **Supporting**: HQ-SAM token applied to lightweight TinyViT pipeline with 3D context.

---

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| Organ query tokens fail to converge (all tokens learn same thing) | Medium | Add cosine diversity loss between query token embeddings |
| Two-stage zoom-in misses adrenal in Pass 1 | Low | Pad ROI by 32px; if no detection, use anatomical prior location |
| ODE conditioning collapses (all organs get same bias) | Low | Monitor per-organ bias norms during training |
| VRAM overflow with 15 parallel decoder heads | Low | Heads share all transformer weights; only output MLPs are per-organ |

---

*Spec approved by user: 2026-04-11*
*Next step: Implementation plan via writing-plans skill*
