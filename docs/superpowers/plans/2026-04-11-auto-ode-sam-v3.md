# Auto-ODE-SAM V3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform ODE-SAM V2 from a box-prompted single-organ model into a fully automatic 15-organ CT segmentation model targeting 0.90–0.92 mean DSC on AMOS22 at 256px.

**Architecture:** Replace the PromptEncoder with 15 learned organ query tokens (DETR-style); add organ-conditioned ODE bias so each organ gets a distinct integration trajectory; add HQ-SAM output token for fine-structure boundary recovery; add two-stage zoom-in at inference for small organs; add foreground oversampling + deep supervision at training time.

**Tech Stack:** PyTorch, OmegaConf, TinyViT-21M, existing `models/`, `training/`, `datasets/`, `inference/` directories in `C:\Users\Raywa\Desktop\VoluFormer3D\`. venv at `C:\Users\Raywa\Desktop\LiteSAM3D\.venv`.

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `models/organ_query_decoder.py` | **Create** | 15 learned organ tokens + HQ token + multi-organ forward pass |
| `models/ode_cross_slice.py` | **Modify** | Add organ-conditioned ODE bias (`dh/dt = f(h,t) + MLP(organ_embed[id])`) |
| `models/ode_sam.py` | **Modify** | Wire OrganQueryDecoder; remove PromptEncoder; pass organ_id to ODE |
| `models/mask_decoder.py` | **Modify** | Add HQ output token + Stage 1 feature fusion |
| `models/__init__.py` | **Modify** | Register `auto_ode_sam` architecture |
| `datasets/amos22.py` | **Modify** | Add `organ_id` to sample dict; add foreground oversampling index |
| `training/losses.py` | **Modify** | Add boundary loss (DT-based); add deep supervision aux loss |
| `training/trainer.py` | **Modify** | Support multi-organ batches; foreground oversampling; organ_id routing |
| `inference/zoom_refine.py` | **Create** | Two-stage zoom-in for small organs at inference time |
| `configs/phase3a_autoodesam_liver.yaml` | **Create** | Single-organ liver validation (21 epochs) |
| `configs/phase3b_autoodesam_full.yaml` | **Create** | Full 15-organ training (100 epochs) |

---

## Task 1: Organ-Conditioned ODE

Add a small organ bias term to each ODEFunction so each organ gets a distinct ODE trajectory. `dh/dt = f_θ(h, t) + MLP(organ_embed[organ_id])`. The merge layer maps `(N, dim)` bias to match the hidden ODE output.

**Files:**
- Modify: `models/ode_cross_slice.py`

- [ ] **Step 1: Add `OrganConditionedODEFunction` class after `ODEFunction`**

Open `models/ode_cross_slice.py`. After the `ODEFunction` class (line ~108), add:

```python
class OrganConditionedODEFunction(nn.Module):
    """
    ODE dynamics with per-organ bias: dh/dt = f_θ(h, t) + MLP(organ_embed[organ_id]).

    The base dynamics f_θ(h, t) are shared across all organs.
    The organ bias term shifts the trajectory so each organ's ODE
    follows a distinct path — liver (wide, slow) vs adrenal (tight, sharp).

    Args:
        dim        (int): Feature channel dimension.
        n_organs   (int): Number of organ classes (15 for AMOS22).
        organ_emb_dim (int): Organ embedding dimension. Default: 32.
        hidden     (int): Hidden width in base dynamics. Default: dim // 4.
        n_freqs    (int): Fourier frequency bands for time. Default: 4.
    """

    def __init__(
        self,
        dim:          int,
        n_organs:     int = 15,
        organ_emb_dim: int = 32,
        hidden:       int = None,
        n_freqs:      int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.n_freqs = n_freqs

        # Base dynamics: shared ODE function
        self.base_ode = ODEFunction(dim=dim, hidden=hidden, n_freqs=n_freqs)

        # Organ embedding: maps organ index → continuous embedding vector
        # organ_id is 1-indexed (1..15); we use it as-is with embedding size n_organs+1
        self.organ_embed = nn.Embedding(n_organs + 1, organ_emb_dim)

        # Bias MLP: organ embedding → dim-dimensional bias on dh/dt
        self.bias_mlp = nn.Sequential(
            nn.Linear(organ_emb_dim, dim),
        )
        # Zero-init: bias = 0 at start → model degrades to base ODE (safe initialisation)
        nn.init.zeros_(self.bias_mlp[0].weight)
        nn.init.zeros_(self.bias_mlp[0].bias)

    def forward(
        self,
        h:         torch.Tensor,  # (N, dim) — current feature state
        t:         float,          # normalised depth in [0, 1]
        organ_id:  torch.Tensor,  # (B,) long tensor — organ index per batch item
                                   # N = B * H * W, so we need to expand
    ) -> torch.Tensor:
        """
        Returns dh_dt: (N, dim).

        organ_id is (B,) but h is (B*H*W, dim). We expand organ_id to (B*H*W,).
        """
        N = h.shape[0]
        B = organ_id.shape[0]
        HW = N // B  # spatial positions per sample

        # Expand organ_id: (B,) → (B*H*W,)
        organ_id_expanded = organ_id.repeat_interleave(HW)  # (N,)
        organ_emb = self.organ_embed(organ_id_expanded)      # (N, organ_emb_dim)
        bias = self.bias_mlp(organ_emb)                      # (N, dim)

        return self.base_ode(h, t) + bias
```

- [ ] **Step 2: Add `OrganConditionedBidirectionalNeuralODE` class after `BidirectionalNeuralODECrossSlice`**

After the `BidirectionalNeuralODECrossSlice` class (end of file), add:

```python
class OrganConditionedBidirectionalNeuralODE(nn.Module):
    """
    Bidirectional Neural ODE with per-organ trajectory conditioning.

    Wraps OrganConditionedODEFunction in the same bidirectional Euler
    integration framework as BidirectionalNeuralODECrossSlice, but
    routes organ_id through both forward and backward ODE functions.

    Interface: forward(features, organ_id) — organ_id is (B,) long tensor.
    """

    def __init__(
        self,
        dim:          int,
        n_organs:     int = 15,
        organ_emb_dim: int = 32,
        ode_hidden:   int = None,
        n_freqs:      int = 4,
        substeps:     int = 2,
    ):
        super().__init__()
        self.substeps = substeps

        self.ode_fwd = OrganConditionedODEFunction(
            dim=dim, n_organs=n_organs, organ_emb_dim=organ_emb_dim,
            hidden=ode_hidden, n_freqs=n_freqs,
        )
        self.ode_bwd = OrganConditionedODEFunction(
            dim=dim, n_organs=n_organs, organ_emb_dim=organ_emb_dim,
            hidden=ode_hidden, n_freqs=n_freqs,
        )
        self.merge = nn.Linear(dim * 2, dim, bias=False)
        self.norm  = nn.LayerNorm(dim)

    def _euler_trajectory(
        self,
        ode_func,
        h0:       torch.Tensor,   # (N, C)
        D:        int,
        organ_id: torch.Tensor,   # (B,)
    ) -> torch.Tensor:
        t_points = [d / max(D - 1, 1) for d in range(D)]
        h = h0
        states = [h]
        for i in range(1, D):
            t_start = t_points[i - 1]
            t_end   = t_points[i]
            sub_dt  = (t_end - t_start) / self.substeps
            for j in range(self.substeps):
                t_curr = t_start + j * sub_dt
                h = h + sub_dt * ode_func(h, t_curr, organ_id)
            states.append(h)
        return torch.stack(states, dim=1)   # (N, D, C)

    def forward(
        self,
        features:  torch.Tensor,  # (B, D, C, H, W)
        organ_id:  torch.Tensor,  # (B,) long — organ index
    ) -> torch.Tensor:
        B, D, C, H, W = features.shape
        x = features.permute(0, 3, 4, 1, 2).reshape(B * H * W, D, C)

        traj_fwd = self._euler_trajectory(self.ode_fwd, x[:, 0],   D, organ_id)
        traj_bwd = self._euler_trajectory(self.ode_bwd, x[:, -1],  D, organ_id)
        traj_bwd = torch.flip(traj_bwd, dims=[1])

        ode_context = self.merge(torch.cat([traj_fwd, traj_bwd], dim=-1))
        ode_context = ode_context.reshape(B, H, W, D, C).permute(0, 3, 4, 1, 2)

        out = features + ode_context
        out = out.permute(0, 1, 3, 4, 2)
        out = self.norm(out)
        return out.permute(0, 1, 4, 2, 3)
```

- [ ] **Step 3: Verify import structure (no external deps needed)**

Run:
```bash
C:\Users\Raywa\Desktop\LiteSAM3D\.venv\Scripts\python -c "from models.ode_cross_slice import OrganConditionedBidirectionalNeuralODE; import torch; m = OrganConditionedBidirectionalNeuralODE(dim=256); f = torch.randn(2,8,256,16,16); ids = torch.tensor([6,6]); print(m(f,ids).shape)"
```
Expected: `torch.Size([2, 8, 256, 16, 16])`

- [ ] **Step 4: Commit**
```bash
cd C:\Users\Raywa\Desktop\VoluFormer3D && git add models/ode_cross_slice.py && git commit -m "feat: add OrganConditionedBidirectionalNeuralODE to ode_cross_slice"
```

---

## Task 2: HQ-SAM Output Token in MaskDecoder

Add one learnable HQ (High-Quality) output token to `MaskDecoder`. It receives a direct skip from Stage 1 encoder features, fuses them with final decoder features, and adds a fine-boundary correction map on top of the base mask. This recovers detail lost in the 16×16 transformer bottleneck.

**Files:**
- Modify: `models/mask_decoder.py`

- [ ] **Step 1: Add HQ token + projection in `MaskDecoder.__init__`**

In `MaskDecoder.__init__`, after the line `self.modality_head = MLP(...)`, add:

```python
        # ----- HQ-SAM output token (NeurIPS 2023, arXiv:2306.01567) -----
        # One extra learnable token appended to the query list. After the
        # two-way transformer, this token fuses Stage 1 encoder features
        # (32×32) with the decoder's upscaled features to recover fine
        # boundary detail lost in the 16×16 bottleneck.
        self.hq_token = nn.Embedding(1, embed_dim)

        # MLP: maps hq_token_out → embed_dim//8 (same dim as mask hypernetworks)
        self.hq_mlp = MLP(embed_dim, embed_dim, embed_dim // 8, depth=3)

        # Projection: fuse Stage 1 features (skip_channels=64) → embed_dim//8
        # for the dot-product with upscaled features
        hq_fuse_in = (skip_channels if skip_channels > 0 else embed_dim // 4)
        self.hq_skip_proj = nn.Sequential(
            nn.Conv2d(hq_fuse_in, embed_dim // 8, kernel_size=1),
            nn.GELU(),
        )
        # Zero-init: HQ path starts neutral
        nn.init.zeros_(self.hq_skip_proj[0].weight)
        nn.init.zeros_(self.hq_skip_proj[0].bias)
```

- [ ] **Step 2: Prepend HQ token to query tokens in `MaskDecoder.forward`**

In `MaskDecoder.forward`, find this line:
```python
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0
        )
```

Replace it with:
```python
        # Append HQ token to the query token list (after iou + mask tokens)
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight, self.hq_token.weight], dim=0
        )  # (1 + num_mask_tokens + 1, embed_dim)
```

Then find `iou_token_out = hs[:, 0, :]` and update the extraction block:

```python
        iou_token_out = hs[:, 0, :]                        # (B, embed_dim)
        mask_tokens_out = hs[:, 1:(1 + self.num_multimask_outputs + 1), :]
        hq_token_out = hs[:, (1 + self.num_multimask_outputs + 1), :]  # (B, embed_dim)
```

- [ ] **Step 3: Compute HQ correction map and add to masks**

After the line `masks = (hyper_in @ upscaled.view(b, c, h * w)).view(b, -1, h, w)`, add:

```python
        # ----- HQ correction mask -----
        # hq_token_out → weights of shape (B, embed_dim//8)
        hq_weights = self.hq_mlp(hq_token_out)  # (B, embed_dim//8)

        if skip_features is not None:
            # Fuse Stage 1 skip features (already edge-enhanced) into HQ path
            skip_for_hq = skip_features
            if skip_for_hq.shape[-2:] != upscaled.shape[-2:]:
                skip_for_hq = F.interpolate(
                    skip_for_hq.float(), size=upscaled.shape[-2:],
                    mode='bilinear', align_corners=False,
                ).to(upscaled.dtype)
            hq_feat = self.hq_skip_proj(skip_for_hq)  # (B, embed_dim//8, H, W)
        else:
            # Fallback: use upscaled features directly (no skip available)
            hq_feat = upscaled  # (B, embed_dim//8, H, W) already

        # Dot-product: hq_weights (B, C) × hq_feat (B, C, H, W) → (B, 1, H, W)
        hq_mask = (hq_weights.unsqueeze(1) @ hq_feat.view(b, c, h * w)).view(b, 1, h, w)

        # Add HQ correction to all mask candidates
        masks = masks + hq_mask
```

- [ ] **Step 4: Verify shapes**

```bash
C:\Users\Raywa\Desktop\LiteSAM3D\.venv\Scripts\python -c "
import torch
from models.mask_decoder import MaskDecoder
m = MaskDecoder(embed_dim=256, num_multimask_outputs=3, skip_channels=64)
imgs = torch.randn(2,256,16,16)
sp = torch.randn(2,5,256)
dp = torch.randn(2,256,16,16)
skip = torch.randn(2,64,32,32)
masks, iou, mod = m(imgs, sp, dp, skip_features=skip)
print('masks:', masks.shape, 'iou:', iou.shape)
"
```
Expected: `masks: torch.Size([2, 3, 64, 64])  iou: torch.Size([2, 3])`

- [ ] **Step 5: Commit**
```bash
cd C:\Users\Raywa\Desktop\VoluFormer3D && git add models/mask_decoder.py && git commit -m "feat: add HQ-SAM output token to MaskDecoder with Stage 1 skip fusion"
```

---

## Task 3: Organ Query Decoder

Create `models/organ_query_decoder.py`. This replaces the `PromptEncoder` with 15 learned organ query tokens and outputs 15 binary masks simultaneously. Each token represents one AMOS22 organ. The existing `TwoWayTransformer` + `MaskDecoder` are reused.

**Files:**
- Create: `models/organ_query_decoder.py`

- [ ] **Step 1: Create `models/organ_query_decoder.py`**

```python
# =============================================================================
# models/organ_query_decoder.py — Organ Query Decoder
#
# Replaces box-prompted PromptEncoder with 15 learned organ query tokens.
# Each token is a learnable embedding that attends to ODE-enriched image
# features through the TwoWayTransformer and decodes one binary mask.
#
# This makes the model fully automatic: no box prompt required at inference.
# All 15 AMOS22 organ masks are predicted in a single forward pass.
#
# Novel framing: "Organ Query Transformer with Neural ODE cross-slice dynamics"
# — combining DETR-style learned queries with continuous ODE trajectory modeling.
# No published paper has combined these two components as of April 2026.
#
# AMOS22 organ IDs:
#   1=spleen, 2=right_kidney, 3=left_kidney, 4=gallbladder, 5=esophagus,
#   6=liver, 7=stomach, 8=aorta, 9=inferior_vena_cava, 10=pancreas,
#   11=right_adrenal_gland, 12=left_adrenal_gland, 13=duodenum,
#   14=bladder, 15=prostate_uterus
# =============================================================================

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mask_decoder import TwoWayTransformer, MLP

# Number of AMOS22 organs — fixed constant
N_ORGANS = 15

# Small organs that benefit from two-stage zoom-in at inference
SMALL_ORGAN_IDS = [4, 5, 11, 12, 13]  # gallbladder, esophagus, adrenal L/R, duodenum


class OrganQueryDecoder(nn.Module):
    """
    Automatic multi-organ decoder using learned organ query tokens.

    Architecture:
      organ_queries (15 × embed_dim)         ← one per AMOS22 organ
      HQ token      (1  × embed_dim)         ← fine-boundary correction
        ↓
      TwoWayTransformer(queries, image_features)
        ↓
      15 parallel MLP heads → 15 mask logits (one per organ)
      HQ head fuses Stage 1 skip → adds correction to all masks

    Each organ query learns WHAT that organ looks like from data alone.
    At inference, all 15 masks are produced simultaneously without any
    user-provided box.

    Args:
        embed_dim             (int): Feature embedding dimension (256).
        n_organs              (int): Number of organs (15 for AMOS22).
        transformer_depth     (int): TwoWayTransformer blocks.
        transformer_mlp_dim   (int): FFN hidden dim in transformer.
        iou_head_depth        (int): Depth of per-organ IoU head MLP.
        iou_head_hidden_dim   (int): Width of per-organ IoU head MLP.
        skip_channels         (int): Stage 1 skip channels (64). 0 = disable.
        diversity_loss_weight (float): Weight for cosine diversity loss between organ tokens.
                                       Prevents all tokens from learning the same thing.
    """

    def __init__(
        self,
        embed_dim:             int   = 256,
        n_organs:              int   = N_ORGANS,
        transformer_depth:     int   = 2,
        transformer_mlp_dim:   int   = 2048,
        iou_head_depth:        int   = 3,
        iou_head_hidden_dim:   int   = 256,
        skip_channels:         int   = 64,
        diversity_loss_weight: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_organs  = n_organs
        self.diversity_loss_weight = diversity_loss_weight

        # ----- 15 organ query tokens -----
        # organ_id is 1-indexed (1..15), mapped to 0-indexed embeddings internally
        self.organ_queries = nn.Embedding(n_organs, embed_dim)
        nn.init.normal_(self.organ_queries.weight, std=0.02)

        # ----- HQ output token (fine-boundary recovery) -----
        self.hq_token = nn.Embedding(1, embed_dim)

        # ----- Shared two-way transformer -----
        # All 15 organ queries share the same transformer weights.
        # This is critical for parameter efficiency — only the query embeddings differ.
        self.transformer = TwoWayTransformer(
            depth=transformer_depth,
            embedding_dim=embed_dim,
            num_heads=8,
            mlp_dim=transformer_mlp_dim,
        )

        # ----- Per-organ mask prediction heads -----
        # Each head maps its query token output → per-pixel weights (embed_dim//8)
        # then dot-product with upscaled features → mask logit
        self.mask_mlps = nn.ModuleList([
            MLP(embed_dim, embed_dim, embed_dim // 8, depth=3)
            for _ in range(n_organs)
        ])

        # ----- Per-organ IoU prediction heads -----
        self.iou_heads = nn.ModuleList([
            MLP(embed_dim, iou_head_hidden_dim, 1, depth=iou_head_depth)
            for _ in range(n_organs)
        ])

        # ----- Upsampling (shared, same as MaskDecoder) -----
        self.upsample_conv1 = nn.ConvTranspose2d(embed_dim, embed_dim // 4, kernel_size=2, stride=2)
        self.upsample_ln    = nn.LayerNorm(embed_dim // 4)
        self.upsample_conv2 = nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=2, stride=2)

        # ----- Stage 1 skip injection (same as MaskDecoder.use_skip) -----
        self.use_skip = skip_channels > 0
        if self.use_skip:
            self.skip_gate = nn.Parameter(torch.zeros(1))
            self.skip_ln   = nn.LayerNorm(skip_channels)

        # ----- HQ path -----
        hq_fuse_in = skip_channels if skip_channels > 0 else embed_dim // 4
        self.hq_mlp = MLP(embed_dim, embed_dim, embed_dim // 8, depth=3)
        self.hq_skip_proj = nn.Sequential(
            nn.Conv2d(hq_fuse_in, embed_dim // 8, kernel_size=1),
            nn.GELU(),
        )
        nn.init.zeros_(self.hq_skip_proj[0].weight)
        nn.init.zeros_(self.hq_skip_proj[0].bias)

    # ------------------------------------------------------------------
    @staticmethod
    def _haar_edge_enhance(x: torch.Tensor) -> torch.Tensor:
        """Zero-parameter Haar wavelet edge enhancement (same as MaskDecoder)."""
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]
        LH = (x00 - x01 + x10 - x11) * 0.25
        HL = (x00 + x01 - x10 - x11) * 0.25
        HH = (x00 - x01 - x10 + x11) * 0.25
        hf = LH.abs() + HL.abs() + HH.abs()
        hf_up = F.interpolate(hf, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return x + 0.5 * hf_up

    # ------------------------------------------------------------------
    def compute_diversity_loss(self) -> torch.Tensor:
        """
        Cosine diversity loss between organ query embeddings.

        Encourages organ tokens to be dissimilar (diverse), preventing
        all 15 tokens from collapsing to the same embedding.

        Loss = mean(cosine_similarity(token_i, token_j)) for i != j
        Minimising this pushes tokens to be orthogonal.
        """
        W = self.organ_queries.weight  # (15, embed_dim)
        W_norm = F.normalize(W, dim=-1)
        sim_matrix = W_norm @ W_norm.T  # (15, 15)
        # Exclude diagonal (self-similarity = 1)
        mask = 1.0 - torch.eye(self.n_organs, device=W.device)
        diversity_loss = (sim_matrix * mask).sum() / (self.n_organs * (self.n_organs - 1))
        return self.diversity_loss_weight * diversity_loss

    # ------------------------------------------------------------------
    def forward(
        self,
        image_embeddings:      torch.Tensor,         # (B, embed_dim, H_feat, W_feat)
        dense_pe:              torch.Tensor,          # (B, embed_dim, H_feat, W_feat) — positional enc
        skip_features:         Optional[torch.Tensor] = None,  # (B, skip_ch, 2*H, 2*W)
        target_organ_ids:      Optional[List[int]] = None,     # subset of [1..15]; None = all
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Produce masks and IoU scores for all (or selected) organs.

        Args:
            image_embeddings:  (B, embed_dim, H_feat, W_feat)
            dense_pe:          positional encoding for image tokens
            skip_features:     Stage 1 encoder features for skip injection
            target_organ_ids:  organ IDs to predict (1-indexed). None = all 15.

        Returns:
            masks:    (B, K, H_out, W_out) — mask logits, K = len(target_organ_ids)
            iou_pred: (B, K) — IoU predictions
            organ_ids_out: list[int] — which organs correspond to each mask channel
        """
        B, C, H_feat, W_feat = image_embeddings.shape

        # Map requested organ IDs to 0-indexed query positions
        if target_organ_ids is None:
            target_organ_ids = list(range(1, self.n_organs + 1))
        query_indices = [oid - 1 for oid in target_organ_ids]  # 0-indexed
        K = len(query_indices)

        # Gather organ query tokens for requested organs: (K, embed_dim)
        organ_q = self.organ_queries.weight[query_indices]  # (K, embed_dim)
        # Add HQ token: (K+1, embed_dim)
        all_queries = torch.cat([organ_q, self.hq_token.weight], dim=0)
        # Expand to batch: (B, K+1, embed_dim)
        queries = all_queries.unsqueeze(0).expand(B, -1, -1)
        query_pe = torch.zeros_like(queries)  # no positional encoding for organ tokens

        # ----- Upscale and apply skip -----
        # (done before transformer so skip-fused features can inform queries)
        upscaled = self.upsample_conv1(image_embeddings)   # (B, C//4, H*2, W*2)
        B_, C_, H_, W_ = upscaled.shape
        upscaled = upscaled.permute(0, 2, 3, 1).reshape(-1, C_)
        upscaled = self.upsample_ln(upscaled)
        upscaled = upscaled.reshape(B_, H_, W_, C_).permute(0, 3, 1, 2)
        upscaled = F.gelu(upscaled)

        if self.use_skip and skip_features is not None:
            sf = skip_features
            if sf.shape[-2:] != upscaled.shape[-2:]:
                sf = F.interpolate(sf.float(), size=upscaled.shape[-2:],
                                   mode='bilinear', align_corners=False).to(upscaled.dtype)
            sf_flat = sf.permute(0, 2, 3, 1).reshape(-1, sf.shape[1])
            sf_flat = self.skip_ln(sf_flat)
            sf = sf_flat.reshape(sf.shape[0], *sf.shape[2:], sf.shape[1]).permute(0, 3, 1, 2)
            sf = self._haar_edge_enhance(sf)
            upscaled = upscaled + self.skip_gate * sf

        upscaled = self.upsample_conv2(upscaled)   # (B, C//8, H*4, W*4)
        upscaled = F.gelu(upscaled)
        b, c, h, w = upscaled.shape

        # ----- Two-way transformer -----
        src_flat  = image_embeddings.flatten(2).permute(0, 2, 1)   # (B, H*W, C)
        pos_flat  = dense_pe.flatten(2).permute(0, 2, 1)            # (B, H*W, C)
        hs, _     = self.transformer(src_flat, pos_flat, queries, query_pe)
        # hs: (B, K+1, embed_dim)

        organ_token_outs = hs[:, :K, :]    # (B, K, embed_dim)
        hq_token_out     = hs[:, K, :]     # (B, embed_dim)

        # ----- Predict masks -----
        masks    = []
        iou_pred = []
        for k_idx, organ_idx in enumerate(query_indices):
            weights = self.mask_mlps[organ_idx](organ_token_outs[:, k_idx, :])  # (B, C//8)
            mask_k  = (weights.unsqueeze(1) @ upscaled.view(b, c, h * w)).view(b, 1, h, w)
            iou_k   = self.iou_heads[organ_idx](organ_token_outs[:, k_idx, :])  # (B, 1)
            masks.append(mask_k)
            iou_pred.append(iou_k)

        masks    = torch.cat(masks, dim=1)     # (B, K, H_out, W_out)
        iou_pred = torch.cat(iou_pred, dim=1)  # (B, K)

        # ----- HQ correction (adds to all mask channels) -----
        hq_weights = self.hq_mlp(hq_token_out)   # (B, C//8)
        if skip_features is not None:
            sf_hq = skip_features
            if sf_hq.shape[-2:] != upscaled.shape[-2:]:
                sf_hq = F.interpolate(sf_hq.float(), size=upscaled.shape[-2:],
                                      mode='bilinear', align_corners=False).to(upscaled.dtype)
            hq_feat = self.hq_skip_proj(sf_hq)   # (B, C//8, H, W)
        else:
            hq_feat = upscaled
        hq_mask = (hq_weights.unsqueeze(1) @ hq_feat.view(b, c, h * w)).view(b, 1, h, w)
        masks = masks + hq_mask

        return masks, iou_pred, target_organ_ids
```

- [ ] **Step 2: Verify forward pass**

```bash
C:\Users\Raywa\Desktop\LiteSAM3D\.venv\Scripts\python -c "
import torch
from models.organ_query_decoder import OrganQueryDecoder
m = OrganQueryDecoder(embed_dim=256, skip_channels=64)
imgs = torch.randn(2,256,16,16)
dp = torch.randn(2,256,16,16)
skip = torch.randn(2,64,32,32)
masks, iou, ids = m(imgs, dp, skip_features=skip)
print('masks:', masks.shape, 'iou:', iou.shape, 'n_organs:', len(ids))
div_loss = m.compute_diversity_loss()
print('diversity_loss:', div_loss.item())
"
```
Expected:
```
masks: torch.Size([2, 15, 64, 64])  iou: torch.Size([2, 15])  n_organs: 15
diversity_loss: <small positive float>
```

- [ ] **Step 3: Commit**
```bash
cd C:\Users\Raywa\Desktop\VoluFormer3D && git add models/organ_query_decoder.py && git commit -m "feat: create OrganQueryDecoder with 15 organ tokens, HQ token, diversity loss"
```

---

## Task 4: Auto-ODE-SAM Model (`models/auto_ode_sam.py`)

Create the main `AutoODESAM` model that assembles:
1. `MultiScaleEncoder` (Stage 2 + Stage 1 skip)
2. `OrganConditionedBidirectionalNeuralODE`
3. `PFESA`
4. `OrganQueryDecoder` (replaces PromptEncoder + MaskDecoder)

Also add deep supervision auxiliary head at ODE midpoint and register architecture in `__init__.py`.

**Files:**
- Create: `models/auto_ode_sam.py`
- Modify: `models/__init__.py`

- [ ] **Step 1: Create `models/auto_ode_sam.py`**

```python
# =============================================================================
# models/auto_ode_sam.py — Auto-ODE-SAM V3
#
# Fully automatic 15-organ CT segmentation.
# No box prompt needed — organ query tokens are learned end-to-end.
#
# Architecture:
#   MultiScaleEncoder (TinyViT Stage 2 + Stage 1 skip)
#     ↓
#   PFESA (zero params)
#     ↓
#   OrganConditionedBidirectionalNeuralODE
#     ↓
#   DeepSupervision head at t=0.5 (auxiliary loss only)
#     ↓
#   OrganQueryDecoder (15 organ tokens + HQ token)
#     ↓
#   15 organ mask logits + IoU scores
# =============================================================================

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.multiscale_encoder import MultiScaleEncoder
from models.ode_cross_slice import OrganConditionedBidirectionalNeuralODE
from models.organ_query_decoder import OrganQueryDecoder
from models.trimamba_sam import PFESASpectralSkip


class DeepSupervisionHead(nn.Module):
    """
    Lightweight segmentation head for auxiliary loss at ODE midpoint.

    Applied to features at t=0.5 (mid-integration) to force the ODE to
    produce meaningful intermediate representations.

    Args:
        embed_dim (int): Input feature dimension.
        n_organs  (int): Number of output classes (organs).
    """

    def __init__(self, embed_dim: int, n_organs: int = 15):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 4, kernel_size=1),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 8, n_organs, kernel_size=2, stride=2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, embed_dim, H_feat, W_feat)
        Returns:
            logits: (B, n_organs, H_out, W_out) — H_out = 4 * H_feat
        """
        return self.conv(features)


class AutoODESAM(nn.Module):
    """
    Auto-ODE-SAM V3: Fully automatic 15-organ CT segmentation.

    Args:
        cfg: OmegaConf config with model.architecture = "auto_ode_sam"
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        mcfg = cfg.model

        stage_index = getattr(
            getattr(mcfg, "multiscale_isa", None), "stage_index", 2
        )

        # ---- 1. Encoder ----
        self.encoder = MultiScaleEncoder(
            model_name=mcfg.encoder_name,
            img_size=mcfg.img_size,
            embed_dim=mcfg.embed_dim,
            pretrained=mcfg.encoder_pretrained,
            stage_index=stage_index,
        )
        feat_size = self.encoder.get_output_size()[0]

        # ---- 2. PFESA (zero params) ----
        pfesa_cfg    = getattr(mcfg, "pfesa", None)
        pfesa_alpha  = getattr(pfesa_cfg, "alpha",           1.0) if pfesa_cfg else 1.0
        pfesa_cutoff = getattr(pfesa_cfg, "highfreq_cutoff", 0.5) if pfesa_cfg else 0.5
        self.pfesa = PFESASpectralSkip(alpha=pfesa_alpha, highfreq_cutoff=pfesa_cutoff)

        # ---- 3. Organ-Conditioned Neural ODE ----
        ode_cfg       = getattr(mcfg, "ode", None)
        ode_hidden    = getattr(ode_cfg, "ode_hidden",     mcfg.embed_dim // 4) if ode_cfg else mcfg.embed_dim // 4
        n_freqs       = getattr(ode_cfg, "n_freqs",        4)  if ode_cfg else 4
        substeps      = getattr(ode_cfg, "substeps",       2)  if ode_cfg else 2
        n_organs      = getattr(ode_cfg, "n_organs",       15) if ode_cfg else 15
        organ_emb_dim = getattr(ode_cfg, "organ_emb_dim",  32) if ode_cfg else 32

        self.ode = OrganConditionedBidirectionalNeuralODE(
            dim=mcfg.embed_dim,
            n_organs=n_organs,
            organ_emb_dim=organ_emb_dim,
            ode_hidden=ode_hidden,
            n_freqs=n_freqs,
            substeps=substeps,
        )

        # ---- 4. Deep supervision auxiliary head ----
        self.deep_sup_head = DeepSupervisionHead(
            embed_dim=mcfg.embed_dim,
            n_organs=n_organs,
        )

        # ---- 5. Organ Query Decoder ----
        dec_cfg = getattr(mcfg, "organ_decoder", None)
        self.decoder = OrganQueryDecoder(
            embed_dim=mcfg.embed_dim,
            n_organs=n_organs,
            transformer_depth=getattr(dec_cfg, "transformer_depth", 2) if dec_cfg else 2,
            transformer_mlp_dim=getattr(dec_cfg, "transformer_mlp_dim", 2048) if dec_cfg else 2048,
            iou_head_depth=getattr(dec_cfg, "iou_head_depth", 3) if dec_cfg else 3,
            iou_head_hidden_dim=getattr(dec_cfg, "iou_head_hidden_dim", 256) if dec_cfg else 256,
            skip_channels=self.encoder.skip_channels,
        )

        # ---- Dense PE buffer (sinusoidal, registered once) ----
        # Used as positional encoding for the two-way transformer image tokens.
        # Shape: (1, embed_dim, feat_size, feat_size) — broadcast over batch.
        self.register_buffer(
            "dense_pe",
            self._make_dense_pe(mcfg.embed_dim, feat_size),
        )

    @staticmethod
    def _make_dense_pe(embed_dim: int, feat_size: int) -> torch.Tensor:
        """Sinusoidal 2D positional encoding for image tokens."""
        half = embed_dim // 2
        grid = torch.arange(feat_size, dtype=torch.float32) / feat_size
        x, y = torch.meshgrid(grid, grid, indexing="ij")
        freqs = torch.arange(half, dtype=torch.float32)
        pe_x = torch.sin(x.unsqueeze(-1) * (2 ** freqs.unsqueeze(0).unsqueeze(0)))
        pe_y = torch.cos(y.unsqueeze(-1) * (2 ** freqs.unsqueeze(0).unsqueeze(0)))
        pe = torch.cat([pe_x, pe_y], dim=-1)  # (feat_size, feat_size, embed_dim)
        return pe.permute(2, 0, 1).unsqueeze(0)  # (1, embed_dim, feat_size, feat_size)

    # ------------------------------------------------------------------
    def encode_image(
        self,
        images:            torch.Tensor,  # (B*D, 3, H, W) or (B, D, 3, H, W)
        organ_id:          torch.Tensor,  # (B,) long — which organ we're training on
        return_all_slices: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode, apply PFESA, run ODE.

        Side-effect: stores deep supervision features in self._deepsup_cache.

        Returns:
            features: (B*D, embed, Hf, Wf) or (B, embed, Hf, Wf)
            skip:     (B*D, skip_ch, Hs, Ws) or (B, skip_ch, Hs, Ws)
        """
        B_D = images.shape[0]
        # Detect if images is 5D (B, D, C, H, W) — reshape to 4D
        if images.dim() == 5:
            B, D, C, H, W = images.shape
            images = images.reshape(B * D, C, H, W)
        else:
            # Infer B from organ_id
            B  = organ_id.shape[0]
            D  = B_D // B

        feat_2d, skip_2d = self.encoder(images, return_skip=True)
        # feat_2d: (B*D, embed, Hf, Wf)

        _, embed_dim, Hf, Wf = feat_2d.shape
        feat_3d = feat_2d.reshape(B, D, embed_dim, Hf, Wf)

        # --- ODE (organ-conditioned) ---
        feat_3d = self.ode(feat_3d, organ_id)  # (B, D, embed, Hf, Wf)

        # --- Deep supervision: take features at mid-depth (t=0.5 ≈ D//2) ---
        mid = D // 2
        mid_feat = feat_3d[:, mid]  # (B, embed, Hf, Wf)
        self._deepsup_cache = mid_feat

        if return_all_slices:
            feat_all = feat_3d.reshape(B * D, embed_dim, Hf, Wf)
            feat_all = self.pfesa(feat_all)
            return feat_all, skip_2d

        center = feat_3d[:, mid]
        center = self.pfesa(center)
        # Center slice skip
        skip_ch, Hs, Ws = skip_2d.shape[1:]
        center_skip = skip_2d.reshape(B, D, skip_ch, Hs, Ws)[:, mid]
        return center, center_skip

    # ------------------------------------------------------------------
    def forward(
        self,
        images:           torch.Tensor,          # (B, D, 3, H, W) or (B, 3, H, W)
        organ_id:         torch.Tensor,           # (B,) long — training organ index
        target_organ_ids: Optional[List[int]] = None,  # organs to predict; None = all 15
        is_3d:            bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Returns:
            dict with keys:
              "masks":        (B, K, H_out, W_out) — K mask logits
              "iou_pred":     (B, K)
              "organ_ids":    list[int] — organ corresponding to each mask channel
              "deepsup_logits": (B, K, H_out, W_out) — auxiliary deep supervision masks
        """
        if is_3d:
            feat, skip = self.encode_image(images, organ_id, return_all_slices=True)
            # All-slice: feat is (B*D, embed, Hf, Wf); we use center for decoder
            B  = organ_id.shape[0]
            D  = feat.shape[0] // B
            embed_dim, Hf, Wf = feat.shape[1], feat.shape[2], feat.shape[3]
            feat_3d = feat.reshape(B, D, embed_dim, Hf, Wf)
            feat    = feat_3d[:, D // 2]   # center for decoder
            skip_ch = skip.shape[1]
            skip    = skip.reshape(B, D, skip_ch, skip.shape[2], skip.shape[3])[:, D // 2]
        else:
            feat, skip = self.encode_image(images, organ_id, return_all_slices=False)
            B = feat.shape[0]

        # Dense PE (broadcast to batch)
        dense_pe = self.dense_pe.expand(B, -1, -1, -1)

        masks, iou_pred, organ_ids_out = self.decoder(
            image_embeddings=feat,
            dense_pe=dense_pe,
            skip_features=skip,
            target_organ_ids=target_organ_ids,
        )

        # Deep supervision auxiliary logits
        deepsup_logits = self.deep_sup_head(self._deepsup_cache)
        # Resize to match masks output
        if deepsup_logits.shape[-2:] != masks.shape[-2:]:
            deepsup_logits = F.interpolate(
                deepsup_logits, size=masks.shape[-2:], mode='bilinear', align_corners=False,
            )

        return {
            "masks":           masks,
            "iou_pred":        iou_pred,
            "organ_ids":       organ_ids_out,
            "deepsup_logits":  deepsup_logits,
        }

    # ------------------------------------------------------------------
    def predict(
        self,
        images:           torch.Tensor,
        organ_id:         torch.Tensor,
        target_organ_ids: Optional[List[int]] = None,
        is_3d:            bool = False,
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Inference. Returns sigmoid masks and organ ID list.

        Returns:
            masks:      (B, K, H, W) — probabilities in [0, 1]
            organ_ids:  list[int]
        """
        with torch.no_grad():
            out = self.forward(images, organ_id, target_organ_ids, is_3d=is_3d)
        return torch.sigmoid(out["masks"]), out["organ_ids"]

    # ------------------------------------------------------------------
    def count_parameters(self) -> Dict[str, int]:
        def count(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {
            "encoder":        count(self.encoder),
            "ode_module":     count(self.ode),
            "decoder":        count(self.decoder),
            "deep_sup_head":  count(self.deep_sup_head),
            "pfesa":          0,
            "total":          count(self),
        }
```

- [ ] **Step 2: Register in `models/__init__.py`**

Add after the `elif arch == "ode_sam":` block:

```python
    elif arch == "auto_ode_sam":
        from models.auto_ode_sam import AutoODESAM
        return AutoODESAM(cfg)
```

- [ ] **Step 3: Verify full forward pass**

```bash
C:\Users\Raywa\Desktop\LiteSAM3D\.venv\Scripts\python -c "
import torch
from models import build_model
from omegaconf import OmegaConf
cfg = OmegaConf.create({
    'model': {
        'architecture': 'auto_ode_sam',
        'encoder_name': 'tiny_vit_21m_224',
        'img_size': 256,
        'embed_dim': 256,
        'encoder_pretrained': False,
        'multiscale_isa': {'stage_index': 2},
        'ode': {'ode_hidden': 64, 'n_freqs': 4, 'substeps': 2, 'n_organs': 15, 'organ_emb_dim': 32},
        'pfesa': {'alpha': 1.0, 'highfreq_cutoff': 0.5},
    }
})
model = build_model(cfg)
imgs = torch.randn(2, 8, 3, 256, 256)
organ_id = torch.tensor([6, 6])
out = model(imgs, organ_id, is_3d=True)
print('masks:', out['masks'].shape, 'iou:', out['iou_pred'].shape)
print('deepsup:', out['deepsup_logits'].shape)
params = model.count_parameters()
print('params:', {k: f'{v/1e6:.2f}M' for k, v in params.items()})
"
```
Expected:
```
masks: torch.Size([2, 15, 64, 64])  iou: torch.Size([2, 15])
deepsup: torch.Size([2, 15, 64, 64])
params: {'total': '~18.xM', ...}
```

- [ ] **Step 4: Commit**
```bash
cd C:\Users\Raywa\Desktop\VoluFormer3D && git add models/auto_ode_sam.py models/__init__.py && git commit -m "feat: create AutoODESAM V3 with organ queries, conditioned ODE, deep supervision"
```

---

## Task 5: Dataset — Organ ID in Samples + Foreground Oversampling

Modify `datasets/amos22.py` to:
1. Include `organ_id` (int) in every sample dict so the trainer can pass it to the ODE
2. Add foreground oversampling: pre-compute a per-organ slice presence index so 33% of batch slices contain small-organ foreground

**Files:**
- Modify: `datasets/amos22.py`

- [ ] **Step 1: Read the full `_build_slice_index` and `__getitem__` methods**

```bash
C:\Users\Raywa\Desktop\LiteSAM3D\.venv\Scripts\python -c "
import inspect
from datasets.amos22 import AMOS22_3D_Dataset
print(inspect.getsource(AMOS22_3D_Dataset._build_slice_index))
"
```
Read the output to understand how `self.samples` is structured. Each entry is a dict.

- [ ] **Step 2: Add `organ_id` to every sample dict in `_build_slice_index`**

Find the line in `_build_slice_index` where items are appended to `samples` (look for `samples.append({...})`). Add `"organ_id": organ_id` to the dict if it is not already there.

Example — if the current append looks like:
```python
samples.append({
    "image_path": str(img_path),
    "label_path": str(lbl_path),
    "slice_idx": slice_idx,
    "organ_id": organ_id,
    "modality": modality,
})
```
If `organ_id` is already there, this step is a no-op. Verify by running:
```bash
C:\Users\Raywa\Desktop\LiteSAM3D\.venv\Scripts\python -c "
from datasets.amos22 import AMOS22_3D_Dataset
from omegaconf import OmegaConf
cfg = OmegaConf.load('configs/phase3_odesam_v2.yaml')
ds = AMOS22_3D_Dataset(cfg, split='val', target_organs=[6])
sample = ds[0]
print(list(sample.keys()))
"
```
Expected: `organ_id` in keys.

- [ ] **Step 3: Add foreground oversampling index to `AMOS22_3D_Dataset.__init__`**

After `self.samples = self._build_slice_index()`, add:

```python
        # Foreground oversampling index — guarantees 33% of batches contain
        # small organ foreground. Pre-computed once at dataset init.
        # SMALL_ORGAN_IDS = [4, 5, 11, 12, 13] (gallbladder, esophagus, adrenals, duodenum)
        SMALL_ORGAN_IDS = {4, 5, 11, 12, 13}
        self._fg_indices     = []   # indices of samples with any target organ foreground
        self._small_fg_indices = []  # indices of samples with small organ foreground
        for i, s in enumerate(self.samples):
            oid = s.get("organ_id", 0)
            if oid in SMALL_ORGAN_IDS:
                self._small_fg_indices.append(i)
            else:
                self._fg_indices.append(i)
        print(f"  Foreground oversampling index: "
              f"{len(self._small_fg_indices)} small-organ slices, "
              f"{len(self._fg_indices)} regular slices")
```

- [ ] **Step 4: Add `get_oversampled_indices` method for use in DataLoader sampler**

After `__len__`, add:

```python
    def get_oversampled_indices(self, n_samples: int, small_organ_fraction: float = 0.33) -> List[int]:
        """
        Return a list of `n_samples` indices with `small_organ_fraction`
        guaranteed to come from small-organ slices (foreground oversampling).

        Used to build a custom sampler in the trainer.
        """
        import random as _random
        n_small  = int(n_samples * small_organ_fraction)
        n_regular = n_samples - n_small

        small_pool   = self._small_fg_indices if self._small_fg_indices else self._fg_indices
        regular_pool = self._fg_indices if self._fg_indices else self._small_fg_indices

        chosen_small   = _random.choices(small_pool,   k=n_small)
        chosen_regular = _random.choices(regular_pool, k=n_regular)
        indices = chosen_small + chosen_regular
        _random.shuffle(indices)
        return indices
```

- [ ] **Step 5: Verify**

```bash
C:\Users\Raywa\Desktop\LiteSAM3D\.venv\Scripts\python -c "
from datasets.amos22 import AMOS22_3D_Dataset
from omegaconf import OmegaConf
cfg = OmegaConf.load('configs/phase3_odesam_v2.yaml')
ds = AMOS22_3D_Dataset(cfg, split='val', target_organs=[6])
sample = ds[0]
print('organ_id in sample:', 'organ_id' in sample)
indices = ds.get_oversampled_indices(100, small_organ_fraction=0.33)
print('oversampled indices len:', len(indices))
"
```

- [ ] **Step 6: Commit**
```bash
cd C:\Users\Raywa\Desktop\VoluFormer3D && git add datasets/amos22.py && git commit -m "feat: add organ_id to dataset samples and foreground oversampling index"
```

---

## Task 6: Loss Functions — Boundary Loss + Deep Supervision

Add two new loss components to `training/losses.py`:
1. **Boundary loss** (distance-transform-based): ramps from 0.0 → 0.3 after epoch 50
2. **Deep supervision auxiliary loss**: weight 0.5 × main loss on ODE midpoint features

**Files:**
- Modify: `training/losses.py`

- [ ] **Step 1: Read the `CombinedLoss` class to understand current structure**

```bash
C:\Users\Raywa\Desktop\LiteSAM3D\.venv\Scripts\python -c "
import inspect
from training.losses import CombinedLoss
print(inspect.getsource(CombinedLoss))
"
```
Note the `__init__` signature and which loss weights are supported.

- [ ] **Step 2: Add `BoundaryLoss` class before `CombinedLoss`**

After the existing loss classes, add:

```python
class BoundaryLoss(nn.Module):
    """
    Distance-transform boundary loss (Kervadec et al., MIDL 2019).

    Minimises the sum of predicted probability × distance-transform of GT mask.
    Regions far from the GT boundary incur higher penalty if predicted as foreground.

    This is computed from the distance transform of the ground truth mask, which
    must be provided as input (precomputed or computed on-the-fly).

    We compute the DT on-the-fly using scipy (fast enough for D=8, H=W=256).

    Args:
        smooth (float): Epsilon for numerical stability.
    """

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    @staticmethod
    def _distance_transform(mask: torch.Tensor) -> torch.Tensor:
        """
        Compute distance transform of binary mask on CPU.
        mask: (B, H, W) binary float tensor
        Returns: (B, H, W) float tensor of normalised DT values in [0, 1]
        """
        from scipy.ndimage import distance_transform_edt
        import numpy as np
        B, H, W = mask.shape
        dt = torch.zeros_like(mask)
        mask_np = (mask.detach().cpu().numpy() == 0).astype(np.uint8)  # DT of background
        for b in range(B):
            raw = distance_transform_edt(mask_np[b]).astype(np.float32)
            max_val = raw.max() + 1e-5
            dt[b] = torch.from_numpy(raw / max_val)
        return dt.to(mask.device)

    def forward(
        self,
        pred:   torch.Tensor,   # (B, H, W) — logits or probabilities
        target: torch.Tensor,   # (B, H, W) — binary GT
    ) -> torch.Tensor:
        pred_prob = torch.sigmoid(pred) if pred.min() < 0 else pred
        dt = self._distance_transform(target)   # (B, H, W)
        # Boundary loss = mean(pred_prob * dt)
        # Penalises predicting foreground far from GT boundary
        return (pred_prob * dt).mean()
```

- [ ] **Step 3: Update `CombinedLoss.__init__` to accept boundary weight**

In `CombinedLoss.__init__`, find where loss weights are stored (e.g., `self.w_dice`, `self.w_focal`). Add:

```python
        self.w_boundary = cfg_loss.get("w_boundary", 0.0) if hasattr(cfg_loss, 'get') else getattr(cfg_loss, 'w_boundary', 0.0)
        self.w_deepsup  = cfg_loss.get("w_deepsup",  0.0) if hasattr(cfg_loss, 'get') else getattr(cfg_loss, 'w_deepsup',  0.0)
        if self.w_boundary > 0:
            self.boundary_loss = BoundaryLoss()
```

- [ ] **Step 4: Update `CombinedLoss.forward` to include boundary and deepsup losses**

In `CombinedLoss.forward`, after the existing loss terms are summed, add:

```python
        if self.w_boundary > 0 and hasattr(self, 'boundary_loss'):
            bl = self.boundary_loss(pred_center, target_center)
            total = total + self.w_boundary * bl
            breakdown["boundary"] = bl.item()

        # Deep supervision loss (auxiliary, from ODE midpoint features)
        if self.w_deepsup > 0 and deepsup_logits is not None and deepsup_organ_idx is not None:
            # deepsup_logits: (B, n_organs, H, W) from deep_sup_head
            # We extract the channel corresponding to this batch's organ and compute Dice
            ds_pred = deepsup_logits[:, deepsup_organ_idx, :, :]  # (B, H, W)
            ds_pred_resized = F.interpolate(
                ds_pred.unsqueeze(1), size=target_center.shape[-2:],
                mode='bilinear', align_corners=False,
            ).squeeze(1)
            ds_loss = self.dice_loss(ds_pred_resized, target_center)
            total = total + self.w_deepsup * ds_loss
            breakdown["deepsup"] = ds_loss.item()
```

Note: `CombinedLoss.forward` will need `deepsup_logits` and `deepsup_organ_idx` as optional kwargs. Add them to the signature:

```python
    def forward(
        self,
        pred:               torch.Tensor,
        target:             torch.Tensor,
        iou_pred:           torch.Tensor = None,
        deepsup_logits:     torch.Tensor = None,
        deepsup_organ_idx:  int          = None,
        current_epoch:      int          = 0,
        boundary_ramp_epoch: int         = 50,
    ) -> Tuple[torch.Tensor, Dict]:
```

Inside the method, before using `self.w_boundary`, ramp it:

```python
        # Boundary loss ramp: 0.0 → w_boundary after epoch boundary_ramp_epoch
        w_boundary_eff = 0.0
        if self.w_boundary > 0 and current_epoch >= boundary_ramp_epoch:
            ramp = min(1.0, (current_epoch - boundary_ramp_epoch) / 10.0)
            w_boundary_eff = self.w_boundary * ramp
```

- [ ] **Step 5: Verify BoundaryLoss**

```bash
C:\Users\Raywa\Desktop\LiteSAM3D\.venv\Scripts\python -c "
import torch
from training.losses import BoundaryLoss
bl = BoundaryLoss()
pred = torch.randn(2, 64, 64)
target = (torch.rand(2, 64, 64) > 0.7).float()
loss = bl(pred, target)
print('boundary loss:', loss.item())
"
```
Expected: a small positive float (typically 0.01–0.3).

- [ ] **Step 6: Commit**
```bash
cd C:\Users\Raywa\Desktop\VoluFormer3D && git add training/losses.py && git commit -m "feat: add BoundaryLoss and deep supervision loss to CombinedLoss"
```

---

## Task 7: Trainer Updates for AutoODESAM

Update `training/trainer.py` to:
1. Pass `organ_id` to model
2. Pass `deepsup_logits` + `deepsup_organ_idx` to loss
3. Use foreground oversampling sampler
4. Support multi-organ batch (one organ per sample in batch)

**Files:**
- Modify: `training/trainer.py`

- [ ] **Step 1: Read full trainer to understand batch loop structure**

```bash
C:\Users\Raywa\Desktop\LiteSAM3D\.venv\Scripts\python -c "
import inspect
from training.trainer import Trainer
# Print the _train_epoch method
print(inspect.getsource(Trainer._train_epoch))
" 2>&1 | head -120
```

- [ ] **Step 2: Extract `organ_id` from batch in `_train_epoch`**

Find where the batch is unpacked (look for `images = batch["images"]` or similar). Add:

```python
            organ_id = batch.get("organ_id", None)
            if organ_id is not None:
                organ_id = organ_id.to(self.device)
                # organ_id is (B,) long. If dataset returns per-slice organ IDs,
                # take the first (they're all the same within a 3D patch).
                if organ_id.dim() > 1:
                    organ_id = organ_id[:, 0]
```

- [ ] **Step 3: Pass `organ_id` to model forward**

Find the model call (e.g., `outputs = self.model(images, boxes, modality_ids, ...)`). For AutoODESAM:

```python
            if hasattr(self.model, 'decoder'):
                # AutoODESAM path — no boxes, uses organ query tokens
                # Use training organ_id for ODE conditioning
                train_organ_ids = organ_id.tolist() if organ_id is not None else None
                # All samples in batch train on same organ (single-organ phase 3a)
                # or mixed organs (phase 3b) — either way organ_id handles it
                target_ids = [int(organ_id[0].item())] if organ_id is not None else None
                outputs = self.model(
                    images, organ_id,
                    target_organ_ids=target_ids,
                    is_3d=True,
                )
            else:
                # Legacy ODESAM / VoluFormer3D path with box prompts
                outputs = self.model(images, boxes, modality_ids, is_3d=True)
```

- [ ] **Step 4: Pass `deepsup_logits` to loss computation**

Find where loss is computed (look for `loss, breakdown = self.criterion(...)`). Add deepsup args:

```python
            deepsup_logits = outputs.get("deepsup_logits", None) if isinstance(outputs, dict) else None
            organ_idx_for_loss = int(organ_id[0].item()) - 1 if organ_id is not None else None

            loss, breakdown = self.criterion(
                pred=pred_masks,
                target=target_masks,
                iou_pred=iou_pred,
                deepsup_logits=deepsup_logits,
                deepsup_organ_idx=organ_idx_for_loss,
                current_epoch=epoch,
                boundary_ramp_epoch=getattr(self.cfg.training, 'boundary_ramp_epoch', 50),
            )

            # Add decoder diversity loss (prevents organ token collapse)
            if hasattr(self.model, 'decoder') and hasattr(self.model.decoder, 'compute_diversity_loss'):
                div_loss = self.model.decoder.compute_diversity_loss()
                loss = loss + div_loss
```

- [ ] **Step 5: Add foreground oversampling to DataLoader construction**

Find where the train DataLoader is created (look for `DataLoader(train_dataset, ...)`). Replace with:

```python
            # Foreground oversampling: 33% of batch slices from small organs
            use_oversampling = getattr(self.cfg.training, 'foreground_oversampling', False)
            if use_oversampling and hasattr(train_dataset, 'get_oversampled_indices'):
                from torch.utils.data import SubsetRandomSampler
                n_samples = len(train_dataset)
                oversampled_indices = train_dataset.get_oversampled_indices(
                    n_samples, small_organ_fraction=0.33
                )
                sampler = SubsetRandomSampler(oversampled_indices)
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.cfg.training.batch_size,
                    sampler=sampler,
                    num_workers=self.cfg.training.num_workers,
                    pin_memory=self.cfg.training.pin_memory,
                )
            else:
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.cfg.training.batch_size,
                    shuffle=True,
                    num_workers=self.cfg.training.num_workers,
                    pin_memory=self.cfg.training.pin_memory,
                )
```

- [ ] **Step 6: Commit**
```bash
cd C:\Users\Raywa\Desktop\VoluFormer3D && git add training/trainer.py && git commit -m "feat: update trainer to support AutoODESAM organ_id, deepsup loss, foreground oversampling"
```

---

## Task 8: Two-Stage Zoom-In Inference

Create `inference/zoom_refine.py` for coarse-to-fine refinement of small organ masks at inference time.

**Files:**
- Create: `inference/zoom_refine.py`

- [ ] **Step 1: Create `inference/` directory and `zoom_refine.py`**

First check if directory exists:
```bash
ls C:\Users\Raywa\Desktop\VoluFormer3D\inference\ 2>/dev/null || echo "Directory missing"
```

If missing:
```bash
mkdir C:\Users\Raywa\Desktop\VoluFormer3D\inference
```

Create `inference/zoom_refine.py`:

```python
# =============================================================================
# inference/zoom_refine.py — Two-Stage Zoom-In for Small Organ Refinement
#
# Small organs (adrenal glands ~10px, esophagus ~3-5px wide) fall below the
# reliable detection threshold at 256px global resolution.
#
# Solution: Two-pass inference
#   Pass 1: Full volume @ 256px → coarse masks for all 15 organs
#   Pass 2: For each small organ, crop the ROI from Pass 1,
#           resize to 256px (4× effective resolution), run inference again,
#           paste refined mask back into original coordinate space.
#
# Reference: PRNet 2025, RAPS-3D — both show +3–8% DSC on small organs
# with zoom-in approaches.
# =============================================================================

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# Organ IDs that get two-stage zoom treatment
SMALL_ORGAN_IDS = [4, 5, 11, 12, 13]   # gallbladder, esophagus, adrenal L/R, duodenum

# AMOS22 organ names for logging
ORGAN_NAMES = {
    1: "spleen", 2: "right_kidney", 3: "left_kidney", 4: "gallbladder",
    5: "esophagus", 6: "liver", 7: "stomach", 8: "aorta",
    9: "inferior_vena_cava", 10: "pancreas", 11: "right_adrenal_gland",
    12: "left_adrenal_gland", 13: "duodenum", 14: "bladder", 15: "prostate_uterus",
}


def extract_roi_bbox(
    mask: torch.Tensor,    # (H, W) binary float mask
    padding: int = 32,     # pixels of padding around detected region
    min_size: int = 48,    # minimum crop size (prevents degenerate crops)
) -> Optional[Tuple[int, int, int, int]]:
    """
    Extract bounding box around non-zero region of mask.

    Args:
        mask:    (H, W) binary mask (values 0 or 1)
        padding: extra pixels around the detected region
        min_size: minimum crop dimension

    Returns:
        (y1, x1, y2, x2) bbox in pixel coordinates, or None if no foreground
    """
    H, W = mask.shape
    fg = (mask > 0.5).nonzero(as_tuple=False)
    if fg.shape[0] == 0:
        return None

    y_min, x_min = fg.min(dim=0).values.tolist()
    y_max, x_max = fg.max(dim=0).values.tolist()

    # Add padding
    y1 = max(0, y_min - padding)
    x1 = max(0, x_min - padding)
    y2 = min(H, y_max + padding + 1)
    x2 = min(W, x_max + padding + 1)

    # Enforce minimum crop size
    if (y2 - y1) < min_size:
        cy = (y1 + y2) // 2
        y1 = max(0, cy - min_size // 2)
        y2 = min(H, y1 + min_size)
    if (x2 - x1) < min_size:
        cx = (x1 + x2) // 2
        x1 = max(0, cx - min_size // 2)
        x2 = min(W, x1 + min_size)

    return y1, x1, y2, x2


def zoom_refine_volume(
    model,
    images:        torch.Tensor,         # (B, D, C, H, W) — original volume slices
    coarse_masks:  torch.Tensor,         # (B, K, H, W) — Pass 1 output (probabilities)
    organ_ids_out: List[int],            # which organs correspond to mask channels
    organ_id:      torch.Tensor,         # (B,) — training organ for ODE conditioning
    device:        torch.device,
    threshold:     float = 0.3,          # detection threshold for Pass 1 mask
    small_organs:  List[int] = SMALL_ORGAN_IDS,
) -> torch.Tensor:
    """
    Two-stage zoom-in: refine small organ masks using ROI crops.

    For each small organ detected in Pass 1:
      1. Find the bounding box of the coarse mask
      2. Crop all D slices of the volume to that ROI
      3. Resize crop to (H, W) = original resolution (4× effective resolution)
      4. Run Pass 2 inference on the crop
      5. Paste refined mask back into the original (H, W) space

    Args:
        model:         AutoODESAM model (eval mode)
        images:        (B, D, C, H, W) original volume
        coarse_masks:  (B, K, H_out, W_out) Pass 1 sigmoid outputs
        organ_ids_out: list of organ IDs corresponding to each mask channel K
        organ_id:      (B,) ODE conditioning organ ID
        device:        target device
        threshold:     minimum probability to consider an organ "detected"
        small_organs:  list of organ IDs to apply zoom-in to

    Returns:
        refined_masks: (B, K, H_out, W_out) — same shape as coarse_masks,
                       with small organ channels replaced by refined predictions
    """
    B, K, H_out, W_out = coarse_masks.shape
    _, D, C, H, W = images.shape

    # Work on a copy
    refined_masks = coarse_masks.clone()

    model.eval()
    with torch.no_grad():
        for k_idx, oid in enumerate(organ_ids_out):
            if oid not in small_organs:
                continue

            for b in range(B):
                coarse_mask_b = coarse_masks[b, k_idx]  # (H_out, W_out)

                # Skip if organ not detected in Pass 1
                if coarse_mask_b.max() < threshold:
                    continue

                # Get ROI bounding box from Pass 1 mask
                bbox = extract_roi_bbox(coarse_mask_b, padding=32)
                if bbox is None:
                    continue
                y1, x1, y2, x2 = bbox

                # Crop volume: (D, C, H, W) → (D, C, crop_h, crop_w)
                crop_h = y2 - y1
                crop_w = x2 - x1
                vol_crop = images[b, :, :, y1:y2, x1:x2]   # (D, C, crop_h, crop_w)

                # Resize to original resolution
                vol_crop_resized = F.interpolate(
                    vol_crop,
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False,
                )  # (D, C, H, W)

                # Add batch dim: (1, D, C, H, W)
                vol_crop_batch = vol_crop_resized.unsqueeze(0).to(device)
                oid_batch = organ_id[b:b+1].to(device)

                # Pass 2: inference on zoom crop
                out = model(
                    vol_crop_batch, oid_batch,
                    target_organ_ids=[oid],
                    is_3d=True,
                )
                refined_prob = torch.sigmoid(out["masks"][0, 0])  # (H_out, W_out)

                # Paste back: resize refined mask to crop size, then place in original space
                refined_crop = F.interpolate(
                    refined_prob.unsqueeze(0).unsqueeze(0),
                    size=(crop_h, crop_w),
                    mode='bilinear',
                    align_corners=False,
                )[0, 0]  # (crop_h, crop_w)

                # Paste into result (take max of coarse and refined)
                refined_masks[b, k_idx, y1:y2, x1:x2] = torch.maximum(
                    refined_masks[b, k_idx, y1:y2, x1:x2],
                    refined_crop,
                )

    return refined_masks
```

- [ ] **Step 2: Verify import**
```bash
C:\Users\Raywa\Desktop\LiteSAM3D\.venv\Scripts\python -c "from inference.zoom_refine import zoom_refine_volume, SMALL_ORGAN_IDS; print('OK, small organs:', SMALL_ORGAN_IDS)"
```
Expected: `OK, small organs: [4, 5, 11, 12, 13]`

- [ ] **Step 3: Commit**
```bash
cd C:\Users\Raywa\Desktop\VoluFormer3D && git add inference/zoom_refine.py && git commit -m "feat: create two-stage zoom-in inference for small organs"
```

---

## Task 9: Training Configs

Create two configs for Phase 3a (single-organ liver validation) and Phase 3b (full 15-organ run).

**Files:**
- Create: `configs/phase3a_autoodesam_liver.yaml`
- Create: `configs/phase3b_autoodesam_full.yaml`

- [ ] **Step 1: Create `configs/phase3a_autoodesam_liver.yaml`**

```yaml
# =============================================================================
# phase3a_autoodesam_liver.yaml — Auto-ODE-SAM V3 Single-Organ Validation
#
# PURPOSE: Validate the full Auto-ODE-SAM V3 architecture on liver before
#          committing to the 3–4 day 15-organ run.
#
# TARGET: ≥ 0.920 DSC on liver (vs ODE-SAM V2 baseline: 0.9358)
# NOTE:   The organ query decoder has LESS information than a tight box prompt
#         (MCP-MedSAM style). A small regression vs 0.9358 is expected.
#         Success = demonstrating the automatic path is competitive (≥0.920).
#
# RUN:
#   cd C:\Users\Raywa\Desktop\VoluFormer3D
#   C:\Users\Raywa\Desktop\LiteSAM3D\.venv\Scripts\python train.py \
#       --config configs/phase3a_autoodesam_liver.yaml
# =============================================================================

experiment:
  name: "phase3a_autoodesam_liver_256px"
  seed: 42
  output_dir: "checkpoints"
  log_dir: "logs"

model:
  architecture: "auto_ode_sam"
  encoder_name: "tiny_vit_21m_224"
  encoder_pretrained: true
  encoder_freeze_backbone: false
  img_size: 256
  embed_dim: 256

  multiscale_isa:
    stage_index: 2

  ode:
    ode_hidden: 64
    n_freqs: 4
    substeps: 2
    n_organs: 15
    organ_emb_dim: 32

  pfesa:
    alpha: 1.0
    highfreq_cutoff: 0.5

  organ_decoder:
    transformer_depth: 2
    transformer_mlp_dim: 2048
    iou_head_depth: 3
    iou_head_hidden_dim: 256

data:
  dataset: "amos22_3d"
  data_root: "C:/Users/Raywa/Desktop/LiteSAM3D/data/amos22"
  modality: "ct"
  num_classes: 1
  target_organs: [6]          # liver only for validation
  clip_range: [-175, 250]
  target_spacing: [1.5, 1.5, 1.5]
  normalize: true
  train_split: "train"
  val_split: "val"
  test_split: "test"
  slices_per_volume: 8
  slice_overlap: 0.5
  cache_rate: 0.0

training:
  epochs: 21
  batch_size: 4
  grad_accumulation_steps: 4
  num_workers: 4
  pin_memory: true
  mixed_precision: true
  optimizer: "adamw"
  lr: 1.0e-4
  lr_min: 1.0e-6
  weight_decay: 0.01
  betas: [0.9, 0.999]
  scheduler: "cosine_warmup"
  warmup_epochs: 2
  warmup_lr_init: 1.0e-6
  grad_clip: 1.0

  # All Phase 2 training fixes
  all_slice_supervision: true
  cutmix_prob: 0.5
  foreground_oversampling: false   # disabled for single-organ liver (no small organs)
  boundary_ramp_epoch: 50

  loss:
    w_dice:      0.5
    w_focal:     0.0
    w_cldice:    0.2
    w_iou:       0.1
    w_boundary:  0.1    # enabled (will ramp in at epoch 50 — beyond 21 epochs, so effectively 0 here)
    w_dice_topk: 0.5
    w_deepsup:   0.3    # auxiliary deep supervision on ODE midpoint features
  focal_alpha: 0.25
  focal_gamma: 2.0
  cldice_iterations: 3

checkpoint:
  save_every_n_epochs: 5
  keep_top_k: 2
  monitor_metric: "val/dice_3d"
  resume_from: null

validation:
  val_every_n_epochs: 5
  vis_every_n_epochs: 10

evaluation:
  metric_space: "3d_volume"
  report_spacing_mm: [1.5, 1.5, 1.5]
  compute_hd95: true
  compute_nsd: true

cross_validation:
  enabled: false
  n_folds: 5
  fold: 0
  stratify_by: "patient"

logging:
  use_wandb: false
  use_tensorboard: true
  wandb_project: "voluformer3d"
  wandb_entity: null
  log_every_n_steps: 50
  log_images: true
```

- [ ] **Step 2: Create `configs/phase3b_autoodesam_full.yaml`**

```yaml
# =============================================================================
# phase3b_autoodesam_full.yaml — Auto-ODE-SAM V3 Full 15-Organ Training
#
# PURPOSE: Full 15-organ training for thesis submission.
#          Target: ≥ 0.90 mean DSC across all 15 AMOS22 organs at 256px.
#
# PREREQUISITES:
#   - phase3a_autoodesam_liver: ≥ 0.920 DSC (architecture validated)
#
# COMPARISON TARGETS (from published papers):
#   MCP-MedSAM (CVPR 2024 challenge): ~0.875 DSC (FLARE, not AMOS22 — approximate)
#   LiteMedSAM: ~0.83–0.86 mean DSC
#   nnU-Net: 0.889–0.924 (NOT fair — 56M params, 3D, no prompts)
#
# NOVELTY: First automatic SAM model with Neural ODE cross-slice dynamics
#          and organ-conditioned trajectory modeling on AMOS22.
#
# RUN:
#   cd C:\Users\Raywa\Desktop\VoluFormer3D
#   C:\Users\Raywa\Desktop\LiteSAM3D\.venv\Scripts\python train.py \
#       --config configs/phase3b_autoodesam_full.yaml
#
# ESTIMATED TIME: ~3–4 days on RTX 3090 (24GB)
# =============================================================================

experiment:
  name: "phase3b_autoodesam_full_256px"
  seed: 42
  output_dir: "checkpoints"
  log_dir: "logs"

model:
  architecture: "auto_ode_sam"
  encoder_name: "tiny_vit_21m_224"
  encoder_pretrained: true
  encoder_freeze_backbone: false
  img_size: 256
  embed_dim: 256

  multiscale_isa:
    stage_index: 2

  ode:
    ode_hidden: 64
    n_freqs: 4
    substeps: 2
    n_organs: 15
    organ_emb_dim: 32

  pfesa:
    alpha: 1.0
    highfreq_cutoff: 0.5

  organ_decoder:
    transformer_depth: 2
    transformer_mlp_dim: 2048
    iou_head_depth: 3
    iou_head_hidden_dim: 256

data:
  dataset: "amos22_3d"
  data_root: "C:/Users/Raywa/Desktop/LiteSAM3D/data/amos22"
  modality: "ct"
  num_classes: 1
  target_organs: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # ALL 15 ORGANS
  clip_range: [-175, 250]
  target_spacing: [1.5, 1.5, 1.5]
  normalize: true
  train_split: "train"
  val_split: "val"
  test_split: "test"
  slices_per_volume: 8
  slice_overlap: 0.5
  cache_rate: 0.0

training:
  epochs: 100
  batch_size: 4
  grad_accumulation_steps: 4
  num_workers: 4
  pin_memory: true
  mixed_precision: true
  optimizer: "adamw"
  lr: 5.0e-5           # slightly lower LR for longer training
  lr_min: 1.0e-6
  weight_decay: 0.01
  betas: [0.9, 0.999]
  scheduler: "cosine_warmup"
  warmup_epochs: 5
  warmup_lr_init: 1.0e-6
  grad_clip: 1.0

  # All Phase 2 training fixes
  all_slice_supervision: true
  cutmix_prob: 0.5
  foreground_oversampling: true    # 33% of batches guaranteed small-organ foreground
  boundary_ramp_epoch: 50          # boundary loss ramps in after epoch 50

  loss:
    w_dice:      0.5
    w_focal:     0.0
    w_cldice:    0.2
    w_iou:       0.1
    w_boundary:  0.3    # will ramp from 0 → 0.3 starting at epoch 50
    w_dice_topk: 0.5
    w_deepsup:   0.3    # auxiliary deep supervision weight
  focal_alpha: 0.25
  focal_gamma: 2.0
  cldice_iterations: 3

checkpoint:
  save_every_n_epochs: 10
  keep_top_k: 3
  monitor_metric: "val/dice_3d"
  resume_from: null    # set to checkpoint path to resume

validation:
  val_every_n_epochs: 10
  vis_every_n_epochs: 20

evaluation:
  metric_space: "3d_volume"
  report_spacing_mm: [1.5, 1.5, 1.5]
  compute_hd95: true
  compute_nsd: true

cross_validation:
  enabled: false
  n_folds: 5
  fold: 0
  stratify_by: "patient"

logging:
  use_wandb: false
  use_tensorboard: true
  wandb_project: "voluformer3d"
  wandb_entity: null
  log_every_n_steps: 50
  log_images: true
```

- [ ] **Step 3: Verify configs parse**
```bash
C:\Users\Raywa\Desktop\LiteSAM3D\.venv\Scripts\python -c "
from omegaconf import OmegaConf
cfg_a = OmegaConf.load('configs/phase3a_autoodesam_liver.yaml')
cfg_b = OmegaConf.load('configs/phase3b_autoodesam_full.yaml')
print('3a arch:', cfg_a.model.architecture)
print('3b organs:', cfg_b.data.target_organs)
print('3b epochs:', cfg_b.training.epochs)
"
```
Expected:
```
3a arch: auto_ode_sam
3b organs: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
3b epochs: 100
```

- [ ] **Step 4: Commit**
```bash
cd C:\Users\Raywa\Desktop\VoluFormer3D && git add configs/phase3a_autoodesam_liver.yaml configs/phase3b_autoodesam_full.yaml && git commit -m "feat: add phase3a and phase3b configs for Auto-ODE-SAM V3"
```

---

## Task 10: End-to-End Integration Test

Run a short smoke test to confirm the full pipeline works before committing to Phase 3a.

**Files:** None (test only)

- [ ] **Step 1: Run 2-epoch smoke test**

```bash
cd C:\Users\Raywa\Desktop\VoluFormer3D && C:\Users\Raywa\Desktop\LiteSAM3D\.venv\Scripts\python train.py --config configs/phase3a_autoodesam_liver.yaml training.epochs=2 training.batch_size=2 validation.val_every_n_epochs=1 checkpoint.save_every_n_epochs=2
```

Expected: Training runs 2 epochs, validation runs, no crashes, loss decreases.

- [ ] **Step 2: Confirm no shape errors in log**

Look for lines like:
```
Epoch 1/2  loss=X.XXX  dice=X.XXX
Epoch 2/2  loss=X.XXX  dice=X.XXX
val dice_3d: X.XXX
```
No `RuntimeError`, no `shape mismatch`, no `CUDA out of memory`.

- [ ] **Step 3: If smoke test passes — queue Phase 3a**

Phase 3a run command:
```bash
cd C:\Users\Raywa\Desktop\VoluFormer3D
C:\Users\Raywa\Desktop\LiteSAM3D\.venv\Scripts\python train.py --config configs/phase3a_autoodesam_liver.yaml
```
Target: ≥ 0.920 DSC on liver (21 epochs, ~2.5h)

---

## Updated Training Phases

| Phase | Config | Status | Target | Command |
|-------|--------|--------|--------|---------|
| 0: Architecture ablations | `test_*_a1.yaml` | ✅ Done | — | — |
| 1: ODE-SAM + TriMamba | `test_ode_sam_a1.yaml` | ✅ Done | — | — |
| 2: Phase 2 training fixes | `phase2_ode_full.yaml` | ✅ Done | — | — |
| 2c: Arch upgrades (V2) | `phase3_odesam_v2.yaml` | ✅ **0.9358 DSC** | beat 0.9183 | — |
| **3a: Auto-ODE-SAM liver** | `phase3a_autoodesam_liver.yaml` | ⏳ Next | ≥ 0.920 | `python train.py --config configs/phase3a_autoodesam_liver.yaml` |
| **3b: Full 15-organ** | `phase3b_autoodesam_full.yaml` | 🔒 After 3a | ≥ 0.90 mean | `python train.py --config configs/phase3b_autoodesam_full.yaml` |
| 4: Analysis & writing | — | 🔒 After 3b | — | Per-organ table, ablations, ODE viz |

---

## Spec Coverage Check

| Spec Requirement | Task |
|-----------------|------|
| Organ Query Decoder (15 tokens) | Task 3 |
| Organ-Conditioned ODE (`dh/dt = f + bias`) | Task 1 |
| HQ-SAM output token | Task 2 |
| Two-stage zoom-in (small organs) | Task 8 |
| Foreground oversampling (33%) | Task 5 |
| Boundary loss (ramp at epoch 50) | Task 6 |
| Deep supervision at t=0.5 | Task 4 + 6 |
| Diversity loss for organ tokens | Task 3 |
| Phase 3a config | Task 9 |
| Phase 3b config | Task 9 |
| End-to-end test | Task 10 |
