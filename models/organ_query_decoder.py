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
        diversity_loss_weight: float = 0.01,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_organs  = n_organs
        self.diversity_loss_weight = diversity_loss_weight

        # ----- 15 organ query tokens -----
        # organ_id is 1-indexed (1..15), mapped to 0-indexed embeddings internally
        self.organ_queries = nn.Embedding(n_organs, embed_dim)
        # ViT-convention init: std = embed_dim**-0.5 (~0.0625 for dim=256).
        # The earlier std=0.02 left queries at L2 ~0.32 forever — diag run on
        # A1 ep_013 showed queries STILL at L2 0.32-0.37, never escaping init
        # magnitude (audit Concern 5). dim**-0.5 gives initial L2 ~1.0 and
        # better gradient signal-to-noise.
        nn.init.trunc_normal_(self.organ_queries.weight, std=embed_dim ** -0.5)

        # ----- HQ output token (fine-boundary recovery) -----
        self.hq_token = nn.Embedding(1, embed_dim)

        # ----- Shared two-way transformer -----
        self.transformer = TwoWayTransformer(
            depth=transformer_depth,
            embedding_dim=embed_dim,
            num_heads=8,
            mlp_dim=transformer_mlp_dim,
        )

        # ----- Per-organ mask prediction heads -----
        self.mask_mlps = nn.ModuleList([
            MLP(embed_dim, embed_dim, embed_dim // 8, depth=3)
            for _ in range(n_organs)
        ])

        # ----- Per-organ IoU prediction heads -----
        self.iou_heads = nn.ModuleList([
            MLP(embed_dim, iou_head_hidden_dim, 1, depth=iou_head_depth)
            for _ in range(n_organs)
        ])

        # ----- Upsampling (shared) -----
        self.upsample_conv1 = nn.ConvTranspose2d(embed_dim, embed_dim // 4, kernel_size=2, stride=2)
        self.upsample_ln    = nn.LayerNorm(embed_dim // 4)
        self.upsample_conv2 = nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=2, stride=2)

        # ----- Stage 1 skip injection -----
        self.use_skip = skip_channels > 0
        if self.use_skip:
            # Small positive init (0.05) opens the skip path from epoch 0.
            # zero-init kept the Stage 1 boundary features permanently dormant in
            # early training because gradients through a zero gate are tiny.
            # 0.05 is conservative enough to not overwhelm main features at init
            # but ensures the path receives gradient signal from step 1.
            self.skip_gate = nn.Parameter(torch.full((1,), 0.05))
            self.skip_ln   = nn.LayerNorm(skip_channels)

        # ----- HQ path -----
        hq_fuse_in = skip_channels if skip_channels > 0 else embed_dim // 4
        self.hq_mlp = MLP(embed_dim, embed_dim, embed_dim // 8, depth=3)
        self.hq_skip_proj = nn.Sequential(
            nn.Conv2d(hq_fuse_in, embed_dim // 8, kernel_size=1),
            nn.GELU(),
        )
        # Zero-init the WEIGHT so hq_feat = 0 at step 0 regardless of bias.
        # Bias is left at default (fan-in) init so the path can bootstrap:
        # once hq_gate accumulates gradient, hq_skip_proj(sf_hq) has nonzero
        # bias signal to start from. Double-zero (weight AND bias) would mean
        # hq_gate.grad = 0 forever, keeping the HQ head permanently dead.
        nn.init.zeros_(self.hq_skip_proj[0].weight)
        # Learnable gate: small positive init so the HQ correction is active
        # from step 1 (unlike zero-init which requires many epochs to open).
        self.hq_gate = nn.Parameter(torch.full((1,), 0.05))

    # ------------------------------------------------------------------
    @staticmethod
    def _haar_edge_enhance(x: torch.Tensor) -> torch.Tensor:
        """Zero-parameter Haar wavelet edge enhancement."""
        # Pad to even spatial dimensions to prevent subsampling length mismatch
        orig_h, orig_w = x.shape[-2], x.shape[-1]
        pad_h = orig_h % 2
        pad_w = orig_w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]
        LH = (x00 - x01 + x10 - x11) * 0.25
        HL = (x00 + x01 - x10 - x11) * 0.25
        HH = (x00 - x01 - x10 + x11) * 0.25
        hf = LH.abs() + HL.abs() + HH.abs()
        hf_up = F.interpolate(hf, size=(orig_h + pad_h, orig_w + pad_w), mode='bilinear', align_corners=False)
        result = x + 0.5 * hf_up
        # Trim back to original spatial dimensions
        return result[:, :, :orig_h, :orig_w]

    # ------------------------------------------------------------------
    def compute_diversity_loss(self) -> torch.Tensor:
        """
        Cosine diversity loss between organ query embeddings.

        Encourages organ tokens to be dissimilar, preventing all 15 tokens
        from collapsing to the same embedding.

        Loss = mean(cosine_similarity(token_i, token_j)) for i != j
        """
        W = self.organ_queries.weight  # (n_organs, embed_dim)
        W_norm = F.normalize(W, dim=-1)
        sim_matrix = W_norm @ W_norm.T  # (n_organs, n_organs)
        mask = 1.0 - torch.eye(self.n_organs, device=W.device)
        # Use absolute cosine similarity so the loss is always ≥ 0.
        # Penalises both positive AND negative correlation between organ tokens,
        # pushing all pairs toward orthogonality (|cos| → 0).
        # Note: abs() penalises both positive AND negative correlation, pushing tokens
        # toward orthogonality. Without abs(), tokens collapse to antipodal pairs
        # (negative similarity is also zero-penalty), which is a degenerate solution.
        # This departs from the plan formula but is strictly better.
        diversity_loss = (sim_matrix.abs() * mask).sum() / (self.n_organs * (self.n_organs - 1))
        return self.diversity_loss_weight * diversity_loss

    # ------------------------------------------------------------------
    def forward(
        self,
        image_embeddings:  torch.Tensor,              # (B, embed_dim, H_feat, W_feat)
        dense_pe:          torch.Tensor,               # (B, embed_dim, H_feat, W_feat)
        skip_features:     Optional[torch.Tensor] = None,  # (B, skip_ch, 2*H, 2*W)
        target_organ_ids:  Optional[List[int]] = None,     # subset of [1..15]; None = all
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Produce masks and IoU scores for all (or selected) organs.

        Args:
            image_embeddings:  (B, embed_dim, H_feat, W_feat)
            dense_pe:          positional encoding for image tokens
            skip_features:     Stage 1 encoder features for skip injection (optional)
            target_organ_ids:  organ IDs to predict (1-indexed). None = all 15.

        Returns:
            masks:         (B, K, H_out, W_out) — mask logits
            iou_pred:      (B, K) — IoU predictions
            organ_ids_out: list[int] — which organs correspond to each mask channel
        """
        B, C, H_feat, W_feat = image_embeddings.shape

        if target_organ_ids is None:
            target_organ_ids = list(range(1, self.n_organs + 1))
        # Map 1-indexed organ IDs to 0-indexed embedding positions
        query_indices = [oid - 1 for oid in target_organ_ids]
        K = len(query_indices)

        # Gather organ query tokens: (K, embed_dim)
        organ_q = self.organ_queries.weight[query_indices]
        # Append HQ token: (K+1, embed_dim)
        all_queries = torch.cat([organ_q, self.hq_token.weight], dim=0)
        # Expand to batch: (B, K+1, embed_dim)
        queries = all_queries.unsqueeze(0).expand(B, -1, -1).contiguous()
        query_pe = torch.zeros_like(queries)

        # ----- Two-way transformer -----
        # Run transformer FIRST so we can upsample the query-attended image tokens.
        # SAM's original design upsamples the transformer-updated image features, not
        # the raw ones — each organ mask is then generated against a canvas that already
        # reflects cross-attention between that organ's query and the image.
        src_flat = image_embeddings.flatten(2).permute(0, 2, 1)   # (B, H*W, C)
        pos_flat = dense_pe.flatten(2).permute(0, 2, 1)            # (B, H*W, C)
        hs, src_updated = self.transformer(src_flat, pos_flat, queries, query_pe)
        # hs:         (B, K+1, embed_dim)  — updated query tokens
        # src_updated: (B, H*W, C)          — updated image tokens (organ-conditioned)

        # ----- Upsample transformer-updated image features -----
        # Reshape updated image tokens back to spatial map before upsampling.
        src_spatial = src_updated.permute(0, 2, 1).reshape(B, C, H_feat, W_feat)
        upscaled = self.upsample_conv1(src_spatial)   # (B, C//4, H*2, W*2)
        B_, C_, H_, W_ = upscaled.shape
        upscaled = upscaled.permute(0, 2, 3, 1).reshape(-1, C_)
        upscaled = self.upsample_ln(upscaled)
        upscaled = upscaled.reshape(B_, H_, W_, C_).permute(0, 3, 1, 2)
        upscaled = F.gelu(upscaled)

        # ----- Stage 1 skip injection -----
        if self.use_skip and skip_features is not None:
            sf = skip_features
            if sf.shape[-2:] != upscaled.shape[-2:]:
                sf = F.interpolate(
                    sf.float(), size=upscaled.shape[-2:],
                    mode='bilinear', align_corners=False,
                ).to(upscaled.dtype)
            sf_flat = sf.permute(0, 2, 3, 1).reshape(-1, sf.shape[1])
            sf_flat = self.skip_ln(sf_flat)
            sf = sf_flat.reshape(sf.shape[0], *sf.shape[2:], sf.shape[1]).permute(0, 3, 1, 2)
            sf = self._haar_edge_enhance(sf)
            upscaled = upscaled + self.skip_gate * sf

        upscaled = self.upsample_conv2(upscaled)   # (B, C//8, H*4, W*4)
        upscaled = F.gelu(upscaled)
        b, c, h, w = upscaled.shape

        assert hs.shape[1] == K + 1, (
            f"Transformer output has {hs.shape[1]} tokens; expected {K + 1} (K organs + HQ)"
        )

        organ_token_outs = hs[:, :K, :]   # (B, K, embed_dim)
        hq_token_out     = hs[:, K, :]    # (B, embed_dim)

        # ----- Predict per-organ masks -----
        masks    = []
        iou_pred = []
        for k_idx, organ_idx in enumerate(query_indices):
            weights = self.mask_mlps[organ_idx](organ_token_outs[:, k_idx, :])  # (B, C//8)
            mask_k  = (weights.unsqueeze(1) @ upscaled.view(b, c, h * w)).view(b, 1, h, w)
            iou_k   = self.iou_heads[organ_idx](organ_token_outs[:, k_idx, :])  # (B, 1)
            masks.append(mask_k)
            iou_pred.append(iou_k)

        masks    = torch.cat(masks,    dim=1)  # (B, K, H_out, W_out)
        iou_pred = torch.cat(iou_pred, dim=1)  # (B, K)

        # ----- HQ correction -----
        hq_weights = self.hq_mlp(hq_token_out)   # (B, C//8)
        if skip_features is not None:
            sf_hq = skip_features
            if sf_hq.shape[-2:] != upscaled.shape[-2:]:
                sf_hq = F.interpolate(
                    sf_hq.float(), size=upscaled.shape[-2:],
                    mode='bilinear', align_corners=False,
                ).to(upscaled.dtype)
            # Apply skip_ln before projecting (consistent with main skip path)
            if self.use_skip:
                sf_hq_flat = sf_hq.permute(0, 2, 3, 1).reshape(-1, sf_hq.shape[1])
                sf_hq_flat = self.skip_ln(sf_hq_flat)
                sf_hq = sf_hq_flat.reshape(
                    sf_hq.shape[0], sf_hq.shape[2], sf_hq.shape[3], sf_hq.shape[1]
                ).permute(0, 3, 1, 2)
            # Gate starts at 0: HQ correction is inactive until training opens it
            hq_feat = self.hq_gate * self.hq_skip_proj(sf_hq)  # (B, C//8, H, W)
        else:
            hq_feat = torch.zeros_like(upscaled)
        b_hq, c_hq, h_hq, w_hq = hq_feat.shape
        hq_mask = (hq_weights.unsqueeze(1) @ hq_feat.view(b_hq, c_hq, h_hq * w_hq)).view(b_hq, 1, h_hq, w_hq)
        masks = masks + hq_mask

        return masks, iou_pred, target_organ_ids
