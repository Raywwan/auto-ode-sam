# =============================================================================
# models/mask_decoder.py — Mask Decoder (Two-Way Transformer)
#
# Decodes image features + prompt embeddings into binary segmentation masks.
#
# Architecture follows SAM's mask decoder with MCP-MedSAM's additions:
#   - Two-way transformer: image tokens attend to prompt tokens AND vice versa
#   - Multi-scale upsampling: 4x upsampling to reach full feature resolution
#   - FiLM conditioning: modality information modulates feature statistics
#   - IoU prediction head: predicts mask quality score (used for best-mask selection)
#   - Multiple mask outputs: outputs 3 masks, IoU head picks the best one
#
# Reference:
#   SAM (Kirillov et al., 2023) — original two-way transformer decoder
#   MCP-MedSAM (CVPR 2024) — FiLM conditioning + modality classification head
# =============================================================================

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoWayAttentionBlock(nn.Module):
    """
    A single block of two-way (bidirectional) attention.

    Two-way means:
      1. Query tokens attend to key-value tokens  (prompt → image)
      2. Key-value tokens attend to query tokens  (image → prompt)

    This bidirectional attention allows prompt and image features to
    mutually condition each other — the image learns where the prompt is,
    and the prompt learns what's in the image.

    Args:
        embedding_dim (int): Dimension for all attention operations.
        num_heads (int): Number of attention heads.
        mlp_dim (int): Hidden dimension in the FFN.
        attention_downsample_rate (int): Reduce dimension inside attention for efficiency.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 8,
        mlp_dim: int = 2048,
        attention_downsample_rate: int = 2,
    ):
        super().__init__()

        self.self_attn = Attention(embedding_dim, num_heads, downsample_rate=1)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLP(embedding_dim, mlp_dim, embedding_dim, depth=2)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm4 = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        queries: torch.Tensor,    # (B, N_q, C) — prompt tokens
        keys: torch.Tensor,       # (B, N_k, C) — image tokens (flattened feat map)
        query_pe: torch.Tensor,   # (B, N_q, C) — positional encoding for prompts
        key_pe: torch.Tensor,     # (B, N_k, C) — positional encoding for image tokens
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            queries: (B, N_q, C) — updated prompt tokens
            keys:    (B, N_k, C) — updated image tokens
        """
        # ---- Step 1: Self-attention among query (prompt) tokens ----
        queries = self.norm1(queries + self.self_attn(
            q=queries + query_pe, k=queries + query_pe, v=queries
        ))

        # ---- Step 2: Cross-attention: prompt tokens query image tokens ----
        queries = self.norm2(queries + self.cross_attn_token_to_image(
            q=queries + query_pe, k=keys + key_pe, v=keys
        ))

        # ---- Step 3: FFN on prompt tokens ----
        queries = self.norm3(queries + self.mlp(queries))

        # ---- Step 4: Cross-attention: image tokens query prompt tokens ----
        keys = self.norm4(keys + self.cross_attn_image_to_token(
            q=keys + key_pe, k=queries + query_pe, v=queries
        ))

        return queries, keys


class Attention(nn.Module):
    """
    Multi-head attention with optional internal dimension reduction.

    Args:
        embedding_dim (int): Input/output dimension.
        num_heads (int): Attention heads.
        downsample_rate (int): Internal dim = embedding_dim // downsample_rate.
    """

    def __init__(self, embedding_dim: int, num_heads: int, downsample_rate: int = 1):
        super().__init__()

        self.num_heads = num_heads
        self.internal_dim = embedding_dim // downsample_rate
        assert self.internal_dim % num_heads == 0

        self.head_dim = self.internal_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, D) -> (B, H, N, D/H)"""
        B, N, D = x.shape
        return x.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def _recombine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, N, D/H) -> (B, N, D)"""
        B, H, N, Dh = x.shape
        return x.permute(0, 2, 1, 3).reshape(B, N, H * Dh)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        Q = self._separate_heads(self.q_proj(q))   # (B, H, N_q, Dh)
        K = self._separate_heads(self.k_proj(k))   # (B, H, N_k, Dh)
        V = self._separate_heads(self.v_proj(v))   # (B, H, N_k, Dh)

        attn = torch.einsum("bhid,bhjd->bhij", Q, K) * self.scale  # (B, H, N_q, N_k)
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attn, V)              # (B, H, N_q, Dh)
        return self.out_proj(self._recombine_heads(out))             # (B, N_q, D)


class MLP(nn.Module):
    """Simple multi-layer perceptron (FFN)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, depth: int):
        super().__init__()
        layers = []
        in_d = input_dim
        for i in range(depth - 1):
            layers += [nn.Linear(in_d, hidden_dim), nn.GELU()]
            in_d = hidden_dim
        layers.append(nn.Linear(in_d, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) conditioning.

    Modulates feature statistics (scale and shift) based on a conditioning signal.
    Used to condition the decoder on modality information.

    FiLM: y = gamma(c) * x + beta(c)
    where c is the conditioning signal (modality embedding).

    Reference: Perez et al., "FiLM: Visual Reasoning with a General Conditioning
    Layer", AAAI 2018.
    """

    def __init__(self, feature_dim: int, condition_dim: int):
        super().__init__()
        # Predict scale (gamma) and shift (beta) from condition
        self.gamma_proj = nn.Linear(condition_dim, feature_dim)
        self.beta_proj = nn.Linear(condition_dim, feature_dim)
        # Initialise to identity (gamma=1, beta=0) for stable training start.
        # For gamma: W=0, bias=1 → gamma(c) = 0·c + 1 = 1 (identity scale).
        # For beta:  W=0, bias=0 → beta(c) = 0·c + 0 = 0 (no shift).
        # BUG FIX: Previous code used nn.init.ones_(gamma_proj.weight) which
        # produces gamma = sum(condition) ≈ large value, NOT 1.
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) — feature map to modulate
            condition: (B, condition_dim) — conditioning signal

        Returns:
            x_modulated: (B, C, H, W)
        """
        gamma = self.gamma_proj(condition)[:, :, None, None]  # (B, C, 1, 1)
        beta = self.beta_proj(condition)[:, :, None, None]     # (B, C, 1, 1)
        return gamma * x + beta


class MaskDecoder(nn.Module):
    """
    Two-way transformer mask decoder with FiLM conditioning and hierarchical
    skip connections from the encoder's Stage 1 features.

    Processes image features + prompt embeddings to produce:
      - num_multimask_outputs binary masks per sample
      - IoU quality score per mask

    The best mask is selected at inference using the IoU scores.

    Hierarchical skip connection (V3+ addition):
      Stage 1 features (32×32 at 256px) are injected after the first
      ConvTranspose2d upsample step, which also produces 32×32 features.
      The skip is gated by a learnable scalar (init=0, model opens it gradually)
      and edge-enhanced via zero-parameter Haar wavelet high-frequency extraction
      before injection.  This gives the decoder fine-grained boundary cues that
      the 16×16 bottleneck loses during the two-way transformer step.

    Args:
        embed_dim (int): Feature embedding dimension.
        num_multimask_outputs (int): Number of mask candidates (SAM uses 3).
        transformer_depth (int): Number of two-way attention blocks.
        transformer_mlp_dim (int): FFN hidden dimension in transformer.
        iou_head_depth (int): Depth of IoU prediction MLP.
        iou_head_hidden_dim (int): Width of IoU prediction MLP.
        num_modalities (int): Number of modalities for FiLM conditioning.
        skip_channels (int): Channels of Stage 1 skip features (embed_dim//4=64).
                             0 disables skip connection (backward compat).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_multimask_outputs: int = 3,
        transformer_depth: int = 2,
        transformer_mlp_dim: int = 2048,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        num_modalities: int = 11,
        skip_channels: int = 0,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_multimask_outputs = num_multimask_outputs
        num_mask_tokens = num_multimask_outputs + 1  # +1 for single-mask mode

        # ----- Learnable mask tokens (query tokens for the transformer) -----
        # Each token will decode into one mask prediction
        self.iou_token = nn.Embedding(1, embed_dim)
        self.mask_tokens = nn.Embedding(num_mask_tokens, embed_dim)

        # ----- Two-way transformer -----
        self.transformer = TwoWayTransformer(
            depth=transformer_depth,
            embedding_dim=embed_dim,
            num_heads=8,
            mlp_dim=transformer_mlp_dim,
        )

        # ----- FiLM conditioning (modality) -----
        # We use the modality embedding from sparse prompts as conditioning signal
        self.film = FiLMLayer(feature_dim=embed_dim, condition_dim=embed_dim)

        # ----- Upsampling head: recover spatial resolution -----
        # Feature map is H/16 x W/16; we upsample 4x to H/4 x W/4
        # Then apply a convolution to produce the final mask logit
        # NOTE: LayerNorm is applied dynamically in forward() because spatial
        # dims vary with input size.  We store the components individually.
        self.upsample_conv1 = nn.ConvTranspose2d(embed_dim, embed_dim // 4, kernel_size=2, stride=2)
        self.upsample_ln = nn.LayerNorm(embed_dim // 4)  # channel-wise, applied in forward
        self.upsample_conv2 = nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=2, stride=2)

        # ----- Hierarchical skip connection from Stage 1 encoder features -----
        # skip_channels == embed_dim // 4 == 64 (projected in MultiScaleEncoder)
        # We inject AFTER upsample_conv1 which also outputs embed_dim // 4 at 32×32.
        self.use_skip = skip_channels > 0
        if self.use_skip:
            # Learnable gate: starts at 0 (closed), model gradually opens it.
            # Using raw parameter (not sigmoid/tanh) so model can freely tune magnitude.
            self.skip_gate = nn.Parameter(torch.zeros(1))
            # LayerNorm to stabilise skip features before injection
            self.skip_ln = nn.LayerNorm(skip_channels)

        # MLP to transform each mask token into per-pixel prediction weights
        self.output_hypernetworks_mlps = nn.ModuleList([
            MLP(embed_dim, embed_dim, embed_dim // 8, depth=3)
            for _ in range(num_mask_tokens)
        ])

        # ----- IoU prediction head -----
        self.iou_prediction_head = MLP(
            embed_dim, iou_head_hidden_dim, num_mask_tokens, depth=iou_head_depth
        )

        # ----- Modality classification head (auxiliary loss, optional) -----
        # Predicts the modality from the decoded features — regularises the
        # modality embedding to be discriminative
        self.modality_head = MLP(embed_dim, embed_dim // 2, num_modalities, depth=2)

        # ----- HQ-SAM output token (NeurIPS 2023, arXiv:2306.01567) -----
        self.hq_token = nn.Embedding(1, embed_dim)

        # MLP: maps hq_token_out → embed_dim//8 (same dim as mask hypernetworks)
        self.hq_mlp = MLP(embed_dim, embed_dim, embed_dim // 8, depth=3)

        # Projection: fuse Stage 1 features (skip_channels=64) → embed_dim//8
        hq_fuse_in = (skip_channels if skip_channels > 0 else embed_dim // 4)
        self.hq_skip_proj = nn.Sequential(
            nn.Conv2d(hq_fuse_in, embed_dim // 8, kernel_size=1),
            nn.GELU(),
        )
        # Zero-init the feature projection only. hq_mask = hq_weights @ hq_feat,
        # so zeroing hq_feat at init guarantees hq_mask = 0 regardless of hq_weights.
        # This is the neutral start — model opens the path as training progresses.
        nn.init.zeros_(self.hq_skip_proj[0].weight)
        nn.init.zeros_(self.hq_skip_proj[0].bias)

    # ------------------------------------------------------------------
    @staticmethod
    def _haar_edge_enhance(x: torch.Tensor) -> torch.Tensor:
        """
        Zero-parameter Haar wavelet edge enhancement.

        Extracts high-frequency (edge) content from skip features and adds it
        back, amplifying boundary detail before decoder injection.

        The Haar DWT decomposes x (B, C, 2H, 2W) into 4 subbands at (H, W):
          LL: low-pass   (smooth regions)
          LH: horizontal edges
          HL: vertical edges
          HH: diagonal edges

        We amplify edge subbands and upsample them back to the skip resolution.
        This is a zero-parameter alternative to learned edge attention — inspired
        by SAMwave (BMVC 2025) but applied to skip connections in 3D medical SAM.
        """
        # x: (B, C, 2H, 2W)
        x00 = x[:, :, 0::2, 0::2]   # top-left
        x01 = x[:, :, 0::2, 1::2]   # top-right
        x10 = x[:, :, 1::2, 0::2]   # bottom-left
        x11 = x[:, :, 1::2, 1::2]   # bottom-right

        LH = (x00 - x01 + x10 - x11) * 0.25   # horizontal detail
        HL = (x00 + x01 - x10 - x11) * 0.25   # vertical detail
        HH = (x00 - x01 - x10 + x11) * 0.25   # diagonal detail

        # Sum absolute HF content — energy at each (h, w) position
        hf = LH.abs() + HL.abs() + HH.abs()  # (B, C, H, W)

        # Upsample HF map back to original resolution and add to skip
        hf_up = F.interpolate(hf, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return x + 0.5 * hf_up  # 0.5 weight keeps HF as additive hint, not dominant

    def forward(
        self,
        image_embeddings: torch.Tensor,   # (B, embed_dim, H_feat, W_feat)
        sparse_prompt_embeddings: torch.Tensor,  # (B, N_sparse, embed_dim)
        dense_prompt_embeddings: torch.Tensor,   # (B, embed_dim, H_feat, W_feat)
        modality_ids: torch.Tensor = None,        # (B,) for FiLM conditioning
        multimask_output: bool = True,
        skip_features: torch.Tensor = None,      # (B, skip_ch, 2*H_feat, 2*W_feat) Stage 1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            image_embeddings: (B, embed_dim, H_feat, W_feat)
            sparse_prompt_embeddings: (B, N_sparse, embed_dim)
            dense_prompt_embeddings: (B, embed_dim, H_feat, W_feat)
            modality_ids: (B,) optional, for FiLM conditioning
            multimask_output: True returns all 3 masks; False returns best 1

        Returns:
            masks: (B, K, H_feat*4, W_feat*4) — K mask logits
            iou_pred: (B, K) — IoU quality predictions
            modality_logits: (B, num_modalities) — auxiliary modality prediction
        """
        B, C, H_feat, W_feat = image_embeddings.shape

        # ----- Prepare output tokens -----
        # Concatenate [iou_token, mask_tokens] → query tokens
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight, self.hq_token.weight], dim=0
        )  # (1 + num_mask_tokens + 1, embed_dim)
        output_tokens = output_tokens.unsqueeze(0).expand(B, -1, -1)
        # (B, 1 + num_mask_tokens + 1, embed_dim)  — +1 for HQ token

        # Prepend output tokens to sparse prompt embeddings
        tokens = torch.cat([output_tokens, sparse_prompt_embeddings], dim=1)
        # (B, 1 + num_mask_tokens + 1 + N_sparse, embed_dim)

        # ----- Apply FiLM conditioning to image features -----
        # Extract modality embedding from sparse prompts (token index 2 = modality)
        modality_emb = sparse_prompt_embeddings[:, 2, :]  # (B, embed_dim) — modality token
        src = self.film(image_embeddings, modality_emb)   # (B, embed_dim, H_feat, W_feat)

        # Add dense prompt (positional) to image features
        src = src + dense_prompt_embeddings  # (B, embed_dim, H_feat, W_feat)

        # ----- Two-way transformer -----
        # Flatten image features to sequence: (B, H_feat*W_feat, embed_dim)
        src_flat = src.flatten(2).permute(0, 2, 1)  # (B, H*W, embed_dim)

        # Dense PE for image tokens (same as the dense prompt)
        pos_src = dense_prompt_embeddings.flatten(2).permute(0, 2, 1)  # (B, H*W, embed_dim)

        # Tokens PE (zeros for mask/iou tokens; sparse prompts carry their own PE)
        token_pe = torch.zeros_like(tokens)

        # Run two-way transformer
        hs, src_flat = self.transformer(src_flat, pos_src, tokens, token_pe)
        assert hs.shape[1] > (1 + self.num_multimask_outputs + 1), (
            f"Transformer output has {hs.shape[1]} tokens; expected at least "
            f"{1 + self.num_multimask_outputs + 2} (iou + {self.num_multimask_outputs + 1} mask + hq)"
        )
        # hs: (B, num_tokens, embed_dim) — updated prompt tokens
        # src_flat: (B, H*W, embed_dim) — updated image tokens

        # Extract iou token output and mask token outputs
        iou_token_out = hs[:, 0, :]                        # (B, embed_dim)
        mask_tokens_out = hs[:, 1:(1 + self.num_multimask_outputs + 1), :]
        # (B, num_mask_tokens, embed_dim)
        hq_token_out = hs[:, (1 + self.num_multimask_outputs + 1), :]  # (B, embed_dim)

        # ----- Upscale image features -----
        src_2d = src_flat.permute(0, 2, 1).reshape(B, C, H_feat, W_feat)
        # Apply upsampling (4x total)
        upscaled = self.upsample_conv1(src_2d)           # (B, C//4, H*2, W*2)
        # LayerNorm on channel dim — reshape to (B*H*W, C) for LayerNorm
        B_, C_, H_, W_ = upscaled.shape
        upscaled = upscaled.permute(0, 2, 3, 1).reshape(-1, C_)
        upscaled = self.upsample_ln(upscaled)            # learned affine LN
        upscaled = upscaled.reshape(B_, H_, W_, C_).permute(0, 3, 1, 2)
        upscaled = F.gelu(upscaled)

        # ----- Hierarchical skip connection (Stage 1 → 32×32) -----
        # skip_features: (B, embed_dim//4, H*2, W*2) — same spatial size as upscaled
        # Edge-enhanced via Haar HF extraction, then gated addition.
        # skip_gate starts at 0 (closed); model learns when/how much to open it.
        if self.use_skip and skip_features is not None:
            # Resize skip to match upscaled in case of minor size mismatch
            if skip_features.shape[-2:] != upscaled.shape[-2:]:
                skip_features = F.interpolate(
                    skip_features.float(), size=upscaled.shape[-2:],
                    mode='bilinear', align_corners=False
                ).to(upscaled.dtype)
            # Normalise skip features channel-wise
            sf = skip_features.permute(0, 2, 3, 1).reshape(-1, skip_features.shape[1])
            sf = self.skip_ln(sf)
            sf = sf.reshape(skip_features.shape[0], *skip_features.shape[2:],
                            skip_features.shape[1]).permute(0, 3, 1, 2)
            # Haar edge enhancement — zero params, amplifies organ boundaries
            sf = self._haar_edge_enhance(sf)
            # Gated addition
            upscaled = upscaled + self.skip_gate * sf

        upscaled = self.upsample_conv2(upscaled)         # (B, C//8, H*4, W*4)
        upscaled = F.gelu(upscaled)
        # upscaled: (B, embed_dim//8, H_feat*4, W_feat*4)

        # ----- Predict masks -----
        # Each mask token generates per-pixel prediction weights via a small MLP
        # Then dot-product with upscaled features to get mask logits
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_multimask_outputs + 1):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )  # (B, embed_dim//8)

        hyper_in = torch.stack(hyper_in_list, dim=1)  # (B, num_mask_tokens, embed_dim//8)

        b, c, h, w = upscaled.shape
        masks = (hyper_in @ upscaled.view(b, c, h * w)).view(b, -1, h, w)
        # (B, num_mask_tokens, H_feat*4, W_feat*4)

        # ----- HQ correction mask -----
        hq_weights = self.hq_mlp(hq_token_out)  # (B, embed_dim//8)

        if skip_features is not None:
            skip_for_hq = skip_features
            if skip_for_hq.shape[-2:] != upscaled.shape[-2:]:
                skip_for_hq = F.interpolate(
                    skip_for_hq.float(), size=upscaled.shape[-2:],
                    mode='bilinear', align_corners=False,
                ).to(upscaled.dtype)
            hq_feat = self.hq_skip_proj(skip_for_hq)  # (B, embed_dim//8, H, W)
        else:
            hq_feat = torch.zeros_like(upscaled)  # (B, embed_dim//8, H, W)

        # Dot-product: hq_weights (B, C) × hq_feat (B, C, H, W) → (B, 1, H, W)
        b_hq, c_hq, h_hq, w_hq = hq_feat.shape
        hq_mask = (hq_weights.unsqueeze(1) @ hq_feat.view(b_hq, c_hq, h_hq * w_hq)).view(b_hq, 1, h_hq, w_hq)

        # Add HQ correction to all mask candidates
        masks = masks + hq_mask

        # ----- IoU prediction -----
        iou_pred = self.iou_prediction_head(iou_token_out)  # (B, num_mask_tokens)

        # ----- Auxiliary modality prediction -----
        modality_logits = self.modality_head(iou_token_out)  # (B, num_modalities)

        # ----- Select masks to return -----
        if multimask_output:
            masks = masks[:, 1:, :, :]       # Return 3 candidate masks (skip index 0)
            iou_pred = iou_pred[:, 1:]
        else:
            # Single-mask mode: pick the mask with highest predicted IoU
            best_idx = iou_pred[:, 1:].argmax(dim=1) + 1  # Offset by 1 (skip index 0)
            masks = masks[
                torch.arange(B, device=masks.device), best_idx
            ].unsqueeze(1)  # (B, 1, H, W)
            iou_pred = iou_pred[
                torch.arange(B, device=iou_pred.device), best_idx
            ].unsqueeze(1)  # (B, 1)

        return masks, iou_pred, modality_logits


class TwoWayTransformer(nn.Module):
    """
    Stack of TwoWayAttentionBlocks with a final cross-attention layer.

    Args:
        depth (int): Number of TwoWayAttentionBlocks.
        embedding_dim (int): Feature dimension.
        num_heads (int): Attention heads.
        mlp_dim (int): FFN hidden dimension.
        attention_downsample_rate (int): Reduce internal attention dim.
    """

    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        attention_downsample_rate: int = 2,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            TwoWayAttentionBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                attention_downsample_rate=attention_downsample_rate,
            )
            for _ in range(depth)
        ])

        # Final cross-attention: token queries attend to image keys
        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: torch.Tensor,   # (B, N_image, embed_dim) — flattened feat map
        image_pe: torch.Tensor,           # (B, N_image, embed_dim)
        point_embedding: torch.Tensor,    # (B, N_tokens, embed_dim) — prompt tokens
        point_pe: torch.Tensor,           # (B, N_tokens, embed_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Note: argument naming follows SAM convention (image_embedding = keys,
        point_embedding = queries).
        """
        queries = point_embedding
        keys = image_embedding

        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_pe,
                key_pe=image_pe,
            )

        # Final cross-attention: queries attend to keys one more time
        q = queries + point_pe
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = self.norm_final_attn(queries + attn_out)

        return queries, keys
