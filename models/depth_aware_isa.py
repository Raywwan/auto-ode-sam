# C:\Users\Raywa\Desktop\LiteSAM3D_v2\models\depth_aware_isa.py
# =============================================================================
# Depth-Aware Inter-Slice Attention (DA-ISA) Module
#
# Improvement over the original ISA (LiteSAM-3D v1):
#
# PROBLEM WITH ORIGINAL ISA:
#   ISA treats all depth positions identically — a query at slice 5 attending
#   to slices 4,5,6 produces the same attention pattern (modulo content) as
#   a query at slice 50 attending to 49,50,51. There is no positional signal
#   for WHERE in the volume a slice lives.
#
# DA-ISA ADDITIONS:
#   1. Absolute Depth Positional Encoding (sinusoidal + linear projection)
#      - Injected into Q and K projections so the model knows which depth
#        position each slice occupies.
#      - Sinusoidal (not learned) for generalization to unseen volume depths.
#      - Mathematical form:
#          PE(pos, 2i)   = sin(pos / 10000^(2i/d))
#          PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
#          depth_pe = Linear(PE(pos)) added to Q, K after projection.
#
#   2. Relative Depth Bias (learnable scalar per relative distance)
#      - Added to attention logits before softmax: A_ij += bias[|i-j|]
#      - Analogous to relative position bias in Swin Transformer (Liu et al.)
#      - Learns a prior: "how important is a context slice that is k steps
#        away?" Independent of content.
#
# EFFICIENCY IMPROVEMENT:
#   Original ISA loops over each context slice individually (N_context forward
#   passes of the attention projections per query slice). DA-ISA batches ALL
#   context slices into a single attention call per query slice using
#   F.scaled_dot_product_attention with proper reshaping.
#
# COMPLEXITY: O(N * w) per layer — same asymptotic cost as original ISA.
#   The depth PE adds O(D * d) for sinusoidal generation + O(D * d^2) for
#   linear projection — negligible compared to attention.
#
# ABLATION COMPATIBILITY:
#   Setting use_depth_pe=False and use_relative_bias=False recovers original
#   ISA behaviour (with the efficiency improvement retained).
#
# Reference:
#   - Sinusoidal PE: Vaswani et al., "Attention Is All You Need", NeurIPS 2017
#   - Relative bias: Liu et al., "Swin Transformer", ICCV 2021
#   - Axial attention: Ho et al., "Axial Attention in Multidimensional
#     Transformers", arXiv 2019
# =============================================================================

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sinusoidal_encoding(max_len: int, dim: int, device: torch.device) -> torch.Tensor:
    """
    Generate sinusoidal positional encoding table.

    Args:
        max_len: Maximum sequence length (number of depth positions).
        dim: Encoding dimension.
        device: Target device.

    Returns:
        Tensor of shape (max_len, dim) with sinusoidal values.

    >>> enc = _sinusoidal_encoding(10, 64, torch.device('cpu'))
    >>> enc.shape
    torch.Size([10, 64])
    >>> float(enc[0, 0])  # sin(0) = 0
    0.0
    """
    position = torch.arange(max_len, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32, device=device)
        * -(math.log(10000.0) / dim)
    )
    pe = torch.zeros(max_len, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    # For even dim: 1::2 has dim//2 slots, div_term has dim//2 elements → exact match.
    # For odd dim: 1::2 has dim//2 slots, div_term has (dim+1)//2 elements → slice to fit.
    pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
    return pe


class DepthAwareISA(nn.Module):
    """
    Single Depth-Aware Inter-Slice Attention layer.

    For a stack of D slices (each with B*H*W spatial tokens), each slice
    attends to its neighbouring slices within a local window. Optionally
    enriched with absolute depth positional encoding and relative depth bias.

    Args:
        dim: Feature dimension (embed_dim from encoder).
        num_heads: Number of attention heads.
        window_size: Local context window size (odd recommended).
            window_size=3 means each slice sees +/-1 neighbour.
        dropout: Dropout on attention weights and FFN.
        use_ff: Whether to include a Feed-Forward Network after attention.
        ff_mult: FFN hidden dimension multiplier.
        use_depth_pe: Enable absolute depth positional encoding.
        use_relative_bias: Enable relative depth bias in attention logits.
        max_depth: Maximum supported volume depth for sinusoidal PE table.
        max_window_size: Maximum window size for relative bias table.
            Supports future ablations without re-initializing.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: int = 3,
        dropout: float = 0.1,
        use_ff: bool = True,
        ff_mult: float = 4.0,
        use_depth_pe: bool = True,
        use_relative_bias: bool = True,
        per_head_bias: bool = False,
        max_depth: int = 512,
        max_window_size: int = 5,
    ) -> None:
        super().__init__()

        assert dim % num_heads == 0, (
            f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        )

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5
        self.use_ff = use_ff
        self.use_depth_pe = use_depth_pe
        self.use_relative_bias = use_relative_bias
        self.per_head_bias = per_head_bias
        self.max_depth = max_depth
        self.max_window_size = max_window_size

        # ----- Attention projections -----
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=True)

        self.attn_drop = nn.Dropout(dropout)

        # ----- Pre-norm layers -----
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)

        # ----- Absolute Depth Positional Encoding -----
        if use_depth_pe:
            # Sinusoidal base (not learned) for generalization to variable depths.
            # Projected through a small linear layer to match dim.
            # We use dim as the sinusoidal dimension for simplicity; the linear
            # layer learns to select/combine frequencies.
            pe_dim = dim  # sinusoidal encoding dimension
            self.register_buffer(
                "_sinusoidal_table",
                _sinusoidal_encoding(max_depth, pe_dim, torch.device("cpu")),
                persistent=False,
            )
            # Project sinusoidal PE to model dim (lightweight: dim -> dim)
            self.depth_pe_proj = nn.Linear(pe_dim, dim, bias=False)
            # Initialize near-zero so PE is a small perturbation at init
            nn.init.normal_(self.depth_pe_proj.weight, std=0.02)

        # ----- Relative Depth Bias -----
        if use_relative_bias:
            # Learnable bias per relative distance from -max_window_size to
            # +max_window_size.  Index 0 = distance -max_window_size.
            # Total entries: 2 * max_window_size + 1.
            #
            # per_head_bias=True: independent bias per head (Swin-v2 style).
            #   Shape: (num_heads, 2*max_window_size+1).  Allows each head to
            #   learn different depth-distance preferences.
            # per_head_bias=False (default): shared scalar across heads.
            #   Shape: (2*max_window_size+1,).
            bias_shape = (
                (num_heads, 2 * max_window_size + 1) if per_head_bias
                else (2 * max_window_size + 1,)
            )
            self.relative_bias = nn.Parameter(torch.zeros(*bias_shape))
            nn.init.normal_(self.relative_bias, std=0.01)

        # ----- Optional Feed-Forward Network -----
        if use_ff:
            ff_dim = int(dim * ff_mult)
            self.ff = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, ff_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, dim),
                nn.Dropout(dropout),
            )

    def _get_depth_pe(self, depth_indices: torch.Tensor) -> torch.Tensor:
        """
        Retrieve and project absolute depth positional encodings.

        Args:
            depth_indices: Integer tensor of shape (n,) with depth positions.

        Returns:
            Projected PE of shape (n, dim).
        """
        # Clamp to valid range
        indices = depth_indices.clamp(0, self.max_depth - 1).long()
        sinusoidal = self._sinusoidal_table[indices]  # (n, pe_dim)
        return self.depth_pe_proj(sinusoidal)  # (n, dim)

    def _get_relative_bias(
        self, query_depth: int, context_depths: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute relative depth bias for attention logits.

        Args:
            query_depth: Absolute depth index of the query slice.
            context_depths: Tensor of shape (n_ctx,) with context depth indices.

        Returns:
            If per_head_bias=False: (n_ctx,) — one scalar per context slice.
            If per_head_bias=True:  (H, n_ctx) — one scalar per (head, context).
        """
        # Relative distances: positive means context is deeper than query
        rel_dist = context_depths - query_depth  # (n_ctx,)
        # Clamp to table range and shift to non-negative index
        rel_idx = (rel_dist + self.max_window_size).clamp(
            0, 2 * self.max_window_size
        ).long()
        if self.per_head_bias:
            return self.relative_bias[:, rel_idx]  # (H, n_ctx)
        return self.relative_bias[rel_idx]  # (n_ctx,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply depth-aware inter-slice cross-attention.

        Args:
            x: Tensor of shape (D, B, C, H, W).
                D = number of slices in the volume.
                B = batch size.
                C = feature channels (must equal self.dim).
                H, W = spatial feature map dimensions.

        Returns:
            Tensor of shape (D, B, C, H, W) with 3D-context-enriched features.

        The algorithm for each query slice i:
            1. Gather context slices within window [i - w//2, i + w//2].
            2. Pre-norm query and context tokens.
            3. Project to Q, K, V.
            4. (Optional) Add absolute depth PE to Q and K.
            5. Compute attention: softmax(Q K^T / sqrt(d) + rel_bias) V.
            6. Residual + LayerNorm.
            7. (Optional) FFN with residual.
        """
        D, B, C, H, W = x.shape
        N = H * W  # spatial tokens per slice
        device = x.device
        half_w = self.window_size // 2

        # Flatten spatial: (D, B*N, C)
        x_flat = x.permute(0, 1, 3, 4, 2).reshape(D, B * N, C)

        # Pre-compute depth PE for all slices if enabled
        if self.use_depth_pe:
            all_depth_indices = torch.arange(D, device=device)
            all_depth_pe = self._get_depth_pe(all_depth_indices)  # (D, dim)

        out_flat = torch.zeros_like(x_flat)

        for i in range(D):
            # ---- Window bounds ----
            w_start = max(0, i - half_w)
            w_end = min(D, i + half_w + 1)
            n_ctx = w_end - w_start

            # ---- Gather tokens ----
            query = x_flat[i]  # (B*N, C)
            context = x_flat[w_start:w_end]  # (n_ctx, B*N, C)

            # ---- Pre-norm ----
            q_norm = self.norm_q(query)  # (B*N, C)
            kv_norm = self.norm_kv(context)  # (n_ctx, B*N, C)

            # ---- Project Q, K, V ----
            Q = self.q_proj(q_norm)  # (B*N, C)
            K = self.k_proj(kv_norm.reshape(n_ctx * B * N, C))  # (n_ctx*B*N, C)
            V = self.v_proj(kv_norm.reshape(n_ctx * B * N, C))  # (n_ctx*B*N, C)

            # ---- Add absolute depth PE to Q and K ----
            if self.use_depth_pe:
                # Q: add PE for depth i, broadcast over all B*N tokens
                q_pe = all_depth_pe[i]  # (dim,)
                Q = Q + q_pe.unsqueeze(0)  # (B*N, C)

                # K: add PE for each context depth, broadcast over B*N tokens
                ctx_depths = torch.arange(w_start, w_end, device=device)
                k_pe = all_depth_pe[ctx_depths]  # (n_ctx, dim)
                # Expand to (n_ctx, B*N, dim) then reshape to (n_ctx*B*N, dim)
                k_pe_expanded = k_pe.unsqueeze(1).expand(n_ctx, B * N, C)
                K = K + k_pe_expanded.reshape(n_ctx * B * N, C)

            # ---- Reshape for multi-head attention ----
            # Q: (B*N, C) -> (B*N, H, d) -> (B*N, H, 1, d)
            Q = Q.view(B * N, self.num_heads, self.head_dim).unsqueeze(2)
            # K: (n_ctx*B*N, C) -> (n_ctx, B*N, H, d) -> (B*N, H, n_ctx, d)
            K = K.view(n_ctx, B * N, self.num_heads, self.head_dim)
            K = K.permute(1, 2, 0, 3)  # (B*N, H, n_ctx, d)
            # V: same reshape as K
            V = V.view(n_ctx, B * N, self.num_heads, self.head_dim)
            V = V.permute(1, 2, 0, 3)  # (B*N, H, n_ctx, d)

            # ---- Compute attention with optional relative bias ----
            if self.use_relative_bias:
                # Compute relative bias per context position
                ctx_depths = torch.arange(w_start, w_end, device=device)
                rel_bias = self._get_relative_bias(i, ctx_depths)
                if self.per_head_bias:
                    # rel_bias: (H, n_ctx) → (1, H, 1, n_ctx) for broadcasting
                    attn_bias = rel_bias.unsqueeze(0).unsqueeze(2)
                else:
                    # rel_bias: (n_ctx,) → (1, 1, 1, n_ctx) for broadcasting
                    attn_bias = rel_bias.view(1, 1, 1, n_ctx)

                # Manual attention with bias (cannot use SDPA with arbitrary bias
                # on all PyTorch versions, so we compute explicitly for correctness)
                attn_logits = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
                # attn_logits: (B*N, H, 1, n_ctx)
                attn_logits = attn_logits + attn_bias
                attn_weights = F.softmax(attn_logits, dim=-1)
                attn_weights = self.attn_drop(attn_weights)
                attended = torch.matmul(attn_weights, V)  # (B*N, H, 1, d)
            else:
                # Use F.scaled_dot_product_attention for flash attention support
                # Q: (B*N, H, 1, d), K: (B*N, H, n_ctx, d), V: (B*N, H, n_ctx, d)
                attended = F.scaled_dot_product_attention(
                    Q, K, V,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                    scale=self.scale,
                )  # (B*N, H, 1, d)

            # ---- Reshape output ----
            # (B*N, H, 1, d) -> (B*N, C)
            attended = attended.squeeze(2).reshape(B * N, C)
            out_flat[i] = self.out_proj(attended)

        # ---- Residual + LayerNorm ----
        out_flat = self.norm_out(out_flat + x_flat)

        # ---- Optional FFN with residual ----
        if self.use_ff:
            out_flat = out_flat + self.ff(out_flat)

        # ---- Reshape back to (D, B, C, H, W) ----
        out = out_flat.reshape(D, B, N, C).permute(0, 1, 3, 2).reshape(D, B, C, H, W)
        return out


class DepthAwareISAStack(nn.Module):
    """
    Stack of N Depth-Aware ISA layers.

    Drop-in replacement for InterSliceAttentionStack from LiteSAM-3D v1.
    With use_depth_pe=False and use_relative_bias=False, this is functionally
    equivalent to the original ISA stack (with improved batched attention).

    Args:
        dim: Feature dimension.
        depth: Number of DA-ISA layers to stack.
        num_heads: Attention heads per layer.
        window_size: Local slice window per layer.
        dropout: Attention and FFN dropout.
        use_depth_pe: Enable absolute depth PE in all layers.
        use_relative_bias: Enable relative depth bias in all layers.
        max_depth: Maximum volume depth for sinusoidal PE table.
        max_window_size: Maximum window size for relative bias table.

    Example:
        >>> import torch
        >>> stack = DepthAwareISAStack(dim=64, depth=2, num_heads=4, window_size=3)
        >>> x = torch.randn(5, 2, 64, 4, 4)  # (D=5, B=2, C=64, H=4, W=4)
        >>> out = stack(x)
        >>> out.shape
        torch.Size([5, 2, 64, 4, 4])
    """

    def __init__(
        self,
        dim: int,
        depth: int = 2,
        num_heads: int = 8,
        window_size: int = 3,
        dropout: float = 0.1,
        use_depth_pe: bool = True,
        use_relative_bias: bool = True,
        per_head_bias: bool = False,
        max_depth: int = 512,
        max_window_size: int = 5,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList([
            DepthAwareISA(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                dropout=dropout,
                use_ff=True,
                ff_mult=4.0,
                use_depth_pe=use_depth_pe,
                use_relative_bias=use_relative_bias,
                per_head_bias=per_head_bias,
                max_depth=max_depth,
                max_window_size=max_window_size,
            )
            for _ in range(depth)
        ])

        # Final layer norm after all DA-ISA layers
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply stacked DA-ISA layers to a volume of slice feature maps.

        Args:
            x: Tensor of shape (D, B, C, H, W).

        Returns:
            Tensor of shape (D, B, C, H, W) with depth-aware 3D features.
        """
        D, B, C, H, W = x.shape

        for layer in self.layers:
            x = layer(x)

        # Apply final norm slice-by-slice
        x_flat = x.permute(0, 1, 3, 4, 2).reshape(D * B, H * W, C)
        x_flat = self.final_norm(x_flat)
        x = x_flat.reshape(D, B, H, W, C).permute(0, 1, 4, 2, 3)

        return x

    def extra_repr(self) -> str:
        """String representation for debugging."""
        layer0 = self.layers[0]
        return (
            f"depth={len(self.layers)}, dim={layer0.dim}, "
            f"window_size={layer0.window_size}, "
            f"use_depth_pe={layer0.use_depth_pe}, "
            f"use_relative_bias={layer0.use_relative_bias}, "
            f"per_head_bias={layer0.per_head_bias}"
        )
