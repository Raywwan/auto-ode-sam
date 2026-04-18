"""Anatomy-Graph Query Decoder.

Differences from V3 `OrganQueryDecoder`:
  - Anatomy-graph message pass on organ queries before the TwoWayTransformer
  - SINGLE shared mask MLP (V3 had 15 separate, which collapsed to identical weights)
  - SINGLE shared IoU MLP
  - HQ token + hq_skip_proj DELETED (dead in every V3 checkpoint)
  - Stage-1 skip + Haar edge enhancement kept (V3 feature that worked)
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mask_decoder import TwoWayTransformer, MLP


# AMOS22 anatomical adjacency prior (1-indexed organ IDs).
# Pairs are symmetric. (organ_a, organ_b).
AMOS22_ADJACENCY = [
    (6, 2), (6, 7), (6, 4),           # liver — right kidney, stomach, gallbladder
    (1, 3), (3, 12), (2, 11),         # spleen — left kidney, kidneys — adrenals
    (8, 9),                           # aorta — IVC
    (10, 7), (10, 13), (7, 13),       # pancreas — stomach, duodenum; stomach — duodenum
    (14, 15),                         # bladder — prostate/uterus
    (5, 7),                           # esophagus — stomach
]


def _build_adjacency_init(n_organs: int) -> torch.Tensor:
    """15×15 tensor of adjacency logits. Init ~2.0 for adjacent, ~-2.0 otherwise.

    sigmoid(2) ≈ 0.88, sigmoid(-2) ≈ 0.12 — nonzero everywhere so gradients flow.
    """
    A = torch.full((n_organs, n_organs), -2.0)
    for a, b in AMOS22_ADJACENCY:
        A[a - 1, b - 1] = 2.0
        A[b - 1, a - 1] = 2.0
    A.fill_diagonal_(0.0)  # no self-loops in message pass
    return A


class AnatomyGraphAttention(nn.Module):
    """One GAT-style message pass with learnable 15×15 adjacency."""

    def __init__(self, n_organs: int = 15, embed_dim: int = 256) -> None:
        super().__init__()
        self.n_organs = n_organs
        self.adjacency = nn.Parameter(_build_adjacency_init(n_organs))
        # Frozen reference for the L_anatomy regularizer (||A - A_init||₂).
        self.register_buffer("_init_adjacency", _build_adjacency_init(n_organs).clone())
        # Off-diagonal mask — zeros the self-loop column so an organ never attends
        # to itself via the message pass (sigmoid(0)=0.5 would otherwise be a 50% self-loop).
        self.register_buffer("_offdiag_mask", 1.0 - torch.eye(n_organs))
        self.w_msg = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.zeros_(self.w_msg.weight)  # start as identity residual

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """q: (n_organs, embed_dim) → (n_organs, embed_dim)."""
        A_soft = torch.sigmoid(self.adjacency) * self._offdiag_mask
        m = A_soft @ q
        return q + self.w_msg(m)


class AnatomyGraphDecoder(nn.Module):
    """TwoWayTransformer + anatomy-graph queries + shared mask head."""

    def __init__(
        self,
        embed_dim: int = 256,
        n_organs: int = 15,
        skip_channels: int = 128,
        transformer_depth: int = 4,
        transformer_mlp_dim: int = 2048,
    ) -> None:
        super().__init__()
        self.n_organs = n_organs
        self.organ_queries = nn.Embedding(n_organs, embed_dim)
        nn.init.normal_(self.organ_queries.weight, std=0.02)
        self.graph = AnatomyGraphAttention(n_organs, embed_dim)
        self.transformer = TwoWayTransformer(
            depth=transformer_depth, embedding_dim=embed_dim,
            num_heads=8, mlp_dim=transformer_mlp_dim,
        )
        self.upsample_conv1 = nn.ConvTranspose2d(embed_dim, embed_dim // 4, kernel_size=2, stride=2)
        self.upsample_ln = nn.LayerNorm(embed_dim // 4)
        self.upsample_conv2 = nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=2, stride=2)
        self.skip_gate = nn.Parameter(torch.full((1,), 0.05))
        self.skip_ln = nn.LayerNorm(skip_channels)
        self.skip_proj = nn.Conv2d(skip_channels, embed_dim // 4, kernel_size=1)
        self.mask_mlp = MLP(embed_dim, embed_dim, embed_dim // 8, depth=3)
        self.iou_mlp = MLP(embed_dim, 256, 1, depth=3)

    @staticmethod
    def _haar_edge(x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2], x.shape[-1]
        pad_h, pad_w = h % 2, w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        x00 = x[:, :, 0::2, 0::2]; x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]; x11 = x[:, :, 1::2, 1::2]
        LH = (x00 - x01 + x10 - x11) * 0.25
        HL = (x00 + x01 - x10 - x11) * 0.25
        HH = (x00 - x01 - x10 + x11) * 0.25
        hf = LH.abs() + HL.abs() + HH.abs()
        up = F.interpolate(hf, size=(h + pad_h, w + pad_w), mode="bilinear", align_corners=False)
        return (x + 0.5 * up)[:, :, :h, :w]

    def forward(
        self,
        image_emb: torch.Tensor,   # (B, C, H, W)
        dense_pe: torch.Tensor,    # (B, C, H, W)
        skip: torch.Tensor,        # (B, skip_ch, 2H, 2W)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = image_emb.shape
        q = self.organ_queries.weight          # (15, C)
        q = self.graph(q)                       # anatomy-graph message pass
        queries = q.unsqueeze(0).expand(B, -1, -1).contiguous()
        q_pe = torch.zeros_like(queries)
        src = image_emb.flatten(2).permute(0, 2, 1)
        pe = dense_pe.flatten(2).permute(0, 2, 1)
        hs, src_updated = self.transformer(src, pe, queries, q_pe)
        src_spatial = src_updated.permute(0, 2, 1).reshape(B, C, H, W)
        up = self.upsample_conv1(src_spatial)
        up = up.permute(0, 2, 3, 1)
        up = self.upsample_ln(up)
        up = up.permute(0, 3, 1, 2)
        up = F.gelu(up)
        if skip.shape[-2:] != up.shape[-2:]:
            skip = F.interpolate(skip.float(), size=up.shape[-2:], mode="bilinear", align_corners=False).to(up.dtype)
        sf_flat = skip.permute(0, 2, 3, 1)
        sf_flat = self.skip_ln(sf_flat).permute(0, 3, 1, 2)
        sf = self._haar_edge(sf_flat)
        sf = self.skip_proj(sf)
        up = up + self.skip_gate * sf
        up = F.gelu(self.upsample_conv2(up))
        b, c, h, w = up.shape
        weights = self.mask_mlp(hs[:, :self.n_organs, :])          # (B, 15, C//8)
        masks = (weights @ up.view(b, c, h * w)).view(b, self.n_organs, h, w)
        iou = self.iou_mlp(hs[:, :self.n_organs, :]).squeeze(-1)   # (B, 15)
        return masks, iou
