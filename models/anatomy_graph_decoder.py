"""Anatomy-Graph Query Decoder — V6.

V6 upgrades over V5:
- FPN-style multi-scale skip fusion: consumes encoder skip (stride-8, 128ch)
  AND skip_fine (stride-4, 64ch).
- 4-stage upsample: 20→40→80→160→320 (previously 16→32→64→128).
- GroupNorm + ResBlock after each upsample stage.
- Multi-scale deep-supervision heads at 3 decoder scales (40, 80, 160).

V5 elements kept:
- OrganAnatomicalPriorQuery (STEP 9)
- AnatomyGraphAttention message pass
- TwoWayTransformer on main features
- Haar edge enhancement on stride-8 skip
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mask_decoder import TwoWayTransformer, MLP


AMOS22_ADJACENCY = [
    (6, 2), (6, 7), (6, 4),
    (1, 3), (3, 12), (2, 11),
    (8, 9),
    (10, 7), (10, 13), (7, 13),
    (14, 15),
    (5, 7),
]

AMOS22_ORGAN_PRIOR_CENTROIDS = [
    (0.35, 0.70, 0.10),   # 1 spleen
    (0.35, 0.25, 0.07),   # 2 right kidney
    (0.35, 0.75, 0.07),   # 3 left kidney
    (0.45, 0.55, 0.03),   # 4 gallbladder
    (0.30, 0.50, 0.02),   # 5 esophagus
    (0.40, 0.35, 0.20),   # 6 liver
    (0.40, 0.55, 0.08),   # 7 stomach
    (0.50, 0.45, 0.03),   # 8 aorta
    (0.50, 0.55, 0.03),   # 9 IVC
    (0.45, 0.50, 0.06),   # 10 pancreas
    (0.40, 0.35, 0.02),   # 11 right adrenal
    (0.40, 0.65, 0.02),   # 12 left adrenal
    (0.50, 0.55, 0.05),   # 13 duodenum
    (0.80, 0.50, 0.05),   # 14 bladder
    (0.85, 0.50, 0.04),   # 15 prostate/uterus
]


class OrganAnatomicalPriorQuery(nn.Module):
    def __init__(self, n_organs: int, embed_dim: int) -> None:
        super().__init__()
        pos_init = torch.tensor(AMOS22_ORGAN_PRIOR_CENTROIDS[:n_organs], dtype=torch.float32)
        self.pos = nn.Parameter(pos_init)
        self.content = nn.Parameter(torch.empty(n_organs, embed_dim))
        nn.init.normal_(self.content, std=0.02)
        self.pos_proj = nn.Linear(3, embed_dim)
        nn.init.xavier_uniform_(self.pos_proj.weight)
        nn.init.zeros_(self.pos_proj.bias)

    @property
    def weight(self) -> torch.Tensor:
        return self.content + self.pos_proj(self.pos)

    def forward(self) -> torch.Tensor:
        return self.weight


def _build_adjacency_init(n_organs: int) -> torch.Tensor:
    A = torch.full((n_organs, n_organs), -2.0)
    for a, b in AMOS22_ADJACENCY:
        A[a - 1, b - 1] = 2.0
        A[b - 1, a - 1] = 2.0
    A.fill_diagonal_(0.0)
    return A


class AnatomyGraphAttention(nn.Module):
    def __init__(self, n_organs: int = 15, embed_dim: int = 256) -> None:
        super().__init__()
        self.n_organs = n_organs
        self.adjacency = nn.Parameter(_build_adjacency_init(n_organs))
        self.register_buffer("_init_adjacency", _build_adjacency_init(n_organs).clone())
        self.register_buffer("_offdiag_mask", 1.0 - torch.eye(n_organs))
        self.w_msg = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.zeros_(self.w_msg.weight)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        A_soft = torch.sigmoid(self.adjacency) * self._offdiag_mask
        m = A_soft @ q
        return q + self.w_msg(m)


class ResBlock(nn.Module):
    """GN-GELU-Conv3x3 ×2 with residual (1x1 conv if channels change)."""
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8) -> None:
        super().__init__()
        g_in = min(groups, in_ch) if in_ch % groups == 0 else 1
        g_out = min(groups, out_ch) if out_ch % groups == 0 else 1
        self.n1 = nn.GroupNorm(g_in, in_ch)
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.n2 = nn.GroupNorm(g_out, out_ch)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.c1(F.gelu(self.n1(x)))
        h = self.c2(F.gelu(self.n2(h)))
        return h + self.skip(x)


class AnatomyGraphDecoder(nn.Module):
    """Multi-scale FPN decoder with anatomy-graph queries, ResBlock fusion,
    and deep-supervision heads at 3 scales (40, 80, 160 when input=320)."""

    def __init__(
        self,
        embed_dim: int = 256,
        n_organs: int = 15,
        skip_channels: int = 128,
        skip_fine_channels: int = 64,
        transformer_depth: int = 8,
        transformer_mlp_dim: int = 4096,
    ) -> None:
        super().__init__()
        self.n_organs = n_organs
        self.embed_dim = embed_dim
        self.organ_queries = OrganAnatomicalPriorQuery(n_organs, embed_dim)
        self.graph = AnatomyGraphAttention(n_organs, embed_dim)
        self.transformer = TwoWayTransformer(
            depth=transformer_depth, embedding_dim=embed_dim,
            num_heads=8, mlp_dim=transformer_mlp_dim,
        )
        C = embed_dim
        # Stage 1: 20→40, C → C/2, fuse stride-8 skip (skip_channels)
        self.up1 = nn.ConvTranspose2d(C, C // 2, 2, 2)
        self.skip1_proj = nn.Conv2d(skip_channels, C // 2, 1)
        self.skip1_ln = nn.GroupNorm(min(8, skip_channels), skip_channels)
        self.fuse1 = ResBlock(C // 2, C // 2)
        self.skip_gate1 = nn.Parameter(torch.full((1,), 0.5))
        # Stage 2: 40→80, C/2 → C/4, fuse stride-4 skip (skip_fine_channels)
        self.up2 = nn.ConvTranspose2d(C // 2, C // 4, 2, 2)
        self.skip2_proj = nn.Conv2d(skip_fine_channels, C // 4, 1)
        self.skip2_ln = nn.GroupNorm(min(8, skip_fine_channels), skip_fine_channels)
        self.fuse2 = ResBlock(C // 4, C // 4)
        self.skip_gate2 = nn.Parameter(torch.full((1,), 0.5))
        # Stage 3: 80→160, C/4 → C/4, no skip
        self.up3 = nn.ConvTranspose2d(C // 4, C // 4, 2, 2)
        self.fuse3 = ResBlock(C // 4, C // 4)
        # Stage 4: 160→320, C/4 → C/4, no skip
        self.up4 = nn.ConvTranspose2d(C // 4, C // 4, 2, 2)
        self.fuse4 = ResBlock(C // 4, C // 4)

        self.mask_mlp = MLP(embed_dim, embed_dim, C // 4, depth=3)
        self.iou_mlp = MLP(embed_dim, 256, 1, depth=3)

        # Multi-scale deep-supervision heads (1×1 conv → n_organs)
        self.ds_head_40 = nn.Conv2d(C // 2, n_organs, 1)
        self.ds_head_80 = nn.Conv2d(C // 4, n_organs, 1)
        self.ds_head_160 = nn.Conv2d(C // 4, n_organs, 1)

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

    def _match_size(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x.float(), size=ref.shape[-2:],
                              mode="bilinear", align_corners=False).to(ref.dtype)
        return x

    def forward(
        self,
        image_emb: torch.Tensor,    # (B, C, H, W)  — H=W=20 at 320px input
        dense_pe: torch.Tensor,     # (B, C, H, W)
        skip: torch.Tensor,         # (B, skip_ch,      2H, 2W)
        skip_fine: torch.Tensor,    # (B, skip_fine_ch, 4H, 4W)
        query_bias: torch.Tensor = None,   # optional (n_organs, C) bias (V8 text-prompt injection)
    ) -> Dict[str, torch.Tensor]:
        B, C, H, W = image_emb.shape
        q = self.organ_queries.weight
        if query_bias is not None:
            q = q + query_bias
        q = self.graph(q)
        queries = q.unsqueeze(0).expand(B, -1, -1).contiguous()
        q_pe = torch.zeros_like(queries)
        src = image_emb.flatten(2).permute(0, 2, 1)
        pe = dense_pe.flatten(2).permute(0, 2, 1)
        hs, src_updated = self.transformer(src, pe, queries, q_pe)
        s0 = src_updated.permute(0, 2, 1).reshape(B, C, H, W)

        # Stage 1: 20→40 + stride-8 skip
        s1 = self.up1(s0)
        skip_proc = self.skip1_ln(skip)
        skip_proc = self._haar_edge(skip_proc)
        skip_proc = self.skip1_proj(skip_proc)
        skip_proc = self._match_size(skip_proc, s1)
        s1 = self.fuse1(s1 + self.skip_gate1 * skip_proc)

        # Stage 2: 40→80 + stride-4 skip
        s2 = self.up2(s1)
        skip_fine_proc = self.skip2_ln(skip_fine)
        skip_fine_proc = self.skip2_proj(skip_fine_proc)
        skip_fine_proc = self._match_size(skip_fine_proc, s2)
        s2 = self.fuse2(s2 + self.skip_gate2 * skip_fine_proc)

        # Stage 3: 80→160
        s3 = self.fuse3(self.up3(s2))
        # Stage 4: 160→320
        s4 = self.fuse4(self.up4(s3))

        b, c, h, w = s4.shape
        weights = self.mask_mlp(hs[:, :self.n_organs, :])           # (B, 15, C//4)
        masks = (weights @ s4.view(b, c, h * w)).view(b, self.n_organs, h, w)
        iou = self.iou_mlp(hs[:, :self.n_organs, :]).squeeze(-1)

        return {
            "masks": masks,
            "iou": iou,
            "ds_s40": self.ds_head_40(s1),
            "ds_s80": self.ds_head_80(s2),
            "ds_s160": self.ds_head_160(s3),
        }
