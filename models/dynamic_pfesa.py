"""V9 Novel #6 — Organ-aware dynamic PFESA++.

Standard PFESA (position-encoded frequency-enhanced spatial attention — see
`models/pfesa_plus.py`) uses fixed spatial kernels. We replace the kernel
weights with ones *generated on-the-fly* from `(organ_embedding, text_emb)`
via a small HyperNetwork. At each forward pass, different organs get
different effective kernels — a form of class-conditional dynamic convolution
that hasn't been applied to medical segmentation PFESA heads before.

Design:
  - Hypernet: (organ_emb ⊕ text_emb) → per-organ (k×k×out×in) kernel weights.
  - Forward: grouped conv2d where each batch element uses its own kernel.
  - For B > 1 we run grouped conv (groups=B) with concatenated kernels, a
    standard dynamic-conv trick. Pure PyTorch, no custom CUDA.

Inputs:
  feat     : (B, C_in, H, W)
  organ_id : (B,) long in [0, n_organs)
  text_emb : (B, text_dim) or None

Output: (B, C_out, H, W)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicPFESA(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        n_organs: int = 15,
        kernel_size: int = 3,
        text_dim: int = 0,
        organ_dim: int = 32,
        hyper_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.k = kernel_size
        self.n_organs = n_organs
        self.text_dim = text_dim

        self.organ_emb = nn.Embedding(n_organs, organ_dim)
        cond_dim = organ_dim + text_dim

        self.hyper = nn.Sequential(
            nn.Linear(cond_dim, hyper_hidden), nn.GELU(),
            nn.Linear(hyper_hidden, out_ch * in_ch * kernel_size * kernel_size),
        )
        nn.init.zeros_(self.hyper[-1].weight)
        nn.init.zeros_(self.hyper[-1].bias)

        self.bias = nn.Parameter(torch.zeros(out_ch))
        self.skip = (
            nn.Identity()
            if in_ch == out_ch
            else nn.Conv2d(in_ch, out_ch, 1)
        )

    def forward(
        self,
        feat: torch.Tensor,
        organ_id: torch.Tensor,
        text_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, C_in, H, W = feat.shape
        assert C_in == self.in_ch, f"expected {self.in_ch} channels, got {C_in}"
        oe = self.organ_emb(organ_id)
        if self.text_dim > 0:
            assert text_emb is not None, "text_dim>0 but text_emb is None"
            cond = torch.cat([oe, text_emb], dim=1)
        else:
            cond = oe

        kernels = self.hyper(cond)
        kernels = kernels.view(
            B * self.out_ch, C_in, self.k, self.k,
        )

        x = feat.reshape(1, B * C_in, H, W)
        out = F.conv2d(x, kernels, bias=None, padding=self.k // 2, groups=B)
        out = out.view(B, self.out_ch, H, W) + self.bias.view(1, -1, 1, 1)
        out = out + self.skip(feat)
        return out
