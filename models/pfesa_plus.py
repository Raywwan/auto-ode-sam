"""PFESA++ — parameterized extension of MICCAI 2025 PFESA.

Learnable amplification α and soft radial high-frequency cutoff.
At init (alpha=1.0, cutoff_logit=0.0, steepness_log=log(10)) the behavior
matches the original PFESA α=1.0 cutoff=0.5 up to the sigmoid soft edge.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class PFESAPlus(nn.Module):
    """FFT-based high-frequency amplifier with 3 learnable scalars."""

    def __init__(self) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.cutoff_logit = nn.Parameter(torch.tensor(0.0))  # σ(0)=0.5
        self.steepness_log = nn.Parameter(torch.tensor(math.log(10.0)))

    def _radial_grid(self, H: int, W: int, device, dtype) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        r = torch.sqrt(xx * xx + yy * yy)
        return r.clamp_max(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → same shape."""
        B, C, H, W = x.shape
        r = self._radial_grid(H, W, x.device, torch.float32)
        cutoff = torch.sigmoid(self.cutoff_logit)
        steep = torch.exp(self.steepness_log)
        mask = torch.sigmoid(steep * (r - cutoff))  # soft high-pass
        x32 = x.float()
        F = torch.fft.fft2(x32, norm="ortho")
        F_enh = F * (1.0 + self.alpha * mask.unsqueeze(0).unsqueeze(0))
        y = torch.fft.ifft2(F_enh, norm="ortho").real
        return y.to(x.dtype)
