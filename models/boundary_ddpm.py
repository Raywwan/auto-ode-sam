"""V9 Novel #8 - Boundary-aware DDPM refiner.

A tiny (~3M param) UNet that runs a small number (typically 5-10) of DDPM
denoising steps *only* on the high-uncertainty boundary band of the coarse
mask. Confident interior/exterior pixels are preserved verbatim - fast at
inference, targets the regions where the refiner disagrees with itself.

Training: sample a random timestep t, corrupt the GT mask with Gaussian
noise, ask the network to predict the noise given (img, coarse_mask, t).

Inference: start from the coarse mask (not pure noise) and run n_steps
reverse-diffusion updates, masking updates outside the boundary band.

Uncertainty band: pixels where coarse in [0.05, 0.95]. (Configurable.)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sinusoidal_time_emb(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(torch.linspace(0, -9.21, half, device=t.device))
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([args.sin(), args.cos()], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class _TinyUNet(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, hidden: int, t_dim: int = 64) -> None:
        super().__init__()
        self.t_proj = nn.Sequential(nn.Linear(t_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.down1 = nn.Sequential(nn.Conv2d(in_ch, hidden, 3, padding=1), nn.GELU())
        self.down2 = nn.Sequential(nn.Conv2d(hidden, hidden * 2, 3, padding=1, stride=2), nn.GELU())
        self.mid = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1), nn.GELU(),
        )
        self.up1 = nn.Sequential(nn.ConvTranspose2d(hidden * 2, hidden, 2, stride=2), nn.GELU())
        self.out = nn.Conv2d(hidden * 2, out_ch, 3, padding=1)
        self.t_dim = t_dim
        self.hidden = hidden

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = _sinusoidal_time_emb(t, self.t_dim)
        temb = self.t_proj(temb)[:, :, None, None]
        h1 = self.down1(x) + temb
        h2 = self.down2(h1)
        h3 = self.mid(h2)
        u1 = self.up1(h3)
        if u1.shape[-2:] != h1.shape[-2:]:
            u1 = F.interpolate(u1, size=h1.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(torch.cat([u1, h1], dim=1))


class BoundaryDDPM(nn.Module):
    def __init__(
        self,
        n_organs: int = 15,
        hidden: int = 64,
        n_timesteps: int = 100,
        uncertainty_lo: float = 0.05,
        uncertainty_hi: float = 0.95,
    ) -> None:
        super().__init__()
        self.K = n_organs
        self.T = n_timesteps
        self.lo = uncertainty_lo
        self.hi = uncertainty_hi
        self.net = _TinyUNet(in_ch=3 + n_organs, out_ch=n_organs, hidden=hidden)

        betas = torch.linspace(1e-4, 0.02, n_timesteps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bar", alpha_bar)

    def _uncertainty_mask(self, coarse: torch.Tensor) -> torch.Tensor:
        return ((coarse > self.lo) & (coarse < self.hi)).float()

    def _soft_uncertainty_weight(self, coarse: torch.Tensor) -> torch.Tensor:
        # Smooth bump peaked at coarse=0.5, zero at 0 / 1. Differentiable in
        # coarse so the DDPM loss backprops into the refiner branch that
        # produced the coarse mask.
        return (4.0 * coarse * (1.0 - coarse)).clamp(0.0, 1.0)

    def training_loss(
        self,
        img: torch.Tensor,
        coarse_mask: torch.Tensor,
        gt_mask: torch.Tensor,
    ) -> torch.Tensor:
        B = img.shape[0]
        device = img.device
        t = torch.randint(0, self.T, (B,), device=device)
        ab = self.alpha_bar[t].view(B, 1, 1, 1)
        noise = torch.randn_like(gt_mask)
        y_t = ab.sqrt() * gt_mask + (1.0 - ab).sqrt() * noise

        x = torch.cat([img, y_t], dim=1)
        t_norm = t.float() / self.T
        pred = self.net(x, t_norm)

        weight = self._soft_uncertainty_weight(coarse_mask)
        denom = weight.sum().clamp_min(1e-3)
        se = (pred - noise) ** 2
        return (se * weight).sum() / denom

    @torch.no_grad()
    def sample(
        self,
        img: torch.Tensor,
        coarse_mask: torch.Tensor,
        n_steps: int = 10,
    ) -> torch.Tensor:
        B = img.shape[0]
        device = img.device
        band = self._uncertainty_mask(coarse_mask)
        y = coarse_mask.clone()

        tsteps = torch.linspace(self.T - 1, 0, n_steps, device=device).long()
        for i, t in enumerate(tsteps):
            ab_t = self.alpha_bar[t]
            x = torch.cat([img, y], dim=1)
            t_norm = (t.float() / self.T).repeat(B)
            eps = self.net(x, t_norm)
            y0 = (y - (1.0 - ab_t).sqrt() * eps) / ab_t.sqrt().clamp_min(1e-4)
            y0 = y0.clamp(0, 1)
            if i < len(tsteps) - 1:
                ab_next = self.alpha_bar[tsteps[i + 1]]
                y = ab_next.sqrt() * y0 + (1.0 - ab_next).sqrt() * eps
            else:
                y = y0

        return torch.where(band > 0, y.clamp(0, 1), coarse_mask)
