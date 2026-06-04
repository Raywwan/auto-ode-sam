"""V9 Novel #2 — Mamba-ODE hybrid cross-slice block.

Our existing flow-matched ODE (`OrganConditionedBidirectionalNeuralODE`)
handles *smooth, local* cross-slice continuity by integrating a learned
vector field along the Z-axis. This is ideal for short depth (D=8) and
organs whose shape evolves slowly across slices (kidneys, liver).

Selective-scan state-space models (Mamba) excel at *long-range, discrete-
transition* dependencies — e.g., organs that appear, disappear, or split
along Z (duodenum, prostate-uterus). We add a Mamba-lite block in parallel
and fuse with the ODE output via a learned gate.

Implementation: pure-PyTorch S6-style selective scan (no `mamba_ssm`
dependency — required for Windows). For D=8 slices the Python loop is
negligible cost; for future 3D patches we can swap in a CUDA kernel.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    """S6-style SSM with input-dependent Δ, B, C. Acts along the L axis.

    Input:  (B, L, C)
    Output: (B, L, C)
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, bias: bool = False):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            groups=self.d_inner, padding=d_conv - 1, bias=bias,
        )
        # Selective projections for Δ, B, C
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        # Initialize dt bias to give ~1.0 time-step prior.
        with torch.no_grad():
            nn.init.uniform_(self.dt_proj.bias, 0.001, 0.1)

        # Diagonal A parametrization (real, negative)
        A_log = torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)
                          .unsqueeze(0).repeat(self.d_inner, 1))
        self.A_log = nn.Parameter(A_log)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        xz = self.in_proj(x)                              # (B, L, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)                     # (B, L, d_inner)

        x_in = x_in.transpose(1, 2)                       # (B, d_inner, L)
        x_in = self.conv1d(x_in)[..., :L]                 # causal in L
        x_in = F.silu(x_in)
        x_in = x_in.transpose(1, 2)                       # (B, L, d_inner)

        x_dbl = self.x_proj(x_in)                         # (B, L, 2*d_state + 1)
        delta = x_dbl[..., :1]                            # (B, L, 1)
        B_sel = x_dbl[..., 1:1 + self.d_state]            # (B, L, d_state)
        C_sel = x_dbl[..., 1 + self.d_state:]             # (B, L, d_state)

        delta = F.softplus(self.dt_proj(delta))           # (B, L, d_inner) > 0

        A = -torch.exp(self.A_log.float())                # (d_inner, d_state)

        # Discretize: ΔA = exp(Δ * A), ΔB = Δ * B_sel
        # For small d_state and short L, run a straightforward Python scan.
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x_in.dtype)
        ys = []
        for l in range(L):
            dl = delta[:, l]                              # (B, d_inner)
            bl = B_sel[:, l]                              # (B, d_state)
            cl = C_sel[:, l]                              # (B, d_state)
            xl = x_in[:, l]                               # (B, d_inner)
            dA = torch.exp(dl[:, :, None] * A[None])      # (B, d_inner, d_state)
            dB = dl[:, :, None] * bl[:, None, :]          # (B, d_inner, d_state)
            h = dA * h + dB * xl[:, :, None]              # update state
            y = (h * cl[:, None, :]).sum(dim=-1)          # (B, d_inner)
            ys.append(y)
        y = torch.stack(ys, dim=1)                        # (B, L, d_inner)
        y = y + x_in * self.D                             # skip via D
        y = y * F.silu(z)                                 # gating
        return self.out_proj(y)


class MambaCrossSliceBlock(nn.Module):
    """Applies a selective SSM across the slice dimension for a 3D feature map.

    Input:  (B, D, C, H, W) — D = number of slices
    Output: (B, D, C, H, W) — same shape
    """

    def __init__(self, n_channels: int, d_state: int = 16, d_conv: int = 2,
                 expand: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(n_channels)
        self.ssm = SelectiveSSM(
            d_model=n_channels, d_state=d_state, d_conv=d_conv, expand=expand,
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        B, D, C, H, W = feat.shape
        # Flatten spatial to batch, keep slice axis as sequence
        x = feat.permute(0, 3, 4, 1, 2).reshape(B * H * W, D, C)
        x_res = x
        x = self.norm(x)
        y = self.ssm(x)
        y = y + x_res
        y = y.reshape(B, H, W, D, C).permute(0, 3, 4, 1, 2).contiguous()
        return y


class MambaODEHybrid(nn.Module):
    """Novel #2 — combine existing ODE output with Mamba SSM via learned gate.

    Used as a drop-in wrapper around the refiner's ODE. We don't replace the
    ODE; we *augment* it. The gate is initialized to trust the ODE more
    (sigmoid(-1.0)≈0.27 Mamba contribution) so the V7 working recipe is
    preserved on day one and Mamba only helps where useful.

    Usage in V9 refiner:
      ode_out = self.ode(feat3d, organ_id)
      fused   = self.mamba_ode(feat3d, ode_out)
    """

    def __init__(self, n_channels: int, d_state: int = 16, init_mamba_bias: float = -1.0):
        super().__init__()
        self.mamba = MambaCrossSliceBlock(n_channels, d_state=d_state)
        # Channel-wise gate: sigmoid(σ(conv(x_ode, x_mamba))) → α Mamba + (1-α) ODE.
        self.gate_conv = nn.Conv3d(n_channels * 2, n_channels, kernel_size=1)
        with torch.no_grad():
            self.gate_conv.bias.fill_(init_mamba_bias)

    def forward(
        self,
        feat_before_ode: torch.Tensor,           # (B, D, C, H, W) pre-ODE features
        feat_after_ode: torch.Tensor,            # (B, D, C, H, W) ODE output
    ) -> torch.Tensor:
        m_out = self.mamba(feat_before_ode)                     # (B, D, C, H, W)
        # Gate is computed from both paths concatenated along C.
        # Move D into batch so we can 2D conv (cheaper); then reshape back.
        B, D, C, H, W = feat_after_ode.shape
        cat = torch.cat([feat_after_ode, m_out], dim=2)         # (B, D, 2C, H, W)
        # Conv3d over (D, H, W) with kernel 1 — but gate_conv is Conv3d, so:
        cat_flat = cat.permute(0, 2, 1, 3, 4)                   # (B, 2C, D, H, W)
        gate = torch.sigmoid(self.gate_conv(cat_flat))          # (B, C, D, H, W)
        gate = gate.permute(0, 2, 1, 3, 4)                      # (B, D, C, H, W)
        return gate * m_out + (1.0 - gate) * feat_after_ode
