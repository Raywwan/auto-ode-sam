# =============================================================================
# models/mamba_pure.py — Pure PyTorch Selective State Space Model (Mamba-style)
#
# Implements a bidirectional selective SSM (S6) block with NO external dependencies.
# Uses sequential scan — correct and fast for short sequences (L ≤ 32).
#
# Why sequential scan is fine here:
#   Axial direction:    L = D = 8  — 8 iterations  (trivial)
#   Spatial directions: L = H = W = 16 — 16 iterations (trivial)
#   Compare: language models need L=8192+ where parallel scan matters
#
# Reference: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective
# State Spaces", arXiv:2312.00752.  This is a simplified clean-room variant
# without hardware-efficient CUDA kernels — equivalent in result for L ≤ 32.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaBidirectional1D(nn.Module):
    """
    Bidirectional selective SSM block for 1D sequences.

    Runs forward + backward selective scan, concatenates the two directions,
    projects back to d_model.  LayerNorm + residual are applied internally.

    Args:
        d_model (int): Input/output channel dimension.
        d_inner (int): Inner SSM state dimension.  Default: d_model // 2.
        d_state (int): Number of SSM recurrent state slots (N in the paper).  Default: 8.
        d_conv  (int): Depth-wise conv kernel size for local token mixing.  Default: 3.
    """

    def __init__(
        self,
        d_model: int,
        d_inner: int = None,
        d_state: int = 8,
        d_conv: int = 3,
    ):
        super().__init__()
        self.d_model  = d_model
        self.d_inner  = d_inner if d_inner is not None else d_model // 2
        self.d_state  = d_state

        # ---- Input projection: split into content (x) and gate (z) ----
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # ---- Local depth-wise conv (symmetric padding → same-length output) ----
        # Serves as short-range token mixing before the SSM scan.
        self.dw_conv = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv // 2, groups=self.d_inner, bias=True,
        )

        # ---- Selective SSM projections ----
        # From x_in, produce: dt_rank (=1 scalar), B_ssm, C_ssm (each d_state)
        self.x_proj  = nn.Linear(self.d_inner, 1 + d_state * 2, bias=False)
        # Expand scalar dt to full d_inner
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        # Initialise dt_proj.bias small so softplus(dt) starts near 0.01
        nn.init.constant_(self.dt_proj.bias, -4.0)

        # ---- Fixed state matrix A (negative real diagonal, stored as log) ----
        # A_n = -n  for n = 1 .. d_state  (discrete-time stable)
        A_log = torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
        self.register_buffer("A_log", A_log)  # (d_state,)

        # ---- Skip connection D  (scales x before adding to SSM output) ----
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # ---- Output projection: [fwd | bwd] → d_model ----
        self.out_proj = nn.Linear(self.d_inner * 2, d_model, bias=False)

        # ---- Residual normalisation ----
        self.norm = nn.LayerNorm(d_model)

    # ------------------------------------------------------------------
    def _ssm_scan(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward-direction sequential selective scan (S6).

        Args:
            x: (N, L, d_inner)  — N = batch × flattened spatial dims

        Returns:
            y: (N, L, d_inner)
        """
        N, L, d_in = x.shape

        # ---- Compute selective parameters from input ----
        xp    = self.x_proj(x)                          # (N, L, 1 + 2*d_state)
        dt_r  = xp[..., :1]                              # (N, L, 1)  — raw dt
        B_ssm = xp[..., 1: 1 + self.d_state]            # (N, L, d_state)
        C_ssm = xp[..., 1 + self.d_state:]               # (N, L, d_state)

        # Expand scalar dt to full d_inner channel width
        dt = F.softplus(self.dt_proj(dt_r))              # (N, L, d_inner)

        # A is negative diagonal: A_n = -exp(A_log_n)
        A_neg = -torch.exp(self.A_log)                   # (d_state,) — negative

        # ---- Sequential scan ----
        h = torch.zeros(N, d_in, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []

        for i in range(L):
            dt_i  = dt[:, i, :]           # (N, d_inner)
            B_i   = B_ssm[:, i, :]        # (N, d_state)
            C_i   = C_ssm[:, i, :]        # (N, d_state)
            x_i   = x[:, i, :]            # (N, d_inner)

            # Discretise: Ā_t = exp(Δ_t · A),  B̄_t = Δ_t · B_t
            # Broadcasting: dt_i (N, d_inner, 1) × A_neg (d_state,)
            A_bar = torch.exp(dt_i.unsqueeze(-1) * A_neg)          # (N, d_inner, d_state)
            # B_bar: (N, d_inner, 1) × (N, 1, d_state)
            B_bar = dt_i.unsqueeze(-1) * B_i.unsqueeze(1)          # (N, d_inner, d_state)

            # State update:  h_t = Ā_t · h_{t-1} + B̄_t · x_t
            h = A_bar * h + B_bar * x_i.unsqueeze(-1)              # (N, d_inner, d_state)

            # Output:  y_t = C_t^T · h_t  +  D · x_t
            y_i = (h * C_i.unsqueeze(1)).sum(-1) + self.D * x_i    # (N, d_inner)
            outputs.append(y_i)

        return torch.stack(outputs, dim=1)  # (N, L, d_inner)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, L, d_model)  — N = batch × flattened spatial dimensions

        Returns:
            out: (N, L, d_model)  — residual + LayerNorm applied
        """
        shortcut = x  # residual connection

        # ---- Split into content and gate ----
        xz    = self.in_proj(x)              # (N, L, 2 * d_inner)
        x_in, z = xz.chunk(2, dim=-1)        # each (N, L, d_inner)

        # ---- Local depth-wise conv ----
        x_c = x_in.permute(0, 2, 1)          # (N, d_inner, L)
        x_c = self.dw_conv(x_c)              # (N, d_inner, L)  — symmetric padding
        x_c = x_c.permute(0, 2, 1)           # (N, L, d_inner)
        x_c = F.silu(x_c)

        # ---- Forward scan ----
        y_fwd = self._ssm_scan(x_c)          # (N, L, d_inner)

        # ---- Backward scan (flip → scan → flip back) ----
        y_bwd = self._ssm_scan(torch.flip(x_c, dims=[1]))
        y_bwd = torch.flip(y_bwd, dims=[1])   # (N, L, d_inner) — aligned to original

        # ---- Gate both directions with SiLU(z) ----
        gate  = F.silu(z)                     # (N, L, d_inner)
        y_fwd = y_fwd * gate
        y_bwd = y_bwd * gate

        # ---- Concat + output projection ----
        out = self.out_proj(
            torch.cat([y_fwd, y_bwd], dim=-1)
        )                                     # (N, L, d_model)

        # ---- Residual + LayerNorm ----
        return self.norm(shortcut + out)

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, d_inner={self.d_inner}, "
            f"d_state={self.d_state}, bidir=True"
        )
