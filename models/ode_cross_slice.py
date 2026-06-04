# =============================================================================
# models/ode_cross_slice.py — Neural ODE Cross-Slice Feature Dynamics
#
# Core idea: CT slices along the Z-axis are samples of a continuously
# evolving 3D anatomical field. Instead of treating them as a discrete
# sequence (attention, Mamba) or frequency signal (FFT), we model the
# evolution of TinyViT features between slices as governed by an
# Ordinary Differential Equation:
#
#       dh/dt = f_θ(h, t)     h(0) = features at first slice
#
# Solved with forward Euler, the ODE trajectory h(t₀), h(t₁), ..., h(t_{D-1})
# gives each slice a "context state" informed by the ENTIRE volume dynamics,
# starting from either end (bidirectional: fwd + bwd).
#
# Why this beats attention / Mamba for this problem:
#   - ODE: continuous, slice-order invariant (can query any z ∈ [0,1])
#   - Attention: quadratic in D, ignores ordering semantics
#   - Mamba: linear but still discrete SSM steps
#   - ODE: smooth trajectory = perfect inductive bias for smooth anatomy
#
# Novelty: Neural ODE applied to SAM cross-slice feature evolution for
# abdominal CT segmentation. Not in any published paper as of 2025.
#
# Reference: Chen et al., "Neural Ordinary Differential Equations",
# NeurIPS 2018.  Our implementation uses simple bidirectional Euler
# integration — no torchdiffeq dependency needed.
# =============================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ODE dynamics function  f_θ(h, t) = dh/dt
# ---------------------------------------------------------------------------

class ODEFunction(nn.Module):
    """
    The right-hand side of the ODE:  dh/dt = f_θ(h, t).

    Time-conditioned via Fourier encoding: t → [sin(2πt·k), cos(2πt·k)]
    for k = 1..n_freqs.  This allows the network to express periodic
    (repeating organ structure) and aperiodic (single-occurrence organ)
    dynamics equally well.

    Architecture: Linear(dim + 2*n_freqs → dim//4) → Tanh → Linear(dim//4 → dim)
    Tanh activation bounds the derivative, preventing exploding ODE states.
    Zero-init on the last layer → derivative = 0 at init → stable training start.

    Args:
        dim     (int): Feature dimension C.
        hidden  (int): Width of hidden layer.  Default: dim // 4.
        n_freqs (int): Number of Fourier frequency bands for time.  Default: 4.
    """

    def __init__(self, dim: int, hidden: int = None, n_freqs: int = 4):
        super().__init__()
        self.dim     = dim
        self.n_freqs = n_freqs
        hidden       = hidden if hidden is not None else dim // 4

        time_dim = 2 * n_freqs  # sin + cos per frequency

        self.net = nn.Sequential(
            nn.Linear(dim + time_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, dim),
        )

        # Zero-init final layer: at init, dh/dt = 0 for all (h, t).
        # This guarantees the ODE is the identity function at epoch 0
        # — the model starts exactly at the Stage2-noISA baseline.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _time_encoding(self, t: float, N: int, device, dtype) -> torch.Tensor:
        """
        Fourier encoding of scalar time t → (N, 2*n_freqs) tensor.
        Repeated N times for batch broadcast.
        """
        freqs = torch.arange(1, self.n_freqs + 1, dtype=dtype, device=device)
        t_val = torch.tensor(t, dtype=dtype, device=device)
        enc   = torch.cat([
            torch.sin(2 * math.pi * freqs * t_val),
            torch.cos(2 * math.pi * freqs * t_val),
        ])  # (2*n_freqs,)
        return enc.unsqueeze(0).expand(N, -1)          # (N, 2*n_freqs)

    # ------------------------------------------------------------------
    def forward(self, h: torch.Tensor, t: float) -> torch.Tensor:
        """
        Args:
            h: (N, dim)  — current feature state at time t
            t: float     — normalised depth ∈ [0, 1]

        Returns:
            dh_dt: (N, dim) — time derivative of features
        """
        t_enc = self._time_encoding(t, h.shape[0], h.device, h.dtype)
        return self.net(torch.cat([h, t_enc], dim=-1))


# ---------------------------------------------------------------------------
# Bidirectional Neural ODE Cross-Slice Module
# ---------------------------------------------------------------------------

class BidirectionalNeuralODECrossSlice(nn.Module):
    """
    Bidirectional Neural ODE cross-slice feature enrichment.

    Two ODE instances run in parallel:
      • Forward ODE:  h(0) = feat[:,0],   evolve t: 0 → 1 (top → bottom)
      • Backward ODE: h(0) = feat[:,D-1], evolve t: 0 → 1 (bottom → top)

    The forward trajectory captures the "top-down anatomical context"
    (e.g., liver first appears high in the abdomen).
    The backward trajectory captures the "bottom-up context"
    (e.g., stomach is above the rectum).
    Combining both gives every slice full-volume context.

    Each ODE is solved with a simple multi-step Euler method:
        h_{i+1} = h_i + Δt · f_θ(h_i, t_i)

    No external ODE solver library needed — the short sequence length
    (D = 8 slices) makes Euler integration accurate and fast.

    Args:
        dim        (int): Feature channel dimension.
        ode_hidden (int): Hidden width in ODEFunction.  Default: dim // 4.
        n_freqs    (int): Fourier frequency bands for time.  Default: 4.
        substeps   (int): Euler sub-steps per slice interval.  Default: 2.
                          2 substeps × 7 intervals = 14 ODE calls per direction.
    """

    def __init__(
        self,
        dim:        int,
        ode_hidden: int = None,
        n_freqs:    int = 4,
        substeps:   int = 2,
    ):
        super().__init__()
        self.substeps = substeps

        # Forward ODE: learns top-to-bottom anatomical dynamics
        self.ode_fwd = ODEFunction(dim, ode_hidden, n_freqs)

        # Backward ODE: learns bottom-to-top anatomical dynamics
        # Separate parameters — the top-down and bottom-up dynamics
        # are not the same (liver vs. rectum initialisation)
        self.ode_bwd = ODEFunction(dim, ode_hidden, n_freqs)

        # Merge bidirectional trajectories → single feature correction
        self.merge   = nn.Linear(dim * 2, dim, bias=False)
        nn.init.zeros_(self.merge.weight)

        # Residual normalisation
        self.norm    = nn.LayerNorm(dim)

    # ------------------------------------------------------------------
    def _heun_trajectory(
        self,
        ode_func: ODEFunction,
        h0:       torch.Tensor,   # (N, C)
        D:        int,
    ) -> torch.Tensor:
        """
        Heun's method (RK2 predictor-corrector) integration from t=0 to t=1.

        Heun's method vs forward Euler:
            Euler:  h_{n+1} = h_n + dt·f(h_n, t_n)
            Heun:   k1 = f(h_n, t_n)
                    k2 = f(h_n + dt·k1, t_n + dt)   # corrector evaluation
                    h_{n+1} = h_n + 0.5·dt·(k1 + k2)

        This halves the local truncation error (O(dt²) vs O(dt³)) at the cost
        of one extra ODE function call per sub-step — worthwhile because D=8
        means only 14–28 total calls per direction, so the extra cost is tiny.

        Args:
            ode_func: the ODE function f_θ(h, t)
            h0:       initial state (N, C)
            D:        number of depth slices

        Returns:
            trajectory: (N, D, C) — states at t = 0, 1/(D-1), ..., 1
        """
        t_points = [d / max(D - 1, 1) for d in range(D)]   # [0, 1/(D-1), ..., 1]
        h        = h0
        states   = [h]

        for i in range(1, D):
            t_start = t_points[i - 1]
            t_end   = t_points[i]
            sub_dt  = (t_end - t_start) / self.substeps

            for j in range(self.substeps):
                t_curr = t_start + j * sub_dt
                k1     = ode_func(h, t_curr)
                h_pred = h + sub_dt * k1
                k2     = ode_func(h_pred, t_curr + sub_dt)
                h      = h + 0.5 * sub_dt * (k1 + k2)

            states.append(h)

        return torch.stack(states, dim=1)   # (N, D, C)

    # ------------------------------------------------------------------
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, D, C, H, W) — volume feature maps

        Returns:
            out: (B, D, C, H, W) — ODE-enriched features (residual connection)
        """
        B, D, C, H, W = features.shape

        # Reshape each spatial position into an independent batch:
        # (B, D, C, H, W) → (B*H*W, D, C)
        x = features.permute(0, 3, 4, 1, 2).reshape(B * H * W, D, C)
        N = x.shape[0]

        # ---- Forward ODE: initial condition = first slice (t=0) ----
        traj_fwd = self._heun_trajectory(self.ode_fwd, x[:, 0], D)    # (N, D, C)

        # ---- Backward ODE: initial condition = last slice (time-reversed) ----
        # We solve a separate ODE that "looks backward" — it starts from the
        # deepest slice and evolves toward the top, giving bottom-up context.
        traj_bwd = self._heun_trajectory(self.ode_bwd, x[:, -1], D)   # (N, D, C)
        traj_bwd = torch.flip(traj_bwd, dims=[1])                        # re-align to [0..D-1]

        # ---- Merge fwd + bwd trajectories ----
        ode_context = self.merge(
            torch.cat([traj_fwd, traj_bwd], dim=-1)
        )  # (N, D, C)

        # ---- Reshape back to volume and apply residual + LayerNorm ----
        # Pre-norm: normalise ode_context BEFORE the residual add.
        # When merge.weight=0 → ode_context=0 → norm(0)=0 → output = features exactly.
        # Post-norm (norm(features + ode_context)) cannot be identity since
        # LayerNorm normalizes even when ode_context=0.
        ode_context = ode_context.reshape(B, H, W, D, C).permute(0, 3, 4, 1, 2)
        # (B, D, C, H, W)

        ode_context = ode_context.permute(0, 1, 3, 4, 2)   # (B, D, H, W, C) for LayerNorm
        ode_context = self.norm(ode_context)
        ode_context = ode_context.permute(0, 1, 4, 2, 3)   # (B, D, C, H, W)
        return features + ode_context                       # residual

    def extra_repr(self) -> str:
        return (
            f"dim={self.ode_fwd.dim}, "
            f"substeps={self.substeps}, "
            f"bidir=True"
        )


# ---------------------------------------------------------------------------
# Organ-Conditioned ODE Function
# ---------------------------------------------------------------------------

class OrganConditionedODEFunction(nn.Module):
    """
    ODE dynamics with per-organ bias: dh/dt = f_θ(h, t) + MLP(organ_embed[organ_id]).

    The base dynamics f_θ(h, t) are shared across all organs.
    The organ bias term shifts the trajectory so each organ's ODE
    follows a distinct path — liver (wide, slow) vs adrenal (tight, sharp).

    Args:
        dim        (int): Feature channel dimension.
        n_organs   (int): Number of organ classes (15 for AMOS22).
        organ_emb_dim (int): Organ embedding dimension. Default: 32.
        hidden     (int): Hidden width in base dynamics. Default: dim // 4.
        n_freqs    (int): Fourier frequency bands for time. Default: 4.
    """

    def __init__(
        self,
        dim:          int,
        n_organs:     int = 15,
        organ_emb_dim: int = 32,
        hidden:       int = None,
        n_freqs:      int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.n_freqs = n_freqs
        self.n_organs = n_organs

        # Base dynamics: shared ODE function
        self.base_ode = ODEFunction(dim=dim, hidden=hidden, n_freqs=n_freqs)

        # Organ embedding: maps organ index → continuous embedding vector
        # organ_id is 1-indexed (1..15); we use it as-is with embedding size n_organs+1
        self.organ_embed = nn.Embedding(n_organs + 1, organ_emb_dim)

        # Bias MLP: organ embedding → dim-dimensional bias on dh/dt
        self.bias_mlp = nn.Sequential(
            nn.Linear(organ_emb_dim, dim),
        )
        # Zero-init: bias = 0 at start → model degrades to base ODE (safe initialisation)
        nn.init.zeros_(self.bias_mlp[0].weight)
        nn.init.zeros_(self.bias_mlp[0].bias)

    def forward(
        self,
        h:         torch.Tensor,  # (N, dim) — current feature state
        t:         float,          # normalised depth in [0, 1]
        organ_id:  torch.Tensor,  # (B,) long tensor — organ index per batch item
                                   # N = B * H * W, so we need to expand
    ) -> torch.Tensor:
        """
        Returns dh_dt: (N, dim).

        organ_id is (B,) but h is (B*H*W, dim). We expand organ_id to (B*H*W,).
        """
        N = h.shape[0]
        B = organ_id.shape[0]
        HW = N // B  # spatial positions per sample

        # Expand organ_id: (B,) → (B*H*W,)
        organ_id_expanded = organ_id.repeat_interleave(HW)  # (N,)
        organ_emb = self.organ_embed(organ_id_expanded)      # (N, organ_emb_dim)
        bias = self.bias_mlp(organ_emb)                      # (N, dim)

        return self.base_ode(h, t) + bias


# ---------------------------------------------------------------------------
# Organ-Conditioned Bidirectional Neural ODE Cross-Slice Module
# ---------------------------------------------------------------------------

class OrganConditionedBidirectionalNeuralODE(nn.Module):
    """
    Bidirectional Neural ODE with per-organ trajectory conditioning.

    Wraps OrganConditionedODEFunction in the same bidirectional Euler
    integration framework as BidirectionalNeuralODECrossSlice, but
    routes organ_id through both forward and backward ODE functions.

    Interface: forward(features, organ_id) — organ_id is (B,) long tensor.
    """

    def __init__(
        self,
        dim:          int,
        n_organs:     int = 15,
        organ_emb_dim: int = 32,
        ode_hidden:   int = None,
        n_freqs:      int = 4,
        substeps:     int = 2,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.substeps = substeps
        self.bidirectional = bidirectional

        self.ode_fwd = OrganConditionedODEFunction(
            dim=dim, n_organs=n_organs, organ_emb_dim=organ_emb_dim,
            hidden=ode_hidden, n_freqs=n_freqs,
        )
        # ode_bwd and merge always allocated for ckpt-shape compatibility.
        # When bidirectional=False, they are unused at forward time.
        self.ode_bwd = OrganConditionedODEFunction(
            dim=dim, n_organs=n_organs, organ_emb_dim=organ_emb_dim,
            hidden=ode_hidden, n_freqs=n_freqs,
        )
        self.merge = nn.Linear(dim * 2, dim, bias=False)
        nn.init.zeros_(self.merge.weight)
        self.norm  = nn.LayerNorm(dim)

    def _heun_trajectory(
        self,
        ode_func,
        h0:       torch.Tensor,   # (N, C)
        D:        int,
        organ_id: torch.Tensor,   # (B,)
    ) -> torch.Tensor:
        """Heun's method (RK2) integration — same as BidirectionalNeuralODECrossSlice."""
        t_points = [d / max(D - 1, 1) for d in range(D)]
        h = h0
        states = [h]
        for i in range(1, D):
            t_start = t_points[i - 1]
            t_end   = t_points[i]
            sub_dt  = (t_end - t_start) / self.substeps
            for j in range(self.substeps):
                t_curr = t_start + j * sub_dt
                k1     = ode_func(h, t_curr, organ_id)
                h_pred = h + sub_dt * k1
                k2     = ode_func(h_pred, t_curr + sub_dt, organ_id)
                h      = h + 0.5 * sub_dt * (k1 + k2)
            states.append(h)
        return torch.stack(states, dim=1)   # (N, D, C)

    def forward(
        self,
        features:  torch.Tensor,  # (B, D, C, H, W)
        organ_id:  torch.Tensor,  # (B,) long — organ index
    ) -> torch.Tensor:
        # Validate inputs once here (not inside the integration loop)
        assert organ_id.min() >= 0 and organ_id.max() <= self.ode_fwd.n_organs, \
            f"organ_id must be in [0, {self.ode_fwd.n_organs}], got min={organ_id.min()}, max={organ_id.max()}"
        assert organ_id.shape[0] == features.shape[0], (
            f"organ_id batch size {organ_id.shape[0]} != features batch size {features.shape[0]}"
        )
        B, D, C, H, W = features.shape
        x = features.permute(0, 3, 4, 1, 2).reshape(B * H * W, D, C)

        traj_fwd = self._heun_trajectory(self.ode_fwd, x[:, 0],   D, organ_id)
        if self.bidirectional:
            traj_bwd = self._heun_trajectory(self.ode_bwd, x[:, -1],  D, organ_id)
            traj_bwd = torch.flip(traj_bwd, dims=[1])
            ode_context = self.merge(torch.cat([traj_fwd, traj_bwd], dim=-1))
        else:
            ode_context = traj_fwd
        ode_context = ode_context.reshape(B, H, W, D, C).permute(0, 3, 4, 1, 2)

        # Pre-norm: normalise ode_context BEFORE the residual add.
        # When merge.weight=0 → ode_context=0 → norm(0)=0 → output = features exactly.
        ode_context = ode_context.permute(0, 1, 3, 4, 2)   # (B, D, H, W, C) for LayerNorm
        ode_context = self.norm(ode_context)
        ode_context = ode_context.permute(0, 1, 4, 2, 3)   # (B, D, C, H, W)
        return features + ode_context                       # residual

    def extra_repr(self) -> str:
        return (
            f"dim={self.ode_fwd.base_ode.dim}, "
            f"n_organs={self.ode_fwd.n_organs}, "
            f"substeps={self.substeps}"
        )


# ===========================================================================
# PA-CODE: Position-Aware Conditioned Neural ODE
#
# See docs/superpowers/specs/2026-04-30-pa-code-design.md
#
# Replaces OrganConditionedODEFunction's additive bias with:
#   (1) identity-initialized FiLM modulation:  dh/dt = γ(e) ⊙ f_θ + β(e)
#   (2) position-augmented dynamics:           f_θ(h, t, p)  where p = (x,y)
#   (3) trajectory state return for distillation
#
# Why: the additive bias_mlp + zero-init in OrganConditionedODEFunction
# competes against a V2-warmstarted base_ode with much larger output
# norm. Diagnostic on A1 ep_013 showed cos(dh/dt across organ_ids) = 0.999
# — i.e., conditioning never escaped zero. FiLM's multiplicative gain γ
# scales the *existing* signal per-organ, which is what every modern
# organ-conditioned med-seg paper since 2023 uses.
#
# Identity init (γ=1, β=0) means at step 0 the model behaves EXACTLY like
# the unconditioned base ODE — important when warmstarting from V2.
# ===========================================================================


def _pos_encoding_2d(
    H:           int,
    W:           int,
    n_pos_freqs: int,
    device,
    dtype,
) -> torch.Tensor:
    """Fourier positional encoding for 2D spatial coordinates.

    Returns: (H*W, 4*n_pos_freqs) tensor of [sin/cos × 2 axes × n_pos_freqs].

    Coordinates normalized to [-1, 1] across H and W independently.
    The depth (z) axis is NOT encoded here — it is already represented
    by the ODE time variable t.
    """
    ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)  # (H,)
    xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)  # (W,)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")          # (H, W)
    coords = torch.stack([grid_y, grid_x], dim=-1).reshape(H * W, 2)  # (HW, 2)

    freqs = torch.arange(1, n_pos_freqs + 1, device=device, dtype=dtype)  # (F,)
    # (HW, 2, 1) * (1, 1, F) -> (HW, 2, F)
    args = math.pi * coords.unsqueeze(-1) * freqs.view(1, 1, -1)
    pe   = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)        # (HW, 2, 2F)
    return pe.reshape(H * W, 4 * n_pos_freqs)                            # (HW, 4F)


class PACodeODEFunction(nn.Module):
    """ODE dynamics with FiLM modulation and position-augmented input.

        dh/dt = γ(e_organ) ⊙ f_θ(h, t, p) + β(e_organ)

    where:
        - f_θ is a 3-layer MLP receiving [h, time_enc(t), pos_enc(x, y)]
        - γ_head, β_head are 2-layer MLPs `(e_organ_dim -> hidden -> dim)`
        - γ identity-initialized to 1.0 at step 0 (zero-weights, bias=1)
        - β identity-initialized to 0.0 at step 0 (zero-weights, bias=0)
        - last layer of f_θ zero-initialized (so dh/dt = β(e) at step 0)

    With γ=1 and β=0 at init AND f_θ output zero-initialized, dh/dt = 0
    at every step — model is identity-equivalent to the V2 ODE at step 0.

    Args:
        dim          (int): Feature channel dimension.
        n_organs     (int): Number of organ classes (15 for AMOS22).
        organ_emb_dim(int): Organ embedding width.  Default: 64.
        hidden       (int): Hidden width inside f_θ.  Default: dim // 4.
        n_freqs      (int): Fourier bands for time.  Default: 4.
        n_pos_freqs  (int): Fourier bands for (x, y) position.  Default: 4.
        film_hidden  (int): Hidden width inside γ/β heads.  Default: dim // 2.
    """

    def __init__(
        self,
        dim:           int,
        n_organs:      int = 15,
        organ_emb_dim: int = 64,
        hidden:        int = None,
        n_freqs:       int = 4,
        n_pos_freqs:   int = 4,
        film_hidden:   int = None,
        gamma_bound:   float = 0.5,
    ):
        super().__init__()
        self.dim         = dim
        self.n_organs    = n_organs
        self.n_freqs     = n_freqs
        self.n_pos_freqs = n_pos_freqs
        self.gamma_bound = float(gamma_bound)
        hidden           = hidden if hidden is not None else dim // 4
        film_hidden      = film_hidden if film_hidden is not None else dim // 2

        # ---- Organ embedding ----
        # n_organs+1 entries because organ_id is 1-indexed (1..15) plus a
        # safety slot at index 0 for "unspecified".
        self.organ_embed = nn.Embedding(n_organs + 1, organ_emb_dim)
        # Larger init than std=0.02: with dim=64 ViT convention is std=0.125
        nn.init.trunc_normal_(self.organ_embed.weight, std=organ_emb_dim ** -0.5)

        # ---- Base dynamics f_θ(h, t, p) ----
        time_dim = 2 * n_freqs
        pos_dim  = 4 * n_pos_freqs   # 2 axes × (sin + cos) × n_pos_freqs
        in_dim   = dim + time_dim + pos_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, dim),
        )
        # Zero-init final layer: f_θ output = 0 at init.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        # ---- FiLM γ head (multiplicative gain) ----
        # Identity init: ONLY the last layer is zero-init (weight=0, bias=1).
        # First layer gets standard fan-in init so the GELU receives a
        # non-zero per-organ signal -- this is what makes the last-layer
        # weight gradient non-zero at step 0 (chain rule: dL/dW2 = dL/dy * h2^T).
        # If we zero-init both layers, dL/dW2 = 0 forever and γ collapses to a
        # uniform scalar across organ_ids -- exactly the A1 additive-bias trap
        # (preflight gate G7 caught this on first run, 2026-05-01).
        # At step 0: y = W2 @ GELU(W1 @ e + b1) + b2 = 0 + 1 = 1, identity preserved.
        self.gamma_head = nn.Sequential(
            nn.Linear(organ_emb_dim, film_hidden),
            nn.GELU(),
            nn.Linear(film_hidden, dim),
        )
        nn.init.trunc_normal_(self.gamma_head[0].weight, std=organ_emb_dim ** -0.5)
        nn.init.zeros_(self.gamma_head[0].bias)
        nn.init.zeros_(self.gamma_head[-1].weight)
        nn.init.ones_(self.gamma_head[-1].bias)   # γ = 1 at step 0

        # ---- FiLM β head (additive shift) ----
        # Same pattern: first layer trunc-normal, last layer zero-init.
        # At step 0: y = 0 + 0 = 0, identity preserved; gradients flow.
        self.beta_head = nn.Sequential(
            nn.Linear(organ_emb_dim, film_hidden),
            nn.GELU(),
            nn.Linear(film_hidden, dim),
        )
        nn.init.trunc_normal_(self.beta_head[0].weight, std=organ_emb_dim ** -0.5)
        nn.init.zeros_(self.beta_head[0].bias)
        nn.init.zeros_(self.beta_head[-1].weight)
        nn.init.zeros_(self.beta_head[-1].bias)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _time_encoding(self, t: float, N: int, device, dtype) -> torch.Tensor:
        freqs = torch.arange(1, self.n_freqs + 1, dtype=dtype, device=device)
        t_val = torch.tensor(t, dtype=dtype, device=device)
        enc   = torch.cat([
            torch.sin(2 * math.pi * freqs * t_val),
            torch.cos(2 * math.pi * freqs * t_val),
        ])  # (2*n_freqs,)
        return enc.unsqueeze(0).expand(N, -1)

    # ------------------------------------------------------------------
    def forward(
        self,
        h:        torch.Tensor,   # (N, dim)
        t:        float,
        pos_enc:  torch.Tensor,   # (N, 4*n_pos_freqs) — precomputed by wrapper
        organ_id: torch.Tensor,   # (B,) — long
    ) -> torch.Tensor:
        """Returns dh/dt: (N, dim)."""
        N = h.shape[0]
        B = organ_id.shape[0]
        HW = N // B

        # ---- Time + position context ----
        t_enc = self._time_encoding(t, N, h.device, h.dtype)
        h_in  = torch.cat([h, t_enc, pos_enc], dim=-1)
        f     = self.net(h_in)                        # (N, dim)

        # ---- Organ-conditioned FiLM ----
        # organ_id is (B,); expand to (B*HW,) by repeat_interleave to align with N
        organ_id_expanded = organ_id.repeat_interleave(HW)
        e = self.organ_embed(organ_id_expanded)       # (N, organ_emb_dim)
        gamma_raw = self.gamma_head(e)                # raw FiLM-gamma output, ~1 at init
        # Bound gamma to [1 - gamma_bound, 1 + gamma_bound] via tanh.
        # 2026-05-01 audit: unbounded gamma drift (γσ-across-organs ran from 0
        # to 0.49 in 5 epochs at LR 1e-4, with per-organ |γ| reaching ±1.78)
        # destroyed V2-decoder feature mapping -> val oscillation. Bounded
        # gamma keeps features within decoder-readable scale.
        # At init gamma_raw = 1 -> tanh(0) = 0 -> gamma = 1 (identity preserved).
        gamma = 1.0 + self.gamma_bound * torch.tanh(gamma_raw - 1.0)
        beta  = self.beta_head(e)                     # (N, dim) — init at 0

        return gamma * f + beta


# ---------------------------------------------------------------------------
# PA-CODE Bidirectional Neural ODE Cross-Slice Module
# ---------------------------------------------------------------------------

class PACodeBidirectionalNeuralODE(nn.Module):
    """Bidirectional Neural ODE with FiLM organ-conditioning + position-augmented
    state. Drop-in replacement for OrganConditionedBidirectionalNeuralODE.

    Args:
        dim           (int): Feature channel dimension.
        n_organs      (int): Number of organ classes.
        organ_emb_dim (int): Organ embedding width.
        ode_hidden    (int): Hidden width inside f_θ.
        n_freqs       (int): Fourier bands for time.
        n_pos_freqs   (int): Fourier bands for (x, y) position.
        substeps      (int): Heun sub-steps per slice interval.

    Trajectory return: forward() returns (B, D, C, H, W) features.
    For trajectory distillation, call get_last_trajectories() after forward()
    to retrieve the full integration states (B*H*W, D, C) for fwd and bwd
    directions. They are cached on the module to avoid duplicate compute.
    """

    def __init__(
        self,
        dim:           int,
        n_organs:      int = 15,
        organ_emb_dim: int = 64,
        ode_hidden:    int = None,
        n_freqs:       int = 4,
        n_pos_freqs:   int = 4,
        substeps:      int = 4,
        gamma_bound:   float = 0.5,
    ):
        super().__init__()
        self.substeps    = substeps
        self.n_pos_freqs = n_pos_freqs

        self.ode_fwd = PACodeODEFunction(
            dim=dim, n_organs=n_organs, organ_emb_dim=organ_emb_dim,
            hidden=ode_hidden, n_freqs=n_freqs, n_pos_freqs=n_pos_freqs,
            gamma_bound=gamma_bound,
        )
        self.ode_bwd = PACodeODEFunction(
            dim=dim, n_organs=n_organs, organ_emb_dim=organ_emb_dim,
            hidden=ode_hidden, n_freqs=n_freqs, n_pos_freqs=n_pos_freqs,
            gamma_bound=gamma_bound,
        )
        self.merge = nn.Linear(dim * 2, dim, bias=False)
        nn.init.zeros_(self.merge.weight)
        self.norm  = nn.LayerNorm(dim)

        # Trajectory cache for distillation.  Populated by forward() if
        # cache_trajectories=True. Cleared at the next forward() call.
        self._cached_traj_fwd = None   # (N, D, C)
        self._cached_traj_bwd = None   # (N, D, C) — already flipped

    def _heun_trajectory(
        self,
        ode_func: PACodeODEFunction,
        h0:       torch.Tensor,   # (N, C)
        D:        int,
        pos_enc:  torch.Tensor,   # (N, 4*n_pos_freqs)
        organ_id: torch.Tensor,   # (B,)
    ) -> torch.Tensor:
        t_points = [d / max(D - 1, 1) for d in range(D)]
        h        = h0
        states   = [h]

        for i in range(1, D):
            t_start = t_points[i - 1]
            t_end   = t_points[i]
            sub_dt  = (t_end - t_start) / self.substeps
            for j in range(self.substeps):
                t_curr = t_start + j * sub_dt
                k1     = ode_func(h, t_curr,         pos_enc, organ_id)
                h_pred = h + sub_dt * k1
                k2     = ode_func(h_pred, t_curr + sub_dt, pos_enc, organ_id)
                h      = h + 0.5 * sub_dt * (k1 + k2)
            states.append(h)

        return torch.stack(states, dim=1)   # (N, D, C)

    def forward(
        self,
        features:           torch.Tensor,  # (B, D, C, H, W)
        organ_id:           torch.Tensor,  # (B,) long
        cache_trajectories: bool = False,
    ) -> torch.Tensor:
        # Validate inputs
        assert organ_id.min() >= 0 and organ_id.max() <= self.ode_fwd.n_organs, \
            f"organ_id must be in [0, {self.ode_fwd.n_organs}], got min={organ_id.min()}, max={organ_id.max()}"
        assert organ_id.shape[0] == features.shape[0], (
            f"organ_id batch size {organ_id.shape[0]} != features batch size {features.shape[0]}"
        )

        B, D, C, H, W = features.shape
        x = features.permute(0, 3, 4, 1, 2).reshape(B * H * W, D, C)
        N = x.shape[0]

        # ---- Build position encoding once for this slab ----
        # pos_enc covers H*W; repeat across batch to (B*H*W, ...)
        pe_hw  = _pos_encoding_2d(H, W, self.ode_fwd.n_pos_freqs,
                                  features.device, features.dtype)         # (HW, 4F)
        pos_enc = pe_hw.unsqueeze(0).expand(B, -1, -1).reshape(N, -1)        # (N, 4F)

        # ---- Bidirectional integration ----
        traj_fwd = self._heun_trajectory(self.ode_fwd, x[:, 0],   D, pos_enc, organ_id)
        traj_bwd = self._heun_trajectory(self.ode_bwd, x[:, -1],  D, pos_enc, organ_id)
        traj_bwd = torch.flip(traj_bwd, dims=[1])

        if cache_trajectories:
            self._cached_traj_fwd = traj_fwd
            self._cached_traj_bwd = traj_bwd
        else:
            self._cached_traj_fwd = None
            self._cached_traj_bwd = None

        ode_context = self.merge(torch.cat([traj_fwd, traj_bwd], dim=-1))
        ode_context = ode_context.reshape(B, H, W, D, C).permute(0, 3, 4, 1, 2)

        # Pre-norm
        ode_context = ode_context.permute(0, 1, 3, 4, 2)
        ode_context = self.norm(ode_context)
        ode_context = ode_context.permute(0, 1, 4, 2, 3)
        return features + ode_context

    def get_cached_trajectories(self):
        """Returns (traj_fwd, traj_bwd) from last forward() call; both (N, D, C).
        Returns (None, None) if cache_trajectories=False was used.
        """
        return self._cached_traj_fwd, self._cached_traj_bwd

    def extra_repr(self) -> str:
        return (
            f"dim={self.ode_fwd.dim}, "
            f"n_organs={self.ode_fwd.n_organs}, "
            f"substeps={self.substeps}, "
            f"n_pos_freqs={self.n_pos_freqs}, "
            f"film=identity-init"
        )
