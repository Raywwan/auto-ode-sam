"""Flow-matched organ-conditioned bidirectional ODE for cross-slice feature evolution.

The module does three things simultaneously on the forward pass:
  1. Integrates the ODE forward from slice 0 and backward from slice D-1 using Heun's method.
  2. Merges the two trajectories with a zero-init Linear (residual add → identity at init).
  3. Collects (v_θ(h_i, t_i), finite-difference target) pairs so the trainer can compute L_flow.

L_flow is NOT computed here — it is computed in training/losses_v4.py using the pairs this
module returns. This keeps the module stateless wrt training schedules.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from models.ode_cross_slice import ODEFunction


class OrganConditionedFlowFunction(nn.Module):
    """dh/dt = base_ode(h, t) + MLP(organ_embed[organ_id])."""

    def __init__(self, dim: int, n_organs: int = 15, organ_emb_dim: int = 32,
                 hidden: Optional[int] = None, n_freqs: int = 4) -> None:
        super().__init__()
        self.dim = dim
        self.n_organs = n_organs
        self.base = ODEFunction(dim=dim, hidden=hidden, n_freqs=n_freqs)
        self.organ_embed = nn.Embedding(n_organs + 1, organ_emb_dim)
        self.bias_mlp = nn.Linear(organ_emb_dim, dim)
        nn.init.zeros_(self.bias_mlp.weight)
        nn.init.zeros_(self.bias_mlp.bias)

    def forward(self, h: torch.Tensor, t: float, organ_id: torch.Tensor) -> torch.Tensor:
        N = h.shape[0]
        B = organ_id.shape[0]
        HW = N // B
        ids = organ_id.repeat_interleave(HW)
        bias = self.bias_mlp(self.organ_embed(ids))
        return self.base(h, t) + bias


class FlowMatchedOrganConditionedODE(nn.Module):
    """Bidirectional organ-conditioned ODE + flow-matching target collection."""

    def __init__(self, dim: int, n_organs: int = 15, organ_emb_dim: int = 32,
                 ode_hidden: Optional[int] = None, n_freqs: int = 4, substeps: int = 4) -> None:
        super().__init__()
        self.substeps = substeps
        self.ode_fwd = OrganConditionedFlowFunction(dim, n_organs, organ_emb_dim, ode_hidden, n_freqs)
        self.ode_bwd = OrganConditionedFlowFunction(dim, n_organs, organ_emb_dim, ode_hidden, n_freqs)
        self.merge = nn.Linear(dim * 2, dim, bias=False)
        nn.init.zeros_(self.merge.weight)
        self.norm = nn.LayerNorm(dim)

    def _heun(self, func, h0, D, organ_id):
        t_pts = [d / max(D - 1, 1) for d in range(D)]
        h, states = h0, [h0]
        for i in range(1, D):
            t0, t1 = t_pts[i - 1], t_pts[i]
            dt = (t1 - t0) / self.substeps
            for j in range(self.substeps):
                tc = t0 + j * dt
                k1 = func(h, tc, organ_id)
                k2 = func(h + dt * k1, tc + dt, organ_id)
                h = h + 0.5 * dt * (k1 + k2)
            states.append(h)
        return torch.stack(states, dim=1)

    def _flow_pairs(self, func, x, D, organ_id, direction: str) -> Dict[str, torch.Tensor]:
        """Collect (v_θ(h_i, t_i), (h_{i±1} - h_i) · (D-1)) pairs at slice timestamps."""
        t_pts = [d / max(D - 1, 1) for d in range(D)]
        preds, targets = [], []
        scale = float(D - 1)
        if direction == "fwd":
            for i in range(D - 1):
                preds.append(func(x[:, i], t_pts[i], organ_id))
                targets.append((x[:, i + 1] - x[:, i]) * scale)
        else:  # bwd
            for i in range(1, D):
                preds.append(func(x[:, i], t_pts[i], organ_id))
                targets.append((x[:, i - 1] - x[:, i]) * scale)
        return {
            "pred": torch.stack(preds, dim=1),
            "target": torch.stack(targets, dim=1),
        }

    def _consistency_pairs(self, func, x, D, organ_id, direction: str) -> Dict[str, torch.Tensor]:
        """Cross-Slice ODE Consistency (XSC) — novel contribution.

        Integrates one inter-slice interval with Heun, so (h_i → h_{i+1}^pred).
        Target is the stop-grad encoder feature at the adjacent slice. This closes
        the flow-matching training loop: flow-matching supervises the velocity,
        XSC supervises the integrated path, teaching the ODE to reconstruct
        adjacent-slice features from one step of its own dynamics.
        """
        t_pts = [d / max(D - 1, 1) for d in range(D)]
        preds, targets = [], []
        if direction == "fwd":
            idxs = range(D - 1)
            step = +1
        else:
            idxs = range(D - 1, 0, -1)
            step = -1
        for i in idxs:
            t0, t1 = t_pts[i], t_pts[i + step]
            dt = (t1 - t0) / self.substeps
            h = x[:, i]
            for j in range(self.substeps):
                tc = t0 + j * dt
                k1 = func(h, tc, organ_id)
                k2 = func(h + dt * k1, tc + dt, organ_id)
                h = h + 0.5 * dt * (k1 + k2)
            preds.append(h)
            targets.append(x[:, i + step].detach())
        return {
            "pred": torch.stack(preds, dim=1),
            "target": torch.stack(targets, dim=1),
        }

    def forward(
        self,
        features: torch.Tensor,       # (B, D, C, H, W)
        organ_id: torch.Tensor,       # (B,) long
        return_flow_targets: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Dict[str, torch.Tensor]]]]:
        assert organ_id.shape[0] == features.shape[0]
        B, D, C, H, W = features.shape
        x = features.permute(0, 3, 4, 1, 2).reshape(B * H * W, D, C)

        traj_fwd = self._heun(self.ode_fwd, x[:, 0], D, organ_id)
        traj_bwd = self._heun(self.ode_bwd, x[:, -1], D, organ_id)
        traj_bwd = torch.flip(traj_bwd, dims=[1])

        ode_ctx = self.merge(torch.cat([traj_fwd, traj_bwd], dim=-1))
        ode_ctx = ode_ctx.reshape(B, H, W, D, C).permute(0, 3, 4, 1, 2)
        ode_ctx = ode_ctx.permute(0, 1, 3, 4, 2)
        ode_ctx = self.norm(ode_ctx)
        ode_ctx = ode_ctx.permute(0, 1, 4, 2, 3)
        out = features + ode_ctx

        flow_targets: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
        if return_flow_targets:
            fwd_cons = self._consistency_pairs(self.ode_fwd, x, D, organ_id, "fwd")
            bwd_cons = self._consistency_pairs(self.ode_bwd, x, D, organ_id, "bwd")
            flow_targets = {
                "fwd_pairs": self._flow_pairs(self.ode_fwd, x, D, organ_id, "fwd"),
                "bwd_pairs": self._flow_pairs(self.ode_bwd, x, D, organ_id, "bwd"),
                "xsc_pairs": {
                    "pred": torch.cat([fwd_cons["pred"], bwd_cons["pred"]], dim=1),
                    "target": torch.cat([fwd_cons["target"], bwd_cons["target"]], dim=1),
                },
            }
        return out, flow_targets
