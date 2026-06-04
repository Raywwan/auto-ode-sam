# =============================================================================
# models/soft_moe.py — Soft Mixture-of-Experts Organ Router
#
# Soft MoE routes each spatial token to a weighted combination of n_experts
# specialised FFNs.  Unlike hard top-k MoE (e.g. Switch Transformer), every
# token uses EVERY expert — just with different learned weights.
#
# Why soft MoE for abdominal CT:
#   - Background tokens (~90% of spatial positions) should use a "background"
#     expert (low activation, cheap).
#   - Organ tokens should use organ-specific experts (liver, spleen, etc.).
#   - Hard routing breaks gradients; soft routing keeps the whole path
#     differentiable with no load-balancing auxiliary loss needed.
#
# Reference design inspired by:
#   Puigcerver et al., "From Sparse to Soft Mixtures of Experts", ICLR 2024.
#   SegMoTE (arXiv:2602.19213) — token-level MoE adapters for SAM.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftMoEOrganRouter(nn.Module):
    """
    Soft Mixture-of-Experts for organ-specific feature routing.

    Each token (spatial position in the feature map) receives a soft
    assignment to n_experts FFNs.  The output is a weighted sum of all
    expert outputs.  Residual + LayerNorm applied.

    Args:
        d_model   (int): Token feature dimension (embed_dim).
        n_experts (int): Number of expert FFNs.  Default: 2 (lightweight).
        d_ff      (int): Hidden dim inside each expert FFN.  Default: d_model.
    """

    def __init__(self, d_model: int, n_experts: int = 2, d_ff: int = None):
        super().__init__()
        d_ff = d_ff if d_ff is not None else d_model
        self.n_experts = n_experts

        # ---- Router: maps each token to a probability distribution ----
        # Linear(d_model → n_experts) + softmax (applied in forward)
        self.router = nn.Linear(d_model, n_experts, bias=False)

        # ---- Expert FFNs ----
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model),
            )
            for _ in range(n_experts)
        ])

        # ---- Residual normalisation ----
        self.norm = nn.LayerNorm(d_model)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, d_model)  — B × spatial tokens × channels.
               B can be B_batch*D_slices when called on a flattened volume.

        Returns:
            out: (B, N, d_model)  with residual + LayerNorm applied
        """
        # ---- Routing weights ----
        # (B, N, n_experts) — each token's soft assignment vector
        gates = torch.softmax(self.router(x), dim=-1)  # (B, N, E)

        # ---- Expert outputs ----
        # Stack into (E, B, N, d_model) for easy einsum
        expert_outs = torch.stack(
            [e(x) for e in self.experts], dim=0
        )  # (E, B, N, d_model)

        # ---- Weighted combination ----
        # sum_e  gates[b,n,e] * expert_outs[e,b,n,:]
        out = torch.einsum("bne,ebnd->bnd", gates, expert_outs)  # (B, N, d_model)

        # ---- Residual + LayerNorm ----
        return self.norm(x + out)

    def extra_repr(self) -> str:
        return f"n_experts={self.n_experts}, d_model={self.experts[0][0].in_features}"
