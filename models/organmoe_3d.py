"""OrganMoE-3D -- Class-Presence-Aware Sparse LoRA Mixture-of-Experts.

Three building blocks:

  1) MoELoRALinear: drop-in replacement for nn.Linear that delegates to
     K LoRA-rank-r experts. Each expert is a (lora_A, lora_B) pair sharing
     the frozen base Linear's weights. Top-k sparse routing.

  2) PresenceConditionedRouter: small MLP that takes (token feature, organ
     presence vector) and outputs K logits per token. Top-k softmax gives
     the per-token gating weights and active expert ids.

  3) PresenceHead: a tiny MLP on globally-pooled low-res encoder features
     that predicts a 15-organ presence vector c in [0, 1]^15.

Injection helper (used by SwinUNETRProposer): replace every named Linear
(qkv/proj/fc1/fc2) inside the swinViT encoder with a MoELoRALinear, sharing
the same router config but distinct expert parameters.
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class _LoRAExpert(nn.Module):
    """One LoRA expert: low-rank delta on a frozen base Linear."""
    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = int(rank)
        self.scale = float(alpha) / max(self.rank, 1)
        self.lora_A = nn.Parameter(torch.zeros(self.rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.lora_B, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features) -> (..., out_features)
        return F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale


class PresenceConditionedRouter(nn.Module):
    """Top-k softmax router over K experts; conditioned on token feature + presence vector."""

    def __init__(self, in_features: int, n_experts: int, n_organs: int, top_k: int = 2,
                 hidden: int = 64) -> None:
        super().__init__()
        self.n_experts = int(n_experts)
        self.top_k = int(top_k)
        self.n_organs = int(n_organs)
        self.proj = nn.Sequential(
            nn.Linear(in_features + n_organs, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_experts),
        )

    def forward(self, x: torch.Tensor, organ_presence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (B, T, D)  organ_presence: (B, n_organs)
        Returns:
          gate: (B, K, T) sparse top-k softmax weights (zero outside top-k)
          topk_idx: (B, T, top_k) expert indices
        """
        B, T, D = x.shape
        # Broadcast presence to per-token: (B, 1, n_organs) -> (B, T, n_organs)
        pres_t = organ_presence.unsqueeze(1).expand(B, T, -1)
        cat = torch.cat([x, pres_t], dim=-1)            # (B, T, D + n_organs)
        logits = self.proj(cat)                          # (B, T, K)
        # Top-k masking
        topk_vals, topk_idx = logits.topk(self.top_k, dim=-1)
        mask = torch.full_like(logits, float("-inf"))
        mask.scatter_(-1, topk_idx, topk_vals)
        gate = F.softmax(mask, dim=-1)                   # (B, T, K) zeroed outside top-k
        gate = gate.transpose(1, 2)                      # (B, K, T) for downstream conv
        return gate, topk_idx


class MoELoRALinear(nn.Module):
    """Drop-in replacement for an nn.Linear: y = base(x) + sum_k g_k(x) * expert_k(x)."""

    def __init__(self, base: nn.Linear, n_experts: int, rank: int, alpha: float,
                 n_organs: int, top_k: int = 2, router_hidden: int = 64) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.n_experts = int(n_experts)
        self.top_k = int(top_k)
        self.experts = nn.ModuleList([
            _LoRAExpert(base, rank=rank, alpha=alpha) for _ in range(n_experts)
        ])
        self.router = PresenceConditionedRouter(
            in_features=self.in_features, n_experts=n_experts,
            n_organs=n_organs, top_k=top_k, hidden=router_hidden,
        )

    def forward(self, x: torch.Tensor, organ_presence: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (..., in_features). organ_presence: (B, n_organs) or None.
        If organ_presence is None, defaults to zeros (uniform routing).
        Returns: (output (..., out_features), gate_weights (B, K, T)).
        """
        # Reshape arbitrary leading dims to (B, T, D)
        orig_shape = x.shape
        if x.dim() == 2:
            x_bt = x.unsqueeze(0)            # (1, T, D)
        elif x.dim() == 3:
            x_bt = x                          # (B, T, D)
        else:
            # Fold all non-feature dims into T.
            B = orig_shape[0]
            x_bt = x.reshape(B, -1, self.in_features)
        B, T, D = x_bt.shape
        if organ_presence is None:
            organ_presence = torch.zeros(B, self.router.n_organs, device=x.device, dtype=x.dtype)
        gate, _topk = self.router(x_bt, organ_presence)        # gate: (B, K, T)

        base_out = self.base(x_bt)                              # (B, T, out)
        expert_out = base_out.new_zeros(B, T, self.out_features)
        for k in range(self.n_experts):
            g_k = gate[:, k, :].unsqueeze(-1)                   # (B, T, 1)
            if g_k.abs().sum() == 0:
                continue
            e_k = self.experts[k](x_bt)                          # (B, T, out)
            expert_out = expert_out + g_k * e_k
        out = base_out + expert_out

        # Restore original leading shape
        if len(orig_shape) == 2:
            out = out.squeeze(0)
        elif len(orig_shape) > 3:
            out = out.reshape(*orig_shape[:-1], self.out_features)
        return out, gate


class PresenceHead(nn.Module):
    """Predict (B, n_organs) presence logits from a low-res encoder feature map."""

    def __init__(self, in_channels: int, n_organs: int, hidden: int = 128) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden), nn.GELU(),
            nn.Linear(hidden, n_organs),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.pool(feat))


def inject_organmoe_into_swin(
    swin_module: nn.Module,
    n_experts: int = 8,
    rank: int = 16,
    alpha: float = 16.0,
    n_organs: int = 15,
    top_k: int = 2,
) -> int:
    """Walk swin_module; replace every Linear named qkv/proj/fc1/fc2 with
    a MoELoRALinear. Returns the count of replaced modules.
    """
    n = 0
    for name, child in list(swin_module.named_children()):
        if isinstance(child, nn.Linear) and name in ("qkv", "proj", "fc1", "fc2"):
            setattr(swin_module, name, MoELoRALinear(
                base=child, n_experts=n_experts, rank=rank, alpha=alpha,
                n_organs=n_organs, top_k=top_k,
            ))
            n += 1
        else:
            n += inject_organmoe_into_swin(
                child, n_experts=n_experts, rank=rank, alpha=alpha,
                n_organs=n_organs, top_k=top_k,
            )
    return n


if __name__ == "__main__":
    # Module-level sanity smoke
    base = nn.Linear(128, 128)
    moe = MoELoRALinear(base=base, n_experts=8, rank=16, alpha=16.0, n_organs=15, top_k=2)
    x = torch.randn(2, 64, 128)
    p = torch.rand(2, 15)
    y, g = moe(x, p)
    print(f"in={x.shape} out={y.shape} gate={g.shape} sum_gate~top_k? "
          f"{g.sum(dim=1).mean().item():.3f} (expected ~1.0)")
    head = PresenceHead(in_channels=384, n_organs=15)
    feat = torch.randn(2, 384, 6, 6, 6)
    print(f"presence={head(feat).shape}")
