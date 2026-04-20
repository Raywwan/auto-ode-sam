"""V9 Novel #7 - Cascade consistency loss.

Both the 3D proposer and the 2D-slab refiner produce (B, K, H, W) probability
maps for the slab's center slice. Without a consistency term the refiner can
drift from the proposer's 3D context (e.g., refine a liver boundary in a slab
where the proposer assigns near-zero liver probability - proposer is usually
right because it saw the whole Z-axis). Penalize this disagreement with a
symmetric KL + soft-Dice loss.

Design discipline:
  - lam_kl = lam_dice = 0 (default) short-circuits to zero.
  - Inputs are expected in [0, 1]; we clamp before log for numerical safety.
  - Used in Stage 2 and Stage 3 training (spec S5). Not used in Stage 1.

Inputs:
  prop : (B, K, H, W) proposer prob (sliced at slab center).
  ref  : (B, K, H, W) refiner prob (sigmoid applied).

Output: {"loss": scalar, "kl": scalar, "dice": scalar}
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


def _sym_kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp(eps, 1.0 - eps)
    q = q.clamp(eps, 1.0 - eps)
    kl_pq = (p * (p.log() - q.log())).mean()
    kl_qp = (q * (q.log() - p.log())).mean()
    return 0.5 * (kl_pq + kl_qp)


def _soft_dice(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    num = 2.0 * (p * q).sum(dim=(-2, -1))
    den = (p * p).sum(dim=(-2, -1)) + (q * q).sum(dim=(-2, -1)) + eps
    return (1.0 - num / den).mean()


class CascadeConsistencyLoss(nn.Module):
    def __init__(self, lam_kl: float = 0.0, lam_dice: float = 0.0) -> None:
        super().__init__()
        self.lam_kl = lam_kl
        self.lam_dice = lam_dice

    def forward(
        self,
        prop: torch.Tensor,
        ref: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.lam_kl == 0.0 and self.lam_dice == 0.0:
            zero = torch.zeros((), device=prop.device)
            return {"loss": zero, "kl": zero, "dice": zero}

        kl = _sym_kl(prop, ref) if self.lam_kl > 0 else torch.zeros((), device=prop.device)
        dice = _soft_dice(prop, ref) if self.lam_dice > 0 else torch.zeros((), device=prop.device)
        loss = self.lam_kl * kl + self.lam_dice * dice
        return {"loss": loss, "kl": kl.detach(), "dice": dice.detach()}
