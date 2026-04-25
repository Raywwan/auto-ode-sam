"""OrganMoELoss = SmallOrganLoss + presence BCE + MoE load balance.

Three terms (each toggleable by setting weight to 0):
  - seg_loss: existing SmallOrganLoss on (B, K+1, Z, H, W) seg logits.
              Optional `use_presence_weighted_dice` injects a per-batch
              presence mask into the class-weight prior.
  - presence_bce: BCEWithLogitsLoss on (B, n_organs) presence prediction.
                  Target derived from `seg_target` (organ present iff >=1 vox).
  - balance: MoE load-balance auxiliary loss following Switch-Transformer.
             `(N_experts * sum_k (P_k * F_k))` where P_k is mean router prob
             across the batch and F_k is the fraction-of-tokens routed to k.
             Encourages uniform expert usage.
"""
from __future__ import annotations
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from training.losses_small_organ import SmallOrganLoss


class OrganMoELoss(nn.Module):
    def __init__(
        self,
        n_classes: int,
        presence_weight: float = 0.1,
        balance_weight: float = 0.01,
        use_presence_weighted_dice: bool = True,
        small_organ_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.n_classes = int(n_classes)
        self.n_organs = int(n_classes) - 1  # exclude bg
        self.pres_w = float(presence_weight)
        self.bal_w = float(balance_weight)
        self.use_presence_weighted_dice = bool(use_presence_weighted_dice)
        kw = dict(small_organ_kwargs or {})
        self.seg_loss = SmallOrganLoss(n_classes=n_classes, **kw)
        self.bce = nn.BCEWithLogitsLoss()

    def _derive_presence_target(self, seg_target: torch.Tensor) -> torch.Tensor:
        """Return (B, n_organs) bool presence target (1.0 if organ >=1 voxel)."""
        B = seg_target.shape[0]
        target = torch.zeros(B, self.n_organs, device=seg_target.device, dtype=torch.float32)
        for b in range(B):
            uniq = torch.unique(seg_target[b])
            for v in uniq.tolist():
                if 0 < v <= self.n_organs:
                    target[b, v - 1] = 1.0
        return target

    def _load_balance(self, gate_weights: torch.Tensor) -> torch.Tensor:
        """Switch-Transformer load balance loss.

        gate_weights: (B, K, ...) — softmax over K experts at each token.
        Returns scalar = K * mean( P_k * F_k ).
        """
        B = gate_weights.shape[0]
        K = gate_weights.shape[1]
        # Flatten spatial dims into "tokens".
        gate_flat = gate_weights.reshape(B, K, -1)              # (B, K, T)
        # P_k = mean router prob over all tokens
        P = gate_flat.mean(dim=(0, 2))                            # (K,)
        # F_k = fraction of tokens whose argmax router is expert k
        argmax = gate_flat.argmax(dim=1)                          # (B, T)
        F_ = torch.zeros_like(P)
        for k in range(K):
            F_[k] = (argmax == k).float().mean()
        return K * (P * F_).sum()

    def forward(
        self,
        seg_logits: torch.Tensor,
        seg_target: torch.Tensor,
        presence_logits: Optional[torch.Tensor] = None,
        gate_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}

        if self.use_presence_weighted_dice:
            present = self._derive_presence_target(seg_target)            # (B, n_organs)
            # Convert to per-class prior (K+1 long; bg always 1.0)
            class_prior = torch.ones(self.n_classes, device=seg_logits.device)
            # If any sample in batch contains an organ, weight stays 1.0;
            # if NO sample contains it, drop weight to a small value (0.1).
            for k in range(self.n_organs):
                if present[:, k].sum() == 0:
                    class_prior[k + 1] = 0.1
            self.seg_loss.class_prior = class_prior
        seg = self.seg_loss(seg_logits, seg_target.long())
        out["seg"] = seg

        if presence_logits is not None and self.pres_w > 0:
            presence_target = self._derive_presence_target(seg_target)
            pres = self.bce(presence_logits, presence_target)
            out["presence_bce"] = pres
        else:
            out["presence_bce"] = seg.new_zeros(())

        if gate_weights is not None and self.bal_w > 0:
            bal = self._load_balance(gate_weights)
            out["balance"] = bal
        else:
            out["balance"] = seg.new_zeros(())

        out["total"] = seg + self.pres_w * out["presence_bce"] + self.bal_w * out["balance"]
        return out


if __name__ == "__main__":
    torch.manual_seed(0)
    crit = OrganMoELoss(n_classes=16)
    logits = torch.randn(2, 16, 8, 16, 16, requires_grad=True)
    target = torch.randint(0, 16, (2, 8, 16, 16))
    presence_logits = torch.randn(2, 15, requires_grad=True)
    gate_weights = torch.randn(2, 8, 8, 16, 16).softmax(dim=1)
    out = crit(logits, target, presence_logits, gate_weights)
    out["total"].backward()
    print(f"total={out['total'].item():.4f} seg={out['seg'].item():.4f} "
          f"pres={out['presence_bce'].item():.4f} bal={out['balance'].item():.4f}")
