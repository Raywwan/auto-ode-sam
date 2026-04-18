"""V4 composite loss:
    L_total = L_mask + λ_flow · L_flow + λ_deepsup · L_deepsup + λ_anatomy · L_anatomy
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss_binary(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-sample soft Dice on sigmoid probabilities. pred: (N, H, W), target: (N, H, W)."""
    p = torch.sigmoid(pred)
    inter = (p * target).flatten(1).sum(1)
    union = p.flatten(1).sum(1) + target.flatten(1).sum(1)
    return 1.0 - (2.0 * inter + eps) / (union + eps)


class V4MultiOrganLoss(nn.Module):
    def __init__(
        self,
        lambda_flow: float = 0.5,
        lambda_deepsup: float = 0.1,
        lambda_anatomy: float = 0.01,
    ) -> None:
        super().__init__()
        self.lambda_flow = lambda_flow
        self.lambda_deepsup = lambda_deepsup
        self.lambda_anatomy = lambda_anatomy

    def forward(
        self,
        pred_masks: torch.Tensor,                                  # (B, 15, H, W) logits
        gt_masks: torch.Tensor,                                     # (B, 15, H, W) or (B, 15, D, H, W) uint8
        present_mask: torch.Tensor,                                 # (B, 15) bool
        flow_targets: Optional[Dict] = None,
        deepsup_logits: Optional[torch.Tensor] = None,
        anatomy_adjacency: Optional[torch.Tensor] = None,
        anatomy_adjacency_init: Optional[torch.Tensor] = None,
        lambda_flow_override: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        if gt_masks.dim() == 5:
            D = gt_masks.shape[2]
            gt_masks = gt_masks[:, :, D // 2, :, :]                 # center slice only for mask loss
        target = gt_masks.float()
        if target.shape[-2:] != pred_masks.shape[-2:]:
            target = F.interpolate(target, size=pred_masks.shape[-2:], mode="nearest")

        B, K, H, W = pred_masks.shape
        flat_pred = pred_masks.reshape(B * K, H, W)
        flat_tgt = target.reshape(B * K, H, W)
        mask_ok = present_mask.reshape(B * K)

        bce = F.binary_cross_entropy_with_logits(flat_pred, flat_tgt, reduction="none").mean(dim=(-1, -2))
        dsc = dice_loss_binary(flat_pred, flat_tgt)

        if mask_ok.any():
            L_mask = (bce[mask_ok].mean() + dsc[mask_ok].mean()) * 0.5
        else:
            L_mask = pred_masks.sum() * 0.0

        components: Dict[str, float] = {"mask": L_mask.item()}
        L_flow = pred_masks.sum() * 0.0
        L_deepsup = pred_masks.sum() * 0.0
        L_anatomy = pred_masks.sum() * 0.0

        if flow_targets is not None:
            fwd = flow_targets["fwd_pairs"]
            bwd = flow_targets["bwd_pairs"]
            L_flow = 0.5 * (F.mse_loss(fwd["pred"], fwd["target"]) + F.mse_loss(bwd["pred"], bwd["target"]))
            components["flow"] = L_flow.item()

        if deepsup_logits is not None:
            ds_tgt = target
            if ds_tgt.shape[-2:] != deepsup_logits.shape[-2:]:
                ds_tgt = F.interpolate(ds_tgt, size=deepsup_logits.shape[-2:], mode="nearest")
            L_deepsup = F.binary_cross_entropy_with_logits(deepsup_logits, ds_tgt)
            components["deepsup"] = L_deepsup.item()

        if anatomy_adjacency is not None and anatomy_adjacency_init is not None:
            L_anatomy = F.mse_loss(anatomy_adjacency, anatomy_adjacency_init)
            components["anatomy"] = L_anatomy.item()

        lf = self.lambda_flow if lambda_flow_override is None else lambda_flow_override
        total = L_mask + lf * L_flow + self.lambda_deepsup * L_deepsup + self.lambda_anatomy * L_anatomy
        components["total"] = total.item()
        return {"loss": total, "components": components}
