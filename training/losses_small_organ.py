"""Small-organ weighted 3D segmentation loss for V9 Phase C.

Rationale (2026-04-24): the FT proposer 3D eval shows weak organs (pancreas,
adrenals, duodenum, gallbladder, stomach) at <0.30 Dice while large organs are
>0.70. Standard CE + soft-Dice is dominated by voxel counts, so the small
organs contribute tiny gradients. This module combines three well-known
ingredients tuned for small-organ recall:

  - Inverse-sqrt-volume weighted soft Dice: the per-class Dice term is scaled
    by 1/sqrt(|gt_c| + eps), normalised to mean 1 across present classes.
  - Tversky (alpha=0.3, beta=0.7): penalises false negatives ~2.3x more than
    false positives; recovers under-segmented tubular/small structures.
  - Focal cross-entropy (gamma=1.5): down-weights easy background voxels, so
    the boundary / rare-class voxels carry the gradient.

Usage:
    criterion = SmallOrganLoss(n_classes=K+1, ignore_bg_in_dice=True)
    loss = criterion(logits, target)   # logits (B, K+1, Z, H, W), target (B, Z, H, W) long

Design notes:
  - Non-novel per se (Tversky, focal, inv-vol are all public), but combined
    behind a single flag and tuned for AMOS22 CT small organs.
  - Background (class 0) is included in CE/focal but excluded from Dice /
    Tversky by default, matching nnU-Net convention.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallOrganLoss(nn.Module):
    def __init__(
        self,
        n_classes: int,
        dice_weight: float = 1.0,
        tversky_weight: float = 1.0,
        focal_weight: float = 1.0,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        focal_gamma: float = 1.5,
        ignore_bg_in_dice: bool = True,
        eps: float = 1e-6,
        class_weight_prior: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        self.n_classes = int(n_classes)
        self.dice_w = float(dice_weight)
        self.tv_w = float(tversky_weight)
        self.fc_w = float(focal_weight)
        self.alpha = float(tversky_alpha)
        self.beta = float(tversky_beta)
        self.gamma = float(focal_gamma)
        self.ignore_bg = bool(ignore_bg_in_dice)
        self.eps = float(eps)
        if class_weight_prior is not None:
            cw = torch.as_tensor(list(class_weight_prior), dtype=torch.float32)
            assert cw.numel() == n_classes
            self.register_buffer("class_prior", cw, persistent=False)
        else:
            self.class_prior = None

    def _one_hot(self, target: torch.Tensor) -> torch.Tensor:
        return F.one_hot(target.clamp(0, self.n_classes - 1), self.n_classes) \
                .permute(0, 4, 1, 2, 3).float()

    def _inv_sqrt_vol_weights(self, oh: torch.Tensor) -> torch.Tensor:
        vols = oh.sum(dim=(0, 2, 3, 4)).clamp(min=self.eps)
        w = 1.0 / torch.sqrt(vols)
        if self.ignore_bg:
            w[0] = 0.0
        present = w > 0
        n_pres = present.sum().clamp(min=1)
        w = torch.where(present, w * (n_pres / w[present].sum().clamp(min=self.eps)), w)
        return w

    def _weighted_soft_dice(self, probs: torch.Tensor, oh: torch.Tensor) -> torch.Tensor:
        w = self._inv_sqrt_vol_weights(oh)
        if self.class_prior is not None:
            w = w * self.class_prior.to(w.device)
        inter = (probs * oh).sum(dim=(0, 2, 3, 4))
        denom = probs.sum(dim=(0, 2, 3, 4)) + oh.sum(dim=(0, 2, 3, 4))
        dice_c = (2.0 * inter + self.eps) / (denom + self.eps)
        loss_c = 1.0 - dice_c
        return (w * loss_c).sum() / w.sum().clamp(min=self.eps)

    def _weighted_tversky(self, probs: torch.Tensor, oh: torch.Tensor) -> torch.Tensor:
        w = self._inv_sqrt_vol_weights(oh)
        if self.class_prior is not None:
            w = w * self.class_prior.to(w.device)
        tp = (probs * oh).sum(dim=(0, 2, 3, 4))
        fp = (probs * (1.0 - oh)).sum(dim=(0, 2, 3, 4))
        fn = ((1.0 - probs) * oh).sum(dim=(0, 2, 3, 4))
        tv_c = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        loss_c = 1.0 - tv_c
        return (w * loss_c).sum() / w.sum().clamp(min=self.eps)

    def _focal_ce(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        target_flat = target.clamp(0, self.n_classes - 1)
        gather_idx = target_flat.unsqueeze(1)
        lp_t = log_probs.gather(1, gather_idx).squeeze(1)
        p_t = lp_t.exp().clamp(min=self.eps, max=1.0 - self.eps)
        focal_term = (1.0 - p_t).pow(self.gamma)
        if self.class_prior is not None:
            prior = self.class_prior.to(logits.device)
            w_pix = prior[target_flat]
            return -(w_pix * focal_term * lp_t).mean()
        return -(focal_term * lp_t).mean()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert logits.shape[1] == self.n_classes, (
            f"logits channel {logits.shape[1]} != n_classes {self.n_classes}"
        )
        probs = F.softmax(logits, dim=1)
        oh = self._one_hot(target.long())
        total = logits.new_zeros(())
        if self.dice_w > 0:
            total = total + self.dice_w * self._weighted_soft_dice(probs, oh)
        if self.tv_w > 0:
            total = total + self.tv_w * self._weighted_tversky(probs, oh)
        if self.fc_w > 0:
            total = total + self.fc_w * self._focal_ce(logits, target.long())
        return total


if __name__ == "__main__":
    torch.manual_seed(0)
    crit = SmallOrganLoss(n_classes=16)
    logits = torch.randn(2, 16, 8, 16, 16, requires_grad=True)
    target = torch.randint(0, 16, (2, 8, 16, 16))
    loss = crit(logits, target)
    loss.backward()
    print(f"loss={loss.item():.4f}  grad_norm={logits.grad.norm().item():.4f}")
