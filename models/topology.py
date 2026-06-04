# =============================================================================
# models/topology.py — Topology-Preserving Loss (clDice)
#
# PURPOSE:
#   Standard Dice loss optimises pixel overlap but ignores connectivity.
#   For vessels, airways, and other tubular structures, a segmentation can have
#   high Dice but still miss critical connections (broken vessel trees).
#
#   clDice (Centreline Dice) addresses this by:
#   1. Computing soft skeletons of both prediction and ground truth
#   2. Measuring how well each skeleton is covered by the other's mask
#   3. Combining into a topologically-aware Dice score
#
#   This loss enforces that:
#   - The predicted skeleton lies inside the GT mask  (precision of connectivity)
#   - The GT skeleton lies inside the predicted mask  (recall of connectivity)
#
# REFERENCE:
#   Shit et al., "clDice - a Novel Topology-Preserving Loss Function for Tubular
#   Structure Segmentation", CVPR 2021.
#   https://arxiv.org/abs/2003.07311
#
# USAGE:
#   Enable by setting w_cldice > 0 in config (recommended: 0.3 for vessels).
#   Not needed for organ segmentation (use standard Dice+Focal there).
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_skeletonize(x: torch.Tensor, num_iterations: int = 10) -> torch.Tensor:
    """
    Differentiable soft skeletonisation via iterative min-pooling erosion.

    The hard skeletonisation algorithm (Zhang-Suen thinning) is non-differentiable.
    This soft version approximates it using:
      - Repeated morphological erosion (min-pooling = morphological erosion)
      - Each iteration removes the outermost "layer" of the foreground

    The result is a soft (differentiable) approximation of the centreline/skeleton.

    Algorithm:
      1. Start with binary-ish prediction (probabilities in [0, 1])
      2. Repeatedly erode by replacing each pixel with the minimum
         of its local neighbourhood (3x3 kernel)
      3. The result after N iterations is a thin, centreline-like structure

    Args:
        x: (B, 1, H, W) — predicted probabilities (after sigmoid)
        num_iterations: Depth of skeletonisation (more iterations = thinner)

    Returns:
        skeleton: (B, 1, H, W) — soft skeleton (values in [0, 1])
    """
    # We iteratively erode by taking the min in a 3x3 neighbourhood
    # Min-pooling = morphological erosion
    skeleton = x.clone()

    for _ in range(num_iterations):
        # Min-pooling: erode by taking minimum in 3x3 neighbourhood
        # Note: -max(-x) = min(x) — a common trick since PyTorch has max_pool
        eroded = -F.max_pool2d(-skeleton, kernel_size=3, stride=1, padding=1)

        # The skeleton is the residual: areas that "disappear" under erosion
        # We keep the maximum between current skeleton and the erosion residual
        skeleton = torch.max(skeleton, eroded + (skeleton - eroded).clamp(min=0))

    return skeleton


class clDiceLoss(nn.Module):
    """
    Centreline Dice (clDice) Loss for topology-preserving segmentation.

    Combines standard Dice (pixel overlap) with centreline Dice (skeleton overlap).

    Final loss:
      L_clDice = 1 - clDice(pred, gt)
      clDice = 2 * T_prec * T_sens / (T_prec + T_sens)

    where:
      T_prec = overlap(skel(pred), gt)  / sum(skel(pred))   — skeleton precision
      T_sens = overlap(skel(gt), pred)  / sum(skel(gt))     — skeleton sensitivity

    Args:
        num_iterations (int): Depth of soft skeletonisation.
        smooth (float): Numerical stability constant.
    """

    def __init__(self, num_iterations: int = 10, smooth: float = 1e-5):
        super().__init__()
        self.num_iterations = num_iterations
        self.smooth = smooth

    def forward(
        self,
        pred: torch.Tensor,    # (B, 1, H, W) — raw logits
        target: torch.Tensor,  # (B, 1, H, W) — binary ground truth
    ) -> torch.Tensor:
        """
        Compute clDice loss.

        Args:
            pred: Raw logits (before sigmoid). Shape: (B, 1, H, W) or (B, H, W).
            target: Binary ground truth. Shape: (B, 1, H, W) or (B, H, W).

        Returns:
            loss: Scalar clDice loss value.
        """
        # Ensure correct shape (B, 1, H, W)
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)

        # Convert logits to probabilities
        pred_prob = torch.sigmoid(pred)
        target = target.float()

        # ----- Compute soft skeletons -----
        skel_pred = soft_skeletonize(pred_prob, self.num_iterations)
        skel_target = soft_skeletonize(target, self.num_iterations)

        # ----- Topology precision: skeleton of pred covered by GT mask -----
        # T_prec = |skel(pred) ∩ GT| / |skel(pred)|
        t_prec = (
            (skel_pred * target).sum()
            / (skel_pred.sum() + self.smooth)
        )

        # ----- Topology sensitivity: skeleton of GT covered by pred mask -----
        # T_sens = |skel(GT) ∩ pred| / |skel(GT)|
        t_sens = (
            (skel_target * pred_prob).sum()
            / (skel_target.sum() + self.smooth)
        )

        # ----- clDice score -----
        cl_dice_score = (2.0 * t_prec * t_sens) / (t_prec + t_sens + self.smooth)

        return 1.0 - cl_dice_score


class TopologyLoss(nn.Module):
    """
    Combined topology loss: Dice + clDice.

    Provides both pixel-level overlap (Dice) and connectivity preservation (clDice).

    Args:
        w_dice (float): Weight for standard Dice loss.
        w_cldice (float): Weight for clDice loss.
        cldice_iterations (int): Skeletonisation depth.
    """

    def __init__(
        self,
        w_dice: float = 0.5,
        w_cldice: float = 0.5,
        cldice_iterations: int = 10,
    ):
        super().__init__()
        self.w_dice = w_dice
        self.w_cldice = w_cldice
        self.cldice = clDiceLoss(num_iterations=cldice_iterations)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        # Standard Dice loss
        pred_prob = torch.sigmoid(pred).flatten(1)
        target_flat = target.float().flatten(1)
        intersection = (pred_prob * target_flat).sum(1)
        dice_loss = 1.0 - (2.0 * intersection + 1e-5) / (
            pred_prob.sum(1) + target_flat.sum(1) + 1e-5
        )
        dice_loss = dice_loss.mean()

        # clDice loss
        cldice_loss = self.cldice(pred, target)

        return self.w_dice * dice_loss + self.w_cldice * cldice_loss
