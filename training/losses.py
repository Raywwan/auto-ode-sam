# =============================================================================
# training/losses.py — Loss Functions
#
# LiteSAM-3D uses a combination of:
#
#   1. DICE LOSS — Optimises pixel overlap (F1 score for segmentation)
#      Good at: handling class imbalance (foreground is usually tiny)
#      Weakness: doesn't care about connectivity
#
#   2. FOCAL LOSS — Variant of BCE that down-weights easy examples
#      Good at: forcing the model to focus on hard/ambiguous pixels
#      Parameter: gamma controls how much hard examples are upweighted
#
#   3. clDICE LOSS (optional) — Topology-preserving loss
#      Good at: preserving vessel connectivity, preventing broken centrelines
#      Enable: set w_cldice > 0 in config (recommended: 0.3 for vessel tasks)
#
#   4. IoU REGRESSION LOSS — Supervises IoU prediction head
#      Good at: making the IoU prediction head reliable for mask selection
#
# Default weighting (organ segmentation):
#   L = 0.5 * Dice + 0.5 * Focal + 0.1 * IoU
#
# Vessel task weighting:
#   L = 0.4 * Dice + 0.3 * Focal + 0.3 * clDice + 0.1 * IoU
# =============================================================================

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.topology import clDiceLoss


# =============================================================================
# Individual Loss Components
# =============================================================================

class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.

    Dice = 2 * |P ∩ T| / (|P| + |T|)
    Loss = 1 - Dice (minimise)

    Args:
        smooth (float): Numerical stability constant (prevents division by zero).
    """

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        pred: torch.Tensor,                           # (B, H, W) or (B, 1, H, W) — raw logits
        target: torch.Tensor,                         # (B, H, W) or (B, 1, H, W) — binary GT
        sample_weight: Optional[torch.Tensor] = None, # (B,) per-sample loss multipliers
    ) -> torch.Tensor:
        # Apply sigmoid to get probabilities
        pred_prob = torch.sigmoid(pred).flatten(1)  # (B, H*W)
        target_flat = target.float().flatten(1)     # (B, H*W)

        # Soft Dice computation
        intersection = (pred_prob * target_flat).sum(dim=1)  # (B,)
        dice = (2.0 * intersection + self.smooth) / (
            pred_prob.sum(dim=1) + target_flat.sum(dim=1) + self.smooth
        )
        loss_per_sample = 1.0 - dice  # (B,)
        if sample_weight is not None:
            loss_per_sample = loss_per_sample * sample_weight
        return loss_per_sample.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for binary segmentation.

    Focal = -alpha * (1 - p_t)^gamma * log(p_t)

    Down-weights easy-to-classify pixels and focuses on hard ones.
    Critical for medical imaging where background >> foreground.

    Args:
        alpha (float): Weighting factor for positive class (foreground).
                       alpha=0.25 means foreground is weighted 0.25,
                       background is weighted 0.75.
        gamma (float): Focusing parameter. gamma=0 → standard BCE.
                       gamma=2 is the standard choice (Lin et al., 2017).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        pred: torch.Tensor,    # (B, H, W) raw logits
        target: torch.Tensor,  # (B, H, W) binary GT
    ) -> torch.Tensor:
        target = target.float()

        # Standard BCE
        bce_loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction="none"
        )  # (B, H, W)

        # p_t: probability of the correct class
        prob = torch.sigmoid(pred)
        p_t = target * prob + (1 - target) * (1 - prob)

        # Alpha weighting: alpha for positive, (1-alpha) for negative
        alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)

        # Focal weighting: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        focal_loss = alpha_t * focal_weight * bce_loss
        return focal_loss.mean()


class IoUPredictionLoss(nn.Module):
    """
    MSE loss for the IoU prediction head.

    The IoU head predicts how good each mask is (0=bad, 1=perfect).
    We supervise it with the actual IoU between the predicted mask and GT.

    This makes the IoU head reliable for mask selection at inference.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,     # (B, H, W) raw mask logit
        target: torch.Tensor,   # (B, H, W) binary GT
        iou_pred: torch.Tensor, # (B,) predicted IoU score
    ) -> torch.Tensor:
        # Compute actual IoU between prediction and target
        pred_bin = (torch.sigmoid(pred) > 0.5).float().flatten(1)  # (B, H*W)
        target_flat = target.float().flatten(1)                     # (B, H*W)

        intersection = (pred_bin * target_flat).sum(dim=1)  # (B,)
        union = pred_bin.sum(dim=1) + target_flat.sum(dim=1) - intersection  # (B,)
        true_iou = (intersection + 1e-5) / (union + 1e-5)  # (B,)

        # Stop gradient on true IoU — we don't want this to train the mask decoder
        true_iou = true_iou.detach()

        # MSE between predicted and actual IoU.
        # iou_pred may be (B,) or (B, N) for N mask candidates — average across N.
        iou_pred_1d = iou_pred.mean(dim=-1) if iou_pred.dim() > 1 else iou_pred.flatten()
        # Cast to float32: MSE is numerically unstable in fp16 for IoU regression
        return F.mse_loss(iou_pred_1d.float(), true_iou.float())


class DiceTopKLoss(nn.Module):
    """
    TopK Dice Loss — focuses on the k% hardest voxels.

    Instead of soft Dice over all voxels, computes Dice only on the top-k%
    voxels with the highest prediction error.  Forces the model to attend to
    the hardest (boundary/ambiguous) regions.

    Shown to be the best single loss in a 20-loss comparison study for
    medical image segmentation (class-imbalanced tasks).

    Args:
        k (float): Fraction of voxels to keep (0 < k ≤ 1). Default 0.5.
        smooth (float): Numerical stability constant.
    """

    def __init__(self, k: float = 0.5, smooth: float = 1e-5):
        super().__init__()
        self.k = k
        self.smooth = smooth

    def forward(
        self,
        pred: torch.Tensor,                           # (B, H, W) raw logits
        target: torch.Tensor,                         # (B, H, W) binary GT
        sample_weight: Optional[torch.Tensor] = None, # (B,) per-sample loss multipliers
    ) -> torch.Tensor:
        pred_prob = torch.sigmoid(pred)
        pred_flat = pred_prob.flatten(1)      # (B, N)
        target_flat = target.float().flatten(1)  # (B, N)

        # Per-voxel prediction error — high where model is most wrong
        voxel_err = (pred_flat - target_flat).abs()

        # Select top-k hardest voxels
        k_count = max(1, int(self.k * pred_flat.shape[1]))
        _, topk_idx = voxel_err.topk(k_count, dim=1)

        pred_topk = pred_flat.gather(1, topk_idx)
        target_topk = target_flat.gather(1, topk_idx)

        # Dice on the hard subset
        intersection = (pred_topk * target_topk).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            pred_topk.sum(dim=1) + target_topk.sum(dim=1) + self.smooth
        )
        loss_per_sample = 1.0 - dice  # (B,)
        if sample_weight is not None:
            loss_per_sample = loss_per_sample * sample_weight
        return loss_per_sample.mean()


class PMDiceLoss(nn.Module):
    """
    Pixel-wise Modulated Dice Loss (Hosseini, arXiv 2506.15744, 2025).

    Upgrades DiceTopKLoss: instead of a hard TopK cutoff (sort + threshold),
    applies a continuous per-voxel modulating weight based on prediction
    confidence.  Uncertain voxels get weight ≈1, confident voxels get weight ≈0.

        m_i = (1 - p_i)^γ   for foreground voxels (GT=1)
        m_i = p_i^γ          for background voxels (GT=0)

    The modulated Dice is then:

        PM-Dice = 1 - (2·Σ(m_i·p_i·g_i) + ε) / (Σ(m_i·p_i) + Σ(m_i·g_i) + ε)

    Advantages over DiceTopKLoss:
      - Fully differentiable (no sorting, no hard threshold)
      - Smooth gradient at organ boundaries — important for small organs
        where boundary voxels are the dominant learning signal
      - Paper reports +1.85–2.66 DSC over baseline Dice on multi-organ tasks
      - Zero computational overhead beyond standard Dice

    Args:
        gamma (float): Modulation exponent. gamma=2.0 matches focal loss convention.
        smooth (float): Numerical stability constant.
    """

    def __init__(self, gamma: float = 2.0, smooth: float = 1e-5):
        super().__init__()
        self.gamma = gamma
        self.smooth = smooth

    def forward(
        self,
        pred: torch.Tensor,                           # (B, H, W) raw logits
        target: torch.Tensor,                         # (B, H, W) binary GT
        sample_weight: Optional[torch.Tensor] = None, # (B,) per-sample loss multipliers
    ) -> torch.Tensor:
        p = torch.sigmoid(pred)
        g = target.float()

        p_flat = p.flatten(1)   # (B, N)
        g_flat = g.flatten(1)   # (B, N)

        # Per-voxel modulating weight: high where model is uncertain
        # m → 1 at wrong/uncertain voxels, m → 0 at confident correct voxels
        m = torch.where(g_flat > 0.5,
                        (1.0 - p_flat) ** self.gamma,   # foreground: penalise missed FG
                        p_flat ** self.gamma)            # background: penalise false FG

        # Modulated Dice numerator/denominator
        intersection = (m * p_flat * g_flat).sum(dim=1)
        union = (m * p_flat).sum(dim=1) + (m * g_flat).sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss_per_sample = 1.0 - dice  # (B,)

        if sample_weight is not None:
            loss_per_sample = loss_per_sample * sample_weight
        return loss_per_sample.mean()


class BoundaryLoss(nn.Module):
    """
    Distance-transform boundary loss (Kervadec et al., MIDL 2019).

    Minimises the sum of predicted probability × distance-transform of GT mask.
    Penalises predicting foreground far from the GT boundary.

    Distance transform computed on-the-fly via scipy (fast for D=8, H=W=256).

    Args:
        smooth (float): Epsilon for numerical stability.
    """

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    @staticmethod
    def _distance_transform(mask: torch.Tensor) -> torch.Tensor:
        """
        Compute normalised distance transform of binary mask on CPU.
        mask: (B, H, W) binary float tensor
        Returns: (B, H, W) float tensor in [0, 1]
        """
        from scipy.ndimage import distance_transform_edt
        import numpy as np
        B, H, W = mask.shape
        dt = torch.zeros_like(mask)
        # DT of background (foreground = 0, background = 1)
        mask_np = (mask.detach().cpu().numpy() == 0).astype(np.uint8)
        for b in range(B):
            raw = distance_transform_edt(mask_np[b]).astype(np.float32)
            max_val = raw.max() + 1e-5
            dt[b] = torch.from_numpy(raw / max_val)
        return dt.to(mask.device)

    def forward(
        self,
        pred:   torch.Tensor,   # (B, H, W) — logits or probabilities
        target: torch.Tensor,   # (B, H, W) — binary GT
    ) -> torch.Tensor:
        pred_prob = torch.sigmoid(pred)
        dt = self._distance_transform(target)   # (B, H, W)
        return (pred_prob * dt).mean()


class HausdorffDTLoss(nn.Module):
    """Hausdorff distance loss via distance transforms (Karimi & Salcudean, IEEE TMI 2020).

    Approximates HD by weighting per-pixel squared error with the alpha-th power
    of the GT and predicted distance transforms:

        L_HD = mean((p - g)^2 * (DT(g)^alpha + DT(p)^alpha))

    Differences from `BoundaryLoss` above (Kervadec MIDL 2019):
      - Kervadec uses `(p * DT(g)).mean()` — linear in p, only one DT, GT-only.
        Best paired with a region loss; well-behaved gradient, but does not
        explicitly punish predictions inside the GT either.
      - Karimi--Salcudean uses `(p - g)^2 * (DT(g)^alpha + DT(p)^alpha)` —
        symmetric (penalises both FN and FP), uses DT of predicted mask too,
        and aligns more directly with the HD metric. alpha=2 is the value
        reported as best in the original paper.

    Both DTs are recomputed per batch on CPU. For 256x256 axial slices this
    is ~0.5 ms/slice on a modern CPU; trivial compared to per-step model fwd.

    Args:
        alpha: Exponent on the distance transform. 2.0 is the original paper's
            recommended value.
        smooth: Numerical safety floor for the DT normalisation.
    """

    def __init__(self, alpha: float = 2.0, smooth: float = 1e-5):
        super().__init__()
        self.alpha = float(alpha)
        self.smooth = float(smooth)

    @staticmethod
    def _dt_normalised(binary_mask_BHW: torch.Tensor) -> torch.Tensor:
        """Distance transform of the COMPLEMENT of the binary mask, normalised
        per-slice to [0, 1] so the loss scale does not blow up on large images.
        Returns DT in the same dtype/device as the input.
        """
        from scipy.ndimage import distance_transform_edt
        import numpy as np

        B, H, W = binary_mask_BHW.shape
        mask_np = (binary_mask_BHW.detach().cpu().numpy() == 0).astype(np.uint8)
        out = np.zeros_like(mask_np, dtype=np.float32)
        for b in range(B):
            raw = distance_transform_edt(mask_np[b]).astype(np.float32)
            max_val = float(raw.max()) + 1e-5
            out[b] = raw / max_val
        return torch.from_numpy(out).to(binary_mask_BHW.device)

    def forward(
        self,
        pred:   torch.Tensor,   # (B, H, W) — raw logits
        target: torch.Tensor,   # (B, H, W) — binary GT
    ) -> torch.Tensor:
        if pred.dim() == 4 and pred.shape[1] == 1:
            pred = pred.squeeze(1)
        if target.dim() == 4 and target.shape[1] == 1:
            target = target.squeeze(1)

        p = torch.sigmoid(pred)
        g = target.float()

        dt_g = self._dt_normalised(g)
        with torch.no_grad():
            p_bin = (p > 0.5).float()
        dt_p = self._dt_normalised(p_bin)

        dist_weight = dt_g.pow(self.alpha) + dt_p.pow(self.alpha)
        return ((p - g).pow(2) * dist_weight).mean()


class BoundaryDoULoss(nn.Module):
    """Boundary Difference-over-Union loss (Sun et al., MICCAI 2023; arXiv:2308.00220).

    Region-only, distance-transform-free generalisation of Dice that puts extra
    weight on the boundary region of large foreground objects. Defined as

        L_DoU = (|G \\cup P| - |G \\cap P|) / (|G \\cup P| - alpha * |G \\cap P|),

    equivalently `S_D / (S_D + (1 - alpha) * S_I)` where `S_D` is the symmetric
    difference (FP+FN) and `S_I` is the intersection (TP) under soft predictions.
    `alpha` in [0, 1) is set per-sample from the GT geometry:

        alpha = clamp(1 - 2 * C / S, 0, alpha_max),

    where `C` is the boundary length of the GT (1-pixel boundary via 3x3
    erosion) and `S` is its area. Large organs (small C/S) get alpha close to
    1 so the loss is dominated by boundary mismatches; small organs get alpha
    near 0 so the loss reduces toward an IoU-like region loss and the entire
    structure is supervised. `alpha_max < 1` keeps the denominator strictly
    positive for very large flat regions (paper figures use alpha=0.8).

    The boundary is extracted via a 3x3 max-pool minus a 3x3 erosion on the
    binarised GT, computed once per forward call on-device with no scipy
    dependency — keeps the loss differentiable through p and side-steps the
    distance-transform CPU bottleneck shared by `BoundaryLoss` and
    `HausdorffDTLoss`.
    """

    def __init__(
        self,
        alpha_max: float = 0.8,
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.alpha_max = float(alpha_max)
        self.smooth = float(smooth)

    @staticmethod
    def _gt_boundary_and_area(target_BHW: torch.Tensor):
        """Per-sample boundary length C (1-pixel ring) and foreground area S.

        Returns:
            C: (B,) float tensor — number of 1-pixel boundary voxels of GT.
            S: (B,) float tensor — number of foreground voxels of GT.
        """
        g = (target_BHW > 0.5).float()  # (B, H, W) binary
        # 3x3 erosion: a voxel survives iff all 9 neighbours are foreground.
        # min-pool == -max-pool(-x); for {0,1} mask this equals neighbourhood
        # min, which is the binary erosion.
        eroded = -F.max_pool2d(-g.unsqueeze(1), kernel_size=3, stride=1, padding=1)
        eroded = eroded.squeeze(1)
        boundary = (g - eroded).clamp_min_(0.0)  # 1-pixel inner boundary
        C = boundary.flatten(1).sum(dim=1)
        S = g.flatten(1).sum(dim=1)
        return C, S

    def forward(
        self,
        pred: torch.Tensor,    # (B, H, W) or (B, 1, H, W) raw logits
        target: torch.Tensor,  # (B, H, W) or (B, 1, H, W) binary GT
        sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if pred.dim() == 4 and pred.shape[1] == 1:
            pred = pred.squeeze(1)
        if target.dim() == 4 and target.shape[1] == 1:
            target = target.squeeze(1)

        prob = torch.sigmoid(pred)
        gt = target.float()

        # Soft intersection / union per-sample
        S_I = (prob * gt).flatten(1).sum(dim=1)
        S_P = prob.flatten(1).sum(dim=1)
        S_G = gt.flatten(1).sum(dim=1)
        S_U = S_P + S_G - S_I                       # |P u G|
        S_D = (S_U - S_I).clamp_min_(0.0)           # symmetric diff

        # Per-sample adaptive alpha from GT geometry
        with torch.no_grad():
            C, S = self._gt_boundary_and_area(target)
            alpha = (1.0 - 2.0 * C / S.clamp_min(1.0))
            # Empty GT (S == 0) → fall back to Dice-like (alpha = 0)
            alpha = torch.where(S < 1.0, torch.zeros_like(alpha), alpha)
            alpha = alpha.clamp(min=0.0, max=self.alpha_max)

        # L_DoU = S_D / (S_D + (1 - alpha) * S_I)
        denom = S_D + (1.0 - alpha) * S_I + self.smooth
        loss_per_sample = S_D / denom

        if sample_weight is not None:
            loss_per_sample = loss_per_sample * sample_weight
        return loss_per_sample.mean()


class ModalityClassificationLoss(nn.Module):
    """
    Cross-entropy loss on the auxiliary modality classification head.

    Predicting the modality from decoded features regularises the modality
    embeddings to be discriminative and semantically meaningful.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        modality_logits: torch.Tensor,  # (B, num_modalities)
        modality_ids: torch.Tensor,     # (B,) integer labels
    ) -> torch.Tensor:
        return F.cross_entropy(modality_logits, modality_ids)


# =============================================================================
# Combined Loss Function
# =============================================================================

class CombinedLoss(nn.Module):
    """
    Combined segmentation loss: Dice + Focal + [clDice] + IoU + [Modality] +
    [Boundary] + [DeepSup].

    This is the main loss function used during training.  Can be constructed
    either from individual keyword arguments (legacy) or from a config object
    (OmegaConf DictConfig / any object with attributes).

    Args:
        cfg_or_w_dice: Either a config object (OmegaConf DictConfig) whose
            ``loss`` sub-key holds the weight settings, or the w_dice float
            for legacy positional use.
        w_focal (float): Weight for Focal loss (legacy kwarg).
        w_cldice (float): Weight for clDice topology loss (0 = disabled).
        w_iou (float): Weight for IoU prediction loss.
        w_modality (float): Weight for auxiliary modality classification loss.
        w_boundary (float): Weight for boundary distance-transform loss.
        w_dice_topk (float): Weight for TopK Dice loss.
        w_deepsup (float): Weight for deep-supervision auxiliary Dice loss.
        focal_alpha (float): Focal loss alpha parameter.
        focal_gamma (float): Focal loss gamma parameter.
        cldice_iterations (int): Soft skeletonisation iterations for clDice.
        dice_topk_k (float): Fraction of hard voxels for TopK Dice.
    """

    def __init__(
        self,
        cfg_or_w_dice=0.5,
        w_focal: float = 0.5,
        w_cldice: float = 0.0,
        w_iou: float = 0.1,
        w_modality: float = 0.05,
        w_boundary: float = 0.0,
        w_dice_topk: float = 0.0,
        w_deepsup: float = 0.0,
        w_hd_loss: float = 0.0,
        w_boundary_dou: float = 0.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        cldice_iterations: int = 3,
        dice_topk_k: float = 0.5,
        hd_loss_alpha: float = 2.0,
        boundary_dou_alpha_max: float = 0.8,
    ):
        super().__init__()

        # ---- Accept a config object OR individual kwargs ----
        # If a config object is passed, read weights from cfg.loss (if present)
        # or directly from cfg, falling back to the kwarg defaults.
        if hasattr(cfg_or_w_dice, '__class__') and not isinstance(cfg_or_w_dice, float):
            cfg = cfg_or_w_dice
            # cfg may have a nested 'loss' sub-config
            cfg_loss = getattr(cfg, 'loss', cfg)
            w_dice         = float(getattr(cfg_loss, 'w_dice',      0.5))
            w_focal        = float(getattr(cfg_loss, 'w_focal',     0.5))
            w_cldice       = float(getattr(cfg_loss, 'w_cldice',    0.0))
            w_iou          = float(getattr(cfg_loss, 'w_iou',       0.1))
            w_modality     = float(getattr(cfg_loss, 'w_modality',  0.05))
            w_boundary     = float(getattr(cfg_loss, 'w_boundary',  0.0))
            w_dice_topk    = float(getattr(cfg_loss, 'w_dice_topk', 0.0))
            w_deepsup      = float(getattr(cfg_loss, 'w_deepsup',   0.0))
            w_hd_loss      = float(getattr(cfg_loss, 'w_hd_loss',   0.0))
            w_boundary_dou = float(getattr(cfg_loss, 'w_boundary_dou', 0.0))
            self.hd_loss_alpha = float(getattr(cfg_loss, 'hd_loss_alpha', 2.0))
            self.boundary_dou_alpha_max = float(getattr(cfg_loss, 'boundary_dou_alpha_max', 0.8))
            # Note: focal_alpha/gamma are read from the loss sub-config directly.
            # If these need to be configurable from training level, pass them explicitly.
            self.focal_alpha = float(getattr(cfg_loss, 'focal_alpha', 0.25))
            self.focal_gamma = float(getattr(cfg_loss, 'focal_gamma', 2.0))
            self.cldice_iterations = int(getattr(cfg_loss, 'cldice_iterations', 3))
            focal_alpha    = self.focal_alpha
            focal_gamma    = self.focal_gamma
            cldice_iterations = self.cldice_iterations
            dice_topk_k    = float(getattr(cfg_loss, 'dice_topk_k', 0.5))
        else:
            # Legacy: first positional arg is w_dice float
            w_dice = float(cfg_or_w_dice)
            self.hd_loss_alpha = float(hd_loss_alpha)
            self.boundary_dou_alpha_max = float(boundary_dou_alpha_max)

        self.w_dice = w_dice
        self.w_focal = w_focal
        self.w_cldice = w_cldice
        self.w_iou = w_iou
        self.w_modality = w_modality
        self.w_boundary = w_boundary
        self.w_dice_topk = w_dice_topk
        self.w_deepsup = w_deepsup
        self.w_hd_loss = float(w_hd_loss)
        self.w_boundary_dou = float(w_boundary_dou)

        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

        if w_dice_topk > 0:
            # PMDiceLoss replaces DiceTopKLoss: same purpose (focus on hard voxels)
            # but continuous modulation vs hard TopK cutoff → smoother gradients,
            # especially beneficial for small-organ boundary voxels.
            self.dice_topk_loss = PMDiceLoss(gamma=2.0)
        else:
            self.dice_topk_loss = None
        self.iou_loss = IoUPredictionLoss()
        self.modality_loss = ModalityClassificationLoss()

        if w_boundary > 0:
            self.boundary_loss_fn = BoundaryLoss()
        else:
            self.boundary_loss_fn = None

        if self.w_hd_loss > 0:
            self.hd_loss_fn = HausdorffDTLoss(alpha=self.hd_loss_alpha)
        else:
            self.hd_loss_fn = None

        if self.w_boundary_dou > 0:
            self.boundary_dou_fn = BoundaryDoULoss(alpha_max=self.boundary_dou_alpha_max)
        else:
            self.boundary_dou_fn = None

        if w_cldice > 0:
            self.cldice_loss = clDiceLoss(num_iterations=cldice_iterations)
        else:
            self.cldice_loss = None

    def forward(
        self,
        pred:               torch.Tensor,                      # (B, H, W) best mask logit
        target:             torch.Tensor,                      # (B, H, W) binary GT
        iou_pred:           Optional[torch.Tensor] = None,    # (B,) IoU prediction
        modality_logits:    Optional[torch.Tensor] = None,    # (B, num_mod)
        modality_ids:       Optional[torch.Tensor] = None,    # (B,)
        deepsup_logits:     Optional[torch.Tensor] = None,    # (B, n_organs, H, W)
        deepsup_organ_idx:  Optional[torch.Tensor] = None,    # (B,) per-item organ channel index
        current_epoch:      int                    = 0,
        boundary_ramp_epoch: int                   = 50,
        hd_loss_ramp_epoch: int                    = 3,
        sample_weight:      Optional[torch.Tensor] = None,    # (B,) per-sample multipliers
    ):
        """
        Compute combined loss.

        Args:
            pred: Raw mask logits for the selected/best mask.
            target: Binary ground truth mask.
            iou_pred: IoU quality prediction (optional).
            modality_logits: Auxiliary modality classification logits (optional).
            modality_ids: Ground truth modality labels (optional).
            deepsup_logits: Deep-supervision multi-organ logits (optional).
            deepsup_organ_idx: Per-item organ channel indices (B,) tensor (optional).
            current_epoch: Current training epoch (for boundary ramp).
            boundary_ramp_epoch: Epoch at which boundary loss starts ramping in.

        Returns:
            (total_loss, breakdown): Scalar loss and dict of component values.
        """
        total_loss = torch.tensor(0.0, device=pred.device)
        breakdown: dict = {}

        # ---- Dice loss ----
        if self.w_dice > 0:
            dl = self.dice_loss(pred, target, sample_weight=sample_weight)
            total_loss = total_loss + self.w_dice * dl
            breakdown["dice"] = dl.item()

        # ---- Focal loss ----
        if self.w_focal > 0:
            fl = self.focal_loss(pred, target)
            total_loss = total_loss + self.w_focal * fl
            breakdown["focal"] = fl.item()

        # ---- Boundary loss (scipy distance-transform, ramped) ----
        if self.w_boundary > 0 and self.boundary_loss_fn is not None:
            w_boundary_eff = 0.0
            if current_epoch >= boundary_ramp_epoch:
                ramp = min(1.0, (current_epoch - boundary_ramp_epoch + 1) / 10.0)
                w_boundary_eff = self.w_boundary * ramp
            if w_boundary_eff > 0:
                bl = self.boundary_loss_fn(pred, target)
                total_loss = total_loss + w_boundary_eff * bl
                breakdown["boundary"] = bl.item()

        # ---- Boundary-DoU loss (Sun et al. MICCAI 2023, distance-transform-free) ----
        # Region-only generalisation of Dice that adaptively up-weights the
        # boundary region of large foreground objects; runs entirely on-device
        # and adds no scipy/CPU overhead.
        if self.w_boundary_dou > 0 and self.boundary_dou_fn is not None:
            bdou = self.boundary_dou_fn(pred, target, sample_weight=sample_weight)
            total_loss = total_loss + self.w_boundary_dou * bdou
            breakdown["boundary_dou"] = bdou.item()

        # ---- Hausdorff DT loss (Karimi-Salcudean 2020, ramped) ----
        # Karimi-Salcudean recommend a short region-loss warm-up so the network
        # has a coarse shape estimate before HD-DT shaping kicks in; we ramp
        # over 3 epochs starting at hd_loss_ramp_epoch.
        if self.w_hd_loss > 0 and self.hd_loss_fn is not None:
            w_hd_eff = 0.0
            if current_epoch >= hd_loss_ramp_epoch:
                ramp = min(1.0, (current_epoch - hd_loss_ramp_epoch + 1) / 3.0)
                w_hd_eff = self.w_hd_loss * ramp
            if w_hd_eff > 0:
                hd = self.hd_loss_fn(pred, target)
                total_loss = total_loss + w_hd_eff * hd
                breakdown["hd_loss"] = hd.item()

        # ---- clDice topology loss (vessels only) ----
        if self.w_cldice > 0 and self.cldice_loss is not None:
            cld = self.cldice_loss(pred, target)
            total_loss = total_loss + self.w_cldice * cld
            breakdown["cldice"] = cld.item()

        # ---- TopK Dice loss ----
        if self.w_dice_topk > 0 and self.dice_topk_loss is not None:
            dtk = self.dice_topk_loss(pred, target, sample_weight=sample_weight)
            total_loss = total_loss + self.w_dice_topk * dtk
            breakdown["dice_topk"] = dtk.item()

        # ---- IoU prediction loss ----
        if self.w_iou > 0 and iou_pred is not None:
            iou_l = self.iou_loss(pred, target, iou_pred)
            total_loss = total_loss + self.w_iou * iou_l
            breakdown["iou"] = iou_l.item()

        # ---- Auxiliary modality classification loss ----
        if self.w_modality > 0 and modality_logits is not None and modality_ids is not None:
            mod_l = self.modality_loss(modality_logits, modality_ids)
            total_loss = total_loss + self.w_modality * mod_l
            breakdown["modality"] = mod_l.item()

        # ---- Deep supervision loss: auxiliary Dice on ODE midpoint features ----
        if (self.w_deepsup > 0
                and deepsup_logits is not None
                and deepsup_organ_idx is not None):
            # deepsup_logits: (B, n_organs, H, W)
            # deepsup_organ_idx: (B,) per-item 0-indexed channel — use gather
            B_ds = deepsup_logits.shape[0]
            H_ds, W_ds = deepsup_logits.shape[-2:]
            ds_pred = deepsup_logits.gather(
                1, deepsup_organ_idx.view(-1, 1, 1, 1).expand(-1, 1, H_ds, W_ds)
            ).squeeze(1)  # (B, H_ds, W_ds)
            if ds_pred.shape[-2:] != target.shape[-2:]:
                ds_pred = F.interpolate(
                    ds_pred.unsqueeze(1), size=target.shape[-2:],
                    mode='bilinear', align_corners=False,
                ).squeeze(1)
            ds_loss = self.dice_loss(ds_pred, target, sample_weight=sample_weight)
            total_loss = total_loss + self.w_deepsup * ds_loss
            breakdown["deepsup"] = ds_loss.item()

        return total_loss, breakdown

    def compute_all(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        iou_pred: Optional[torch.Tensor] = None,
        modality_logits: Optional[torch.Tensor] = None,
        modality_ids: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Same as forward() but also returns individual loss components for logging.

        Returns:
            dict: {"total": ..., "dice": ..., "focal": ..., "iou": ..., ...}
        """
        components = {}

        dice = self.dice_loss(pred, target)
        components["dice"] = dice.item()
        total = self.w_dice * dice

        if self.w_focal > 0:
            focal = self.focal_loss(pred, target)
            total = total + self.w_focal * focal
            components["focal"] = focal.item()

        if self.w_boundary > 0 and self.boundary_loss_fn is not None:
            bnd = self.boundary_loss_fn(pred, target)
            total = total + self.w_boundary * bnd
            components["boundary"] = bnd.item()

        if self.w_hd_loss > 0 and self.hd_loss_fn is not None:
            hd = self.hd_loss_fn(pred, target)
            total = total + self.w_hd_loss * hd
            components["hd_loss"] = hd.item()

        if self.w_cldice > 0 and self.cldice_loss is not None:
            cld = self.cldice_loss(pred, target)
            total = total + self.w_cldice * cld
            components["cldice"] = cld.item()

        if self.w_dice_topk > 0 and self.dice_topk_loss is not None:
            dtk = self.dice_topk_loss(pred, target)
            total = total + self.w_dice_topk * dtk
            components["dice_topk"] = dtk.item()

        if self.w_iou > 0 and iou_pred is not None:
            iou_l = self.iou_loss(pred, target, iou_pred)
            total = total + self.w_iou * iou_l
            components["iou"] = iou_l.item()

        if self.w_modality > 0 and modality_logits is not None and modality_ids is not None:
            mod_l = self.modality_loss(modality_logits, modality_ids)
            total = total + self.w_modality * mod_l
            components["modality"] = mod_l.item()

        components["total"] = total.item()
        return total, components
