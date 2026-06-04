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


def tversky_loss(pred: torch.Tensor, target: torch.Tensor,
                 alpha: float = 0.7, beta: float = 0.3, eps: float = 1e-6) -> torch.Tensor:
    """Per-sample Tversky loss. alpha>beta penalizes FN (good for small organs).
    At alpha=beta=0.5 this reduces to Dice."""
    p = torch.sigmoid(pred)
    tp = (p * target).flatten(1).sum(1)
    fn = ((1 - p) * target).flatten(1).sum(1)
    fp = (p * (1 - target)).flatten(1).sum(1)
    tv = (tp + eps) / (tp + alpha * fn + beta * fp + eps)
    return 1.0 - tv


def focal_bce_loss(pred: torch.Tensor, target: torch.Tensor,
                   alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """Per-sample focal BCE. Down-weights well-classified pixels, focusing on hard cases.
    Helpful when the foreground is tiny (adrenals: 2 pixels / 65k)."""
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    p = torch.sigmoid(pred)
    pt = torch.where(target > 0.5, p, 1 - p)
    alpha_t = torch.where(target > 0.5, alpha, 1 - alpha)
    focal = alpha_t * (1 - pt).clamp(min=1e-6).pow(gamma) * bce
    return focal.flatten(1).mean(1)


def boundary_weighted_bce(pred: torch.Tensor, target: torch.Tensor,
                          boundary_weight: float = 3.0, kernel: int = 3) -> torch.Tensor:
    """BCE with higher weight on boundary pixels of the target mask.

    Boundary is identified cheaply with a 3x3 avg-pool mismatch check —
    any foreground pixel whose 3x3 neighbourhood is not uniform is treated as
    a boundary pixel. Background pixels adjacent to foreground are also weighted.
    """
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    tgt = target.float()
    # (N, H, W) -> (N, 1, H, W) for pooling
    was_3d = tgt.dim() == 3
    if was_3d:
        tgt_u = tgt.unsqueeze(1)
    else:
        tgt_u = tgt
    pooled = F.avg_pool2d(tgt_u, kernel_size=kernel, stride=1, padding=kernel // 2)
    boundary = ((pooled > 0.05) & (pooled < 0.95)).float()  # transition zone
    if was_3d:
        boundary = boundary.squeeze(1)
    weight = 1.0 + (boundary_weight - 1.0) * boundary
    weighted_bce = (bce * weight).flatten(1).mean(1)
    return weighted_bce


class V4MultiOrganLoss(nn.Module):
    def __init__(
        self,
        lambda_flow: float = 0.5,
        lambda_deepsup: float = 0.1,
        lambda_anatomy: float = 0.01,
        lambda_xsc: float = 0.25,
        lambda_ds_scale: float = 0.4,
        lambda_focal: float = 0.3,
        lambda_boundary: float = 0.2,
        tversky_alpha: float = 0.7,
        tversky_beta: float = 0.3,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        boundary_weight: float = 3.0,
    ) -> None:
        super().__init__()
        self.lambda_flow = lambda_flow
        self.lambda_deepsup = lambda_deepsup
        self.lambda_anatomy = lambda_anatomy
        self.lambda_xsc = lambda_xsc
        self.lambda_ds_scale = lambda_ds_scale
        self.lambda_focal = lambda_focal
        self.lambda_boundary = lambda_boundary
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.boundary_weight = boundary_weight

    def forward(
        self,
        pred_masks: torch.Tensor,                                  # (B, 15, H, W) logits
        gt_masks: torch.Tensor,                                     # (B, 15, H, W) or (B, 15, D, H, W) uint8
        present_mask: torch.Tensor,                                 # (B, 15) bool
        flow_targets: Optional[Dict] = None,
        deepsup_logits: Optional[torch.Tensor] = None,
        deepsup_slice_indices: Optional[list] = None,
        deepsup_center_slice: Optional[int] = None,
        anatomy_adjacency: Optional[torch.Tensor] = None,
        anatomy_adjacency_init: Optional[torch.Tensor] = None,
        xsc_pairs: Optional[Dict] = None,
        ds_scale_logits: Optional[Dict[str, torch.Tensor]] = None,
        lambda_flow_override: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        # Keep full 5D GT available for all-slice deep-sup; collapse separately for L_mask.
        gt_full = gt_masks.float()
        if gt_full.dim() == 5:
            D_total = gt_full.shape[2]
            target = gt_full[:, :, D_total // 2, :, :]
        else:
            D_total = None
            target = gt_full
        # STEP 3: compute loss at GT res — upsample preds, not downsample target.
        if pred_masks.shape[-2:] != target.shape[-2:]:
            pred_masks = F.interpolate(pred_masks, size=target.shape[-2:], mode="bilinear", align_corners=False)

        B, K, H, W = pred_masks.shape
        flat_pred = pred_masks.reshape(B * K, H, W)
        flat_tgt = target.reshape(B * K, H, W)
        mask_ok = present_mask.reshape(B * K)

        components: Dict[str, float] = {}
        bce = F.binary_cross_entropy_with_logits(flat_pred, flat_tgt, reduction="none").mean(dim=(-1, -2))
        tv  = tversky_loss(flat_pred, flat_tgt, alpha=self.tversky_alpha, beta=self.tversky_beta)

        if mask_ok.any():
            L_mask = (bce[mask_ok].mean() + tv[mask_ok].mean()) * 0.5
            if self.lambda_focal > 0.0:
                fo = focal_bce_loss(flat_pred, flat_tgt, alpha=self.focal_alpha, gamma=self.focal_gamma)
                L_mask = L_mask + self.lambda_focal * fo[mask_ok].mean()
                components["focal"] = fo[mask_ok].mean().item()
            if self.lambda_boundary > 0.0:
                bw = boundary_weighted_bce(flat_pred, flat_tgt, boundary_weight=self.boundary_weight)
                L_mask = L_mask + self.lambda_boundary * bw[mask_ok].mean()
                components["boundary"] = bw[mask_ok].mean().item()
        else:
            L_mask = pred_masks.sum() * 0.0

        components["mask"] = L_mask.item()
        L_flow = pred_masks.sum() * 0.0
        L_deepsup = pred_masks.sum() * 0.0
        L_anatomy = pred_masks.sum() * 0.0

        if flow_targets is not None:
            fwd = flow_targets["fwd_pairs"]
            bwd = flow_targets["bwd_pairs"]
            L_flow = 0.5 * (F.mse_loss(fwd["pred"], fwd["target"]) + F.mse_loss(bwd["pred"], bwd["target"]))
            components["flow"] = L_flow.item()

        if deepsup_logits is not None:
            if deepsup_logits.dim() == 5 and gt_full.dim() == 5 and deepsup_slice_indices is not None:
                # STEP 4: all-slice deep-sup with quadratic weight falloff from center.
                B_, N_, K_, Hp, Wp = deepsup_logits.shape
                ds_tgt = gt_full[:, :, deepsup_slice_indices].permute(0, 2, 1, 3, 4).contiguous()  # (B, N, K, H, W)
                if ds_tgt.shape[-2:] != deepsup_logits.shape[-2:]:
                    ds_tgt_flat = ds_tgt.reshape(B_ * N_, K_, ds_tgt.shape[-2], ds_tgt.shape[-1])
                    ds_tgt_flat = F.interpolate(ds_tgt_flat, size=deepsup_logits.shape[-2:], mode="nearest")
                    ds_tgt = ds_tgt_flat.reshape(B_, N_, K_, Hp, Wp)
                mid = deepsup_center_slice if deepsup_center_slice is not None else (D_total // 2)
                half_span = max(N_ // 2, 1)
                w = torch.tensor(
                    [max(0.0, 1.0 - ((float(i - mid)) / float(half_span)) ** 2) for i in deepsup_slice_indices],
                    device=deepsup_logits.device, dtype=deepsup_logits.dtype,
                )
                w = w / (w.sum() + 1e-6)                                # (N,)
                bce = F.binary_cross_entropy_with_logits(deepsup_logits, ds_tgt, reduction="none")
                bce_per_slice = bce.mean(dim=(0, 2, 3, 4))              # (N,)
                L_deepsup = (bce_per_slice * w).sum()
            else:
                ds_tgt = target
                if ds_tgt.shape[-2:] != deepsup_logits.shape[-2:]:
                    ds_tgt = F.interpolate(ds_tgt, size=deepsup_logits.shape[-2:], mode="nearest")
                L_deepsup = F.binary_cross_entropy_with_logits(deepsup_logits, ds_tgt)
            components["deepsup"] = L_deepsup.item()

        if anatomy_adjacency is not None and anatomy_adjacency_init is not None:
            L_anatomy = F.mse_loss(anatomy_adjacency, anatomy_adjacency_init)
            components["anatomy"] = L_anatomy.item()

        # V6: multi-scale decoder deep-supervision (nnU-Net-style).
        # Scale weights ∝ resolution: higher-res scales get more weight.
        L_ds_scale = pred_masks.sum() * 0.0
        if ds_scale_logits is not None and mask_ok.any():
            scale_weights = {"s40": 0.143, "s80": 0.286, "s160": 0.571}
            per_scale = {}
            for key, logits in ds_scale_logits.items():
                tgt = F.interpolate(target, size=logits.shape[-2:], mode="nearest")
                B_s, K_s, H_s, W_s = logits.shape
                flat_p = logits.reshape(B_s * K_s, H_s, W_s)
                flat_t = tgt.reshape(B_s * K_s, H_s, W_s)
                bce_s = F.binary_cross_entropy_with_logits(flat_p, flat_t, reduction="none").mean(dim=(-1, -2))
                tv_s  = tversky_loss(flat_p, flat_t, alpha=self.tversky_alpha, beta=self.tversky_beta)
                L_s = (bce_s[mask_ok].mean() + tv_s[mask_ok].mean()) * 0.5
                per_scale[key] = L_s
                components[f"ds_{key}"] = L_s.item()
            L_ds_scale = sum(scale_weights[k] * per_scale[k] for k in per_scale)

        # STEP 10: Cross-Slice ODE Consistency (XSC) — novel contribution.
        # Penalises disagreement between one-step-integrated ODE prediction
        # and the stop-grad encoder feature at the next slice.
        L_xsc = pred_masks.sum() * 0.0
        if xsc_pairs is None and flow_targets is not None:
            xsc_pairs = flow_targets.get("xsc_pairs")
        if xsc_pairs is not None:
            pred_next = xsc_pairs["pred"].float()
            real_next = xsc_pairs["target"].float()
            L_xsc = F.mse_loss(pred_next, real_next)
            components["xsc"] = L_xsc.item()

        lf = self.lambda_flow if lambda_flow_override is None else lambda_flow_override
        total = (L_mask
                 + lf * L_flow
                 + self.lambda_deepsup * L_deepsup
                 + self.lambda_ds_scale * L_ds_scale
                 + self.lambda_anatomy * L_anatomy
                 + self.lambda_xsc * L_xsc)
        components["total"] = total.item()
        return {"loss": total, "components": components}
