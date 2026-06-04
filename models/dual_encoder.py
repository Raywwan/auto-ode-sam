"""DINOv2 × MedSAM2 dual-encoder fusion for V8.

Extracts dense features from both MedSAM2 (organ-tuned, SAM2 lineage) and
DINOv2 (self-supervised on LVD-142M, very strong mid-level features) and
fuses them with a gated 1x1 conv before feeding PFESA++.

Why both:
  - MedSAM2: medical-domain alignment, SAM2-like object boundaries.
  - DINOv2:  broad mid-level features, excellent for weakly-supervised organs
             (adrenals, esophagus) where in-domain pretrain is thin.

The gating lets the network trade them off per-channel; init biases the gate
toward MedSAM2 so we don't destabilise warm behaviour on day one.

Lazy loading:
  - DINOv2 loads via torch.hub on first use. If hub is unavailable the module
    falls back to an identity passthrough of the MedSAM2 features so the
    pipeline still runs while we wait for the download.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv2FeatureExtractor(nn.Module):
    """Wraps a DINOv2 backbone. Produces dense (B, C, H/patch, W/patch) feats.

    We use ViT-Small/14 by default (22M params, 384-dim features, 1.5 GB). The
    caller can pass ``variant='dinov2_vitb14'`` for a bigger model.
    """

    def __init__(
        self,
        variant: str = "dinov2_vits14",
        freeze: bool = True,
        img_size: int = 336,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.img_size = img_size
        self.available = False
        self.backbone: Optional[nn.Module] = None
        self.feat_dim = 384 if "vits" in variant else 768
        try:
            mdl = torch.hub.load("facebookresearch/dinov2", variant, pretrained=True)
            self.backbone = mdl
            self.available = True
        except Exception:
            # Keep module in pipeline but inert.
            self.backbone = None
            self.available = False
        if freeze and self.backbone is not None:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """x: (B, 3, H, W). Return (B, C, H/14, W/14) or None if unavailable."""
        if self.backbone is None:
            return None
        B, _, H, W = x.shape
        # DINOv2 expects images sized to a multiple of its patch size (14).
        target = (self.img_size, self.img_size)
        if (H, W) != target:
            x = F.interpolate(x, size=target, mode="bilinear", align_corners=False)
        out = self.backbone.forward_features(x)
        # DINOv2 returns a dict with x_norm_patchtokens: (B, N, C).
        patches = out["x_norm_patchtokens"]
        n = patches.shape[1]
        grid = int(round(n ** 0.5))
        feats = patches.transpose(1, 2).reshape(B, self.feat_dim, grid, grid)
        return feats


class GatedFusion(nn.Module):
    """Gated channel-wise fusion of two feature tensors at matching (H, W).

    out = a * medsam + (1 - a) * dino_proj, a = sigmoid(bias + gate_conv([m, d]))
    """

    def __init__(self, medsam_dim: int, dino_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj_dino = nn.Conv2d(dino_dim, out_dim, 1)
        self.proj_medsam = nn.Conv2d(medsam_dim, out_dim, 1) if medsam_dim != out_dim else nn.Identity()
        self.gate = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(out_dim // 4, out_dim, 1),
        )
        # Bias gate toward MedSAM2 (sigmoid(+1.5) ≈ 0.82) so initial behaviour
        # is close to the MedSAM2-only baseline.
        with torch.no_grad():
            self.gate[-1].bias.fill_(1.5)

    def forward(self, m: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        mp = self.proj_medsam(m)
        if d.shape[-2:] != mp.shape[-2:]:
            d = F.interpolate(d, size=mp.shape[-2:], mode="bilinear", align_corners=False)
        dp = self.proj_dino(d)
        a = torch.sigmoid(self.gate(torch.cat([mp, dp], dim=1)))
        return a * mp + (1.0 - a) * dp


class DualEncoderFusion(nn.Module):
    """MedSAM2 features + optional DINOv2 features → fused features.

    Caller supplies an already-built MedSAM2Encoder. Dual encoder wraps it
    and calls forward() with the same signature; output is a drop-in
    replacement for MedSAM2's ``feat`` tensor.
    """

    def __init__(
        self,
        medsam_encoder: nn.Module,
        out_dim: int = 256,
        dino_variant: str = "dinov2_vits14",
        dino_img_size: int = 336,
    ) -> None:
        super().__init__()
        self.medsam = medsam_encoder
        self.dino = DINOv2FeatureExtractor(variant=dino_variant, img_size=dino_img_size)
        self.fusion = GatedFusion(
            medsam_dim=out_dim,
            dino_dim=self.dino.feat_dim,
            out_dim=out_dim,
        )
        self.available = self.dino.available

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """x: (B, 3, H, W). Return (fused_feat, medsam_aux_dict).

        ``medsam_aux_dict`` forwards any skip-connection outputs from the
        original MedSAM2 encoder (skip_coarse / skip_fine) so downstream
        decoder still has multi-scale access.
        """
        med_out = self.medsam(x)
        # MedSAM2Encoder returns either a single tensor or a dict — normalise.
        if isinstance(med_out, dict):
            med_feat = med_out.get("feat", med_out.get("dense_feat"))
            aux = {k: v for k, v in med_out.items() if k != "feat" and k != "dense_feat"}
        else:
            med_feat = med_out
            aux = {}

        if not self.dino.available:
            return med_feat, aux

        dino_feat = self.dino(x)
        if dino_feat is None:
            return med_feat, aux
        fused = self.fusion(med_feat, dino_feat)
        return fused, aux
