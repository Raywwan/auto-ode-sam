# =============================================================================
# models/multiscale_encoder.py — MultiScaleISA Image Encoder
#
# Modification of V2's TinyViTEncoder: taps Stage 2 (32×32 at 512px,
# 16×16 at 256px) instead of Stage 3 (16×16 at 512px).
#
# Root cause of V2 failure: Stage 3 at 256px = 8×8 = 64 tokens (too coarse
# for ISA to capture meaningful cross-slice structure). Stage 2 gives 4×
# more tokens at the same input resolution.
#
# Ablation A1: Stage 2 only (start here)
# Ablation A2: Stage 2 + Stage 3 via FPN (next step if A1 succeeds)
#
# Output: (B, embed_dim, H/16, W/16) at 512px  ← same shape as V2
#         (B, embed_dim, H/16, W/16) at 256px  ← 16×16 instead of 8×8
# =============================================================================

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import timm


class MultiScaleEncoder(nn.Module):
    """
    TinyViT encoder tapping Stage 2 for richer spatial tokens.

    Stage 2 channels: 384 (at 512px → 32×32; at 256px → 16×16)
    Stage 3 channels: 576 (at 512px → 16×16; at 256px → 8×8)  ← V2 used this

    Using Stage 2 gives 4× more tokens at 512px (1024 vs 256) and
    ISA has meaningful spatial structure to attend across.

    When stage_index > 1, Stage 1 features are also extracted and projected
    to a skip connection tensor for hierarchical decoder fusion.

    Args:
        model_name: timm model name (default: tiny_vit_21m_224)
        img_size: Input resolution (256 for test runs, 512 for full training)
        embed_dim: SAM-compatible embedding dimension (256)
        pretrained: Load ImageNet weights
        stage_index: Which TinyViT stage to tap (2=Stage2, 3=Stage3/V2 default)
    """

    def __init__(
        self,
        model_name: str = "tiny_vit_21m_224",
        img_size: int = 512,
        embed_dim: int = 256,
        pretrained: bool = True,
        stage_index: int = 2,  # Stage 2 for MultiScaleISA
    ):
        super().__init__()

        self.img_size = img_size
        self.embed_dim = embed_dim
        self.stage_index = stage_index

        # ------------------------------------------------------------------
        # Tap Stage 2 (or specified stage) of TinyViT.
        # When stage_index > 1 we also extract Stage 1 for skip connections.
        # ------------------------------------------------------------------
        self._has_skip = stage_index > 1
        out_indices = [1, stage_index] if self._has_skip else [stage_index]

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )

        # feature_info is indexed by stage number (not by out_indices position).
        # feature_info[-1] would always give the LAST stage (576ch for stage 3)
        # regardless of out_indices. Use stage_index directly.
        backbone_out_channels = self.backbone.feature_info[stage_index]["num_chs"]
        self.output_reduction = self.backbone.feature_info[stage_index]["reduction"]

        # ------------------------------------------------------------------
        # Projection: backbone_channels → embed_dim
        # ------------------------------------------------------------------
        self.proj = nn.Sequential(
            nn.Conv2d(backbone_out_channels, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

        # ------------------------------------------------------------------
        # Neck: 2× 3×3 conv with residual (same as V2)
        # ------------------------------------------------------------------
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
        )

        # ------------------------------------------------------------------
        # Stage 1 skip connection projection (embed_dim//4 channels)
        # Stage 1: 192ch at H/8 × W/8 (32×32 at 256px, 64×64 at 512px)
        # Projected to embed_dim//4 = 64ch to match decoder's 1st upsample out
        # ------------------------------------------------------------------
        if self._has_skip:
            skip1_channels = self.backbone.feature_info[1]["num_chs"]  # 192
            self.skip_proj = nn.Sequential(
                nn.Conv2d(skip1_channels, embed_dim // 4, kernel_size=1, bias=False),
                nn.BatchNorm2d(embed_dim // 4),
                nn.GELU(),
            )
            self.skip_channels = embed_dim // 4  # 64
        else:
            self.skip_proj = None
            self.skip_channels = 0

    def forward(
        self,
        x: torch.Tensor,
        return_skip: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: (B, 1 or 3, H, W)
            return_skip: if True, also return projected Stage 1 features for
                         hierarchical skip connection in the decoder.
        Returns:
            features:  (B, embed_dim, H/output_reduction, W/output_reduction)
            skip_feat: (B, embed_dim//4, H/(output_reduction/2), W/…)
                       — only returned when return_skip=True and _has_skip=True
        """
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        feat_list = self.backbone(x)
        if self._has_skip:
            stage1_feat = feat_list[0]   # (B, 192, H/8, W/8)
            stage2_feat = feat_list[1]   # (B, 384, H/16, W/16)
        else:
            stage2_feat = feat_list[0]

        features = self.proj(stage2_feat)      # (B, embed_dim, H/16, W/16)
        features = features + self.neck(features)

        if return_skip and self._has_skip:
            skip_feat = self.skip_proj(stage1_feat)  # (B, embed_dim//4, H/8, W/8)
            return features, skip_feat

        return features

    def get_output_size(self) -> Tuple[int, int]:
        sz = self.img_size // self.output_reduction
        return (sz, sz)
