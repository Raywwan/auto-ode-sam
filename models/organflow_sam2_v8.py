"""OrganFlow-SAM2 V8 — "MCP-Killer".

Superset of V7 with three new capabilities:
  1. BiomedCLIP text-prompt organ embedding (adds medical-language prior).
  2. DINOv2 × MedSAM2 dual-encoder fusion (broadens feature coverage).
  3. 3D-eval-ready outputs — API unchanged so sliding-window + TTA + CC
     postproc modules plug in directly.

Inherits from OrganFlowSAM2 so all of V7's already-trained weights can be
warm-loaded with strict=False (only the new modules stay at init).
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.organflow_sam2 import OrganFlowSAM2
from models.organ_text_encoder import OrganTextEncoder
from models.dual_encoder import DINOv2FeatureExtractor, GatedFusion


class OrganFlowSAM2V8(OrganFlowSAM2):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        v8cfg = getattr(cfg.model, "v8", None)

        # --- Text encoder ---
        use_text = bool(getattr(v8cfg, "use_text", True)) if v8cfg is not None else True
        if use_text:
            self.text_encoder = OrganTextEncoder(
                n_organs=self.n_organs,
                out_dim=self.embed_dim,
                text_dim=512,
            )
        else:
            self.text_encoder = None

        # --- DINOv2 dual encoder (operates on 2D slices, same input as MedSAM2) ---
        use_dino = bool(getattr(v8cfg, "use_dino", True)) if v8cfg is not None else True
        dino_variant = str(getattr(v8cfg, "dino_variant", "dinov2_vits14")) if v8cfg is not None else "dinov2_vits14"
        dino_img_size = int(getattr(v8cfg, "dino_img_size", 336)) if v8cfg is not None else 336
        if use_dino:
            self.dino = DINOv2FeatureExtractor(variant=dino_variant, img_size=dino_img_size)
            self.dino_fuse = GatedFusion(
                medsam_dim=self.embed_dim,
                dino_dim=self.dino.feat_dim,
                out_dim=self.embed_dim,
            )
        else:
            self.dino = None
            self.dino_fuse = None

    def _text_query_bias(self, dtype: torch.dtype, device: torch.device) -> Optional[torch.Tensor]:
        if self.text_encoder is None:
            return None
        tok = self.text_encoder.forward()                     # (K, embed_dim), fp32
        return tok.to(dtype=dtype, device=device)

    def forward(
        self,
        images: torch.Tensor,         # (B, D, 3, H, W) or (B*D, 3, H, W)
        organ_id: torch.Tensor,       # (B,) long
        is_3d: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if images.dim() == 5:
            B, D, C, H, W = images.shape
            images_2d = images.reshape(B * D, C, H, W)
        else:
            images_2d = images
            B = organ_id.shape[0]
            D = images_2d.shape[0] // B

        enc_out = self.encoder(images_2d)
        feat2d = enc_out["main"]
        skip2d = enc_out["skip"]
        skip_fine2d = enc_out["skip_fine"]

        # Dual-encoder fusion: DINOv2 features at the center of every window.
        if self.dino is not None and self.dino.available:
            with torch.no_grad():
                dino_feat = self.dino(images_2d)              # (B*D, dC, gh, gw)
            if dino_feat is not None:
                feat2d = self.dino_fuse(feat2d, dino_feat.to(feat2d.dtype))

        feat2d = self.pfesa(feat2d)
        feat3d = feat2d.reshape(B, D, self.embed_dim, feat2d.shape[-2], feat2d.shape[-1])

        feat3d_out, flow_targets = self.ode(feat3d, organ_id, return_flow_targets=self.training)

        mid = D // 2
        center_feat = feat3d_out[:, mid]
        center_skip = skip2d.reshape(B, D, skip2d.shape[1], skip2d.shape[2], skip2d.shape[3])[:, mid]
        center_skip_fine = skip_fine2d.reshape(B, D, skip_fine2d.shape[1],
                                                skip_fine2d.shape[2], skip_fine2d.shape[3])[:, mid]

        dense_pe = self.dense_pe.to(center_feat.dtype).expand(B, -1, -1, -1)
        if dense_pe.shape[-2:] != center_feat.shape[-2:]:
            dense_pe = F.interpolate(dense_pe.float(), size=center_feat.shape[-2:],
                                      mode="bilinear", align_corners=False).to(center_feat.dtype)

        q_bias = self._text_query_bias(center_feat.dtype, center_feat.device)
        dec_out = self.decoder(center_feat, dense_pe, center_skip, center_skip_fine,
                               query_bias=q_bias)

        slice_range = 2
        start = max(0, mid - slice_range)
        stop = min(D, mid + slice_range + 1)
        sl = list(range(start, stop))
        deepsup_feats = feat3d_out[:, sl].contiguous()
        B_, N_, C_, H_, W_ = deepsup_feats.shape
        deepsup_flat = self.deep_sup(deepsup_feats.reshape(B_ * N_, C_, H_, W_))
        deepsup_logits = deepsup_flat.reshape(B_, N_, self.n_organs, deepsup_flat.shape[-2], deepsup_flat.shape[-1])

        return {
            "masks": dec_out["masks"],
            "iou_pred": dec_out["iou"],
            "deepsup_logits": deepsup_logits,
            "deepsup_slice_indices": sl,
            "deepsup_center_slice": mid,
            "flow_targets": flow_targets,
            "ds_scale_logits": {
                "s40":  dec_out["ds_s40"],
                "s80":  dec_out["ds_s80"],
                "s160": dec_out["ds_s160"],
            },
        }
