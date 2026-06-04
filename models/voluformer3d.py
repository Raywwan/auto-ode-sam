# =============================================================================
# models/voluformer3d.py — VoluFormer3D Main Model (MultiScaleISA)
#
# Architecture:
#   MultiScaleEncoder (Stage 2, 4× richer tokens) → DA-ISA → Decoder
#
# Ablation A1 (current test): Stage 2 only, zero-init would require
# matching V2 weights — instead we train from scratch with pretrained
# TinyViT backbone. The α=0 trick is deferred to A2 (Stage 2+3 FPN).
#
# Compatible with V2's trainer.py, datasets/, evaluation/ — drop-in swap.
# =============================================================================

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from models.multiscale_encoder import MultiScaleEncoder
from models.depth_aware_isa import DepthAwareISAStack
from models.prompt_encoder import PromptEncoder
from models.mask_decoder import MaskDecoder


class VoluFormer3D(nn.Module):
    """
    VoluFormer3D: MultiScaleISA — DA-ISA at TinyViT Stage 2.

    Key difference from LiteSAM3D V2:
      - Encoder taps Stage 2 (32×32 at 512px) instead of Stage 3 (16×16)
      - ISA receives 4× more spatial tokens → richer cross-slice context
      - All other components (prompt encoder, decoder) identical to V2

    Args:
        cfg: OmegaConf config. New keys:
            model.architecture: "multiscale_isa"
            model.multiscale_isa.stage_index: 2 (which TinyViT stage to tap)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg.model

        stage_index = getattr(
            getattr(model_cfg, "multiscale_isa", None), "stage_index", 2
        )

        # ---- Image Encoder (Stage 2) ----
        self.encoder = MultiScaleEncoder(
            model_name=model_cfg.encoder_name,
            img_size=model_cfg.img_size,
            embed_dim=model_cfg.embed_dim,
            pretrained=model_cfg.encoder_pretrained,
            stage_index=stage_index,
        )

        feat_size = self.encoder.get_output_size()[0]

        # ---- DA-ISA ----
        self.use_isa = model_cfg.isa.enabled
        if self.use_isa:
            self.isa = DepthAwareISAStack(
                dim=model_cfg.embed_dim,
                depth=model_cfg.isa.depth,
                num_heads=model_cfg.isa.num_heads,
                window_size=model_cfg.isa.window_size,
                dropout=model_cfg.isa.dropout,
                use_depth_pe=getattr(model_cfg.isa, "use_depth_pe", True),
                use_relative_bias=getattr(model_cfg.isa, "use_relative_bias", True),
                per_head_bias=getattr(model_cfg.isa, "per_head_bias", False),
            )
        else:
            self.isa = None

        # ---- Prompt Encoder ----
        self.prompt_encoder = PromptEncoder(
            embed_dim=model_cfg.embed_dim,
            image_size=model_cfg.img_size,
            image_embedding_size=feat_size,
            num_modalities=model_cfg.prompt_encoder.num_modalities,
            modality_embed_dim=model_cfg.prompt_encoder.modality_embed_dim,
            content_embed_dim=model_cfg.prompt_encoder.content_embed_dim,
        )

        # ---- Mask Decoder ----
        self.mask_decoder = MaskDecoder(
            embed_dim=model_cfg.embed_dim,
            num_multimask_outputs=model_cfg.mask_decoder.num_multimask_outputs,
            iou_head_depth=model_cfg.mask_decoder.iou_head_depth,
            iou_head_hidden_dim=model_cfg.mask_decoder.iou_head_hidden_dim,
            num_modalities=model_cfg.prompt_encoder.num_modalities,
        )

    def encode_image(
        self,
        images: torch.Tensor,
        is_3d: bool = False,
        return_all_slices: bool = False,
    ) -> torch.Tensor:
        if is_3d:
            B, D, C, H, W = images.shape
            images_2d = images.reshape(B * D, C, H, W)
            features_2d = self.encoder(images_2d)  # (B*D, embed_dim, Hf, Wf)

            if self.use_isa and self.isa is not None:
                _, embed_dim, Hf, Wf = features_2d.shape
                features_3d = features_2d.reshape(B, D, embed_dim, Hf, Wf)
                features_3d = features_3d.permute(1, 0, 2, 3, 4)  # (D, B, C, Hf, Wf)
                features_3d = self.isa(features_3d)
                features_3d = features_3d.permute(1, 0, 2, 3, 4)  # (B, D, C, Hf, Wf)
                features_2d = features_3d.reshape(B * D, embed_dim, Hf, Wf)

            if return_all_slices:
                return features_2d
            else:
                _, embed_dim, Hf, Wf = features_2d.shape
                features_vol = features_2d.reshape(B, D, embed_dim, Hf, Wf)
                center = D // 2
                return features_vol[:, center]  # (B, embed_dim, Hf, Wf)
        else:
            return self.encoder(images)

    def forward(
        self,
        images: torch.Tensor,
        boxes: torch.Tensor,
        modality_ids: torch.Tensor,
        is_3d: bool = False,
        multimask_output: bool = True,
        organ_id: torch.Tensor = None,   # accepted for API compatibility; not used
    ) -> Dict[str, torch.Tensor]:
        features = self.encode_image(images, is_3d=is_3d, return_all_slices=False)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            boxes=boxes,
            modality_ids=modality_ids,
            image_features=features,
        )

        masks, iou_pred, modality_logits = self.mask_decoder(
            image_embeddings=features,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        return {
            "masks": masks,
            "iou_pred": iou_pred,
            "modality_logits": modality_logits,
        }

    def predict_all_slices(
        self,
        images: torch.Tensor,
        boxes: torch.Tensor,
        modality_ids: torch.Tensor,
        organ_id: torch.Tensor = None,   # accepted for API compatibility; not used
    ) -> torch.Tensor:
        with torch.no_grad():
            features = self.encode_image(images, is_3d=True, return_all_slices=True)
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                boxes=boxes,
                modality_ids=modality_ids,
                image_features=features,
            )
            masks, iou_pred, _ = self.mask_decoder(
                image_embeddings=features,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
            )

        best_idx = iou_pred.argmax(dim=1)
        best_masks = masks[
            torch.arange(masks.shape[0], device=masks.device), best_idx
        ].unsqueeze(1)
        return torch.sigmoid(best_masks)

    def predict(
        self,
        images: torch.Tensor,
        boxes: torch.Tensor,
        modality_ids: torch.Tensor,
        is_3d: bool = False,
        organ_id: torch.Tensor = None,   # accepted for API compatibility; not used
    ) -> torch.Tensor:
        if is_3d and modality_ids.dim() == 2:
            D = modality_ids.shape[1]
            modality_ids = modality_ids[:, D // 2]

        with torch.no_grad():
            out = self.forward(
                images, boxes, modality_ids, is_3d=is_3d, multimask_output=True
            )

        masks = out["masks"]
        iou_pred = out["iou_pred"]
        best_idx = iou_pred.argmax(dim=1)
        best_masks = masks[
            torch.arange(masks.shape[0], device=masks.device), best_idx
        ].unsqueeze(1)
        return torch.sigmoid(best_masks)

    def count_parameters(self) -> Dict[str, int]:
        def count(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        counts = {
            "encoder": count(self.encoder),
            "prompt_encoder": count(self.prompt_encoder),
            "mask_decoder": count(self.mask_decoder),
        }
        if self.use_isa and self.isa is not None:
            counts["da_isa"] = count(self.isa)
        counts["total"] = sum(counts.values())
        return counts
