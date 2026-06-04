# =============================================================================
# models/ode_sam.py — ODE-SAM: Continuous Organ Dynamics SAM
#
# Novel contribution: The FIRST application of Neural ODEs to model
# cross-slice feature dynamics in a SAM-based abdominal CT segmentation model.
#
# Biological motivation:
#   Abdominal organs are spatially continuous structures. A 1.5mm-spaced
#   CT scan with D=8 slices samples this continuous anatomy at 8 points
#   along Z. All existing cross-slice methods (attention, Mamba, FFT) treat
#   these as discrete tokens. We model them as samples of a continuous
#   dynamical system where the ODE function f_θ(h, t) learns the smooth
#   anatomical transition laws from data.
#
# Architecture:
#   TinyViT Stage2 Encoder (same as best Stage2-noISA baseline)
#     ↓
#   BidirectionalNeuralODECrossSlice — the novel module
#     Forward ODE:  f_fwd(h, t),  h(0)=top slice,   t: 0→1
#     Backward ODE: f_bwd(h, t),  h(0)=bottom slice, t: 0→1
#     → trajectory merged → residual → ODE-enriched features
#     ↓
#   PromptEncoder → MaskDecoder (same as all VoluFormer3D models)
#
# Parameter count (approximate):
#   Encoder:      ~13.3M  (TinyViT pretrained)
#   ODE module:   ~230K   (two ODEFunctions + merge)
#   PromptEnc:    ~205K
#   MaskDecoder:  ~4.2M
#   TOTAL:        ~18.0M  (lightest cross-slice model in this project)
# =============================================================================

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.multiscale_encoder import MultiScaleEncoder
from models.prompt_encoder import PromptEncoder
from models.mask_decoder import MaskDecoder
from models.ode_cross_slice import BidirectionalNeuralODECrossSlice
from models.trimamba_sam import PFESASpectralSkip


class ODESAM(nn.Module):
    """
    ODE-SAM: Continuous Organ Dynamics SAM.

    Replaces the cross-slice attention (DA-ISA) with a bidirectional
    Neural ODE that models the smooth evolution of encoder features
    along the depth axis of a CT volume.

    API is identical to VoluFormer3D, FCASAM, ACMSAM, and TriMambaSAM —
    works as a drop-in replacement in the existing training pipeline.

    Args:
        cfg: OmegaConf config.  New optional keys under model.ode:
            model.ode.ode_hidden  (default embed_dim // 4)
            model.ode.n_freqs     (default 4)
            model.ode.substeps    (default 2)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        mcfg     = cfg.model

        stage_index = getattr(
            getattr(mcfg, "multiscale_isa", None), "stage_index", 2
        )

        # ---- 1. Encoder (Stage 2 — same as Stage2-noISA baseline) ----
        self.encoder = MultiScaleEncoder(
            model_name=mcfg.encoder_name,
            img_size=mcfg.img_size,
            embed_dim=mcfg.embed_dim,
            pretrained=mcfg.encoder_pretrained,
            stage_index=stage_index,
        )
        feat_size = self.encoder.get_output_size()[0]

        # ---- 2. Bidirectional Neural ODE Cross-Slice Module ----
        ode_cfg    = getattr(mcfg, "ode", None)
        ode_hidden = getattr(ode_cfg, "ode_hidden",  mcfg.embed_dim // 4) if ode_cfg else mcfg.embed_dim // 4
        n_freqs    = getattr(ode_cfg, "n_freqs",     4)                   if ode_cfg else 4
        substeps   = getattr(ode_cfg, "substeps",    2)                   if ode_cfg else 2

        self.ode = BidirectionalNeuralODECrossSlice(
            dim=mcfg.embed_dim,
            ode_hidden=ode_hidden,
            n_freqs=n_freqs,
            substeps=substeps,
        )

        # ---- 3. Prompt Encoder ----
        self.prompt_encoder = PromptEncoder(
            embed_dim=mcfg.embed_dim,
            image_size=mcfg.img_size,
            image_embedding_size=feat_size,
            num_modalities=mcfg.prompt_encoder.num_modalities,
            modality_embed_dim=mcfg.prompt_encoder.modality_embed_dim,
            content_embed_dim=mcfg.prompt_encoder.content_embed_dim,
        )

        # ---- 4. Mask Decoder (with Stage 1 hierarchical skip) ----
        self.mask_decoder = MaskDecoder(
            embed_dim=mcfg.embed_dim,
            num_multimask_outputs=mcfg.mask_decoder.num_multimask_outputs,
            iou_head_depth=mcfg.mask_decoder.iou_head_depth,
            iou_head_hidden_dim=mcfg.mask_decoder.iou_head_hidden_dim,
            num_modalities=mcfg.prompt_encoder.num_modalities,
            skip_channels=self.encoder.skip_channels,  # 64 when stage_index > 1
        )

        # ---- 5. PFESA Spectral Enhancement (zero params) ----
        # Amplifies high-frequency (boundary) content in encoder features
        # before they enter the decoder. Proven +3.3% in TriMamba-SAM.
        pfesa_cfg = getattr(mcfg, "pfesa", None)
        pfesa_alpha  = getattr(pfesa_cfg, "alpha",           1.0) if pfesa_cfg else 1.0
        pfesa_cutoff = getattr(pfesa_cfg, "highfreq_cutoff", 0.5) if pfesa_cfg else 0.5
        self.pfesa = PFESASpectralSkip(alpha=pfesa_alpha, highfreq_cutoff=pfesa_cutoff)

    # ------------------------------------------------------------------
    def encode_image(
        self,
        images:            torch.Tensor,
        is_3d:             bool = False,
        return_all_slices: bool = False,
    ) -> torch.Tensor:
        """
        Encode images through backbone → ODE → PFESA.

        Side effect: caches Stage 1 skip features in self._skip_cache for
        use by mask_decoder (either in forward() or trainer's all-slice path).

          self._skip_cache shape:
            - return_all_slices=True:  (B*D, skip_ch, Hs, Ws)
            - return_all_slices=False: (B, skip_ch, Hs, Ws)  center slice only
        """
        if not is_3d:
            feat, skip = self.encoder(images, return_skip=True)
            feat = self.pfesa(feat)
            self._skip_cache = skip
            return feat

        B, D, C, H, W = images.shape
        images_2d = images.reshape(B * D, C, H, W)
        feat_2d, skip_2d = self.encoder(images_2d, return_skip=True)
        # feat_2d:  (B*D, embed, Hf, Wf)
        # skip_2d:  (B*D, skip_ch, Hs, Ws)  where Hs = 2*Hf at 256px

        _, embed_dim, Hf, Wf = feat_2d.shape
        feat_3d = feat_2d.reshape(B, D, embed_dim, Hf, Wf)

        # ---- Neural ODE cross-slice enrichment ----
        feat_3d = self.ode(feat_3d)                      # (B, D, embed, Hf, Wf)

        if return_all_slices:
            feat_all = feat_3d.reshape(B * D, embed_dim, Hf, Wf)
            feat_all = self.pfesa(feat_all)
            self._skip_cache = skip_2d                   # (B*D, skip_ch, Hs, Ws)
            return feat_all

        # Center slice
        center_feat = feat_3d[:, D // 2]                 # (B, embed, Hf, Wf)
        center_feat = self.pfesa(center_feat)
        skip_ch, Hs, Ws = skip_2d.shape[1], skip_2d.shape[2], skip_2d.shape[3]
        self._skip_cache = skip_2d.reshape(B, D, skip_ch, Hs, Ws)[:, D // 2]
        # _skip_cache: (B, skip_ch, Hs, Ws)
        return center_feat

    # ------------------------------------------------------------------
    def forward(
        self,
        images:           torch.Tensor,
        boxes:            torch.Tensor,
        modality_ids:     torch.Tensor,
        is_3d:            bool = False,
        multimask_output: bool = True,
        organ_id:         torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:

        features = self.encode_image(images, is_3d=is_3d)
        skip_feat = getattr(self, '_skip_cache', None)

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
            skip_features=skip_feat,
        )

        return {
            "masks":           masks,
            "iou_pred":        iou_pred,
            "modality_logits": modality_logits,
        }

    # ------------------------------------------------------------------
    def predict(
        self,
        images:       torch.Tensor,
        boxes:        torch.Tensor,
        modality_ids: torch.Tensor,
        is_3d:        bool = False,
        organ_id:     torch.Tensor = None,
    ) -> torch.Tensor:
        if is_3d and modality_ids.dim() == 2:
            modality_ids = modality_ids[:, modality_ids.shape[1] // 2]
        with torch.no_grad():
            out = self.forward(images, boxes, modality_ids, is_3d=is_3d,
                               multimask_output=True)
        best_idx   = out["iou_pred"].argmax(dim=1)
        best_masks = out["masks"][
            torch.arange(out["masks"].shape[0], device=out["masks"].device), best_idx
        ].unsqueeze(1)
        return torch.sigmoid(best_masks)

    # ------------------------------------------------------------------
    def predict_all_slices(
        self,
        images:       torch.Tensor,
        boxes:        torch.Tensor,
        modality_ids: torch.Tensor,
        organ_id:     torch.Tensor = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            features = self.encode_image(images, is_3d=True, return_all_slices=True)
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                boxes=boxes, modality_ids=modality_ids, image_features=features,
            )
            masks, iou_pred, _ = self.mask_decoder(
                image_embeddings=features,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
            )
        best_idx   = iou_pred.argmax(dim=1)
        best_masks = masks[
            torch.arange(masks.shape[0], device=masks.device), best_idx
        ].unsqueeze(1)
        return torch.sigmoid(best_masks)

    # ------------------------------------------------------------------
    def count_parameters(self) -> Dict[str, int]:
        def count(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {
            "encoder":        count(self.encoder),
            "ode_module":     count(self.ode),
            "prompt_encoder": count(self.prompt_encoder),
            "mask_decoder":   count(self.mask_decoder),
            "pfesa":          0,  # zero learnable params
            "total":          count(self),
        }
