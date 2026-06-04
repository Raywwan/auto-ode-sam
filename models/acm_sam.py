# =============================================================================
# models/acm_sam.py — ACM-SAM: Anatomical Context Memory SAM
#
# Organ-specific prototype memory bank with EMA updates:
#   1. Spatial average-pool each slice's feature map → slice token (D*B, C)
#   2. Look up organ's K prototypes from memory bank (register_buffer)
#   3. Multi-head cross-attention: slice tokens query organ prototypes
#   4. Gate result with organ embedding → add to spatial features (broadcast)
#   5. LayerNorm on output
#   6. EMA update memory bank during training (no gradient, torch.no_grad)
#
# Why: The memory bank lets the model accumulate "what liver looks like
# across all slices in a volume" during training. At inference the bank
# is frozen, acting as a learned organ prior.
#
# Distinct from MedSAM-2 (Zhu): that uses a diversity-sorted generic bank.
# ACM-SAM maintains 15 SEPARATE banks — one per organ, updated with EMA.
# =============================================================================

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.multiscale_encoder import MultiScaleEncoder
from models.prompt_encoder import PromptEncoder
from models.mask_decoder import MaskDecoder


class AnatomicalContextMemory(nn.Module):
    """
    Organ-specific prototype memory bank with multi-head cross-attention.

    The memory bank (register_buffer, not a Parameter) stores K prototype
    vectors per organ. Updated via EMA during training so gradients never
    flow through the bank itself.

    Args:
        dim: Feature channels (embed_dim).
        num_organs: Number of organ classes. AMOS22 has 15.
        k_prototypes: Prototype vectors per organ (K=8).
        ema_decay: EMA decay for bank updates (0.99 = very slow update).
        num_heads: Attention heads for cross-attention.
    """

    def __init__(
        self,
        dim: int = 256,
        num_organs: int = 15,
        k_prototypes: int = 8,
        ema_decay: float = 0.99,
        num_heads: int = 8,
    ):
        super().__init__()
        self.dim = dim
        self.num_organs = num_organs
        self.k_prototypes = k_prototypes
        self.ema_decay = ema_decay
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Memory bank: (num_organs, k_prototypes, dim)
        # register_buffer: moved to GPU with .to(device), NOT a trainable param,
        # IS saved in state_dict / checkpoints.
        self.register_buffer(
            "memory_bank",
            torch.zeros(num_organs, k_prototypes, dim),
        )
        # Track which organs have been initialized (plain Python bool — not saved)
        self._initialized = set()

        # Organ embedding (0-indexed internally, organ_id-1 when indexing)
        self.organ_embed = nn.Embedding(num_organs, 64)

        # Cross-attention projections
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

        # Gate: σ(linear(organ_emb)) ∈ (0,1)^C — per-channel gating
        self.gate_proj = nn.Linear(64, dim)

        self.norm = nn.LayerNorm(dim)

    def _bank_idx(self, organ_id: torch.Tensor) -> torch.Tensor:
        """Convert 1-indexed organ IDs (1..15) to 0-indexed bank indices (0..14)."""
        return (organ_id - 1).clamp(0, self.num_organs - 1)

    def forward(
        self,
        x: torch.Tensor,           # (D, B, C, H, W)
        organ_id: torch.Tensor,    # (B,) with values 1..15
    ) -> torch.Tensor:
        D, B, C, H, W = x.shape

        # ---- Spatial pool → slice tokens: (D*B, C) ----
        x_perm = x.permute(0, 1, 3, 4, 2)               # (D, B, H, W, C)
        x_tokens = x_perm.reshape(D * B, H * W, C).mean(dim=1)  # (D*B, C)

        # ---- Expand organ_id to all D slices: (D*B,) ----
        organ_id_exp = organ_id.unsqueeze(0).expand(D, -1).reshape(-1)  # (D*B,)
        bidx = self._bank_idx(organ_id_exp)                              # (D*B,)

        # ---- Lazy initialization: set prototypes to first-batch features ----
        for oi in bidx.unique().tolist():
            if oi not in self._initialized:
                mask = bidx == oi
                mean_feat = x_tokens[mask].mean(0).detach()   # (C,)
                with torch.no_grad():
                    self.memory_bank[oi] = mean_feat.unsqueeze(0).expand(
                        self.k_prototypes, -1
                    )
                self._initialized.add(oi)

        # ---- Prototype lookup: (D*B, K, C) ----
        prototypes = self.memory_bank[bidx]              # (D*B, K, C)

        # ---- Multi-head cross-attention: slice tokens query prototypes ----
        Q = self.q_proj(x_tokens)                        # (D*B, C)
        K_p = self.k_proj(prototypes)                    # (D*B, K, C)
        V_p = self.v_proj(prototypes)                    # (D*B, K, C)

        # Reshape to multi-head: Q(D*B, H, 1, dh), K/V(D*B, H, K, dh)
        Q = Q.view(D * B, self.num_heads, self.head_dim).unsqueeze(2)
        K_p = K_p.view(D * B, self.k_prototypes, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V_p = V_p.view(D * B, self.k_prototypes, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = F.softmax(torch.matmul(Q, K_p.transpose(-2, -1)) * self.scale, dim=-1)
        ctx = torch.matmul(attn, V_p)                    # (D*B, H, 1, dh)
        ctx = ctx.squeeze(2).reshape(D * B, C)           # (D*B, C)
        ctx = self.out_proj(ctx)                          # (D*B, C)

        # ---- Gate with organ embedding ----
        organ_emb = self.organ_embed(bidx)               # (D*B, 64)
        gate = torch.sigmoid(self.gate_proj(organ_emb))  # (D*B, C)
        ctx_gated = gate * ctx                           # (D*B, C)

        # ---- Broadcast-add gated context to all spatial positions ----
        ctx_spatial = ctx_gated.view(D, B, C, 1, 1).expand(-1, -1, -1, H, W)
        x_out = x + ctx_spatial                          # (D, B, C, H, W)

        # ---- LayerNorm ----
        x_perm2 = x_out.permute(0, 1, 3, 4, 2)          # (D, B, H, W, C)
        x_perm2 = self.norm(x_perm2)
        x_out = x_perm2.permute(0, 1, 4, 2, 3)          # (D, B, C, H, W)

        # ---- EMA update (training only, no gradient) ----
        if self.training:
            self._ema_update(x_tokens.detach(), bidx)

        return x_out

    @torch.no_grad()
    def _ema_update(self, features: torch.Tensor, bidx: torch.Tensor):
        """
        EMA update: for each organ seen in this batch, average all its slice
        features and move all K prototypes toward that mean.
        Simple uniform EMA — all K prototypes converge to same mean.
        A future version can maintain diverse prototypes with top-K selection.
        """
        for oi in bidx.unique():
            mask = bidx == oi
            if mask.sum() == 0:
                continue
            mean_feat = features[mask].mean(0)           # (C,)
            target = mean_feat.unsqueeze(0).expand_as(self.memory_bank[oi])
            self.memory_bank[oi] = (
                self.ema_decay * self.memory_bank[oi] + (1 - self.ema_decay) * target
            )

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, num_organs={self.num_organs}, "
            f"k_prototypes={self.k_prototypes}, ema_decay={self.ema_decay}"
        )


class ACMSAM(nn.Module):
    """
    ACM-SAM: Anatomical Context Memory SAM.

    Identical to VoluFormer3D except DA-ISA is replaced with
    AnatomicalContextMemory, which is conditioned on organ_id.
    The trainer passes organ_id from the batch (AMOS22Dataset provides it).

    Args:
        cfg: OmegaConf config. Uses same keys as multiscale_isa.
             Optional keys:
               model.num_organs (default 15)
               model.acm.k_prototypes (default 8)
               model.acm.ema_decay (default 0.99)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg.model

        stage_index = getattr(
            getattr(model_cfg, "multiscale_isa", None), "stage_index", 2
        )

        # ---- Image Encoder ----
        self.encoder = MultiScaleEncoder(
            model_name=model_cfg.encoder_name,
            img_size=model_cfg.img_size,
            embed_dim=model_cfg.embed_dim,
            pretrained=model_cfg.encoder_pretrained,
            stage_index=stage_index,
        )

        feat_size = self.encoder.get_output_size()[0]

        # ---- Anatomical Context Memory ----
        acm_cfg = getattr(model_cfg, "acm", None)
        k_proto = getattr(acm_cfg, "k_prototypes", 8) if acm_cfg is not None else 8
        ema_decay = getattr(acm_cfg, "ema_decay", 0.99) if acm_cfg is not None else 0.99
        num_organs = getattr(model_cfg, "num_organs", 15)
        self.acm = AnatomicalContextMemory(
            dim=model_cfg.embed_dim,
            num_organs=num_organs,
            k_prototypes=k_proto,
            ema_decay=ema_decay,
            num_heads=8,
        )

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
        organ_id: torch.Tensor = None,
    ) -> torch.Tensor:
        if is_3d:
            B, D, C, H, W = images.shape
            images_2d = images.reshape(B * D, C, H, W)
            features_2d = self.encoder(images_2d)              # (B*D, embed_dim, Hf, Wf)

            _, embed_dim, Hf, Wf = features_2d.shape
            features_3d = features_2d.reshape(B, D, embed_dim, Hf, Wf)
            features_3d = features_3d.permute(1, 0, 2, 3, 4)  # (D, B, C, Hf, Wf)

            if organ_id is not None:
                features_3d = self.acm(features_3d, organ_id)  # memory cross-attention

            features_3d = features_3d.permute(1, 0, 2, 3, 4)  # (B, D, C, Hf, Wf)
            features_2d = features_3d.reshape(B * D, embed_dim, Hf, Wf)

            if return_all_slices:
                return features_2d
            else:
                features_vol = features_2d.reshape(B, D, embed_dim, Hf, Wf)
                return features_vol[:, D // 2]
        else:
            return self.encoder(images)

    def forward(
        self,
        images: torch.Tensor,
        boxes: torch.Tensor,
        modality_ids: torch.Tensor,
        is_3d: bool = False,
        multimask_output: bool = True,
        organ_id: torch.Tensor = None,   # (B,) organ IDs 1..15, required for ACM
    ) -> Dict[str, torch.Tensor]:
        features = self.encode_image(images, is_3d=is_3d, organ_id=organ_id)

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

        return {"masks": masks, "iou_pred": iou_pred, "modality_logits": modality_logits}

    def predict(
        self,
        images: torch.Tensor,
        boxes: torch.Tensor,
        modality_ids: torch.Tensor,
        is_3d: bool = False,
        organ_id: torch.Tensor = None,
    ) -> torch.Tensor:
        if is_3d and modality_ids.dim() == 2:
            modality_ids = modality_ids[:, modality_ids.shape[1] // 2]
        with torch.no_grad():
            out = self.forward(images, boxes, modality_ids, is_3d=is_3d,
                               multimask_output=True, organ_id=organ_id)
        best_idx = out["iou_pred"].argmax(dim=1)
        best_masks = out["masks"][
            torch.arange(out["masks"].shape[0], device=out["masks"].device), best_idx
        ].unsqueeze(1)
        return torch.sigmoid(best_masks)

    def predict_all_slices(
        self,
        images: torch.Tensor,
        boxes: torch.Tensor,
        modality_ids: torch.Tensor,
        organ_id: torch.Tensor = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            features = self.encode_image(images, is_3d=True, return_all_slices=True,
                                          organ_id=organ_id)
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                boxes=boxes, modality_ids=modality_ids, image_features=features,
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

    def count_parameters(self) -> Dict[str, int]:
        def count(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {
            "encoder": count(self.encoder),
            "acm": count(self.acm),
            "prompt_encoder": count(self.prompt_encoder),
            "mask_decoder": count(self.mask_decoder),
            "total": count(self),
        }
