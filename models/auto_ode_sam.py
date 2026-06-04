# =============================================================================
# models/auto_ode_sam.py — Auto-ODE-SAM V3
#
# Fully automatic 15-organ CT segmentation.
# No box prompt needed — organ query tokens are learned end-to-end.
#
# Architecture:
#   MultiScaleEncoder (TinyViT Stage 2 + Stage 1 skip)
#     ↓
#   OrganConditionedBidirectionalNeuralODE
#     ↓
#   PFESA (zero params, applied post-ODE)
#     ↓
#   DeepSupervisionHead at ODE midpoint (auxiliary loss only)
#     ↓
#   OrganQueryDecoder (15 organ tokens + HQ token)
#     ↓
#   15 organ mask logits + IoU scores
# =============================================================================

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.multiscale_encoder import MultiScaleEncoder
from models.ode_cross_slice import (
    OrganConditionedBidirectionalNeuralODE,
    PACodeBidirectionalNeuralODE,
)
from models.organ_query_decoder import OrganQueryDecoder
from models.trimamba_sam import PFESASpectralSkip


class DeepSupervisionHead(nn.Module):
    """
    Lightweight segmentation head for auxiliary loss at ODE midpoint.

    Applied to features at t=0.5 (mid-integration, center depth slice)
    to force the ODE to produce meaningful intermediate representations.

    Args:
        embed_dim (int): Input feature dimension.
        n_organs  (int): Number of output classes (organs).
    """

    def __init__(self, embed_dim: int, n_organs: int = 15):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 4, kernel_size=1),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 8, n_organs, kernel_size=2, stride=2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, embed_dim, H_feat, W_feat)
        Returns:
            logits: (B, n_organs, H_feat*4, W_feat*4)
        """
        return self.conv(features)


class AutoODESAM(nn.Module):
    """
    Auto-ODE-SAM V3: Fully automatic 15-organ CT segmentation.

    Args:
        cfg: OmegaConf config with model.architecture = "auto_ode_sam"
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        mcfg = cfg.model

        stage_index = getattr(
            getattr(mcfg, "multiscale_isa", None), "stage_index", 2
        )

        # ---- 1. Encoder ----
        self.encoder = MultiScaleEncoder(
            model_name=mcfg.encoder_name,
            img_size=mcfg.img_size,
            embed_dim=mcfg.embed_dim,
            pretrained=mcfg.encoder_pretrained,
            stage_index=stage_index,
        )
        feat_size = self.encoder.get_output_size()[0]

        # ---- 2. PFESA (zero params) ----
        pfesa_cfg    = getattr(mcfg, "pfesa", None)
        pfesa_alpha  = getattr(pfesa_cfg, "alpha",           1.0) if pfesa_cfg else 1.0
        pfesa_cutoff = getattr(pfesa_cfg, "highfreq_cutoff", 0.5) if pfesa_cfg else 0.5
        self.pfesa = PFESASpectralSkip(alpha=pfesa_alpha, highfreq_cutoff=pfesa_cutoff)

        # ---- 3. Organ-Conditioned Neural ODE ----
        ode_cfg       = getattr(mcfg, "ode", None)
        ode_hidden    = getattr(ode_cfg, "ode_hidden",    mcfg.embed_dim // 4) if ode_cfg else mcfg.embed_dim // 4
        n_freqs       = getattr(ode_cfg, "n_freqs",       4)  if ode_cfg else 4
        substeps      = getattr(ode_cfg, "substeps",      2)  if ode_cfg else 2
        n_organs      = getattr(ode_cfg, "n_organs",      15) if ode_cfg else 15
        organ_emb_dim = getattr(ode_cfg, "organ_emb_dim", 32) if ode_cfg else 32
        use_pa_code   = bool(getattr(ode_cfg, "use_pa_code", False)) if ode_cfg else False
        n_pos_freqs   = getattr(ode_cfg, "n_pos_freqs",   4)  if ode_cfg else 4
        gamma_bound   = float(getattr(ode_cfg, "gamma_bound", 0.5)) if ode_cfg else 0.5
        bidirectional = bool(getattr(ode_cfg, "bidirectional", True)) if ode_cfg else True

        self.use_pa_code = use_pa_code
        if use_pa_code:
            # PA-CODE: identity-init FiLM dynamics + position-augmented state.
            # See docs/superpowers/specs/2026-04-30-pa-code-design.md
            self.ode = PACodeBidirectionalNeuralODE(
                dim=mcfg.embed_dim,
                n_organs=n_organs,
                organ_emb_dim=organ_emb_dim,
                ode_hidden=ode_hidden,
                n_freqs=n_freqs,
                n_pos_freqs=n_pos_freqs,
                substeps=substeps,
                gamma_bound=gamma_bound,
            )
        else:
            self.ode = OrganConditionedBidirectionalNeuralODE(
                dim=mcfg.embed_dim,
                n_organs=n_organs,
                organ_emb_dim=organ_emb_dim,
                ode_hidden=ode_hidden,
                n_freqs=n_freqs,
                substeps=substeps,
                bidirectional=bidirectional,
            )

        # ---- 4. Deep supervision auxiliary head ----
        self.deep_sup_head = DeepSupervisionHead(
            embed_dim=mcfg.embed_dim,
            n_organs=n_organs,
        )

        # ---- 5. Organ Query Decoder ----
        dec_cfg = getattr(mcfg, "organ_decoder", None)
        qd_cfg  = getattr(mcfg, "organ_query_decoder", None)
        diversity_loss_weight = (
            getattr(qd_cfg, "diversity_loss_weight", 0.01) if qd_cfg else 0.01
        )
        self.decoder = OrganQueryDecoder(
            embed_dim=mcfg.embed_dim,
            n_organs=n_organs,
            transformer_depth=getattr(dec_cfg, "transformer_depth",   2)    if dec_cfg else 2,
            transformer_mlp_dim=getattr(dec_cfg, "transformer_mlp_dim", 2048) if dec_cfg else 2048,
            iou_head_depth=getattr(dec_cfg, "iou_head_depth",         3)    if dec_cfg else 3,
            iou_head_hidden_dim=getattr(dec_cfg, "iou_head_hidden_dim", 256) if dec_cfg else 256,
            skip_channels=self.encoder.skip_channels,
            diversity_loss_weight=diversity_loss_weight,
        )

        # ---- Dense PE buffer (sinusoidal 2D, registered once) ----
        self.register_buffer(
            "dense_pe",
            self._make_dense_pe(mcfg.embed_dim, feat_size),
        )
        # Initialized to None; set by encode_image() on each forward pass
        self._deepsup_cache: Optional[torch.Tensor] = None

        # PA-CODE trajectory cache flag (toggled by trainer per-step):
        # when True, encode_image() asks self.ode to retain its integration
        # states so they can be pulled out for trajectory distillation.
        self._cache_ode_trajectories: bool = False

    # ------------------------------------------------------------------
    def get_last_ode_trajectories(self):
        """Return (traj_fwd, traj_bwd) from the most recent forward() call.
        Each is a (B*H*W, D, C) tensor. Returns (None, None) if PA-CODE is
        not enabled or trajectory caching was off.
        """
        if not self.use_pa_code:
            return None, None
        return self.ode.get_cached_trajectories()

    # ------------------------------------------------------------------
    @staticmethod
    def _make_dense_pe(embed_dim: int, feat_size: int) -> torch.Tensor:
        """
        Sinusoidal 2D positional encoding with temperature scaling.
        Matches SAM convention: i-th freq = 1 / (10000 ^ (2i / half)).
        This gives geometrically decaying frequencies that maintain spatial
        coherence across all channels (avoids aliasing that 2**i causes for
        feat_size=16 at index > 14).
        """
        half = embed_dim // 2
        grid = torch.arange(feat_size, dtype=torch.float32) / feat_size
        x, y = torch.meshgrid(grid, grid, indexing="ij")
        dim_t = torch.arange(half, dtype=torch.float32)
        dim_t = 10000 ** (2 * dim_t / half)   # denominators: geometrically spaced
        pe_x = torch.sin(x.unsqueeze(-1) / dim_t.unsqueeze(0).unsqueeze(0))
        pe_y = torch.cos(y.unsqueeze(-1) / dim_t.unsqueeze(0).unsqueeze(0))
        pe = torch.cat([pe_x, pe_y], dim=-1)  # (feat_size, feat_size, embed_dim)
        return pe.permute(2, 0, 1).unsqueeze(0)  # (1, embed_dim, feat_size, feat_size)

    # ------------------------------------------------------------------
    def encode_image(
        self,
        images:            torch.Tensor,  # (B*D, 3, H, W) or (B, D, 3, H, W)
        organ_id:          torch.Tensor,  # (B,) long — ODE conditioning
        return_all_slices: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode → PFESA → ODE. Caches deep supervision features.

        Side effect: stores mid-depth features in self._deepsup_cache.

        Returns:
            features: center slice (B, embed, Hf, Wf) or all slices (B*D, embed, Hf, Wf)
            skip:     center slice (B, skip_ch, Hs, Ws) or all slices (B*D, skip_ch, Hs, Ws)
        """
        if images.dim() == 5:
            B, D, C, H, W = images.shape
            images_2d = images.reshape(B * D, C, H, W)
        else:
            images_2d = images
            B = organ_id.shape[0]
            D = images_2d.shape[0] // B

        feat_2d, skip_2d = self.encoder(images_2d, return_skip=True)
        _, embed_dim, Hf, Wf = feat_2d.shape
        feat_3d = feat_2d.reshape(B, D, embed_dim, Hf, Wf)

        # ODE cross-slice enrichment (organ-conditioned).
        # If PA-CODE + trajectory distillation is on, cache the integration
        # states so the trainer can pull them out for L_traj computation.
        if self.use_pa_code and getattr(self, "_cache_ode_trajectories", False):
            feat_3d = self.ode(feat_3d, organ_id, cache_trajectories=True)
        else:
            feat_3d = self.ode(feat_3d, organ_id)  # (B, D, embed, Hf, Wf)

        # Deep supervision: center depth = t≈0.5
        mid = D // 2
        # Cache mid-depth ODE output PRE-PFESA for deep supervision.
        # Supervising raw ODE output trains the cross-slice trajectory independently
        # of PFESA's spectral post-processing.
        self._deepsup_cache = feat_3d[:, mid]

        if return_all_slices:
            feat_all = feat_3d.reshape(B * D, embed_dim, Hf, Wf)
            feat_all = self.pfesa(feat_all)
            return feat_all, skip_2d

        center_feat = self.pfesa(feat_3d[:, mid])
        skip_ch, Hs, Ws = skip_2d.shape[1], skip_2d.shape[2], skip_2d.shape[3]
        center_skip = skip_2d.reshape(B, D, skip_ch, Hs, Ws)[:, mid]
        return center_feat, center_skip

    # ------------------------------------------------------------------
    def forward(
        self,
        images:           torch.Tensor,
        organ_id:         torch.Tensor,           # (B,) long
        is_3d:            bool = False,
        target_organ_ids: Optional[List[int]] = None,
    ) -> Dict[str, object]:
        """
        Forward pass.

        Returns dict with keys:
          "masks":           (B, K, H_out, W_out)
          "iou_pred":        (B, K)
          "organ_ids":       list[int]
          "deepsup_logits":  (B, n_organs, H_out, W_out)
        """
        # Always decode the center slice only — ODE already runs over all D slices
        # internally. return_all_slices=True would compute PFESA on all B*D feature
        # maps and then discard D-1 per sample (pure waste on every forward pass).
        feat, skip = self.encode_image(images, organ_id, return_all_slices=False)

        # encode_image always returns the center slice (B, C, Hf, Wf) — is_3d only
        # affects whether the ODE runs (which it always does inside encode_image).
        B = feat.shape[0]

        # Resize PE if feature size changed (e.g. Phase 3b 512px → feat 32×32 vs default 16×16)
        if feat.shape[-2:] != self.dense_pe.shape[-2:]:
            dense_pe_base = F.interpolate(
                self.dense_pe.float(),
                size=feat.shape[-2:],
                mode='bilinear',
                align_corners=False,
            ).to(feat.dtype)
        else:
            dense_pe_base = self.dense_pe.to(feat.dtype)
        dense_pe = dense_pe_base.expand(B, -1, -1, -1)

        masks, iou_pred, organ_ids_out = self.decoder(
            image_embeddings=feat,
            dense_pe=dense_pe,
            skip_features=skip,
            target_organ_ids=target_organ_ids,
        )

        deepsup_logits = self.deep_sup_head(self._deepsup_cache)
        if deepsup_logits.shape[-2:] != masks.shape[-2:]:
            deepsup_logits = F.interpolate(
                deepsup_logits, size=masks.shape[-2:], mode='bilinear', align_corners=False,
            )

        return {
            "masks":          masks,
            "iou_pred":       iou_pred,
            "organ_ids":      organ_ids_out,
            "deepsup_logits": deepsup_logits,
        }

    # ------------------------------------------------------------------
    def predict(
        self,
        images:           torch.Tensor,
        organ_id:         torch.Tensor,
        target_organ_ids: Optional[List[int]] = None,
        is_3d:            bool = False,
    ) -> Tuple[torch.Tensor, List[int]]:
        """Inference. Returns sigmoid masks and organ ID list."""
        with torch.no_grad():
            out = self.forward(images, organ_id, is_3d=is_3d, target_organ_ids=target_organ_ids)
        return torch.sigmoid(out["masks"]), out["organ_ids"]

    # ------------------------------------------------------------------
    def count_parameters(self) -> Dict[str, int]:
        def count(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {
            "encoder":        count(self.encoder),
            "ode_module":     count(self.ode),
            "decoder":        count(self.decoder),
            "deep_sup_head":  count(self.deep_sup_head),
            "pfesa":          count(self.pfesa),
            "total":          count(self),
        }
