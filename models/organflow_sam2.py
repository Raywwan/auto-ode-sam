"""OrganFlow-SAM2 — V4 top-level model.

Wires MedSAM2Encoder → PFESAPlus → FlowMatchedOrganConditionedODE →
AnatomyGraphDecoder plus a DeepSupervisionHead at ODE midpoint.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.medsam2_encoder import MedSAM2Encoder
from models.pfesa_plus import PFESAPlus
from models.flow_cross_slice import FlowMatchedOrganConditionedODE
from models.anatomy_graph_decoder import AnatomyGraphDecoder


class DeepSupervisionHead(nn.Module):
    def __init__(self, embed_dim: int, n_organs: int = 15) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 4, 1),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, 2, 2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 8, n_organs, 2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class OrganFlowSAM2(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        mcfg = cfg.model
        self.cfg = cfg
        self.embed_dim = mcfg.embed_dim
        self.n_organs = mcfg.ode.n_organs

        skip_fine_channels = getattr(mcfg, "skip_fine_channels", 64)
        img_size = getattr(mcfg, "img_size", 320)
        self.encoder = MedSAM2Encoder(
            embed_dim=mcfg.embed_dim,
            skip_channels=mcfg.skip_channels,
            skip_fine_channels=skip_fine_channels,
            lora_rank=mcfg.encoder.lora_rank,
            pretrained=mcfg.encoder.pretrained,
            img_size=img_size,
        )
        self.pfesa = PFESAPlus()
        self.ode = FlowMatchedOrganConditionedODE(
            dim=mcfg.embed_dim,
            n_organs=mcfg.ode.n_organs,
            organ_emb_dim=mcfg.ode.organ_emb_dim,
            ode_hidden=mcfg.ode.ode_hidden,
            n_freqs=mcfg.ode.n_freqs,
            substeps=mcfg.ode.substeps,
        )
        self.deep_sup = DeepSupervisionHead(mcfg.embed_dim, n_organs=mcfg.ode.n_organs)
        self.decoder = AnatomyGraphDecoder(
            embed_dim=mcfg.embed_dim,
            n_organs=mcfg.decoder.n_organs,
            skip_channels=mcfg.skip_channels,
            skip_fine_channels=skip_fine_channels,
            transformer_depth=mcfg.decoder.transformer_depth,
            transformer_mlp_dim=mcfg.decoder.transformer_mlp_dim,
        )
        feat_size = self.encoder.get_output_size()[0]
        self.register_buffer("dense_pe", self._make_dense_pe(mcfg.embed_dim, feat_size))

    @staticmethod
    def _make_dense_pe(embed_dim: int, feat_size: int) -> torch.Tensor:
        half = embed_dim // 2
        grid = torch.arange(feat_size, dtype=torch.float32) / feat_size
        x, y = torch.meshgrid(grid, grid, indexing="ij")
        dim_t = torch.arange(half, dtype=torch.float32)
        dim_t = 10000 ** (2 * dim_t / half)
        pe_x = torch.sin(x.unsqueeze(-1) / dim_t.unsqueeze(0).unsqueeze(0))
        pe_y = torch.cos(y.unsqueeze(-1) / dim_t.unsqueeze(0).unsqueeze(0))
        pe = torch.cat([pe_x, pe_y], dim=-1)
        return pe.permute(2, 0, 1).unsqueeze(0)

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

        dec_out = self.decoder(center_feat, dense_pe, center_skip, center_skip_fine)

        # STEP 4: all-slice deep-sup over ±2 slices from center (quadratic weighting done in loss).
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
            # Pre-decoder centre-slice feature (B, embed_dim, H_f, W_f) — used by
            # Novel #5 (CrossModalInfoNCE) to pool an organ-conditioned image
            # embedding against BiomedCLIP text embeddings.
            "center_feat": center_feat,
            "ds_scale_logits": {
                "s40":  dec_out["ds_s40"],
                "s80":  dec_out["ds_s80"],
                "s160": dec_out["ds_s160"],
            },
        }

    def count_parameters(self) -> Dict[str, int]:
        """Return trainable + total param counts, as expected by the trainer."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}
