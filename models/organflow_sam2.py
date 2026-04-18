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

        self.encoder = MedSAM2Encoder(
            embed_dim=mcfg.embed_dim,
            skip_channels=mcfg.skip_channels,
            lora_rank=mcfg.encoder.lora_rank,
            pretrained=mcfg.encoder.pretrained,
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

        feat2d, skip2d = self.encoder(images_2d)        # (B*D, 256, 16, 16), (B*D, 128, 32, 32)
        feat2d = self.pfesa(feat2d)
        feat3d = feat2d.reshape(B, D, self.embed_dim, feat2d.shape[-2], feat2d.shape[-1])

        feat3d_out, flow_targets = self.ode(feat3d, organ_id, return_flow_targets=self.training)

        mid = D // 2
        center_feat = feat3d_out[:, mid]
        center_skip = skip2d.reshape(B, D, skip2d.shape[1], skip2d.shape[2], skip2d.shape[3])[:, mid]

        dense_pe = self.dense_pe.to(center_feat.dtype).expand(B, -1, -1, -1)
        if dense_pe.shape[-2:] != center_feat.shape[-2:]:
            dense_pe = F.interpolate(dense_pe.float(), size=center_feat.shape[-2:],
                                      mode="bilinear", align_corners=False).to(center_feat.dtype)

        masks, iou = self.decoder(center_feat, dense_pe, center_skip)
        deepsup = self.deep_sup(feat3d_out[:, mid].contiguous())
        if deepsup.shape[-2:] != masks.shape[-2:]:
            deepsup = F.interpolate(deepsup, size=masks.shape[-2:], mode="bilinear", align_corners=False)

        return {
            "masks": masks,
            "iou_pred": iou,
            "deepsup_logits": deepsup,
            "flow_targets": flow_targets,
        }
