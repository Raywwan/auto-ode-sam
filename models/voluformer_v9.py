"""VoluFormer3D V9 — Tier B cascade model.

Architecture: SwinUNETR-3D proposer → (ROI sampling) → OrganFlowSAM2 refiner →
AnatomicalCascadeFusion. Implements Novel Contribution #1 (learned fusion).

Stage 1 (training): proposer alone, Dice + CE on 3D mask.
Stage 2: proposer frozen, refiner trained with slab-ROI conditioning + Novel
         losses (#3, #4, #5, #6, #7, #8).
Stage 3: full joint fine-tune with cascade consistency up-weighted.

Inference (all stages): proposer runs sliding-window 96³; refiner runs slab-wise
on ROIs derived from the proposal; fusion combines.

This module is the entry point imported by `models/__init__.py` under
architecture name `voluformer_v9`.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.swin_unetr_3d import SwinUNETRProposer
from models.organflow_sam2 import OrganFlowSAM2


class AnatomicalCascadeFusion(nn.Module):
    """Novel #1 — learned gating between 3D proposer prob and 2D refiner prob.

    Input shapes are conformed to the slab level (B, K, H, W). The proposer's
    3D prob at the slab's center slice is interpolated + passed alongside the
    refiner's output; a per-pixel gate `a ∈ [0, 1]` weighs the two.

    We condition the gate on a small learned organ embedding so different
    organs can trust proposer vs refiner differently (liver → trust refiner
    more, esophagus → trust proposer more).
    """

    def __init__(self, n_organs: int = 15, hidden: int = 32):
        super().__init__()
        self.n_organs = n_organs
        self.organ_emb = nn.Embedding(n_organs, hidden)
        # (prop, refiner, |prop-refiner|, organ_emb[:, None, None]) → gate
        in_ch = 3 + hidden
        self.gate = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        # Bias init toward refiner (sigmoid(-1.0) ≈ 0.27 weight on proposer).
        with torch.no_grad():
            self.gate[-1].bias.fill_(-1.0)

    def forward(
        self,
        prop_slice: torch.Tensor,        # (B, K, H, W) proposer prob at slab center slice
        refiner_mask: torch.Tensor,      # (B, K, H, W) refiner prob
        organ_id: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, K, H, W = prop_slice.shape
        if refiner_mask.shape[-2:] != (H, W):
            refiner_mask = F.interpolate(
                refiner_mask, size=(H, W), mode="bilinear", align_corners=False,
            )
        # Per organ channel: stack (prop_k, ref_k, |prop_k - ref_k|, organ_emb).
        # We loop K to keep the gate parameter-light; K=15 is small.
        fused_per_organ = []
        gates_per_organ = []
        for k in range(K):
            if organ_id is None:
                oid = torch.tensor([k], device=prop_slice.device).expand(B)
            else:
                oid = organ_id
            emb = self.organ_emb(oid)                                # (B, hidden)
            emb_spatial = emb[:, :, None, None].expand(-1, -1, H, W)
            pk = prop_slice[:, k:k+1]
            rk = refiner_mask[:, k:k+1]
            stack = torch.cat([pk, rk, (pk - rk).abs(), emb_spatial], dim=1)
            a = torch.sigmoid(self.gate(stack))                      # (B, 1, H, W)
            fused = a * pk + (1.0 - a) * rk
            fused_per_organ.append(fused)
            gates_per_organ.append(a)
        fused_all = torch.cat(fused_per_organ, dim=1)                # (B, K, H, W)
        gates_all = torch.cat(gates_per_organ, dim=1)                # (B, K, H, W)
        return {"fused": fused_all, "gate": gates_all}


class VoluFormerV9(nn.Module):
    """V9 Tier B cascade. Entry point for `build_model({arch: voluformer_v9})`.

    Training-time forward accepts a joint sample dict:
      - `volume`: (B, 1, D_vol, H_vol, W_vol) isotropic-resampled CT, patch-crop
        supplied by the dataset (96³).
      - `slab`: (B, D_slab, 3, H_img, W_img) — the 2D-slab view of the same
        volume, shared with V7 refiner path.
      - `organ_id`: (B,) long.

    The module does NOT itself do sliding-window reassembly — that happens in
    inference/v9_inference.py.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        n_organs = cfg.model.get("n_organs", 15)

        # Stage 1 — SwinUNETR-3D proposer.
        prop_cfg = cfg.model.proposer
        self.proposer = SwinUNETRProposer(
            n_organs=n_organs,
            patch_size=prop_cfg.get("patch_size", 96),
            feature_size=prop_cfg.get("feature_size", 48),
            use_v2=prop_cfg.get("use_v2", True),
            pretrained_weights=prop_cfg.get("pretrained_weights", None),
        )

        # Stage 2 — OrganFlowSAM2 refiner.
        self.refiner = OrganFlowSAM2(cfg)

        # Fusion head (Novel #1).
        self.fusion = AnatomicalCascadeFusion(n_organs=n_organs)

        # Stage freeze switches (controlled by training script).
        self.freeze_proposer = False
        self.freeze_refiner = False

    def set_stage(self, stage: int) -> None:
        """Turn on/off gradients for proposer/refiner.
        Stage 1: only proposer trains.
        Stage 2: refiner + fusion train; proposer frozen.
        Stage 3: everything trains.
        """
        if stage == 1:
            for p in self.proposer.parameters(): p.requires_grad = True
            for p in self.refiner.parameters():  p.requires_grad = False
            for p in self.fusion.parameters():   p.requires_grad = False
            self.freeze_proposer = False
            self.freeze_refiner = True
        elif stage == 2:
            for p in self.proposer.parameters(): p.requires_grad = False
            for p in self.refiner.parameters():  p.requires_grad = True
            for p in self.fusion.parameters():   p.requires_grad = True
            self.freeze_proposer = True
            self.freeze_refiner = False
        elif stage == 3:
            for p in self.proposer.parameters(): p.requires_grad = True
            for p in self.refiner.parameters():  p.requires_grad = True
            for p in self.fusion.parameters():   p.requires_grad = True
            self.freeze_proposer = False
            self.freeze_refiner = False
        else:
            raise ValueError(f"unknown stage {stage}; expected 1, 2, or 3")
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[V9] stage={stage} trainable={n_train/1e6:.1f}M")

    def forward(
        self,
        sample: Dict[str, torch.Tensor],
        stage: int = 2,
        return_fusion: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Return a dict with proposer+refiner+fused outputs.

        sample keys:
            volume (B, 1, D_v, H_v, W_v)  — 3D patch for proposer
            slab   (B, D_s, 3, H_s, W_s)  — 2D slab for refiner
            organ_id (B,)
            slab_center_z (B,) — integer slice index of slab center in *volume*
                                 coords, used to align proposer prob for fusion.
        """
        out: Dict[str, torch.Tensor] = {}

        # ---- Stage 1: proposer -------------------------------------------
        prop_ctx = torch.no_grad() if (stage == 2 and self.freeze_proposer) else torch.enable_grad()
        with prop_ctx:
            prop = self.proposer(sample["volume"])
        out["proposer"] = prop  # dict with logits/probs/...

        if stage == 1:
            return out  # Stage 1 training only cares about the 3D proposer.

        # ---- Stage 2: refiner on slab ------------------------------------
        refiner_out = self.refiner(
            images=sample["slab"],
            organ_id=sample["organ_id"],
            is_3d=True,
        )
        out["refiner"] = refiner_out

        # ---- Fusion: align proposer prob to slab center slice -----------
        if return_fusion and "slab_center_z" in sample:
            B = sample["slab"].shape[0]
            # Proposer prob: (B, K, D_v, H_v, W_v). Pick z-slice per sample.
            # Assume proposer and slab are in the same H×W (after internal resize
            # in the dataset). Dataset is responsible for this.
            K = prop["probs"].shape[1]
            D_v, H_v, W_v = prop["probs"].shape[-3:]
            idx = sample["slab_center_z"].clamp(0, D_v - 1)           # (B,)
            # Gather: gather along dim=2
            idx_exp = idx.view(B, 1, 1, 1, 1).expand(-1, K, 1, H_v, W_v)
            prop_slice = prop["probs"].gather(2, idx_exp).squeeze(2)   # (B, K, H_v, W_v)

            # Refiner mask: (B, K, H_r, W_r). Sigmoid applied.
            refiner_mask = torch.sigmoid(refiner_out["masks"])

            # Align proposer → refiner spatial resolution.
            if prop_slice.shape[-2:] != refiner_mask.shape[-2:]:
                prop_slice = F.interpolate(
                    prop_slice, size=refiner_mask.shape[-2:],
                    mode="bilinear", align_corners=False,
                )

            fused = self.fusion(prop_slice, refiner_mask, sample["organ_id"])
            out["fused"] = fused               # dict with `fused` and `gate`
            out["_prop_slice_for_consistency"] = prop_slice

        return out

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}
