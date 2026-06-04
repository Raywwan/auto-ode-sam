"""V9 three-stage trainer.

Stage 1: proposer only.
Stage 2: refiner + fusion (proposer frozen). Novel losses on by lambda.
Stage 3: joint fine-tune.

Design:
  - lambda=0 short-circuits each novel loss; no forward compute cost.
  - Stop-rule (spec S7c): abort if val/dice_3d < best - 0.02 on two ticks.
  - Per-component loss values returned dict, caller sums and backprops.
  - Checkpoint dir comes from cfg.experiment.output_dir. Caller sets it per
    stage (checkpoints/v9_stage1, /v9_stage2, /v9_stage3).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from models.voluformer_v9 import VoluFormerV9
from losses.flow_shape_prior import FlowShapePriorLoss
from losses.cascade_consistency import CascadeConsistencyLoss
from losses.teacher_distill import TeacherDistillLoss
from losses.cross_modal_contrastive import CrossModalInfoNCE
from models.boundary_ddpm import BoundaryDDPM


class V9Trainer:
    def __init__(self, model: VoluFormerV9, cfg) -> None:
        self.model = model
        self.cfg = cfg
        self.stage = int(cfg.training.stage)
        self.model.set_stage(self.stage)

        lw = cfg.loss
        self.w = {
            "flow_shape_prior":    float(getattr(lw, "flow_shape_prior", 0.0)),
            "cascade_kl":          float(getattr(lw, "cascade_consistency_kl", 0.0)),
            "cascade_dice":        float(getattr(lw, "cascade_consistency_dice", 0.0)),
            "boundary_ddpm":       float(getattr(lw, "boundary_ddpm", 0.0)),
            "teacher_distill":     float(getattr(lw, "teacher_distill", 0.0)),
            "cross_modal_infonce": float(getattr(lw, "cross_modal_infonce", 0.0)),
        }

        self.losses: Dict[str, torch.nn.Module] = {}
        if self.w["flow_shape_prior"] > 0:
            self.losses["flow_shape_prior"] = FlowShapePriorLoss(
                n_organs=cfg.model.n_organs,
                lam=self.w["flow_shape_prior"],
            )
        if self.w["cascade_kl"] > 0 or self.w["cascade_dice"] > 0:
            self.losses["cascade_consistency"] = CascadeConsistencyLoss(
                lam_kl=self.w["cascade_kl"],
                lam_dice=self.w["cascade_dice"],
            )
        if self.w["boundary_ddpm"] > 0:
            self.losses["boundary_ddpm"] = BoundaryDDPM(n_organs=cfg.model.n_organs)

        # Novel #3 — teacher-ensemble distillation. Teachers are optional; the
        # loss returns zero + logs a warning if none load. The proposer's
        # distill hooks are registered on first forward pass (see set_stage).
        if self.w["teacher_distill"] > 0:
            fs = int(getattr(cfg.model.proposer, "feature_size", 48))
            # SwinUNETR v2 MLP output channels double per stage: fs, 2*fs, 4*fs.
            student_taps = {
                "swinViT.layers1.blocks.1.mlp": fs,
                "swinViT.layers2.blocks.1.mlp": 2 * fs,
                "swinViT.layers3.blocks.1.mlp": 4 * fs,
            }
            self.losses["teacher_distill"] = TeacherDistillLoss(
                student_taps=student_taps,
                teachers=("sam2", "dinov2", "biomedclip"),
                n_organs=cfg.model.n_organs,
            )
            try:
                model.proposer.register_distill_hooks(tuple(student_taps.keys()))
                print("[V9Trainer] registered SwinUNETR distill hooks")
            except Exception as e:  # pragma: no cover
                print(f"[V9Trainer] WARN: failed to register distill hooks: {e}")

        # Novel #5 — cross-modal InfoNCE. BiomedCLIP lazy-loaded on first call.
        if self.w["cross_modal_infonce"] > 0:
            self.losses["cross_modal_infonce"] = CrossModalInfoNCE(
                embed_dim=int(cfg.model.embed_dim),
                n_organs=cfg.model.n_organs,
            )

        self._best_val = -1.0
        self._drops_in_a_row = 0
        self._abort_delta = float(getattr(getattr(cfg, "safety", object()), "abort_if_val_drops", 0.02))

    def step_losses(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        extra: Dict[str, torch.Tensor] = {}

        if self.stage == 1:
            return extra

        refiner = out.get("refiner", {})
        fused = out.get("fused", {})
        organ_id = batch["organ_id"]

        if "flow_shape_prior" in self.losses:
            extra["l_shape"] = self.losses["flow_shape_prior"](
                refiner["masks"], organ_id,
            )["loss"]

        if "cascade_consistency" in self.losses and fused:
            prop_slice = out.get("_prop_slice_for_consistency", None)
            ref_sig = torch.sigmoid(refiner["masks"])
            if prop_slice is not None:
                extra["l_cascade"] = self.losses["cascade_consistency"](
                    prop_slice, ref_sig,
                )["loss"]

        if "boundary_ddpm" in self.losses and "slab" in batch:
            # slab midplane = slab-local depth//2; slab_center_z is in
            # volume-patch coords and is NOT a valid slab index.
            center = batch["slab"].shape[1] // 2
            img2d = batch["slab"][:, center]
            coarse = torch.sigmoid(refiner["masks"])
            gt = batch["mask_slab"][:, :, center]
            extra["l_ddpm"] = self.w["boundary_ddpm"] * \
                self.losses["boundary_ddpm"].training_loss(img2d, coarse, gt)

        # Novel #3 — teacher-ensemble distillation. Only fires when hooks were
        # registered AND at least one teacher loaded.
        if "teacher_distill" in self.losses:
            prop = out.get("proposer", {})
            feats = prop.get("distill_feats", {}) if isinstance(prop, dict) else {}
            td_mod = self.losses["teacher_distill"]
            if feats and not td_mod.is_noop():
                td_out = td_mod(feats, batch["volume"], batch["organ_id"])
                extra["l_distill"] = self.w["teacher_distill"] * td_out["loss"]

        # Novel #5 — cross-modal (organ-text) InfoNCE. Pool `center_feat` over
        # the slab-centre organ mask to get an organ-conditioned image feat.
        if "cross_modal_infonce" in self.losses and "slab" in batch:
            cfeat = refiner.get("center_feat", None)
            if cfeat is not None:
                xm_mod = self.losses["cross_modal_infonce"]
                # slab midplane = slab-local depth//2.
                center = batch["mask_slab"].shape[2] // 2
                mask_center = batch["mask_slab"][:, :, center]     # (B, K, H, W)
                B = cfeat.shape[0]
                k_idx = organ_id.clamp(0, mask_center.shape[1] - 1)
                organ_mask = mask_center[torch.arange(B, device=cfeat.device), k_idx]
                fh, fw = cfeat.shape[-2:]
                gt_lr = F.interpolate(
                    organ_mask[:, None].float(), size=(fh, fw), mode="area",
                )
                area = gt_lr.sum(dim=(2, 3)).clamp_min(1e-6)
                pooled = (cfeat * gt_lr).sum(dim=(2, 3)) / area   # (B, C)
                xm_out = xm_mod(pooled, organ_id)
                extra["l_xmodal"] = self.w["cross_modal_infonce"] * xm_out["loss"]

        return extra

    def stop_rule_update(self, val_dice_3d: float) -> bool:
        if val_dice_3d > self._best_val:
            self._best_val = val_dice_3d
            self._drops_in_a_row = 0
            return False
        if val_dice_3d < self._best_val - self._abort_delta:
            self._drops_in_a_row += 1
            if self._drops_in_a_row >= 2:
                return True
        else:
            self._drops_in_a_row = 0
        return False

    def save_ckpt(self, path: str, epoch: int, val_dice_3d: Optional[float]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "stage": self.stage,
                "epoch": epoch,
                "val_dice_3d": val_dice_3d,
                "cfg_name": self.cfg.experiment.name,
            },
            path,
        )
