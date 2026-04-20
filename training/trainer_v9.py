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

from models.voluformer_v9 import VoluFormerV9
from losses.flow_shape_prior import FlowShapePriorLoss
from losses.cascade_consistency import CascadeConsistencyLoss
from models.boundary_ddpm import BoundaryDDPM


class V9Trainer:
    def __init__(self, model: VoluFormerV9, cfg) -> None:
        self.model = model
        self.cfg = cfg
        self.stage = int(cfg.training.stage)
        self.model.set_stage(self.stage)

        lw = cfg.loss
        self.w = {
            "flow_shape_prior":  float(getattr(lw, "flow_shape_prior", 0.0)),
            "cascade_kl":        float(getattr(lw, "cascade_consistency_kl", 0.0)),
            "cascade_dice":      float(getattr(lw, "cascade_consistency_dice", 0.0)),
            "boundary_ddpm":     float(getattr(lw, "boundary_ddpm", 0.0)),
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
            center = int(batch["slab_center_z"][0].item())
            img2d = batch["slab"][:, center]
            coarse = torch.sigmoid(refiner["masks"])
            gt = batch["mask_slab"][:, :, center]
            extra["l_ddpm"] = self.w["boundary_ddpm"] * \
                self.losses["boundary_ddpm"].training_loss(img2d, coarse, gt)

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
