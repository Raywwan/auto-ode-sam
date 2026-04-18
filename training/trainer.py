# =============================================================================
# training/trainer.py — Training Loop (v2)
#
# v2 changes vs v1:
#   - AMOS22_3D_Dataset always used (cfg.data.dataset = "amos22_3d")
#   - modality_ids from 3D dataset has shape (B, D) — extract center slice
#   - Model uses center-slice features for supervised loss (ISA still runs on all D)
#   - Imports evaluation.metrics (2D slice Dice) for training-time monitoring only.
#     Final reported metrics come from evaluate_3d.py (volumetric).
# =============================================================================

import os
import random
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import build_model
from datasets import get_dataset
from training.losses import CombinedLoss

# Per-organ loss weights: upweight small/rare organs to counteract their
# 5-50× smaller voxel count vs liver/spleen.
#
# Calibration (Apr 2026): based on approximate AMOS22 organ volume ratios.
# Adrenal glands are ~15-20× smaller than liver by voxel count; 2× was
# insufficient — model consistently under-attends them at epoch 10-20.
# AMOS22 best CT-only adrenal DSC is 0.81/0.84 (R/L) — our target.
# Increasing to 3.0× for adrenals, 2.5× for gallbladder, 2.0× for
# esophagus/pancreas/duodenum to better reflect the true signal imbalance.
_ORGAN_LOSS_WEIGHTS: dict = {
    4: 2.5,   # gallbladder   (~25ml — variable shape/absence)
    5: 2.0,   # esophagus     (~thin tubular, hard boundaries)
    10: 2.0,  # pancreas      (~80ml — irregular, hard)
    11: 3.0,  # right adrenal (~4ml  — tiny, AMOS22 SOTA 0.81 DSC)
    12: 3.0,  # left adrenal  (~4ml  — tiny, AMOS22 SOTA 0.84 DSC)
    13: 2.0,  # duodenum      (~thin loop, hard to delineate)
    # all other organs default to 1.0
}
from evaluation.metrics import SegmentationMetrics
from utils.checkpoint import CheckpointManager
from utils.logger import ExperimentLogger, make_segmentation_overlay


def cutmix_batch(
    images: torch.Tensor,       # (B, D, 3, H, W)
    masks: torch.Tensor,        # (B, H, W) — center slice
    masks_all: torch.Tensor,    # (B, D, H, W) or None
    boxes: torch.Tensor,        # (B, 4)
    prob: float = 0.5,
) -> tuple:
    """
    CutMix augmentation at batch level.

    For each sample in the batch (with probability `prob`), pastes a random
    rectangular region from another sample.  Labels in the pasted region come
    from the source sample.  Bounding box is recomputed from the new mask.

    Achieves +4.9% DSC on AMOS22 (arXiv:2602.03555).
    """
    B, D, C, H, W = images.shape

    for i in range(B):
        if random.random() > prob:
            continue
        j = random.randint(0, B - 1)
        while j == i:
            j = random.randint(0, B - 1)

        # Random rectangular cut — area fraction drawn from Beta(1,1) ~ Uniform
        lam = np.random.beta(1.0, 1.0)
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = max(1, int(H * cut_ratio))
        cut_w = max(1, int(W * cut_ratio))
        cx = np.random.randint(0, H)
        cy = np.random.randint(0, W)
        x1 = max(0, cx - cut_h // 2)
        x2 = min(H, x1 + cut_h)
        y1 = max(0, cy - cut_w // 2)
        y2 = min(W, y1 + cut_w)

        # Paste all D slices of image region from j → i
        images[i, :, :, x1:x2, y1:y2] = images[j, :, :, x1:x2, y1:y2]
        # Paste center mask region
        masks[i, x1:x2, y1:y2] = masks[j, x1:x2, y1:y2]
        # Paste all-slice masks if available
        if masks_all is not None:
            masks_all[i, :, x1:x2, y1:y2] = masks_all[j, :, x1:x2, y1:y2]

        # Recompute bounding box from new center mask
        fg = masks[i] > 0.5
        if fg.any():
            rows = fg.any(dim=1).nonzero(as_tuple=False).squeeze(1)
            cols = fg.any(dim=0).nonzero(as_tuple=False).squeeze(1)
            r0, r1 = rows[0].item(), rows[-1].item() + 1
            c0, c1 = cols[0].item(), cols[-1].item() + 1
            # Box format: (x1, y1, x2, y2) where x=col, y=row
            boxes[i] = torch.tensor(
                [c0, r0, c1, r1], dtype=boxes.dtype, device=boxes.device
            )
        # If no foreground after mix, keep original box — model learns empty-mask case

    return images, masks, masks_all, boxes


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False   # False = allow cuDNN auto-tune
    torch.backends.cudnn.benchmark = True        # True  = 10-30% faster on fixed input sizes


class WarmupCosineScheduler:
    """Linear warmup → cosine annealing LR schedule."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        base_lr: float,
        lr_min: float = 1e-6,
        warmup_lr_init: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = base_lr
        self.lr_min = lr_min
        self.warmup_lr_init = warmup_lr_init

    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            lr = self.warmup_lr_init + (self.base_lr - self.warmup_lr_init) * (
                epoch / max(1, self.warmup_epochs)
            )
        else:
            progress = (epoch - self.warmup_epochs) / max(
                1, self.max_epochs - self.warmup_epochs
            )
            lr = self.lr_min + 0.5 * (self.base_lr - self.lr_min) * (
                1 + np.cos(np.pi * progress)
            )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def get_last_lr(self):
        return [g["lr"] for g in self.optimizer.param_groups]

    def state_dict(self):
        return {k: getattr(self, k) for k in
                ("warmup_epochs", "max_epochs", "base_lr", "lr_min", "warmup_lr_init")}

    def load_state_dict(self, state: dict):
        for k, v in state.items():
            setattr(self, k, v)


def _extract_center_modality(modality_ids, is_3d: bool):
    """
    AMOS22_3D_Dataset returns modality_id of shape (B, D) — same modality replicated.
    Extract center slice to get (B,) for the model.

    V4 multi-organ dataset doesn't emit modality_id at all, so None is a valid input
    (caller sets modality_ids to None when the batch key is absent).
    """
    if modality_ids is None:
        return None
    if is_3d and modality_ids.dim() == 2:
        D = modality_ids.shape[1]
        return modality_ids[:, D // 2]
    return modality_ids


class Trainer:
    """Full training pipeline for LiteSAM-3D v2."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        set_seed(cfg.experiment.seed)

        self.exp_dir = Path(cfg.experiment.output_dir) / cfg.experiment.name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self.logger = ExperimentLogger(cfg)
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Experiment: {cfg.experiment.name}")

        # ---- Model ----
        self.logger.info("Building model...")
        self.model = build_model(cfg).to(self.device)

        # Optionally freeze TinyViT backbone (proj + neck remain trainable)
        # Makes ablation faster and more controlled: pure ISA effect, no encoder drift
        if getattr(cfg.model, "encoder_freeze_backbone", False):
            for param in self.model.encoder.backbone.parameters():
                param.requires_grad = False
            frozen = sum(p.numel() for p in self.model.encoder.backbone.parameters())
            self.logger.info(f"Encoder backbone FROZEN ({frozen:,} params, proj+neck still train)")

        param_counts = self.model.count_parameters()
        self.logger.info(f"Parameter counts: {param_counts}")
        self.logger.info(f"Total params: {param_counts['total']:,}")

        # ---- Datasets ----
        # v2: always amos22_3d so ISA is active during training
        self.logger.info("Loading datasets...")
        _copypaste_prob = getattr(cfg.training, "copypaste_prob", 0.0)
        train_dataset = get_dataset(cfg, split="train", copypaste_prob=_copypaste_prob)
        val_dataset   = get_dataset(cfg, split="val",   copypaste_prob=0.0)

        _num_workers = cfg.training.num_workers
        _persistent = _num_workers > 0
        _val_num_workers = _num_workers  # match train — val was ignoring config
        _val_persistent = _val_num_workers > 0
        # For 3D datasets, val samples are (D, 3, H, W) — much larger than 2D.
        # Use same batch size as training (not 4x) to avoid OOM.
        _is_3d_dataset = cfg.data.dataset.lower().endswith("_3d")
        _val_batch_size = cfg.training.batch_size if _is_3d_dataset else cfg.training.batch_size * 4

        # Foreground oversampling (AutoODESAM / small-organ configs)
        use_oversampling = getattr(cfg.training, 'foreground_oversampling', False)
        _train_sampler = None
        if use_oversampling and hasattr(train_dataset, 'get_oversampled_indices'):
            from torch.utils.data import SubsetRandomSampler
            oversampled_indices = train_dataset.get_oversampled_indices(
                len(train_dataset), small_organ_fraction=0.33
            )
            _train_sampler = SubsetRandomSampler(oversampled_indices)
            self.logger.info(
                f"Foreground oversampling enabled: {len(oversampled_indices)} indices "
                f"(dataset size={len(train_dataset)})"
            )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=(_train_sampler is None),   # mutually exclusive with sampler
            sampler=_train_sampler,
            num_workers=_num_workers,
            pin_memory=cfg.training.pin_memory,
            drop_last=True,
            persistent_workers=_persistent,
            prefetch_factor=2 if _num_workers > 0 else None,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=_val_batch_size,
            shuffle=False,
            num_workers=_val_num_workers,
            pin_memory=cfg.training.pin_memory,
            persistent_workers=_val_persistent,
            prefetch_factor=2 if _val_num_workers > 0 else None,
        )
        self.logger.info(
            f"Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples"
        )

        # ---- Loss ----
        self.criterion = CombinedLoss(cfg.training.loss)

        # ---- Phase 2 training flags ----
        self.all_slice_supervision = getattr(cfg.training, "all_slice_supervision", False)
        self.cutmix_prob = getattr(cfg.training, "cutmix_prob", 0.0)

        # ---- Optimiser — layer-wise learning rate ----
        # Encoder (TinyViT backbone) fine-tunes at a lower rate to preserve
        # pretrained features. Decoder + ODE modules train at full LR.
        # encoder_lr_scale: 0.1 → encoder LR = 1e-5 when base LR = 1e-4.
        encoder_lr_scale = getattr(cfg.training, "encoder_lr_scale", 0.1)
        base_lr   = cfg.training.lr
        enc_lr    = base_lr * encoder_lr_scale
        wd        = cfg.training.weight_decay

        # Build a set of parameter full-names that must skip weight decay.
        # Substring match ("embed", "bias") misses nn.Embedding subclasses
        # whose parameters are named "weight" (e.g. organ_queries, hq_token).
        # Use isinstance() to catch ALL Embedding / normalization parameters.
        no_wd_param_names: set = set()
        for mname, module in self.model.named_modules():
            if isinstance(module, (nn.Embedding, nn.LayerNorm, nn.GroupNorm,
                                   nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                for pname, _ in module.named_parameters(recurse=False):
                    full = f"{mname}.{pname}" if mname else pname
                    no_wd_param_names.add(full)

        enc_decay, enc_nodecay, dec_decay, dec_nodecay = [], [], [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            is_encoder = name.startswith("encoder.")
            is_nodecay  = (
                any(nd in name for nd in ("bias", "norm", "bn"))
                or name in no_wd_param_names
            )
            if is_encoder:
                (enc_nodecay if is_nodecay else enc_decay).append(param)
            else:
                (dec_nodecay if is_nodecay else dec_decay).append(param)

        self.optimizer = AdamW(
            [
                {"params": enc_decay,   "lr": enc_lr,  "weight_decay": wd},
                {"params": enc_nodecay, "lr": enc_lr,  "weight_decay": 0.0},
                {"params": dec_decay,   "lr": base_lr, "weight_decay": wd},
                {"params": dec_nodecay, "lr": base_lr, "weight_decay": 0.0},
            ],
            lr=base_lr,
            betas=tuple(cfg.training.betas),
        )

        # ---- Scheduler ----
        self.scheduler = WarmupCosineScheduler(
            optimizer=self.optimizer,
            warmup_epochs=cfg.training.warmup_epochs,
            max_epochs=cfg.training.epochs,
            base_lr=cfg.training.lr,
            lr_min=cfg.training.lr_min,
            warmup_lr_init=cfg.training.warmup_lr_init,
        )

        # ---- AMP ----
        self.use_amp = cfg.training.mixed_precision and self.device.type == "cuda"
        self.scaler = GradScaler("cuda") if self.use_amp else None
        if self.use_amp:
            self.logger.info("Mixed precision (fp16) enabled")

        self.grad_accum_steps = cfg.training.grad_accumulation_steps

        # ---- Checkpoint ----
        self.ckpt_manager = CheckpointManager(
            save_dir=str(self.exp_dir),
            keep_top_k=cfg.checkpoint.keep_top_k,
            monitor_metric="val_dice",
            experiment_name=cfg.experiment.name,
        )

        self.start_epoch = 0
        resume_path = cfg.checkpoint.resume_from
        if resume_path:
            self.logger.info(f"Resuming from: {resume_path}")
            self.start_epoch, prev_metrics = self.ckpt_manager.load(
                resume_path, self.model, self.optimizer, self.scheduler,
                self.scaler, self.device,
            )
            self.start_epoch += 1
            self.logger.info(f"Resumed at epoch {self.start_epoch}")
            self.logger.reinit_tensorboard(purge_step=self.start_epoch)

        self.logger.setup_layout()
        self.global_step = 0

    def train(self):
        """Main training loop."""
        self.logger.info(f"Starting training for {self.cfg.training.epochs} epochs")
        best_dice = 0.0

        for epoch in range(self.start_epoch, self.cfg.training.epochs):
            epoch_start = time.time()
            current_lr = self.scheduler.step(epoch)
            self.logger.log({"train/lr": current_lr}, step=epoch)

            train_metrics = self._train_epoch(epoch)

            is_last_epoch = epoch == self.cfg.training.epochs - 1
            if epoch % self.cfg.validation.val_every_n_epochs == 0 or is_last_epoch:
                val_metrics = self._val_epoch(epoch)
                current_dice = val_metrics.get("val_dice", 0.0)
            else:
                val_metrics = {}
                current_dice = best_dice

            if epoch % self.cfg.checkpoint.save_every_n_epochs == 0 or current_dice > best_dice:
                self.ckpt_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    metrics={**train_metrics, **val_metrics},
                    scaler=self.scaler,
                )

            if current_dice > best_dice:
                best_dice = current_dice
                self.logger.info(f"  New best Dice: {best_dice:.4f} at epoch {epoch}")

            elapsed = time.time() - epoch_start
            self.logger.info(
                f"Epoch {epoch:3d} | "
                f"Loss: {train_metrics.get('train_loss', 0):.4f} | "
                f"Val Dice: {val_metrics.get('val_dice', 0):.4f} | "
                f"LR: {current_lr:.2e} | Time: {elapsed:.0f}s"
            )

        self.logger.info(f"Training complete. Best Dice: {best_dice:.4f}")
        self.logger.finish()

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """One training epoch."""
        self.model.train()
        total_loss = 0.0
        total_dice = 0.0
        total_grad_norm = 0.0
        loss_component_totals = {}
        n_batches = 0
        n_grad_steps = 0

        self.optimizer.zero_grad()
        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch}",
                    leave=False, dynamic_ncols=True)

        is_organflow = getattr(self.cfg.model, "architecture", None) == "organflow_sam2"

        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(self.device)        # (B, D, 3, H, W) for 3D
            # V4 dataset ships {image, masks, present_mask} only — no mask/box/modality_id.
            masks = batch["mask"].to(self.device) if "mask" in batch else None
            boxes = batch["box"].to(self.device) if "box" in batch else None
            modality_ids = batch["modality_id"].to(self.device) if "modality_id" in batch else None
            is_3d = batch.get("is_3d", torch.tensor([True if is_organflow else False]))
            if isinstance(is_3d, torch.Tensor):
                is_3d = is_3d[0].item()

            # Load all-slice masks if dataset provides them
            masks_all = None
            if "masks_all" in batch:
                masks_all = batch["masks_all"].to(self.device)  # (B, D, H, W)

            # ---- Detect model type (needed for path-specific augmentation) ----
            is_auto_ode = (
                hasattr(self.model, 'decoder')
                and not hasattr(self.model, 'prompt_encoder')
            )

            # ---- CutMix augmentation (train only) ----
            # DISABLED for AutoODESAM: CutMix mixes items with different organ_ids.
            # The pasted region carries item-B's mask (e.g. adrenal) into item-A's
            # training (e.g. liver), producing wrong ground truth in the pasted area.
            # AutoODESAM has copy-paste augmentation for small organs instead.
            if self.cutmix_prob > 0 and is_3d and not is_auto_ode:
                images, masks, masks_all, boxes = cutmix_batch(
                    images, masks, masks_all, boxes, prob=self.cutmix_prob
                )

            # v2 fix: extract center modality if AMOS22_3D_Dataset returns (B, D)
            modality_ids_center = _extract_center_modality(modality_ids, is_3d)

            # Extract organ_id if dataset provides it (used by AutoODESAM / ACM-SAM)
            organ_id = batch.get("organ_id", None)
            if organ_id is not None:
                organ_id = organ_id.to(self.device)
                if organ_id.dim() > 1:
                    # Use center slice (D//2), consistent with modality_ids_center
                    organ_id = organ_id[:, organ_id.shape[1] // 2]

            with autocast("cuda", enabled=self.use_amp):

                # Per-sample loss weights: adrenal 3×, gallbladder 2.5×, others 2×.
                # Computed BEFORE criterion call so weighting happens inside the
                # per-sample loss before .mean(), not on the batch-averaged loss.
                sample_weight = None
                if organ_id is not None:
                    sample_weight = torch.tensor(
                        [_ORGAN_LOSS_WEIGHTS.get(int(o), 1.0) for o in organ_id],
                        dtype=torch.float32, device=self.device,
                    )

                # ---- OrganFlowSAM2 (V4) path: multi-organ flow-matched ODE ----
                if is_organflow:
                    from training.losses_v4 import V4MultiOrganLoss
                    if not hasattr(self, "_v4_criterion"):
                        self._v4_criterion = V4MultiOrganLoss(
                            lambda_flow=getattr(self.cfg.training, "lambda_flow", 0.5),
                            lambda_deepsup=getattr(self.cfg.training, "lambda_deepsup", 0.1),
                            lambda_anatomy=getattr(self.cfg.training, "lambda_anatomy", 0.01),
                        )
                    images_v4 = batch["image"].to(self.device)                  # (B, D, 1, H, W)
                    if images_v4.shape[2] == 1:
                        images_v4 = images_v4.repeat(1, 1, 3, 1, 1)
                    masks_v4 = batch["masks"].to(self.device)                   # (B, 15, D, H, W)
                    present_v4 = batch["present_mask"].to(self.device)          # (B, 15) bool
                    organ_id_v4 = torch.argmax(present_v4.int(), dim=1) + 1     # 1-indexed primary organ per sample
                    outputs = self.model(images_v4, organ_id_v4, is_3d=True)
                    warmup_eps = int(getattr(self.cfg.training, "flow_warmup_epochs", 3))
                    lf_target = float(getattr(self.cfg.training, "lambda_flow", 0.5))
                    lf_now = lf_target * min(1.0, epoch / max(warmup_eps, 1))
                    out = self._v4_criterion(
                        pred_masks=outputs["masks"],
                        gt_masks=masks_v4,
                        present_mask=present_v4,
                        flow_targets=outputs.get("flow_targets"),
                        deepsup_logits=outputs.get("deepsup_logits"),
                        anatomy_adjacency=self.model.decoder.graph.adjacency,
                        anatomy_adjacency_init=getattr(self.model.decoder.graph, "_init_adjacency", None),
                        lambda_flow_override=lf_now,
                    )
                    loss = out["loss"]
                    loss_components = out["components"]
                    loss_components["total"] = loss.item()

                    # Per-batch Dice monitor: gather primary organ channel per sample.
                    with torch.no_grad():
                        idx = (organ_id_v4 - 1).long()
                        pm = outputs["masks"]                                    # (B, 15, h, w)
                        H_out, W_out = pm.shape[-2:]
                        best_masks = pm.gather(
                            1, idx.view(-1, 1, 1, 1).expand(-1, 1, H_out, W_out)
                        ).squeeze(1)
                        D_v4 = masks_v4.shape[2]
                        gt_center = masks_v4[:, :, D_v4 // 2, :, :]              # (B, 15, H, W)
                        Hg, Wg = gt_center.shape[-2:]
                        masks_resized = gt_center.gather(
                            1, idx.view(-1, 1, 1, 1).expand(-1, 1, Hg, Wg)
                        ).squeeze(1).float()
                        if masks_resized.shape[-2:] != best_masks.shape[-2:]:
                            masks_resized = F_nn.interpolate(
                                masks_resized.unsqueeze(1),
                                size=best_masks.shape[-2:], mode="nearest",
                            ).squeeze(1)

                # ---- AutoODESAM path (no prompt_encoder, uses organ_id) ----
                elif is_auto_ode:
                    outputs = self.model(images, organ_id, is_3d=is_3d)

                    # AutoODESAM returns masks shape (B, n_organs, H_out, W_out).
                    # Per-item gather: each batch item selects its own organ channel
                    # (1-indexed organ_id → 0-indexed).
                    pred_masks_all = outputs["masks"]  # (B, n_organs, H_out, W_out)
                    iou_pred_all = outputs["iou_pred"]  # (B, n_organs)
                    deepsup_logits = outputs.get("deepsup_logits", None)

                    organ_idxs = (organ_id - 1).long()  # (B,) 0-indexed
                    H_out, W_out = pred_masks_all.shape[-2:]
                    best_masks = pred_masks_all.gather(
                        1, organ_idxs.view(-1, 1, 1, 1).expand(-1, 1, H_out, W_out)
                    ).squeeze(1)  # (B, H_out, W_out)
                    iou_pred_organ = iou_pred_all.gather(
                        1, organ_idxs.unsqueeze(1)
                    ).squeeze(1)  # (B,)

                    if best_masks.shape[-2:] != masks.shape[-2:]:
                        masks_resized = F_nn.interpolate(
                            masks.unsqueeze(1).float(),
                            size=best_masks.shape[-2:],
                            mode="nearest",
                        ).squeeze(1)
                    else:
                        masks_resized = masks.float()

                    loss, loss_components = self.criterion(
                        pred=best_masks,
                        target=masks_resized,
                        iou_pred=iou_pred_organ,
                        deepsup_logits=deepsup_logits,
                        deepsup_organ_idx=organ_idxs,  # (B,) — per-item 0-indexed
                        current_epoch=epoch,
                        boundary_ramp_epoch=getattr(self.cfg.training, 'boundary_ramp_epoch', 50),
                        sample_weight=sample_weight,
                    )
                    # Diversity regularisation from OrganQueryDecoder
                    if hasattr(self.model.decoder, 'compute_diversity_loss'):
                        div_loss = self.model.decoder.compute_diversity_loss()
                        loss = loss + div_loss
                        loss_components["diversity"] = div_loss.item()

                    # Record "total" AFTER diversity so logged loss matches backpropped loss
                    loss_components["total"] = loss.item()

                # ---- All-slice supervision path (ODE-SAM / VoluFormer3D only) ----
                elif not is_auto_ode and self.all_slice_supervision and is_3d \
                        and masks_all is not None \
                        and hasattr(self.model, "encode_image"):
                    B_sz, D_sz = images.shape[0], images.shape[1]

                    # Encode all D slices in one pass (encoder + ODE runs once)
                    feat_all = self.model.encode_image(
                        images, is_3d=True, return_all_slices=True
                    )  # (B*D, C, Hf, Wf)

                    # Retrieve Stage 1 skip features if the model cached them
                    # (set by encode_image in models with hierarchical skip connections)
                    skip_all = getattr(self.model, '_skip_cache', None)

                    # Expand boxes and modality to (B*D,) — same box for all D slices
                    boxes_all = boxes.unsqueeze(1).expand(-1, D_sz, -1).reshape(B_sz * D_sz, 4)
                    mod_all = modality_ids_center.unsqueeze(1).expand(-1, D_sz).reshape(B_sz * D_sz)

                    sparse_emb, dense_emb = self.model.prompt_encoder(
                        boxes=boxes_all,
                        modality_ids=mod_all,
                        image_features=feat_all,
                    )
                    masks_pred_all, iou_pred_all, modality_logits = self.model.mask_decoder(
                        image_embeddings=feat_all,
                        sparse_prompt_embeddings=sparse_emb,
                        dense_prompt_embeddings=dense_emb,
                        multimask_output=True,
                        skip_features=skip_all,
                    )  # (B*D, 3, H_out, W_out)

                    best_idx_all = iou_pred_all.argmax(dim=1)
                    best_masks_all = masks_pred_all[
                        torch.arange(B_sz * D_sz, device=self.device), best_idx_all
                    ]  # (B*D, H_out, W_out)

                    # GT masks for all slices — resize to decoder output size
                    masks_all_flat = masks_all.reshape(B_sz * D_sz, masks_all.shape[-2], masks_all.shape[-1])
                    if best_masks_all.shape[-2:] != masks_all_flat.shape[-2:]:
                        masks_all_flat = F_nn.interpolate(
                            masks_all_flat.unsqueeze(1).float(),
                            size=best_masks_all.shape[-2:],
                            mode="nearest",
                        ).squeeze(1)

                    loss, loss_components = self.criterion.compute_all(
                        pred=best_masks_all,
                        target=masks_all_flat,
                        iou_pred=iou_pred_all[
                            torch.arange(B_sz * D_sz, device=self.device), best_idx_all
                        ],
                        modality_logits=modality_logits,
                        modality_ids=mod_all,
                    )

                    # Use center slice for the per-batch Dice monitor.
                    # Layout of best_masks_all: [b0d0, b0d1, ..., b0dD, b1d0, ...]
                    # Center slice index for batch item i: i*D + D//2
                    center_d = D_sz // 2
                    center_indices = torch.arange(B_sz, device=self.device) * D_sz + center_d
                    best_masks = best_masks_all[center_indices]
                    masks_resized = masks_all_flat[center_indices]

                # ---- Standard center-slice path ----
                else:
                    output = self.model(
                        images=images,
                        boxes=boxes,
                        modality_ids=modality_ids_center,
                        is_3d=is_3d,
                        multimask_output=True,
                        organ_id=organ_id,
                    )

                    masks_pred = output["masks"]      # (B, 3, H_out, W_out)
                    iou_pred = output["iou_pred"]     # (B, 3)
                    modality_logits = output["modality_logits"]

                    best_idx = iou_pred.argmax(dim=1)
                    best_masks = masks_pred[
                        torch.arange(images.shape[0], device=self.device), best_idx
                    ]  # (B, H_out, W_out)

                    if best_masks.shape[-2:] != masks.shape[-2:]:
                        masks_resized = F_nn.interpolate(
                            masks.unsqueeze(1).float(),
                            size=best_masks.shape[-2:],
                            mode="nearest",
                        ).squeeze(1)
                    else:
                        masks_resized = masks.float()

                    loss, loss_components = self.criterion.compute_all(
                        pred=best_masks,
                        target=masks_resized,
                        iou_pred=iou_pred[
                            torch.arange(images.shape[0], device=self.device), best_idx
                        ],
                        modality_logits=modality_logits,
                        modality_ids=modality_ids_center,
                    )

                loss = loss / self.grad_accum_steps

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.training.grad_clip
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.training.grad_clip
                    )
                    self.optimizer.step()
                self.optimizer.zero_grad()
                total_grad_norm += grad_norm.item()
                n_grad_steps += 1

            with torch.no_grad():
                pred_bin = (torch.sigmoid(best_masks) > 0.5).float()
                intersection = (pred_bin * masks_resized).sum(dim=[1, 2])
                dice = (2 * intersection + 1e-5) / (
                    pred_bin.sum(dim=[1, 2]) + masks_resized.sum(dim=[1, 2]) + 1e-5
                )
                batch_dice = dice.mean().item()

            total_loss += loss_components["total"]
            total_dice += batch_dice
            for k, v in loss_components.items():
                loss_component_totals[k] = loss_component_totals.get(k, 0.0) + v
            n_batches += 1
            self.global_step += 1

            if self.global_step % self.cfg.logging.log_every_n_steps == 0:
                log_dict = {f"train/{k}": v for k, v in loss_components.items()}
                log_dict["train/dice_batch"] = batch_dice
                if n_grad_steps > 0:
                    log_dict["train/grad_norm"] = total_grad_norm / n_grad_steps
                self.logger.log(log_dict, step=self.global_step)

            pbar.set_postfix({
                "loss": f"{loss_components['total']:.4f}",
                "dice": f"{batch_dice:.4f}",
            })

        avg_loss = total_loss / max(n_batches, 1)
        avg_dice = total_dice / max(n_batches, 1)
        avg_grad_norm = total_grad_norm / max(n_grad_steps, 1)

        epoch_log = {
            "train/loss_epoch": avg_loss,
            "train/dice_epoch": avg_dice,
            "train/grad_norm_epoch": avg_grad_norm,
            "train/gpu_mem_gb": torch.cuda.memory_reserved() / 1e9 if self.device.type == "cuda" else 0.0,
        }
        for k, v in loss_component_totals.items():
            if k != "total":
                epoch_log[f"train/{k}_epoch_comp"] = v / max(n_batches, 1)
        self.logger.log(epoch_log, step=epoch)

        return {"train_loss": avg_loss, "train_dice": avg_dice}

    def _val_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Validation epoch.

        NOTE: This computes 2D slice-level Dice for training monitoring.
        For final paper metrics, run evaluate_3d.py (volumetric 3D DSC/HD95/NSD).
        """
        self.model.eval()
        _spacing = float(getattr(self.cfg.evaluation, "report_spacing_mm", [1.0])[0])
        val_metrics = SegmentationMetrics(spacing_mm=_spacing)
        total_loss = 0.0
        n_batches = 0

        # Detect AutoODESAM: has .decoder but no .prompt_encoder
        is_auto_ode = hasattr(self.model, 'decoder') and not hasattr(self.model, 'prompt_encoder')
        is_organflow = getattr(self.cfg.model, "architecture", None) == "organflow_sam2"

        pbar = tqdm(self.val_loader, desc=f"Val Epoch {epoch}",
                    leave=False, dynamic_ncols=True)

        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                images = batch["image"].to(self.device)
                # V4 dataset returns "masks" (plural, 5-D) — read legacy "mask" only if present.
                masks = batch["mask"].to(self.device) if "mask" in batch else None
                is_3d = batch.get("is_3d", torch.tensor([True if is_organflow else False]))
                if isinstance(is_3d, torch.Tensor):
                    is_3d = is_3d[0].item()

                if is_organflow:
                    # ---- OrganFlowSAM2 (V4) validation path ----
                    images_v4 = images                                             # (B, D, 1, H, W)
                    if images_v4.shape[2] == 1:
                        images_v4 = images_v4.repeat(1, 1, 3, 1, 1)
                    masks_v4 = batch["masks"].to(self.device)                      # (B, 15, D, H, W)
                    present_v4 = batch["present_mask"].to(self.device)             # (B, 15) bool
                    organ_id_v4 = torch.argmax(present_v4.int(), dim=1) + 1

                    with autocast("cuda", enabled=self.use_amp):
                        out = self.model(images_v4, organ_id_v4, is_3d=True)

                    pm = out["masks"]                                              # (B, 15, h, w)
                    idx = (organ_id_v4 - 1).long()
                    H_v, W_v = pm.shape[-2:]
                    best_masks = pm.gather(
                        1, idx.view(-1, 1, 1, 1).expand(-1, 1, H_v, W_v)
                    ).squeeze(1)
                    best_masks = torch.sigmoid(best_masks)

                    D_v4 = masks_v4.shape[2]
                    gt_center = masks_v4[:, :, D_v4 // 2, :, :]                    # (B, 15, H, W)
                    Hg, Wg = gt_center.shape[-2:]
                    masks = gt_center.gather(
                        1, idx.view(-1, 1, 1, 1).expand(-1, 1, Hg, Wg)
                    ).squeeze(1).float()

                elif is_auto_ode:
                    # ---- AutoODESAM validation path ----
                    organ_id_val = batch.get("organ_id", None)
                    if organ_id_val is not None:
                        organ_id_val = organ_id_val.to(self.device)
                        if organ_id_val.dim() > 1:
                            # Use center slice — consistent with train path and modality_ids_center
                            organ_id_val = organ_id_val[:, organ_id_val.shape[1] // 2]

                    with autocast("cuda", enabled=self.use_amp):
                        out = self.model(images, organ_id_val, is_3d=is_3d)

                    masks_pred_all = out["masks"]   # (B, 15, H_out, W_out)
                    iou_pred_all = out["iou_pred"]  # (B, 15)

                    # Per-item channel selection (same logic as train path)
                    val_organ_idxs = (organ_id_val - 1).long()  # (B,) 0-indexed
                    H_v, W_v = masks_pred_all.shape[-2:]
                    best_masks = masks_pred_all.gather(
                        1, val_organ_idxs.view(-1, 1, 1, 1).expand(-1, 1, H_v, W_v)
                    ).squeeze(1)  # (B, H_v, W_v)
                    best_masks = torch.sigmoid(best_masks)

                else:
                    # ---- Legacy / standard validation path ----
                    boxes = batch["box"].to(self.device)
                    modality_ids = batch["modality_id"].to(self.device)
                    modality_ids = _extract_center_modality(modality_ids, is_3d)

                    organ_id = None
                    if "organ_id" in batch:
                        organ_id = batch["organ_id"].to(self.device)  # (B,)

                    with autocast("cuda", enabled=self.use_amp):
                        output = self.model(
                            images=images,
                            boxes=boxes,
                            modality_ids=modality_ids,
                            is_3d=is_3d,
                            multimask_output=True,
                            organ_id=organ_id,
                        )

                    masks_pred = output["masks"]
                    iou_pred = output["iou_pred"]

                    best_idx = iou_pred.argmax(dim=1)
                    best_masks = masks_pred[
                        torch.arange(images.shape[0], device=self.device), best_idx
                    ]
                    best_masks = torch.sigmoid(best_masks)

                # ---- Shared post-processing for both paths ----
                if best_masks.shape[-2:] != masks.shape[-2:]:
                    masks_resized = torch.nn.functional.interpolate(
                        masks.unsqueeze(1).float(),
                        size=best_masks.shape[-2:],
                        mode="nearest",
                    ).squeeze(1)
                else:
                    masks_resized = masks.float()

                # Compute val loss using logit-safe path:
                # best_masks already has sigmoid applied, compute dice/focal directly
                loss, _ = self.criterion.compute_all(
                    torch.logit(best_masks.clamp(1e-6, 1 - 1e-6)), masks_resized
                )
                total_loss += loss.item() if isinstance(loss, torch.Tensor) else loss
                n_batches += 1

                pred_probs = best_masks.cpu().numpy()
                masks_np = masks_resized.cpu().numpy()
                val_metrics.update(pred_probs, masks_np)

                if batch_idx == 0 and epoch % self.cfg.validation.vis_every_n_epochs == 0:
                    self._log_val_images(
                        images=images,
                        masks_gt=masks_resized,
                        masks_pred=best_masks,
                        epoch=epoch,
                    )

        summary = val_metrics.summary()
        metrics = {
            "val_loss": total_loss / max(n_batches, 1),
            "val_dice": summary["dice_mean"],
            "val_iou": summary["iou_mean"],
            "val_hd95": summary["hd95_mean"],
            "val_nsd": summary["nsd_mean"],
        }
        self.logger.log({
            "val/loss": metrics["val_loss"],
            "val/dice": summary["dice_mean"],
            "val/dice_std": summary["dice_std"],
            "val/iou": summary["iou_mean"],
            "val/hd95": summary["hd95_mean"],
            "val/nsd": summary["nsd_mean"],
        }, step=epoch)
        val_metrics.print_summary()
        return metrics

    def _log_val_images(self, images, masks_gt, masks_pred, epoch):
        import torch.nn.functional as F_nn

        # Handle 3D input: extract center slice for visualization
        if images.dim() == 5:  # (B, D, C, H, W)
            D = images.shape[1]
            img = images[0, D // 2].cpu().numpy()  # (3, H, W)
        else:
            img = images[0].cpu().numpy()

        H, W = img.shape[1], img.shape[2]
        gt = masks_gt[0].cpu()
        pred = masks_pred[0].cpu()

        if gt.shape != (H, W):
            gt = F_nn.interpolate(
                gt.unsqueeze(0).unsqueeze(0).float(), size=(H, W), mode="nearest"
            ).squeeze().numpy()
        else:
            gt = gt.numpy()

        if pred.shape != (H, W):
            pred = F_nn.interpolate(
                pred.unsqueeze(0).unsqueeze(0).float(), size=(H, W),
                mode="bilinear", align_corners=False,
            ).squeeze().numpy()
        else:
            pred = pred.numpy()

        img_gray = img[0]
        overlay = make_segmentation_overlay(img_gray, gt, pred)
        self.logger.log_image("val/overlay", overlay, step=epoch,
                               caption=f"Epoch {epoch} | Green=TP, Red=FP, Blue=FN")
