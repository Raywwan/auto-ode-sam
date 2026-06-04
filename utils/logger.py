# =============================================================================
# utils/logger.py — Experiment Logger
#
# Wraps both Weights & Biases (W&B) and TensorBoard in a unified interface.
# You only need to change the config to switch between them.
#
# Features:
#   - Scalar logging (loss, metrics)
#   - Image logging (segmentation overlays)
#   - Hyperparameter logging
#   - Console logging (always on)
# =============================================================================

import logging
import os
from typing import Any, Dict, Optional

import numpy as np
import torch


def setup_console_logger(name: str = "litesam3d") -> logging.Logger:
    """
    Set up a basic console logger with timestamps.

    Returns a standard Python logger. All training progress is printed here
    regardless of whether W&B or TensorBoard is enabled.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


class ExperimentLogger:
    """
    Unified logger for training experiments.

    Supports:
      - W&B (Weights & Biases): rich visualisations, comparison dashboard
      - TensorBoard: local logging, no account needed
      - Console: always active

    Usage:
        logger = ExperimentLogger(cfg)
        logger.log({"train/loss": 0.5, "train/dice": 0.75}, step=100)
        logger.log_image("val/pred", image_array, step=100)
        logger.finish()
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.use_wandb = cfg.logging.use_wandb
        self.use_tensorboard = cfg.logging.use_tensorboard
        self.console = setup_console_logger()

        self._wandb = None
        self._tb_writer = None

        # Initialise W&B
        if self.use_wandb:
            self._init_wandb()

        # Initialise TensorBoard
        if self.use_tensorboard:
            self._init_tensorboard()

    def _init_wandb(self):
        """Initialise W&B run."""
        try:
            import wandb
            self._wandb = wandb
            wandb.init(
                project=self.cfg.logging.wandb_project,
                entity=getattr(self.cfg.logging, "wandb_entity", None),
                name=self.cfg.experiment.name,
                config=self._cfg_to_dict(self.cfg),
                dir=self.cfg.experiment.log_dir,
            )
            self.console.info(f"W&B run: {wandb.run.url}")
        except ImportError:
            self.console.warning("wandb not installed. Run: pip install wandb")
            self.use_wandb = False
        except Exception as e:
            self.console.warning(f"W&B init failed: {e}")
            self.use_wandb = False

    def _init_tensorboard(self, purge_step: int = None):
        """Initialise TensorBoard SummaryWriter."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = os.path.join(
                self.cfg.experiment.log_dir, self.cfg.experiment.name
            )
            os.makedirs(log_dir, exist_ok=True)
            self._tb_writer = SummaryWriter(log_dir=log_dir, purge_step=purge_step)
            self.console.info(f"TensorBoard logs: {log_dir}")
        except ImportError:
            self.console.warning("tensorboard not installed. Run: pip install tensorboard")
            self.use_tensorboard = False

    def reinit_tensorboard(self, purge_step: int):
        """Reinitialise TensorBoard writer after resuming, to continue curves seamlessly."""
        if not self.use_tensorboard:
            return
        if self._tb_writer is not None:
            self._tb_writer.close()
        self._init_tensorboard(purge_step=purge_step)

    def setup_layout(self):
        """
        Configure TensorBoard custom scalar layout.
        Groups related metrics into named panels for a cleaner dashboard.
        Call once after initialisation (and after reinit_tensorboard).
        """
        if not self.use_tensorboard or self._tb_writer is None:
            return

        layout = {
            "Overview": {
                "Loss — train vs val": ["Multiline", ["train/loss_epoch", "val/loss"]],
                "Dice — train vs val": ["Multiline", ["train/dice_epoch", "val/dice"]],
            },
            "Loss Components": {
                "Per-step losses": ["Multiline", ["train/dice", "train/focal", "train/iou", "train/total"]],
                "Per-epoch losses": ["Multiline", ["train/dice_epoch_comp", "train/focal_epoch_comp"]],
            },
            "Validation Metrics": {
                "Overlap (DSC / IoU)": ["Multiline", ["val/dice", "val/iou"]],
                "Surface (HD95 / NSD)": ["Multiline", ["val/hd95", "val/nsd"]],
            },
            "Training Dynamics": {
                "Learning rate": ["Multiline", ["train/lr"]],
                "Gradient norm": ["Multiline", ["train/grad_norm"]],
                "GPU memory (GB)": ["Multiline", ["train/gpu_mem_gb"]],
            },
        }

        self._tb_writer.add_custom_scalars(layout)

    def log(self, metrics: Dict[str, Any], step: int = None):
        """
        Log scalar metrics.

        Args:
            metrics: Dict of metric name → value
            step: Global step count (epoch or iteration)
        """
        if self.use_wandb and self._wandb:
            self._wandb.log(metrics, step=step)

        if self.use_tensorboard and self._tb_writer:
            for k, v in metrics.items():
                if isinstance(v, (int, float, torch.Tensor)):
                    val = v.item() if isinstance(v, torch.Tensor) else v
                    self._tb_writer.add_scalar(k, val, global_step=step)

    def log_image(
        self,
        name: str,
        image: np.ndarray,     # (H, W, 3) or (H, W) in [0, 255] uint8
        step: int = None,
        caption: str = "",
    ):
        """
        Log a segmentation visualisation image.

        Args:
            name: Image key (e.g., "val/prediction")
            image: RGB or grayscale image array
            step: Global step
            caption: Caption for the image
        """
        if not self.cfg.logging.log_images:
            return

        if self.use_wandb and self._wandb:
            self._wandb.log({
                name: self._wandb.Image(image, caption=caption)
            }, step=step)

        if self.use_tensorboard and self._tb_writer:
            if image.ndim == 3:
                img_t = torch.from_numpy(image).permute(2, 0, 1)  # (3, H, W)
            else:
                img_t = torch.from_numpy(image).unsqueeze(0)      # (1, H, W)
            self._tb_writer.add_image(name, img_t, global_step=step)

    def log_hyperparams(self, params: Dict[str, Any]):
        """Log hyperparameters (called once at start of training)."""
        if self.use_wandb and self._wandb:
            self._wandb.config.update(params)

    def info(self, message: str):
        """Log a message to console."""
        self.console.info(message)

    def finish(self):
        """Clean up logger resources."""
        if self.use_wandb and self._wandb:
            self._wandb.finish()
        if self.use_tensorboard and self._tb_writer:
            self._tb_writer.close()

    def _cfg_to_dict(self, cfg) -> Dict:
        """Convert OmegaConf config to plain dict for W&B."""
        from omegaconf import OmegaConf
        return OmegaConf.to_container(cfg, resolve=True)


def make_segmentation_overlay(
    image: np.ndarray,       # (H, W) float32 in [0, 1]
    mask_gt: np.ndarray,     # (H, W) binary
    mask_pred: np.ndarray,   # (H, W) probability in [0, 1]
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Create an RGB overlay for visualising segmentation results.

    Colour coding:
      - Green: True Positive (both GT and pred = 1)
      - Red:   False Positive (pred=1, GT=0)
      - Blue:  False Negative (pred=0, GT=1)
      - Gray:  True Negative  (both = 0)

    Args:
        image: Input image (H, W) in [0, 1]
        mask_gt: Ground truth binary mask (H, W)
        mask_pred: Predicted probability map (H, W) in [0, 1]
        threshold: Binarisation threshold for prediction

    Returns:
        overlay: (H, W, 3) uint8 RGB image
    """
    pred_bin = (mask_pred > threshold).astype(bool)
    gt_bin = mask_gt.astype(bool)

    # Create RGB image from grayscale input
    img_rgb = np.stack([image, image, image], axis=-1)  # (H, W, 3)
    overlay = (img_rgb * 255).astype(np.uint8)

    alpha = 0.4  # Transparency of the overlay

    # True Positive: Green
    tp = gt_bin & pred_bin
    overlay[tp] = (
        overlay[tp] * (1 - alpha) + np.array([0, 200, 0]) * alpha
    ).astype(np.uint8)

    # False Positive: Red
    fp = (~gt_bin) & pred_bin
    overlay[fp] = (
        overlay[fp] * (1 - alpha) + np.array([200, 0, 0]) * alpha
    ).astype(np.uint8)

    # False Negative: Blue
    fn = gt_bin & (~pred_bin)
    overlay[fn] = (
        overlay[fn] * (1 - alpha) + np.array([0, 0, 200]) * alpha
    ).astype(np.uint8)

    return overlay
