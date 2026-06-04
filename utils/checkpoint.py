# =============================================================================
# utils/checkpoint.py — Checkpoint Management
#
# Handles saving and loading model checkpoints.
#
# Features:
#   - Save best N checkpoints (by validation Dice)
#   - Periodic checkpoint every N epochs
#   - Resume training from checkpoint
#   - Load encoder-only weights (for fine-tuning from ImageNet pretrained)
# =============================================================================

import os
import heapq
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class CheckpointManager:
    """
    Manages model checkpoints during training.

    Automatically:
      - Saves the best K checkpoints (highest validation Dice)
      - Saves periodic checkpoints every N epochs
      - Deletes old checkpoints to save disk space

    Args:
        save_dir: Directory to save checkpoints
        keep_top_k: Number of best checkpoints to keep
        monitor_metric: Metric name to monitor (higher = better)
        experiment_name: Prefix for checkpoint filenames
    """

    def __init__(
        self,
        save_dir: str,
        keep_top_k: int = 3,
        monitor_metric: str = "val_dice",
        experiment_name: str = "litesam3d",
        min_delta: float = 0.0,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_top_k = keep_top_k
        self.monitor_metric = monitor_metric
        self.experiment_name = experiment_name
        # Hysteresis: best.pt only updates if new metric > best + min_delta.
        # Set non-zero (e.g. 0.005) when the monitor metric is noisy (e.g.
        # n_vols=4 mini-eval) to prevent lucky-volume overwrites.
        self.min_delta = float(min_delta)

        # Min-heap of (metric_value, epoch, filepath) — we keep the TOP k
        # Using negative values because heapq is a min-heap
        self._best_checkpoints: list = []
        self.best_metric: float = -float("inf")
        self.best_epoch: int = 0

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        epoch: int,
        metrics: Dict[str, float],
        scaler=None,  # GradScaler for mixed precision
    ) -> str:
        """
        Save a checkpoint.

        Always saves as "latest.pt" for easy resuming.
        If this is the best epoch, also saves "best.pt".
        Manages the top-K checkpoint list.

        Args:
            model: The model to save
            optimizer: Optimizer state
            scheduler: LR scheduler state
            epoch: Current epoch number
            metrics: Dict of metric_name → value
            scaler: GradScaler (if using fp16)

        Returns:
            Path to saved checkpoint file.
        """
        metric_val = metrics.get(self.monitor_metric, 0.0)

        # Build checkpoint dict
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": metrics,
            "monitor_metric": self.monitor_metric,
        }
        if scaler is not None:
            ckpt["scaler_state_dict"] = scaler.state_dict()

        # Always save latest
        latest_path = self.save_dir / f"{self.experiment_name}_latest.pt"
        torch.save(ckpt, str(latest_path))

        # Epoch-specific checkpoint
        epoch_path = self.save_dir / f"{self.experiment_name}_epoch{epoch:03d}.pt"
        torch.save(ckpt, str(epoch_path))

        # Track best checkpoints (top-K)
        heapq.heappush(self._best_checkpoints, (metric_val, epoch, str(epoch_path)))

        # If we have more than K checkpoints, remove the worst
        if len(self._best_checkpoints) > self.keep_top_k:
            worst_val, worst_epoch, worst_path = heapq.heappop(self._best_checkpoints)
            # Don't delete if it's the current one we just saved
            if worst_path != str(epoch_path) and os.path.exists(worst_path):
                os.remove(worst_path)

        # Save best checkpoint (with hysteresis to handle noisy monitor metrics)
        if metric_val > self.best_metric + self.min_delta:
            self.best_metric = metric_val
            self.best_epoch = epoch
            best_path = self.save_dir / f"{self.experiment_name}_best.pt"
            torch.save(ckpt, str(best_path))

        return str(epoch_path)

    def load(
        self,
        path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler=None,
        scaler=None,
        device: torch.device = torch.device("cuda"),
    ) -> Tuple[int, Dict[str, float]]:
        """
        Load a checkpoint and restore model (and optionally optimizer) state.

        Args:
            path: Path to checkpoint file
            model: Model to load weights into
            optimizer: If provided, load optimizer state (for resuming training)
            scheduler: If provided, load scheduler state
            scaler: If provided, load scaler state
            device: Device to load tensors to

        Returns:
            epoch: The epoch this checkpoint was saved at
            metrics: The metrics at this checkpoint
        """
        ckpt = torch.load(path, map_location=device, weights_only=False)

        # Load model weights
        model.load_state_dict(ckpt["model_state_dict"])

        # Optionally restore training state
        if optimizer is not None and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if scheduler is not None and "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"]:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        if scaler is not None and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])

        epoch = ckpt.get("epoch", 0)
        metrics = ckpt.get("metrics", {})

        return epoch, metrics

    def get_best_checkpoint(self) -> Optional[str]:
        """Return path to the best checkpoint (None if none saved yet)."""
        best_path = self.save_dir / f"{self.experiment_name}_best.pt"
        return str(best_path) if best_path.exists() else None

    def get_latest_checkpoint(self) -> Optional[str]:
        """Return path to the latest checkpoint (None if none saved yet)."""
        latest_path = self.save_dir / f"{self.experiment_name}_latest.pt"
        return str(latest_path) if latest_path.exists() else None


def load_pretrained_encoder(
    model: nn.Module,
    checkpoint_path: str,
    strict: bool = False,
) -> nn.Module:
    """
    Load only the encoder weights from a pre-trained checkpoint.
    Useful when starting from a previously trained model but with a different decoder.

    Args:
        model: Target model (full LiteSAM3D)
        checkpoint_path: Path to checkpoint (can be full LiteSAM3D or encoder-only)
        strict: Whether to require all keys to match

    Returns:
        model with encoder weights loaded
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    # Filter to encoder keys only
    encoder_state = {
        k.replace("encoder.", "", 1): v
        for k, v in state_dict.items()
        if k.startswith("encoder.")
    }

    model.encoder.load_state_dict(encoder_state, strict=strict)
    return model
