# =============================================================================
# train.py — Training Entry Point
#
# Usage:
#   python train.py                              # Uses base config
#   python train.py --config configs/amos22.yaml # AMOS22 config
#   python train.py --config configs/amos22.yaml model.isa.enabled=false  # Ablation
#
# Config overrides can be passed as key=value arguments (OmegaConf style):
#   python train.py training.lr=5e-5 experiment.name=my_run
# =============================================================================

import argparse
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train LiteSAM-3D for medical image segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on AMOS22 CT with default settings
  python train.py --config configs/amos22.yaml

  # Ablation: disable ISA module (tests MCP-MedSAM baseline)
  python train.py --config configs/amos22.yaml model.isa.enabled=false

  # Train on vessels with topology loss
  python train.py --config configs/vessel.yaml

  # Resume from checkpoint
  python train.py --config configs/amos22.yaml checkpoint.resume_from=checkpoints/latest.pt
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/amos22.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides in key=value format (e.g., training.lr=1e-4)",
    )
    return parser.parse_args()


def load_config(config_path: str, overrides: list):
    """
    Load base config, merge with dataset-specific config,
    then apply command-line overrides.
    """
    base_cfg_path = Path("configs/base.yaml")
    dataset_cfg_path = Path(config_path)

    # Start with base config
    if base_cfg_path.exists():
        cfg = OmegaConf.load(str(base_cfg_path))
    else:
        cfg = OmegaConf.create({})

    # Merge dataset-specific config (overrides base)
    if dataset_cfg_path.exists():
        dataset_cfg = OmegaConf.load(str(dataset_cfg_path))
        # Remove the "defaults" key if present (it's just documentation)
        if "defaults" in dataset_cfg:
            dataset_cfg = OmegaConf.masked_copy(
                dataset_cfg,
                [k for k in dataset_cfg if k != "defaults"]
            )
        cfg = OmegaConf.merge(cfg, dataset_cfg)
    else:
        print(f"Warning: Config file not found: {dataset_cfg_path}")

    # Apply command-line overrides (highest priority)
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    return cfg


def main():
    args = parse_args()

    # ---- Load config ----
    cfg = load_config(args.config, args.overrides)

    # ---- Print config summary ----
    print("\n" + "="*60)
    print("  VoluFormer3D Training")
    print("="*60)
    print(f"  Experiment:  {cfg.experiment.name}")
    print(f"  Architecture:{getattr(cfg.model, 'architecture', 'multiscale_isa')}")
    print(f"  Dataset:     {cfg.data.dataset} ({cfg.data.modality.upper()})")
    print(f"  Encoder:     {cfg.model.encoder_name}")
    print(f"  ISA:         {'ENABLED' if cfg.model.isa.enabled else 'DISABLED (ablation)'}")
    print(f"  Epochs:      {cfg.training.epochs}")
    print(f"  Batch size:  {cfg.training.batch_size} x {cfg.training.grad_accumulation_steps} = "
          f"{cfg.training.batch_size * cfg.training.grad_accumulation_steps}")
    print(f"  Device:      {'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
    print("="*60 + "\n")

    # ---- Create output directories ----
    os.makedirs(cfg.experiment.output_dir, exist_ok=True)
    os.makedirs(cfg.experiment.log_dir, exist_ok=True)

    # ---- Save config snapshot ----
    exp_dir = Path(cfg.experiment.output_dir) / cfg.experiment.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, str(exp_dir / "config.yaml"))
    print(f"Config saved: {exp_dir / 'config.yaml'}")

    # ---- Run training ----
    from training.trainer import Trainer
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
