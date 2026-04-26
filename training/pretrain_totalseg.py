"""TotalSegmentator pretraining for OrganMoE-3D Phase I.

Wraps train_v9.train_stage1 (alias of train) with TotalSegmentatorDataset
substituted for AMOS22V9Dataset. Saves to checkpoints/voco_totalseg_pretrain/.

Usage:
    python training/pretrain_totalseg.py --cfg configs/organmoe_phase_i_pretrain.yaml
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from omegaconf import OmegaConf
from datasets.totalsegmentator import TotalSegmentatorDataset
from training.train_v9 import train_stage1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training.epochs from CLI.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
    if args.epochs is not None:
        cfg.training.epochs = int(args.epochs)
    if cfg.data.dataset != "totalsegmentator":
        raise ValueError(
            f"This script requires data.dataset=totalsegmentator, got {cfg.data.dataset}"
        )

    train_ds = TotalSegmentatorDataset(
        data_root=cfg.data.data_root,
        split="train",
        volume_patch=int(cfg.data.volume_patch),
        hu_clip=tuple(cfg.data.hu_clip),
        slabs_per_volume=int(cfg.data.slabs_per_volume),
        seed=int(cfg.experiment.seed),
    )
    val_ds = TotalSegmentatorDataset(
        data_root=cfg.data.data_root,
        split="val",
        volume_patch=int(cfg.data.volume_patch),
        hu_clip=tuple(cfg.data.hu_clip),
        slabs_per_volume=int(cfg.data.slabs_per_volume),
        seed=int(cfg.experiment.seed) + 1,
    )
    print(f"[totalseg-pretrain] train={len(train_ds.volume_ids)} vols  "
          f"val={len(val_ds.volume_ids)} vols")

    train_stage1(cfg, train_ds=train_ds, val_ds=val_ds)


if __name__ == "__main__":
    main()
