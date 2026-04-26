"""OrganMoE-3D AMOS22 fine-tune driver (Phase I — Task 12).

Wraps `train_v9.train_stage1` with:
  - AMOS22V9Dataset (15-organ CT)
  - BalancedBatchSampler (rare-organ anchor per batch)
  - OrganMoELoss (SmallOrganLoss + presence BCE + load-balance)
  - Warm-start from the TotalSeg pretrain checkpoint produced by
    `pretrain_totalseg.py`.

Usage:
    python training/train_organmoe_amos22.py \
        --cfg configs/organmoe_phase_i_ft.yaml \
        --proposer-ckpt checkpoints/voco_totalseg_pretrain/best.pt
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from omegaconf import OmegaConf

from datasets.amos22_v9 import AMOS22V9Dataset
from datasets.balanced_sampler import BalancedBatchSampler
from training.losses_organmoe import OrganMoELoss
from training.train_v9 import train_stage1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, type=Path)
    parser.add_argument("--proposer-ckpt", required=True, type=Path,
                        help="TotalSeg-pretrained proposer checkpoint to warm-start from.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training.epochs from CLI.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
    if args.epochs is not None:
        cfg.training.epochs = int(args.epochs)
    if cfg.data.dataset != "amos22_v9":
        raise ValueError(
            f"This script requires data.dataset=amos22_v9, got {cfg.data.dataset}"
        )

    train_ds = AMOS22V9Dataset(
        data_root=cfg.data.data_root,
        split="train",
        img_size=int(cfg.data.img_size),
        depth=int(cfg.data.depth),
        volume_patch=int(cfg.data.volume_patch),
        modality="ct",
        hu_clip=tuple(cfg.data.hu_clip),
        pos_frac=float(cfg.data.pos_frac),
        neg_frac=float(cfg.data.neg_frac),
        mix_frac=float(cfg.data.mix_frac),
        slabs_per_volume=int(cfg.data.slabs_per_volume),
        seed=int(cfg.experiment.seed),
        copy_paste_prob=float(getattr(cfg.data, "copy_paste_prob", 0.0)),
    )
    val_ds = AMOS22V9Dataset(
        data_root=cfg.data.data_root,
        split="val",
        img_size=int(cfg.data.img_size),
        depth=int(cfg.data.depth),
        volume_patch=int(cfg.data.volume_patch),
        modality="ct",
        hu_clip=tuple(cfg.data.hu_clip),
        pos_frac=float(cfg.data.pos_frac),
        neg_frac=float(cfg.data.neg_frac),
        mix_frac=float(cfg.data.mix_frac),
        slabs_per_volume=int(cfg.data.slabs_per_volume),
        seed=int(cfg.experiment.seed) + 1,
    )
    print(f"[organmoe-amos22] train={len(train_ds.volume_ids)} vols  "
          f"val={len(val_ds.volume_ids)} vols")

    train_sampler = None
    if bool(getattr(cfg.data, "balanced_batch_sampler", False)):
        rare = list(cfg.data.rare_organs)
        train_sampler = BalancedBatchSampler(
            dataset=train_ds,
            batch_size=int(cfg.training.batch_size),
            rare_organs=rare,
            shuffle=True,
            seed=int(cfg.experiment.seed),
        )
        print(f"[organmoe-amos22] BalancedBatchSampler ON  rare_organs={rare}  "
              f"pools={ {o: len(p) for o, p in train_sampler._pool.items()} }")

    loss_fn = None
    if bool(getattr(cfg.training, "loss_organmoe", False)):
        kw = OmegaConf.to_container(
            getattr(cfg.training, "loss_organmoe_cfg", {}), resolve=True
        ) or {}
        # n_classes = n_organs + 1 (background).
        n_classes = int(cfg.model.n_organs) + 1
        loss_fn = OrganMoELoss(n_classes=n_classes, **kw)
        print(f"[organmoe-amos22] OrganMoELoss ON  n_classes={n_classes}  cfg={kw}")

    train_stage1(
        cfg,
        train_ds=train_ds,
        val_ds=val_ds,
        train_sampler=train_sampler,
        loss_fn=loss_fn,
        warmstart_proposer_ckpt=str(args.proposer_ckpt),
    )


if __name__ == "__main__":
    main()
