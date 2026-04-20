"""V9 end-to-end smoke - one training step through the full cascade.

Exercises:
  - V9 config loading
  - VoluFormerV9 construction
  - Synthetic V9 dataset (no real AMOS22 needed)
  - V9Trainer.step_losses
  - backward + optimizer step
  - Stop-rule behaviour
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.voluformer_v9 import VoluFormerV9
from training.trainer_v9 import V9Trainer
from datasets.amos22_v9 import AMOS22V9Dataset


class _FakeDS(AMOS22V9Dataset):
    def __init__(self, cfg) -> None:
        self.img_size = cfg.data.img_size
        self.depth = cfg.data.depth
        self.volume_patch = cfg.data.volume_patch
        self.n_organs = cfg.model.n_organs
        self.pos_frac = cfg.data.pos_frac
        self.neg_frac = cfg.data.neg_frac
        self.mix_frac = cfg.data.mix_frac
        self.slabs_per_volume = 1
        self.hu_clip = tuple(cfg.data.hu_clip)
        self.length = 4
        self._seed = 0
        self.volume_ids = ["fake"]

    def __len__(self) -> int:
        return self.length

    def _load_volume(self, idx):
        rng = np.random.default_rng(idx)
        vol = rng.normal(size=(128, 256, 256)).astype(np.float32) * 50
        lab = np.zeros((128, 256, 256), dtype=np.int64)
        lab[30:60, 80:150, 80:150] = 1
        lab[50:80, 100:170, 100:170] = 6
        return vol, lab


def main() -> None:
    cfg_dict = yaml.safe_load(open(ROOT / "configs/v9_tierB.yaml"))
    cfg_dict["training"]["stage"] = 2
    cfg_dict["loss"]["flow_shape_prior"] = 0.1
    cfg_dict["loss"]["cascade_consistency_kl"] = 0.1
    cfg_dict["loss"]["cascade_consistency_dice"] = 0.1
    # Shrink spatial dims so CPU smoke completes quickly.
    cfg_dict["data"]["img_size"] = 64
    cfg_dict["data"]["depth"] = 4
    cfg_dict["data"]["volume_patch"] = 64
    cfg_dict["model"]["img_size"] = 64
    cfg_dict["model"]["proposer"]["patch_size"] = 64
    cfg_dict["model"]["proposer"]["feature_size"] = 24
    cfg = OmegaConf.create(cfg_dict)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    model = VoluFormerV9(cfg).to(device)
    trainer = V9Trainer(model, cfg)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4,
    )

    ds = _FakeDS(cfg)
    raw = ds[0]
    batch = {}
    for k, v in raw.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.unsqueeze(0).to(device)
        else:
            batch[k] = v

    out = model(
        {
            "volume": batch["volume"],
            "slab": batch["slab"],
            "organ_id": batch["organ_id"],
            "slab_center_z": batch["slab_center_z"],
        },
        stage=2,
    )
    base = out["refiner"]["masks"].mean()
    extras = trainer.step_losses(out, batch)
    total = base + sum(v for v in extras.values())
    opt.zero_grad()
    total.backward()
    opt.step()

    assert torch.isfinite(total), "total loss not finite"
    print(f"[V9 full smoke] stage=2 total_loss={total.item():.4f}")
    print(f"  extras: {list(extras.keys())}")

    aborted = trainer.stop_rule_update(val_dice_3d=0.1)
    assert not aborted
    trainer.stop_rule_update(val_dice_3d=0.05)
    aborted2 = trainer.stop_rule_update(val_dice_3d=0.04)
    assert aborted2, "stop-rule should fire after 2 drops"
    print("[V9 full smoke] stop-rule verified")
    print("[V9 full smoke] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
