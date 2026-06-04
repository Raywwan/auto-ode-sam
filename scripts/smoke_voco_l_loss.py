"""3-step smoke test: VoCo-L + bf16 autocast on real AMOS22 batch.

Purpose: confirm loss is finite (not NaN) before committing to the full
60-100 h Week-1 run. This catches the fp16-overflow issue we hit on the
first launch.

Pass criterion: all 3 steps produce finite total loss AND decoder outputs
have non-NaN softmax.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from datasets.amos22_v9 import AMOS22V9Dataset
from models.voluformer_v9 import VoluFormerV9
from training.train_v9 import stage1_loss


def main():
    cfg = OmegaConf.load("configs/v10_voco_l.yaml")
    device = torch.device("cuda")

    # Build a tiny dataset (skip copy-paste, single worker).
    ds = AMOS22V9Dataset(
        data_root=cfg.data.data_root,
        split="train",
        img_size=cfg.data.img_size,
        depth=cfg.data.depth,
        volume_patch=cfg.data.volume_patch,
        hu_clip=tuple(cfg.data.hu_clip),
        pos_frac=cfg.data.pos_frac,
        neg_frac=cfg.data.neg_frac,
        mix_frac=cfg.data.mix_frac,
        slabs_per_volume=cfg.data.slabs_per_volume,
        copy_paste_bank=None,
        copy_paste_prob=0.0,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

    model = VoluFormerV9(cfg).to(device)
    model.train()

    # Match train_v9 AMP setup.
    amp_dtype_str = str(getattr(cfg.training, "amp_dtype", "fp16")).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_str in ("bf16", "bfloat16") else torch.float16
    use_scaler = (amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print(f"amp_dtype={amp_dtype}  use_scaler={use_scaler}")
    print(f"n_train_samples={len(ds)}")

    it = iter(loader)
    for step in range(3):
        batch = next(it)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            out = model(
                {
                    "volume": batch["volume"],
                    "slab": batch["slab"],
                    "organ_id": batch["organ_id"],
                    "slab_center_z": batch["slab_center_z"],
                },
                stage=1,
            )
            losses = stage1_loss(out, batch, small_organ=None, deep_sup_weight=0.0)
            total = losses["total"]

        finite = torch.isfinite(total).item()
        vol_stats = batch["volume"].float()
        logits = out["proposer"]["full_logits"]
        probs_finite = torch.isfinite(F.softmax(logits.float(), dim=1)).all().item()

        print(
            f"step {step}: total={total.item():.4f}  finite={finite}  "
            f"ce={losses.get('ce', torch.tensor(float('nan'))).item():.4f}  "
            f"dice={losses.get('dice', torch.tensor(float('nan'))).item():.4f}  "
            f"softmax_finite={probs_finite}  "
            f"vol_range=[{vol_stats.min().item():.2f},{vol_stats.max().item():.2f}]"
        )

        if not finite:
            print(f"FAIL: step {step} produced NaN/Inf loss")
            sys.exit(1)

        if use_scaler:
            scaler.scale(total).backward()
            scaler.step(opt)
            scaler.update()
        else:
            total.backward()
            opt.step()

    print("RESULT: PASS — 3 steps all finite, safe to relaunch W1 training")


if __name__ == "__main__":
    main()
