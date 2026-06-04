"""Diagnose Stage 2 val plateau by inspecting the refiner's raw sigmoid
distribution and Dice at multiple thresholds.

Loads the most recent Stage 2 epoch checkpoint and runs 10 val batches.
Reports per-organ sigmoid min/mean/max on foreground pixels, and Dice at
thresholds {0.3, 0.4, 0.5} plus soft-Dice (no threshold).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.amos22_v9 import AMOS22V9Dataset
from models.voluformer_v9 import VoluFormerV9


def main() -> None:
    cfg_dict = yaml.safe_load(open(ROOT / "configs/v9_tierB.yaml"))
    cfg_dict["training"]["stage"] = 2
    cfg = OmegaConf.create(cfg_dict)

    ck_dir = ROOT / "checkpoints/v9_stage2"
    ckpts = sorted(ck_dir.glob("epoch_*.pt"))
    if not ckpts:
        print("[inspect] no stage2 epoch_*.pt yet")
        return
    ckpt = ckpts[-1]
    print(f"[inspect] loading {ckpt.name}")

    model = VoluFormerV9(cfg).cuda().eval()
    sd = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    model.load_state_dict(sd["model"], strict=False)

    val_ds = AMOS22V9Dataset(
        data_root=cfg.data.data_root,
        split="val",
        img_size=cfg.data.img_size,
        depth=cfg.data.depth,
        volume_patch=cfg.data.volume_patch,
        hu_clip=tuple(cfg.data.hu_clip),
        pos_frac=1.0, neg_frac=0.0, mix_frac=0.0,
        slabs_per_volume=1,
        seed=cfg.experiment.seed + 1,
    )
    loader = DataLoader(val_ds, batch_size=1, num_workers=0)

    dice_sums = {t: [0.0] * 15 for t in ["0.3", "0.4", "0.5", "soft"]}
    dice_ns   = {t: [0] * 15 for t in ["0.3", "0.4", "0.5", "soft"]}
    fg_sigmoid_vals = [[] for _ in range(15)]

    with torch.no_grad():
        for i, b in enumerate(loader):
            if i >= 10:
                break
            b = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in b.items()}
            out = model(
                {"volume": b["volume"], "slab": b["slab"],
                 "organ_id": b["organ_id"], "slab_center_z": b["slab_center_z"]},
                stage=2,
            )
            prob = torch.sigmoid(out["refiner"]["masks"])    # (1, 15, H, W)
            # slab midplane = slab-local depth//2; slab_center_z is a
            # volume-patch coord and would blow up when used as a slab index.
            center = b["mask_slab"].shape[2] // 2
            gt = b["mask_slab"][:, :, center]
            if gt.shape[-2:] != prob.shape[-2:]:
                gt = F.interpolate(gt, size=prob.shape[-2:], mode="nearest")

            for k in range(15):
                gk = gt[:, k]                              # (1, H, W)
                pk = prob[:, k]
                present = gk.sum().item() > 0
                if not present:
                    continue

                # Collect sigmoid distribution on foreground pixels.
                fg_vals = pk[gk > 0.5]
                if fg_vals.numel() > 0:
                    fg_sigmoid_vals[k].append((
                        float(fg_vals.min()), float(fg_vals.mean()), float(fg_vals.max()),
                    ))

                # Threshold Dice.
                for t in [0.3, 0.4, 0.5]:
                    pbin = (pk > t).float()
                    inter = (pbin * gk).sum()
                    d = (2 * inter) / (pbin.sum() + gk.sum()).clamp_min(1e-6)
                    dice_sums[f"{t}"][k] += float(d.item())
                    dice_ns[f"{t}"][k]   += 1
                # Soft dice.
                inter = (pk * gk).sum()
                d = (2 * inter) / (pk.sum() + gk.sum()).clamp_min(1e-6)
                dice_sums["soft"][k] += float(d.item())
                dice_ns["soft"][k]   += 1

    ORGAN_NAMES = [
        "spleen", "R_kid", "L_kid", "gallbld", "esoph",
        "liver", "stomach", "aorta", "ivc", "pancreas",
        "R_adr", "L_adr", "duod", "bladder", "pros_ut",
    ]

    print(f"\n{'organ':<8} {'th=0.3':>7} {'th=0.4':>7} {'th=0.5':>7} {'soft':>7}  sigmoid(min/mean/max on FG)")
    for k in range(15):
        def _mean(t):
            return dice_sums[t][k] / dice_ns[t][k] if dice_ns[t][k] > 0 else float("nan")
        fgv = fg_sigmoid_vals[k]
        if fgv:
            mn = sum(v[0] for v in fgv) / len(fgv)
            mu = sum(v[1] for v in fgv) / len(fgv)
            mx = sum(v[2] for v in fgv) / len(fgv)
            sig_str = f"{mn:.2f}/{mu:.2f}/{mx:.2f}"
        else:
            sig_str = "-"
        print(
            f"{ORGAN_NAMES[k]:<8} "
            f"{_mean('0.3'):>7.3f} {_mean('0.4'):>7.3f} {_mean('0.5'):>7.3f} "
            f"{_mean('soft'):>7.3f}  {sig_str}"
        )


if __name__ == "__main__":
    main()
