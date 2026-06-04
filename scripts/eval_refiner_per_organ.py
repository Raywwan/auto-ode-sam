"""Correct per-organ val for the organ-conditioned refiner.

For each val sample, loops over all 15 organ_ids, runs the refiner for each,
and extracts only the channel matching that organ_id. This is the faithful
evaluation protocol because OrganFlowSAM2's ODE head applies an organ-specific
drift field to the features — non-chosen channels are unreliable side-effects.
"""
from __future__ import annotations

import argparse
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

ORGAN_NAMES = [
    "spleen", "R_kid", "L_kid", "gallbld", "esoph",
    "liver", "stomach", "aorta", "ivc", "pancreas",
    "R_adr", "L_adr", "duod", "bladder", "pros_ut",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/v9_stage2/epoch_006.pt")
    ap.add_argument("--max_batches", type=int, default=15,
                    help="number of val volumes to sample (organ loop is 15x each)")
    args = ap.parse_args()

    cfg_dict = yaml.safe_load(open(ROOT / "configs/v9_tierB.yaml"))
    cfg_dict["training"]["stage"] = 2
    cfg = OmegaConf.create(cfg_dict)

    model = VoluFormerV9(cfg).cuda().eval()
    sd = torch.load(str(ROOT / args.ckpt), map_location="cpu", weights_only=False)
    model.load_state_dict(sd["model"], strict=False)
    print(f"[eval] loaded {args.ckpt} (stage={sd.get('stage')}, ep={sd.get('epoch')})")

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

    K = 15
    thresholds = [0.3, 0.4, 0.5]
    d_sums = {t: [0.0] * K for t in thresholds + ["soft"]}
    d_ns   = {t: [0] * K for t in thresholds + ["soft"]}

    with torch.no_grad():
        for i, b in enumerate(loader):
            if i >= args.max_batches:
                break
            b = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in b.items()}
            volume = b["volume"]
            slab   = b["slab"]
            # slab midplane is always depth // 2 of the slab tensor; the
            # `slab_center_z` field is in volume-patch coordinates and is not
            # interchangeable with the slab-local index.
            center = b["mask_slab"].shape[2] // 2
            gt     = b["mask_slab"][:, :, center]              # (1, K, H, W)

            for k in range(K):
                oid = torch.tensor([k], device=volume.device, dtype=torch.long)
                out = model(
                    {"volume": volume, "slab": slab,
                     "organ_id": oid, "slab_center_z": b["slab_center_z"]},
                    stage=2,
                )
                logits = out["refiner"]["masks"][:, k:k+1]     # only this organ's channel
                prob = torch.sigmoid(logits)
                gk = gt[:, k:k+1]
                if gk.shape[-2:] != prob.shape[-2:]:
                    gk = F.interpolate(gk, size=prob.shape[-2:], mode="nearest")
                if gk.sum().item() == 0:
                    continue                                   # skip empty GT channels

                dims = (1, 2, 3)
                for t in thresholds:
                    pbin = (prob > t).float()
                    inter = (pbin * gk).sum(dim=dims)
                    d = (2 * inter) / (pbin.sum(dim=dims) + gk.sum(dim=dims)).clamp_min(1e-6)
                    d_sums[t][k] += float(d.item())
                    d_ns[t][k]   += 1
                inter = (prob * gk).sum(dim=dims)
                d = (2 * inter) / (prob.sum(dim=dims) + gk.sum(dim=dims)).clamp_min(1e-6)
                d_sums["soft"][k] += float(d.item())
                d_ns["soft"][k]   += 1
            if (i + 1) % 5 == 0:
                print(f"  [{i+1}/{args.max_batches}] samples done")

    print(f"\n{'organ':<8} {'th=0.3':>7} {'th=0.4':>7} {'th=0.5':>7} {'soft':>7} {'n':>4}")
    print("-" * 45)
    means = {t: [] for t in thresholds + ["soft"]}
    for k in range(K):
        def _m(t):
            return d_sums[t][k] / d_ns[t][k] if d_ns[t][k] > 0 else float("nan")
        print(
            f"{ORGAN_NAMES[k]:<8} "
            f"{_m(0.3):>7.3f} {_m(0.4):>7.3f} {_m(0.5):>7.3f} {_m('soft'):>7.3f} {d_ns[0.3][k]:>4}"
        )
        for t in thresholds + ["soft"]:
            v = _m(t)
            if v == v:
                means[t].append(v)
    print("-" * 45)
    print(
        f"{'mean':<8} "
        f"{sum(means[0.3])/max(1,len(means[0.3])):>7.3f} "
        f"{sum(means[0.4])/max(1,len(means[0.4])):>7.3f} "
        f"{sum(means[0.5])/max(1,len(means[0.5])):>7.3f} "
        f"{sum(means['soft'])/max(1,len(means['soft'])):>7.3f}"
    )


if __name__ == "__main__":
    main()
