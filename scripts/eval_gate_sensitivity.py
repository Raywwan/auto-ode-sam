"""Post-hoc sensitivity of cascade Dice to the fusion gate bias.

Reloads a Stage 2 checkpoint, sweeps the output-bias of the gate head across
{-2, -1, 0, +1, +2, +4} (= sigmoid weight on proposer in {0.12, 0.27, 0.5,
0.73, 0.88, 0.98}), and reports fused Dice for each.

If the refiner is weak, pushing bias positive should recover proposer-class Dice.
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/v9_stage2/epoch_006.pt")
    ap.add_argument("--max_batches", type=int, default=20)
    args = ap.parse_args()

    cfg_dict = yaml.safe_load(open(ROOT / "configs/v9_tierB.yaml"))
    cfg_dict["training"]["stage"] = 2
    cfg = OmegaConf.create(cfg_dict)

    model = VoluFormerV9(cfg).cuda().eval()
    sd = torch.load(str(ROOT / args.ckpt), map_location="cpu", weights_only=False)
    model.load_state_dict(sd["model"], strict=False)
    print(f"[eval] {args.ckpt} (ep={sd.get('epoch')})")

    val_ds = AMOS22V9Dataset(
        data_root=cfg.data.data_root, split="val",
        img_size=cfg.data.img_size, depth=cfg.data.depth,
        volume_patch=cfg.data.volume_patch, hu_clip=tuple(cfg.data.hu_clip),
        pos_frac=1.0, neg_frac=0.0, mix_frac=0.0,
        slabs_per_volume=1, seed=cfg.experiment.seed + 1,
    )
    loader = DataLoader(val_ds, batch_size=1, num_workers=0)

    biases = [-2.0, -1.0, 0.0, 1.0, 2.0, 4.0]
    results = {b: {"prop": [], "ref": [], "fus": []} for b in biases}

    with torch.no_grad():
        for i, b in enumerate(loader):
            if i >= args.max_batches:
                break
            b = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in b.items()}
            for bias_val in biases:
                model.fusion.gate[-1].bias.fill_(bias_val)
                out = model(
                    {"volume": b["volume"], "slab": b["slab"],
                     "organ_id": b["organ_id"], "slab_center_z": b["slab_center_z"]},
                    stage=2,
                )
                prop_probs_3d = out["proposer"]["probs"]
                prop_slice = prop_probs_3d[:, :, prop_probs_3d.shape[2] // 2]
                ref_mask = torch.sigmoid(out["refiner"]["masks"])
                if prop_slice.shape[-2:] != ref_mask.shape[-2:]:
                    prop_slice = F.interpolate(prop_slice, size=ref_mask.shape[-2:],
                                               mode="bilinear", align_corners=False)
                fused_dict = model.fusion(prop_slice, ref_mask, b["organ_id"])
                fus_mask = fused_dict["fused"]
                # slab midplane = slab-local depth//2; slab_center_z is the
                # volume-patch coordinate used by the model's proposer gather.
                center = b["mask_slab"].shape[2] // 2
                gt = b["mask_slab"][:, :, center]
                if gt.shape[-2:] != ref_mask.shape[-2:]:
                    gt = F.interpolate(gt, size=ref_mask.shape[-2:], mode="nearest")

                for name, pred in (("prop", prop_slice), ("ref", ref_mask), ("fus", fus_mask)):
                    pbin = (pred > 0.5).float()
                    dims = (2, 3)
                    inter = (pbin * gt).sum(dim=dims)
                    denom = pbin.sum(dim=dims) + gt.sum(dim=dims)
                    d = (2 * inter) / denom.clamp_min(1e-6)
                    empty = gt.sum(dim=dims) == 0
                    d[empty] = float("nan")
                    vals = d[~empty]
                    if vals.numel() > 0:
                        results[bias_val][name].append(float(vals.mean().item()))
            if (i + 1) % 5 == 0:
                print(f"  [{i+1}/{args.max_batches}] done")

    print(f"\n{'gate_bias':>10} {'prop_w':>8} {'prop':>8} {'ref':>8} {'fus':>8}")
    print("-" * 50)
    for bias_val in biases:
        w = torch.sigmoid(torch.tensor(bias_val)).item()
        r = results[bias_val]
        m = lambda xs: sum(xs) / max(1, len(xs))
        print(f"{bias_val:>10.1f} {w:>8.2f} {m(r['prop']):>8.3f} {m(r['ref']):>8.3f} {m(r['fus']):>8.3f}")


if __name__ == "__main__":
    main()
