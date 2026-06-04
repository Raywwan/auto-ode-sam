"""Canonical V4 val evaluator — modern multi-organ metric.

Scoring rules (MONAI/nnU-Net aligned):
  - Upsample logits 64 -> 256 bilinearly, then sigmoid.
  - For each volume, compute per-organ Dice ONLY over present organs.
  - Primary: soft Dice (probs * GT / probs + GT), threshold-free.
  - Secondary: binary Dice @0.5 for paper parity.
  - Mean over present organs gives per-volume score; mean over volumes gives
    the reported val metric.

Usage:
  python scripts/eval_v4_ckpt.py --ckpt <path.pt> [--n 0 for full set]
"""
from __future__ import annotations
import argparse, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

from models.organflow_sam2 import OrganFlowSAM2
from datasets.amos22_multiorgan import AMOS22MultiOrgan3D_Dataset


def per_organ_dice(
    prob: torch.Tensor,        # (15, H, W) in [0,1]
    gt:   torch.Tensor,        # (15, H, W) in {0,1}
    present: torch.Tensor,     # (15,) bool
    threshold: float | None,   # None = soft dice; float = binary
    eps: float = 1e-6,
) -> list[float]:
    if threshold is not None:
        pred = (prob > threshold).float()
    else:
        pred = prob
    dices = []
    for k in range(15):
        if not present[k]:
            continue
        p = pred[k].flatten()
        g = gt[k].flatten().float()
        inter = (p * g).sum()
        denom = p.sum() + g.sum()
        if denom.item() < eps:
            continue
        d = (2.0 * inter + eps) / (denom + eps)
        dices.append(d.item())
    return dices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=r"C:\Users\Raywa\Desktop\VoluFormer3D_V4\configs\v4_organflow_sam2_256px.yaml")
    ap.add_argument("--n", type=int, default=0, help="0 = full val set")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {args.ckpt}")
    model = OrganFlowSAM2(cfg).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    sd = state.get("model_state_dict", state.get("model", state))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  loaded, missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval()

    ds = AMOS22MultiOrgan3D_Dataset(
        data_root=cfg.data.data_root, split="val",
        img_size=cfg.data.img_size, depth=cfg.data.slices_per_volume,
        modality=cfg.data.modality,
    )
    n = len(ds) if args.n == 0 else min(args.n, len(ds))
    print(f"Evaluating on {n}/{len(ds)} val volumes")

    per_vol_soft, per_vol_bin05, per_vol_bin03 = [], [], []
    per_organ_soft = {k: [] for k in range(15)}

    with torch.no_grad():
        for i in tqdm(range(n), desc="eval"):
            s = ds[i]
            img = s["image"].unsqueeze(0).to(device)
            if img.shape[2] == 1:
                img = img.repeat(1, 1, 3, 1, 1)
            masks_gt = s["masks"].unsqueeze(0).to(device)          # (1, 15, D, H, W)
            present  = s["present_mask"].to(device)                # (15,) bool
            organ_id = torch.argmax(present.int(), dim=0).unsqueeze(0) + 1

            out = model(img, organ_id, is_3d=True)
            pm_logits = out["masks"]                                # (1, 15, 64, 64)

            # Upsample LOGITS to GT size, then sigmoid (modern practice)
            D = masks_gt.shape[2]
            gt_center = masks_gt[0, :, D // 2].float()              # (15, 256, 256)
            Hg, Wg = gt_center.shape[-2:]
            pm_logits_up = F.interpolate(pm_logits, size=(Hg, Wg), mode="bilinear", align_corners=False)
            prob = torch.sigmoid(pm_logits_up[0])                    # (15, 256, 256)

            d_soft = per_organ_dice(prob, gt_center, present, threshold=None)
            d_bin5 = per_organ_dice(prob, gt_center, present, threshold=0.5)
            d_bin3 = per_organ_dice(prob, gt_center, present, threshold=0.3)

            if d_soft:
                per_vol_soft.append(sum(d_soft) / len(d_soft))
            if d_bin5:
                per_vol_bin05.append(sum(d_bin5) / len(d_bin5))
            if d_bin3:
                per_vol_bin03.append(sum(d_bin3) / len(d_bin3))

            # Per-organ soft dice (for breakdown)
            p_idx = present.nonzero(as_tuple=False).squeeze(-1).tolist()
            d_soft_full = per_organ_dice(prob, gt_center, present, threshold=None)
            for j, k in enumerate(p_idx):
                if j < len(d_soft_full):
                    per_organ_soft[k].append(d_soft_full[j])

    def mean(v): return sum(v) / max(len(v), 1)

    print("\n" + "=" * 70)
    print("MODERN VAL METRICS (all present organs, full-res)")
    print("=" * 70)
    print(f"  Primary   — soft Dice       : {mean(per_vol_soft):.4f}  (n={len(per_vol_soft)} vols)")
    print(f"  Secondary — binary@0.5      : {mean(per_vol_bin05):.4f}")
    print(f"  Diagnostic— binary@0.3      : {mean(per_vol_bin03):.4f}")
    print(f"\nGo-gate threshold = 0.15 on soft Dice")
    status = "PASS" if mean(per_vol_soft) >= 0.15 else "FAIL"
    print(f"  ep5 gate : {status}")

    print("\nPer-organ soft Dice breakdown:")
    ORGANS = ["spleen", "r_kidney", "l_kidney", "gallbladder", "esophagus",
              "liver", "stomach", "aorta", "ivc", "pancreas",
              "r_adrenal", "l_adrenal", "duodenum", "bladder", "prostate/uterus"]
    for k in range(15):
        vals = per_organ_soft[k]
        if vals:
            print(f"  organ{k+1:2d} {ORGANS[k]:<18}: {mean(vals):.4f}  (n={len(vals)})")


if __name__ == "__main__":
    main()
