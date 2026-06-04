"""V9 proposer 3D evaluation on the preprocessed AMOS22 val split.

Loads preprocessed .pt tensors (RAS, 1.5 mm isotropic, HU-normalised to [0, 1]
by preprocess_amos.py) and runs the SwinUNETR proposer with a 96^3 sliding
window + Gaussian blend + optional 4-way flip TTA + largest-CC postproc.
Reports per-organ 3D Dice (present-only and all-channel) to match the
AMOS22 / MCP-MedSAM evaluation protocol.

The val split matches AMOS22V9Dataset(split='val'): last 10 percent of sorted
CT IDs (amos_0001..amos_0500 ∩ preprocessed/*.pt).

Why this script exists (2026-04-22):
  evaluation/eval_v9_3d.py feeds RAW NIfTI into the model, skipping
  Orientationd(RAS) and using HU_clip=[-200,250] vs training's [-175,250].
  On a training volume (amos_0008) it returned mean_dice=0.12 vs the
  patch-level training val of 0.8433. Root cause was an orientation /
  normalisation mismatch. This wrapper bypasses all nifti preprocessing
  and feeds training-ready tensors directly.

Usage:
    python -m scripts.eval_v9_proposer_preproc \
        --ckpt checkpoints/v9_stage1/last.pt \
        --cfg configs/v9_tierB.yaml \
        --max-volumes 1            # smoke test first
    python -m scripts.eval_v9_proposer_preproc \
        --ckpt checkpoints/v9_stage1/last.pt \
        --cfg configs/v9_tierB.yaml            # full val set
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.voluformer_v9 import VoluFormerV9
from models.trissr_v9 import TriSSRV9
from evaluation.eval_v9_3d import _gauss3d, _cc_keeplargest
from evaluation.metrics_3d import (
    dice_score_3d,
    hausdorff_distance_95_mm,
    normalised_surface_dice_3d,
    ORGAN_NSD_TOLERANCES_MM,
)


ORGAN_NAMES = [
    "spleen", "r_kidney", "l_kidney", "gallbladder", "esophagus",
    "liver", "stomach", "aorta", "ivc", "pancreas",
    "r_adrenal", "l_adrenal", "duodenum", "bladder", "prostate_uterus",
]
PRESENCE_MIN_VOXELS = 50

# Channels (in preds shape (K+1, Z, H, W), bg at 0) that must be swapped when
# the prediction has been flipped along the L-R axis. Preprocessing uses RAS
# (axis 0 = R), and this script transposes (H, W, D) -> (D, H, W), so the
# model's axis 2 of preds is the L-R body axis. Flipping that axis and
# un-flipping puts r_kidney anatomy under the l_kidney channel (and vice
# versa). The fix is to swap these paired channels before accumulating.
LR_SWAP_CHANNELS = [
    (2, 3),    # r_kidney <-> l_kidney
    (11, 12),  # r_adrenal <-> l_adrenal
]


def discover_val_ids(preproc_dir: Path) -> list[str]:
    all_ids = []
    for p in preproc_dir.glob("*.pt"):
        m = re.match(r"amos_(\d+)", p.stem)
        if not m:
            continue
        n = int(m.group(1))
        if n > 500:
            continue
        all_ids.append((n, p.stem))
    all_ids.sort()
    cut = int(round(len(all_ids) * 0.9))
    return [stem for _, stem in all_ids[cut:]]


@torch.no_grad()
def sliding_window_proposer(
    model: VoluFormerV9,
    img01: np.ndarray,
    device: str,
    patch: int = 96,
    stride: int = 48,
    trissr: TriSSRV9 | None = None,
) -> np.ndarray:
    """Accumulate the proposer's full (K+1)-way softmax (bg + organs) across
    the sliding window. Returns shape (K+1, Z, H, W). The caller applies
    argmax in post-processing — the proposer uses 16-way softmax, so a
    per-channel > 0.5 threshold misses any organ whose softmax mass is
    competing with other organs or background.

    If `trissr` is provided, the proposer's pre-softmax logits are refined by
    TriSSR before softmax, implementing the fine-tuned-proposer + TriSSR cascade.
    """
    Z, H, W = img01.shape
    K = model.fusion.n_organs
    acc = np.zeros((K + 1, Z, H, W), dtype=np.float32)
    wsum = np.zeros((Z, H, W), dtype=np.float32)
    gauss = _gauss3d(patch)

    def starts(n: int) -> list:
        if n <= patch:
            return [0]
        s = list(range(0, n - patch + 1, stride))
        if s[-1] + patch < n:
            s.append(n - patch)
        return s

    for z in starts(Z):
        for y in starts(H):
            for x in starts(W):
                zp = img01[z:z + patch, y:y + patch, x:x + patch]
                pz, py, px = zp.shape
                if (pz, py, px) != (patch, patch, patch):
                    zp = np.pad(
                        zp,
                        [(0, patch - pz), (0, patch - py), (0, patch - px)],
                    )
                v = torch.from_numpy(zp)[None, None].to(device).float()
                if trissr is None:
                    full_probs = model.proposer(v)["full_probs"].cpu().numpy()[0]
                else:
                    full_logits = model.proposer(v)["full_logits"]
                    refined = trissr(full_logits.float(), v.float())
                    full_probs = F.softmax(refined, dim=1).cpu().numpy()[0]
                acc[:, z:z + pz, y:y + py, x:x + px] += (
                    full_probs[:, :pz, :py, :px] * gauss[None, :pz, :py, :px]
                )
                wsum[z:z + pz, y:y + py, x:x + px] += gauss[:pz, :py, :px]

    wsum = np.maximum(wsum, 1e-6)
    return acc / wsum[None]


def eval_one_volume(
    model: VoluFormerV9,
    pt_path: Path,
    device: str,
    tta: bool,
    cc: bool,
    trissr: TriSSRV9 | None = None,
) -> tuple[dict, dict]:
    d = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    img = d["image"][0].numpy()                                    # (H, W, D) in [0, 1]
    lab = d["label"][0].numpy().astype(np.int64)                    # (H, W, D)
    img = np.ascontiguousarray(np.transpose(img, (2, 0, 1))).astype(np.float32)  # (Z, H, W)
    lab = np.ascontiguousarray(np.transpose(lab, (2, 0, 1)))

    preds = sliding_window_proposer(model, img, device=device, trissr=trissr)  # (K+1, Z, H, W)
    if tta:
        flip_axes = [[1], [2], [1, 2]]
        accum = preds.copy()
        for flips in flip_axes:
            imgf = np.flip(img, axis=[a - 1 for a in flips]).copy()
            p = sliding_window_proposer(model, imgf, device=device, trissr=trissr)
            p = np.flip(p, axis=flips).copy()
            if 2 in flips:
                p_sw = p.copy()
                for a, b in LR_SWAP_CHANNELS:
                    p_sw[a] = p[b]
                    p_sw[b] = p[a]
                p = p_sw
            accum += p
        preds = accum / (1.0 + len(flip_axes))

    # 16-way argmax: channel 0 = background, channels 1..K = organs.
    pred_class = preds.argmax(axis=0).astype(np.int64)            # (Z, H, W)
    K = preds.shape[0] - 1
    bin_preds = np.zeros((K,) + pred_class.shape, dtype=np.uint8)
    for k in range(K):
        bin_preds[k] = (pred_class == k + 1).astype(np.uint8)
    if cc:
        bin_preds = _cc_keeplargest(bin_preds)

    per_organ_dice = {}
    per_organ_hd95 = {}
    per_organ_nsd = {}
    gt_sizes = {}
    spacing_mm = (1.5, 1.5, 1.5)   # preprocessing is 1.5 mm isotropic
    for k in range(K):
        gt_k = (lab == k + 1).astype(np.uint8)
        gt_sizes[k + 1] = int(gt_k.sum())
        per_organ_dice[k + 1] = float(dice_score_3d(bin_preds[k], gt_k))
        if gt_k.sum() > PRESENCE_MIN_VOXELS:
            tol = ORGAN_NSD_TOLERANCES_MM.get(k + 1, 2.0)
            per_organ_hd95[k + 1] = float(hausdorff_distance_95_mm(bin_preds[k], gt_k, spacing_mm))
            per_organ_nsd[k + 1]  = float(normalised_surface_dice_3d(bin_preds[k], gt_k, spacing_mm, tol))
        else:
            per_organ_hd95[k + 1] = float("nan")
            per_organ_nsd[k + 1]  = float("nan")
    return per_organ_dice, per_organ_hd95, per_organ_nsd, gt_sizes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument(
        "--preproc-dir",
        default="C:/Users/Raywa/Desktop/LiteSAM3D/data/amos22/preprocessed",
    )
    ap.add_argument("--max-volumes", type=int, default=-1)
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--no-cc", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", default="reports/v9_proposer_3d_eval.json")
    ap.add_argument("--trissr-ckpt", default=None,
                    help="Optional TriSSR-V9 checkpoint to refine proposer logits.")
    ap.add_argument("--trissr-embed-dim", type=int, default=48)
    ap.add_argument("--trissr-n-blocks", type=int, default=3)
    ap.add_argument("--trissr-d-state",  type=int, default=8)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.cfg)
    model = VoluFormerV9(cfg).to(args.device).eval()
    sd = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    state = sd["model"] if "model" in sd else sd
    miss, unex = model.load_state_dict(state, strict=False)
    prop_missing = [k for k in miss if k.startswith("proposer.")]
    print(
        f"[eval] loaded {args.ckpt} | missing={len(miss)} (proposer: "
        f"{len(prop_missing)}) unexpected={len(unex)}"
    )
    if prop_missing:
        print(f"[eval] WARNING proposer keys missing: {prop_missing[:5]}")

    trissr = None
    if args.trissr_ckpt:
        trissr = TriSSRV9(
            n_organs=cfg.model.n_organs,
            embed_dim=args.trissr_embed_dim,
            n_blocks=args.trissr_n_blocks,
            d_state=args.trissr_d_state,
        ).to(args.device).eval()
        tsd = torch.load(args.trissr_ckpt, map_location=args.device, weights_only=False)
        tstate = tsd["model"] if "model" in tsd else tsd
        tmiss, tunex = trissr.load_state_dict(tstate, strict=False)
        print(
            f"[eval] TriSSR loaded {args.trissr_ckpt} | "
            f"missing={len(tmiss)} unexpected={len(tunex)}"
        )

    ids = discover_val_ids(Path(args.preproc_dir))
    if args.max_volumes > 0:
        ids = ids[: args.max_volumes]
    print(
        f"[eval] evaluating {len(ids)} val volumes | tta={not args.no_tta} "
        f"cc={not args.no_cc}"
    )

    all_dice:  dict[int, list[float]] = {k: [] for k in range(1, 16)}
    all_hd95:  dict[int, list[float]] = {k: [] for k in range(1, 16)}
    all_nsd:   dict[int, list[float]] = {k: [] for k in range(1, 16)}
    per_volume_results: list[dict] = []
    t_all = time.time()
    for i, vid in enumerate(ids):
        pt = Path(args.preproc_dir) / f"{vid}.pt"
        t0 = time.time()
        per_organ, per_organ_hd95, per_organ_nsd, gt_sizes = eval_one_volume(
            model, pt, args.device, tta=not args.no_tta, cc=not args.no_cc,
            trissr=trissr,
        )
        for k, dsc in per_organ.items():
            if gt_sizes[k] > PRESENCE_MIN_VOXELS:
                all_dice[k].append(dsc)
                if not np.isnan(per_organ_hd95[k]):
                    all_hd95[k].append(per_organ_hd95[k])
                if not np.isnan(per_organ_nsd[k]):
                    all_nsd[k].append(per_organ_nsd[k])
        present_dices = [
            dsc for k, dsc in per_organ.items() if gt_sizes[k] > PRESENCE_MIN_VOXELS
        ]
        all_channel_dices = list(per_organ.values())
        present_mean = float(np.mean(present_dices)) if present_dices else 0.0
        all_mean = float(np.mean(all_channel_dices))
        print(
            f"[eval] ({i + 1}/{len(ids)}) {vid}  present_mean={present_mean:.4f}  "
            f"all_mean={all_mean:.4f}  n_present={len(present_dices)}  "
            f"{time.time() - t0:.1f}s"
        )
        per_volume_results.append({
            "vol_id": vid,
            "per_organ_dice": {
                ORGAN_NAMES[k - 1]: per_organ[k] for k in per_organ
            },
            "per_organ_hd95_mm": {
                ORGAN_NAMES[k - 1]: per_organ_hd95[k] for k in per_organ_hd95
            },
            "per_organ_nsd": {
                ORGAN_NAMES[k - 1]: per_organ_nsd[k] for k in per_organ_nsd
            },
            "gt_voxels": {
                ORGAN_NAMES[k - 1]: gt_sizes[k] for k in gt_sizes
            },
            "present_mean": present_mean,
            "all_channel_mean": all_mean,
        })

    per_organ_dice_mean = {
        k: float(np.mean(v)) if v else 0.0 for k, v in all_dice.items()
    }
    per_organ_hd95_mean = {
        k: float(np.mean(v)) if v else float("nan") for k, v in all_hd95.items()
    }
    per_organ_nsd_mean = {
        k: float(np.mean(v)) if v else float("nan") for k, v in all_nsd.items()
    }
    organ_means_nonzero = [v for v in per_organ_dice_mean.values() if v > 0]
    dice_present_only = (
        float(np.mean(organ_means_nonzero)) if organ_means_nonzero else 0.0
    )
    dice_all_channel = float(
        np.mean([r["all_channel_mean"] for r in per_volume_results])
    )
    hd95_valid = [v for v in per_organ_hd95_mean.values() if not np.isnan(v)]
    nsd_valid  = [v for v in per_organ_nsd_mean.values()  if not np.isnan(v)]
    hd95_mean = float(np.mean(hd95_valid)) if hd95_valid else float("nan")
    nsd_mean  = float(np.mean(nsd_valid))  if nsd_valid  else float("nan")

    total_min = (time.time() - t_all) / 60.0
    report = {
        "checkpoint": args.ckpt,
        "trissr_checkpoint": args.trissr_ckpt,
        "n_volumes": len(ids),
        "tta": not args.no_tta,
        "cc": not args.no_cc,
        "per_organ_dice_present_only": {
            ORGAN_NAMES[k - 1]: per_organ_dice_mean[k] for k in range(1, 16)
        },
        "per_organ_hd95_mm": {
            ORGAN_NAMES[k - 1]: per_organ_hd95_mean[k] for k in range(1, 16)
        },
        "per_organ_nsd": {
            ORGAN_NAMES[k - 1]: per_organ_nsd_mean[k] for k in range(1, 16)
        },
        "per_organ_present_counts": {
            ORGAN_NAMES[k - 1]: len(all_dice[k]) for k in range(1, 16)
        },
        "mean_dice_present_only": dice_present_only,
        "mean_dice_all_channel": dice_all_channel,
        "mean_hd95_mm": hd95_mean,
        "mean_nsd": nsd_mean,
        "total_time_min": total_min,
        "per_volume": per_volume_results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== PER-ORGAN 3D METRICS (present-only across val volumes) ===")
    print(f"{'organ':<18} {'dice':>8} {'hd95_mm':>10} {'nsd':>8} {'#vols':>8}")
    for k in range(1, 16):
        name = ORGAN_NAMES[k - 1]
        hd = per_organ_hd95_mean[k]
        ns = per_organ_nsd_mean[k]
        print(
            f"{name:<18} {per_organ_dice_mean[k]:>8.4f} "
            f"{hd:>10.2f} {ns:>8.4f} "
            f"{len(all_dice[k]):>8}"
        )
    print(f"\n[eval] mean dice (present-only):  {dice_present_only:.4f}")
    print(f"[eval] mean dice (all-channel):   {dice_all_channel:.4f}")
    print(f"[eval] mean HD95 (mm):            {hd95_mean:.2f}")
    print(f"[eval] mean NSD:                  {nsd_mean:.4f}")
    print(f"[eval] total time {total_min:.1f} min")
    print(f"[eval] report saved to: {out_path}")


if __name__ == "__main__":
    main()
