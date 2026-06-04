"""Path A1 — Diagnostic: K=4 decoder collapse investigation.

Loads epoch008.pt and inspects:
  1. Organ query embeddings — are queries 0/1/2/5 differentiated or collapsed?
  2. Per-channel mask statistics on a single val batch — what is each
     of the 15 channels actually producing?
  3. ODE organ-conditioning sanity — feed the SAME image with different
     organ_ids and check whether the encoded features differ.
  4. Cross-channel mask correlation — if all 15 channels output ~the same
     mask, queries are collapsed regardless of token diversity.
  5. Mask MLP head weight norms — has any per-organ head decayed to ~0?

Output: prints a compact ASCII report to stdout. No file writes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf

from datasets.amos22 import AMOS22_3D_Dataset
from models import build_model

import os
CKPT_NAME = os.environ.get("DIAG_CKPT", "path_a1_big_solid_epoch008.pt")
CKPT_PATH = ROOT / "checkpoints" / "path_a1_big_solid" / CKPT_NAME
A1_CONFIG = ROOT / "configs" / "path_a1_big_solid.yaml"
BASE_CONFIG = ROOT / "configs" / "base.yaml"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_ORGANS = [1, 2, 3, 6]   # spleen, R-kidney, L-kidney, liver
ORGAN_NAMES = {1: "spleen", 2: "r_kidney", 3: "l_kidney", 6: "liver"}


def load_cfg():
    cfg = OmegaConf.load(str(BASE_CONFIG)) if BASE_CONFIG.exists() else OmegaConf.create({})
    a1 = OmegaConf.load(str(A1_CONFIG))
    if "defaults" in a1:
        a1 = OmegaConf.masked_copy(a1, [k for k in a1 if k != "defaults"])
    return OmegaConf.merge(cfg, a1)


def main() -> None:
    print("=" * 78)
    print("A1 K=4 DECODER COLLAPSE DIAGNOSTIC")
    print(f"  ckpt: {CKPT_PATH.name}")
    print("=" * 78)

    cfg = load_cfg()
    model = build_model(cfg).to(DEVICE).eval()
    ckpt = torch.load(str(CKPT_PATH), map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    print(f"  loaded ckpt epoch={ckpt.get('epoch', '?')}  metrics={ckpt.get('metrics', {})}")
    print()

    # ----- 1. Organ query embedding analysis -----
    print("[1/5] Organ query embeddings (all 15 organs)")
    organ_q = model.decoder.organ_queries.weight.detach().float()   # (15, embed_dim)
    organ_q_norm = F.normalize(organ_q, dim=-1)
    # cosine sim between every pair
    sim = organ_q_norm @ organ_q_norm.T
    print(f"  query norms (per-organ L2):")
    for i in range(15):
        marker = " <-- supervised" if i in [0, 1, 2, 5] else ""
        organ_label = ORGAN_NAMES.get(i + 1, f"org{i+1}")
        print(f"    [{i:2d}] {organ_label:14s}  ||q|| = {organ_q[i].norm().item():.4f}{marker}")

    print()
    print("  cosine sim (4 supervised ones vs each other):")
    sup_idx = [0, 1, 2, 5]
    sup_names = ["spleen", "r_kidney", "l_kidney", "liver"]
    print(f"           {'  '.join(f'{n:>10s}' for n in sup_names)}")
    for i_idx, i in enumerate(sup_idx):
        row = [f"{sim[i, j].item():>10.4f}" for j in sup_idx]
        print(f"  {sup_names[i_idx]:8s} {'  '.join(row)}")

    # check for collapse: if any pair has |cos| > 0.95, queries are degenerate
    collapsed = []
    for ii, i in enumerate(sup_idx):
        for jj, j in enumerate(sup_idx):
            if j > i and abs(sim[i, j].item()) > 0.95:
                collapsed.append((sup_names[ii], sup_names[jj], sim[i, j].item()))
    if collapsed:
        print(f"  WARNING: {len(collapsed)} supervised query pairs have |cos| > 0.95:")
        for a, b, c in collapsed:
            print(f"    {a} <-> {b}: cos={c:.4f}")
    else:
        print("  PASS: all supervised query pairs have |cos| < 0.95 (no collapse)")

    print()

    # ----- 2. Mask MLP head weight norms -----
    print("[2/5] Per-organ mask MLP head weight norms")
    for i in range(15):
        head = model.decoder.mask_mlps[i]
        # MLP has internal layers; sum L2 of all params
        norm = sum(p.detach().float().norm().item() ** 2 for p in head.parameters()) ** 0.5
        marker = " <-- supervised" if i in [0, 1, 2, 5] else ""
        organ_label = ORGAN_NAMES.get(i + 1, f"org{i+1}")
        print(f"    [{i:2d}] {organ_label:14s}  ||W|| = {norm:.4f}{marker}")

    print()

    # ----- 3. Build a small val batch (1 image, 4 organ_ids) -----
    print("[3/5] Loading val volume for live forward-pass diagnostic")
    cfg_data = cfg.data
    val_ds = AMOS22_3D_Dataset(
        cfg=cfg,
        split="val",
        modality=getattr(cfg_data, "modality", "ct"),
        target_organs=list(cfg_data.target_organs),
        n_slices=int(cfg_data.slices_per_volume),
        center_slice_only=True,
        copypaste_prob=0.0,
    )
    print(f"  val_ds samples: {len(val_ds)}")

    # find one sample per supervised organ
    samples_by_organ = {}
    for idx in range(min(len(val_ds), 200)):
        s = val_ds[idx]
        oid = int(s["organ_id"])
        if oid in TARGET_ORGANS and oid not in samples_by_organ:
            samples_by_organ[oid] = s
        if len(samples_by_organ) == 4:
            break
    print(f"  found samples for organ_ids: {sorted(samples_by_organ.keys())}")

    # use one image (let's say liver one) and run K=4 forward passes
    liver_sample = samples_by_organ[6]
    img = liver_sample["image"].unsqueeze(0).to(DEVICE)   # (1, D, 3, H, W)
    print(f"  liver sample image shape: {img.shape}")
    print()

    # ----- 4. ODE conditioning sanity: same image, 4 different organ_ids -----
    print("[4/5] ODE conditioning differential test")
    print("  Feeding SAME image with each of organ_id={1,2,3,6}")
    print("  Comparing the encoded center features (should differ if ODE conditions correctly)")
    feats = {}
    with torch.no_grad():
        for oid in TARGET_ORGANS:
            organ_id_t = torch.tensor([oid], dtype=torch.long, device=DEVICE)
            feat, skip = model.encode_image(img, organ_id_t, return_all_slices=False)
            feats[oid] = feat.detach().float().cpu()  # (1, C, Hf, Wf)

    print(f"  feature shape: {feats[6].shape}")
    print(f"  per-organ feat L2 (mean over channels):")
    for oid in TARGET_ORGANS:
        print(f"    organ_id={oid:1d} {ORGAN_NAMES[oid]:9s} L2={feats[oid].norm().item():.4f}")

    # Pairwise cosine sim of flattened features
    print(f"  pairwise cosine sim of features:")
    flat = {oid: feats[oid].flatten() for oid in TARGET_ORGANS}
    flat_n = {oid: F.normalize(flat[oid], dim=0) for oid in TARGET_ORGANS}
    print(f"           {'  '.join(f'{ORGAN_NAMES[o]:>9s}' for o in TARGET_ORGANS)}")
    for o1 in TARGET_ORGANS:
        row = []
        for o2 in TARGET_ORGANS:
            c = (flat_n[o1] * flat_n[o2]).sum().item()
            row.append(f"{c:>9.4f}")
        print(f"  {ORGAN_NAMES[o1]:9s} {'  '.join(row)}")

    # If ODE conditioning works, off-diagonal should be < 0.99 (features differ)
    max_offdiag = 0.0
    for o1 in TARGET_ORGANS:
        for o2 in TARGET_ORGANS:
            if o1 != o2:
                c = (flat_n[o1] * flat_n[o2]).sum().item()
                max_offdiag = max(max_offdiag, c)
    if max_offdiag > 0.99:
        print(f"  WARNING: max off-diag cos = {max_offdiag:.4f} >= 0.99 — ODE conditioning IS NOT DISCRIMINATIVE")
    else:
        print(f"  PASS: max off-diag cos = {max_offdiag:.4f} < 0.99 (ODE features differ per organ)")
    print()

    # ----- 5. Per-channel mask analysis -----
    print("[5/5] Per-channel mask output on the SAME liver image with organ_id=6")
    print("  Decoder predicts ALL 15 channels; we inspect each channel's stats")
    organ_id_t = torch.tensor([6], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        out = model(img, organ_id_t, is_3d=True)
    masks = out["masks"][0].float().cpu()    # (15, H, W)
    iou_pred = out["iou_pred"][0].float().cpu()  # (15,)
    print(f"  mask shape: {masks.shape}")
    print(f"  IoU pred per channel: {iou_pred.numpy().round(3).tolist()}")

    print(f"  per-channel logit stats:")
    for i in range(15):
        m = masks[i]
        prob = torch.sigmoid(m)
        mean_prob = prob.mean().item()
        max_prob = prob.max().item()
        active_frac = (prob > 0.5).float().mean().item()
        marker = " <-- target (liver)" if i == 5 else (" <-- supervised" if i in [0,1,2] else "")
        organ_label = ORGAN_NAMES.get(i + 1, f"org{i+1}")
        print(f"    [{i:2d}] {organ_label:14s}  logit[min={m.min().item():>7.3f} max={m.max().item():>7.3f}]  "
              f"prob[mean={mean_prob:.3f} max={max_prob:.3f} act_frac={active_frac:.3f}]{marker}")

    # Cross-channel correlation: are all 15 channels outputting near-identical masks?
    print()
    print("  cross-channel mask correlation (SUPERVISED only: 0,1,2,5)")
    sup_masks = masks[[0, 1, 2, 5]].view(4, -1)
    sup_norm = F.normalize(sup_masks, dim=-1)
    sim_m = sup_norm @ sup_norm.T
    print(f"           {'  '.join(f'{n:>10s}' for n in sup_names)}")
    for i_idx, name in enumerate(sup_names):
        row = [f"{sim_m[i_idx, j].item():>10.4f}" for j in range(4)]
        print(f"  {name:8s} {'  '.join(row)}")

    if max(sim_m[i, j].abs().item() for i in range(4) for j in range(4) if i != j) > 0.95:
        print("  WARNING: supervised channels output near-identical masks (mask-level collapse)")
    else:
        print("  channels output sufficiently different masks")

    # Compare against GT to compute per-organ DSC on this single sample
    print()
    print("  GT-vs-pred DSC (liver image, all 4 supervised channels):")
    masks_gt_all = liver_sample.get("masks_all", None)  # might exist; check shape
    if masks_gt_all is not None:
        # masks_all expected (15, H, W) or similar
        print(f"    masks_all shape: {masks_gt_all.shape}")
        gt = masks_gt_all.float()
        if gt.shape[-2:] != masks.shape[-2:]:
            gt = F.interpolate(gt.unsqueeze(0), size=masks.shape[-2:],
                              mode="nearest").squeeze(0)
        for k_idx, k in enumerate([0, 1, 2, 5]):
            pred_bin = (torch.sigmoid(masks[k]) > 0.5).float()
            gt_k = (gt[k] > 0.5).float()
            inter = (pred_bin * gt_k).sum().item()
            denom = pred_bin.sum().item() + gt_k.sum().item()
            dsc = 2 * inter / max(denom, 1.0) if denom > 0 else 0.0
            print(f"    [{k:2d}] {sup_names[k_idx]:9s}: DSC={dsc:.4f}  pred_pos={pred_bin.sum().item():.0f}  "
                  f"gt_pos={gt_k.sum().item():.0f}")
    else:
        print(f"    masks_all not in sample — keys: {list(liver_sample.keys())}")

    print()
    print("=" * 78)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
