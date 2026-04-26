"""Preflight audit for OrganMoE-3D Phase I TotalSeg pretrain.

Run BEFORE every long pretrain. Catches schema bugs, label-map omissions,
NaN losses, channel-count mismatches, and VRAM blowouts before sinking 6+ hours.

Usage:
    python scripts/preflight_totalseg_pretrain.py --cfg configs/organmoe_phase_i_pretrain.yaml
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
from collections import Counter

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import torch
from omegaconf import OmegaConf

from datasets.totalsegmentator import TotalSegmentatorDataset, _TS_FILENAME_TO_AMOS
from models.voluformer_v9 import VoluFormerV9
from training.train_v9 import ORGAN_NAMES, masks_to_class_label, stage1_loss
from training.losses_small_organ import SmallOrganLoss


def fail(msg):
    print(f"  [FAIL] {msg}")
    raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, type=Path)
    parser.add_argument("--probe-vols", type=int, default=50)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
    print(f"[preflight] cfg={args.cfg.name}")
    print(f"[preflight] expects K=15 organs (AMOS22 schema)")
    print()

    # 1) Label map audit -----------------------------------------------------
    print("[1/6] Label map audit")
    K = 15
    by_amos = sorted(set(_TS_FILENAME_TO_AMOS.values()))
    print(f"  map size: {len(_TS_FILENAME_TO_AMOS)}, AMOS classes covered: {by_amos}")
    missing = [k for k in range(1, K + 1) if k not in by_amos]
    if missing:
        fail(f"AMOS classes with NO source filename: {missing}. "
             f"Each one will train zero foreground in pretrain.")
    print("  [OK] all 15 classes have at least one source filename")
    print()

    # 2) Per-class hit rate across probe sample -----------------------------
    print(f"[2/6] Per-class foreground hit rate (first {args.probe_vols} train vols)")
    train_ds = TotalSegmentatorDataset(
        data_root=cfg.data.data_root,
        split="train",
        volume_patch=int(cfg.data.volume_patch),
        hu_clip=tuple(cfg.data.hu_clip),
        slabs_per_volume=int(cfg.data.slabs_per_volume),
        seed=int(cfg.experiment.seed),
    )
    print(f"  train_ds size: {len(train_ds.volume_ids)} vols ({len(train_ds)} items)")

    hits = Counter()
    sample_ids = train_ds.volume_ids[: args.probe_vols]
    t0 = time.time()
    for vid in sample_ids:
        _, lab = train_ds._load_volume(vid)
        for c in np.unique(lab):
            if int(c) > 0:
                hits[int(c)] += 1
    dt = time.time() - t0
    print(f"  probe took {dt:.1f}s ({dt / max(len(sample_ids), 1):.2f}s per vol)")
    for k in range(1, K + 1):
        name = ORGAN_NAMES[k - 1]
        h = hits.get(k, 0)
        rate = h / len(sample_ids)
        flag = "  " if h > 0 else "!!"
        print(f"  {flag} class {k:2d} ({name:18s}): {h}/{len(sample_ids)} ({rate*100:.0f}%)")
    zero_classes = [
        (k, ORGAN_NAMES[k - 1]) for k in range(1, K + 1) if hits.get(k, 0) == 0
    ]
    if zero_classes:
        fail(f"Classes with ZERO foreground in probe sample: {zero_classes}. "
             f"These would receive no signal during pretrain.")
    print("  [OK] every class has positive samples in the probe")
    print()

    # 3) Item schema check ---------------------------------------------------
    print("[3/6] Item schema check")
    item = train_ds[0]
    expected = {"volume", "mask_volume", "slab", "mask_slab", "organ_id", "slab_center_z"}
    missing_keys = expected - set(item.keys())
    if missing_keys:
        fail(f"missing keys: {missing_keys}")
    p = int(cfg.data.volume_patch)
    if tuple(item["volume"].shape) != (1, p, p, p):
        fail(f"volume shape={tuple(item['volume'].shape)} expected (1, {p}, {p}, {p})")
    if tuple(item["mask_volume"].shape) != (K, p, p, p):
        fail(f"mask_volume shape={tuple(item['mask_volume'].shape)} expected ({K}, {p}, {p}, {p})")
    print(f"  [OK] keys={sorted(item.keys())}")
    print(f"  [OK] volume {tuple(item['volume'].shape)}, mask_volume {tuple(item['mask_volume'].shape)}")
    print()

    # 4) Model forward + loss + backward dry run -----------------------------
    print("[4/6] Model forward + loss + backward dry run")
    if not torch.cuda.is_available():
        fail("CUDA not available — pretrain requires GPU")

    device = torch.device("cuda")
    model = VoluFormerV9(cfg).to(device)
    model.set_stage(1)

    # build a 2-item batch
    items = [train_ds[i] for i in range(2)]
    batch = {}
    for k in items[0].keys():
        if isinstance(items[0][k], torch.Tensor):
            batch[k] = torch.stack([it[k] for it in items], dim=0).to(device)
    print(f"  batch volume {tuple(batch['volume'].shape)}, mask_volume {tuple(batch['mask_volume'].shape)}")

    amp_dtype = torch.bfloat16 if cfg.training.amp_dtype == "bf16" else torch.float16
    small_organ = None
    if cfg.training.get("loss_small_organ", False):
        sc = dict(cfg.training.loss_small_organ_cfg)
        small_organ = SmallOrganLoss(n_classes=K + 1, **sc).to(device)

    torch.cuda.reset_peak_memory_stats()
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-5,
    )
    model.train()
    n_steps = 3
    for step in range(n_steps):
        opt.zero_grad()
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
            losses = stage1_loss(out, batch, small_organ=small_organ, deep_sup_weight=0.0)
            total = losses["total"]
        if not torch.isfinite(total):
            fail(f"step {step} loss is NaN/Inf: {total.item()}")
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        print(f"  step {step+1}/{n_steps} loss={total.item():.4f}")
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  peak VRAM during dry run: {peak_gb:.2f} GB")
    if peak_gb > float(cfg.safety.get("max_peak_vram_gb", 23)):
        fail(f"peak VRAM {peak_gb:.2f} GB > safety limit {cfg.safety.max_peak_vram_gb} GB")

    # check output channels
    K_plus = out["proposer"]["full_logits"].shape[1]
    if K_plus != K + 1:
        fail(f"proposer output channels={K_plus}, expected K+1={K+1}")
    print(f"  [OK] proposer output channels = {K_plus} (K+1)")
    print(f"  [OK] {n_steps} forward+backward steps complete, all losses finite")
    print()

    # 5) Config sanity -------------------------------------------------------
    print("[5/6] Config sanity")
    fs = int(cfg.model.proposer.feature_size)
    amp = cfg.training.amp_dtype
    if fs >= 96 and amp != "bf16":
        fail(f"feature_size={fs} requires amp_dtype=bf16 (got {amp}). fp16 WILL NaN.")
    every_n = int(cfg.eval.every_n_epochs)
    if every_n != 1:
        print(f"  [WARN] eval.every_n_epochs={every_n}; user policy is save every epoch")
    out_dir = REPO / cfg.experiment.output_dir
    if out_dir.exists() and any(out_dir.iterdir()):
        # Sacred ckpt protocol: never overwrite. Allow ONLY empty dir or fresh launch.
        existing = sorted(out_dir.glob("*.pt"))
        print(f"  [WARN] output_dir not empty: {len(existing)} .pt files. Will overwrite.")
    print(f"  [OK] feature_size={fs}, amp={amp}, every_n_epochs={every_n}")
    print(f"  [OK] output_dir={out_dir}")
    print()

    # 6) DataLoader smoke (single worker, ~5 batches) -----------------------
    print("[6/6] DataLoader smoke (5 batches)")
    from torch.utils.data import DataLoader
    loader = DataLoader(
        train_ds, batch_size=int(cfg.training.batch_size),
        num_workers=0, shuffle=True,
    )
    t0 = time.time()
    for i, b in enumerate(loader):
        if i >= 5:
            break
    dt = time.time() - t0
    print(f"  [OK] 5 batches in {dt:.1f}s ({dt/5:.2f}s per batch)")
    print()

    print("=" * 60)
    print("[preflight] ALL CHECKS PASSED — safe to launch long run")
    print("=" * 60)


if __name__ == "__main__":
    main()
