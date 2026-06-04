"""Fetch teacher-ensemble weights used by Novel #3 (teacher distillation) and
Novel #5 (cross-modal InfoNCE). Idempotent — skips anything already present.

Teachers:
  - SAM2 (Hiera-L)       ~ 900 MB   Meta release (facebookresearch/sam2)
  - DINOv2 ViT-L/14      ~ 1.1 GB   torch.hub (facebookresearch/dinov2)
  - BiomedCLIP PMB+ViT-B ~ 500 MB   HuggingFace via open_clip_torch
  - TotalSegmentator v2            pip-installed weights auto-downloaded on first use

We cache all teacher weights under `checkpoints/teachers/`.
"""
from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEACHERS = ROOT / "checkpoints" / "teachers"
TEACHERS.mkdir(parents=True, exist_ok=True)

SAM2_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/072824/"
    "sam2_hiera_large.pt"
)
SAM2_DEST = TEACHERS / "sam2_hiera_large.pt"


def _hook(name: str):
    def inner(n: int, bs: int, total: int) -> None:
        pct = (n * bs) / max(1, total) * 100.0
        sys.stdout.write(f"\r  [{name}] {pct:5.1f}%")
        sys.stdout.flush()
    return inner


def fetch_sam2() -> None:
    if SAM2_DEST.exists() and SAM2_DEST.stat().st_size > 1 << 28:
        print(f"[sam2] present: {SAM2_DEST} ({SAM2_DEST.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"[sam2] downloading -> {SAM2_DEST}")
    urllib.request.urlretrieve(SAM2_URL, str(SAM2_DEST), reporthook=_hook("sam2"))
    sys.stdout.write("\n")
    print(f"[sam2] done: {SAM2_DEST.stat().st_size / 1e6:.1f} MB")


def fetch_dinov2() -> None:
    """DINOv2 is loaded through torch.hub — this just primes the cache."""
    import torch
    try:
        m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
        print(f"[dinov2] loaded ViT-L/14 ({sum(p.numel() for p in m.parameters())/1e6:.1f}M params)")
    except Exception as e:
        print(f"[dinov2] WARN: load failed: {e}")


def fetch_biomedclip() -> None:
    """BiomedCLIP via open_clip_torch. Downloads to HF hub cache."""
    try:
        import open_clip
    except ImportError:
        print("[biomedclip] open_clip_torch not installed — skipping")
        return
    try:
        model, _, _ = open_clip.create_model_and_transforms(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        )
        n = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[biomedclip] loaded ({n:.1f}M params)")
    except Exception as e:
        print(f"[biomedclip] WARN: load failed: {e}")


def fetch_totalsegv2() -> None:
    """TotalSegmentator v2 weights auto-download on first inference call."""
    try:
        import totalsegmentator  # noqa
        print(f"[totalseg] v2 package installed — weights will auto-download on first use")
    except ImportError:
        print("[totalseg] WARN: totalsegmentator package not installed")


def main() -> int:
    print(f"[fetch_teachers] target dir: {TEACHERS}")
    fetch_sam2()
    fetch_dinov2()
    fetch_biomedclip()
    fetch_totalsegv2()
    print("[fetch_teachers] all teacher sources primed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
