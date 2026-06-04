"""V9 CUDA smoke (spec S7c gate 1).

Runs one forward + backward through the full VoluFormerV9 cascade on CUDA
and asserts peak VRAM < 22 GB and loss is finite. Skips if CUDA missing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    if not torch.cuda.is_available():
        print("[CUDA smoke] SKIPPED - no CUDA device available")
        return

    import yaml
    from omegaconf import OmegaConf
    cfg_dict = yaml.safe_load(open(ROOT / "configs/v9_tierB.yaml"))
    cfg = OmegaConf.create(cfg_dict)

    from models.voluformer_v9 import VoluFormerV9
    torch.manual_seed(0)
    model = VoluFormerV9(cfg).cuda()
    model.set_stage(2)

    B = 2
    sample = {
        "volume": torch.randn(B, 1, 96, 96, 96, device="cuda"),
        "slab": torch.randn(
            B, cfg.data.depth, 3, cfg.data.img_size, cfg.data.img_size,
            device="cuda",
        ),
        "organ_id": torch.tensor([0, 3], device="cuda"),
        "slab_center_z": torch.tensor(
            [cfg.data.depth // 2] * B, device="cuda",
        ),
    }
    torch.cuda.reset_peak_memory_stats()
    out = model(sample, stage=2)
    ref_mask = out["refiner"]["masks"]
    loss = ref_mask.sum()
    loss.backward()
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"[CUDA smoke] peak VRAM: {peak_gb:.2f} GB")
    assert peak_gb < 22.0, f"peak VRAM {peak_gb:.1f} GB exceeds 22 GB budget"
    assert torch.isfinite(loss), "loss is not finite"
    print("[CUDA smoke] PASSED")


if __name__ == "__main__":
    main()
