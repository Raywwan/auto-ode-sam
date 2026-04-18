"""End-to-end V4 smoke test on 2 volumes.

Builds the model, runs 1 forward + backward pass on 2 volumes, verifies:
  - No NaNs or Infs
  - All trainable params receive gradient
  - Loss components all finite
  - Peak VRAM usage < 22 GB (leaves headroom for batch size 4)

Config-key note: aligned to trainer convention, so reads
`cfg.data.data_root` / `cfg.data.slices_per_volume` not `root` / `depth`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.amos22_multiorgan import AMOS22MultiOrgan3D_Dataset
from models.organflow_sam2 import OrganFlowSAM2
from training.losses_v4 import V4MultiOrganLoss


def main() -> None:
    cfg = OmegaConf.load("configs/v4_organflow_sam2_256px.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    ds = AMOS22MultiOrgan3D_Dataset(
        data_root=cfg.data.data_root,
        split="train",
        img_size=cfg.data.img_size,
        depth=cfg.data.slices_per_volume,
        modality="ct",
    )
    assert len(ds) >= 2, f"Need >=2 volumes, have {len(ds)}"
    batch = [ds[0], ds[1]]
    images = torch.stack([b["image"] for b in batch]).to(device)          # (2, D, 1, H, W)
    masks = torch.stack([b["masks"] for b in batch]).to(device)           # (2, 15, D, H, W)
    present = torch.stack([b["present_mask"] for b in batch]).to(device)  # (2, 15)
    images = images.repeat(1, 1, 3, 1, 1)                                  # expand 1→3 channels
    organ_id = torch.argmax(present.int(), dim=1) + 1

    model = OrganFlowSAM2(cfg).to(device)
    model.train()
    criterion = V4MultiOrganLoss(
        lambda_flow=0.5, lambda_deepsup=0.1, lambda_anatomy=0.01,
    )

    with torch.amp.autocast("cuda", enabled=device == "cuda"):
        out = model(images, organ_id, is_3d=True)
        loss_out = criterion(
            pred_masks=out["masks"].float(),
            gt_masks=masks,
            present_mask=present,
            flow_targets=out.get("flow_targets"),
            deepsup_logits=(
                out.get("deepsup_logits").float()
                if out.get("deepsup_logits") is not None else None
            ),
            anatomy_adjacency=model.decoder.graph.adjacency,
            anatomy_adjacency_init=model.decoder.graph._init_adjacency,
        )

    loss = loss_out["loss"]
    print(f"Loss components: {loss_out['components']}")
    for name, v in loss_out["components"].items():
        assert torch.isfinite(torch.tensor(v)), f"Non-finite {name}={v}"
    loss.backward()

    # iou_mlp is intentionally unsupervised in V4 (loss uses mask/flow/deepsup/anatomy only).
    # The head is kept for future IoU-guided 3D inference. Whitelist it here.
    IOU_HEAD_PREFIX = "decoder.iou_mlp"
    missing_grad = [
        n for n, p in model.named_parameters()
        if p.requires_grad and p.grad is None and not n.startswith(IOU_HEAD_PREFIX)
    ]
    assert not missing_grad, f"Trainable params without grad: {missing_grad[:5]}..."

    if device == "cuda":
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak VRAM: {peak_gb:.2f} GB")
        assert peak_gb < 22.0, "Peak VRAM too high - drop batch size or D"

    print("SMOKE TEST PASS")


if __name__ == "__main__":
    main()
