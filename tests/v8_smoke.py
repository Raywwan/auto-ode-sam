"""V8 smoke test: build model, forward + backward, check output shapes."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from omegaconf import OmegaConf

from models import build_model


def main() -> None:
    cfg = OmegaConf.load(ROOT / "configs" / "v8_mcp_killer.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device={device}")

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[smoke] params total={n_params/1e6:.2f}M, trainable={n_train/1e6:.2f}M")
    print(f"[smoke] text available = {model.text_encoder is not None and model.text_encoder.weights_available}")
    print(f"[smoke] dino available = {model.dino is not None and model.dino.available}")

    B, D, C, H, W = 1, 8, 3, 320, 320
    x = torch.randn(B, D, C, H, W, device=device)
    oid = torch.zeros(B, dtype=torch.long, device=device)

    model.train()
    with torch.amp.autocast("cuda", enabled=(device == "cuda")):
        out = model(x, oid, is_3d=True)
    print(f"[smoke] masks={tuple(out['masks'].shape)}")
    print(f"[smoke] deepsup={tuple(out['deepsup_logits'].shape)}")
    print(f"[smoke] ds_s40={tuple(out['ds_scale_logits']['s40'].shape)}")
    print(f"[smoke] ds_s80={tuple(out['ds_scale_logits']['s80'].shape)}")
    print(f"[smoke] ds_s160={tuple(out['ds_scale_logits']['s160'].shape)}")

    loss = out["masks"].float().mean()
    loss.backward()
    has_grad = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
    total_train = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"[smoke] grad coverage: {has_grad}/{total_train}")

    if device == "cuda":
        alloc = torch.cuda.max_memory_allocated() / 1e9
        print(f"[smoke] peak VRAM: {alloc:.2f} GB")
    print("[smoke] OK")


if __name__ == "__main__":
    main()
