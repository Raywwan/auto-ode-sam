"""VRAM pre-flight for VoCo-L on 4090.

Instantiates the SwinUNETRProposer with feature_size=96, loads VoComni_L.pt,
runs one fwd+bwd+optimizer step at batch=1 on a 96^3 patch with AMP, and
reports peak VRAM. Must be <22 GB to green-light v10_voco_l.yaml launch.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.swin_unetr_3d import SwinUNETRProposer


def main():
    assert torch.cuda.is_available(), "CUDA unavailable"
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)

    proposer = SwinUNETRProposer(
        n_organs=15,
        patch_size=96,
        feature_size=96,
        use_v2=True,
        pretrained_weights="checkpoints/pretrained/voco/VoComni_L.pt",
        deep_supervision=False,
    ).to(device)

    n_params = sum(p.numel() for p in proposer.parameters()) / 1e6
    print(f"params: {n_params:.1f}M")

    opt = torch.optim.AdamW(proposer.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    x = torch.randn(1, 1, 96, 96, 96, device=device)

    proposer.train()
    opt.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        out = proposer(x)
        logits = out["full_logits"] if isinstance(out, dict) else out
        loss = logits.float().abs().mean()
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()

    peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3
    reserved_gb = torch.cuda.max_memory_reserved(device) / 1024**3
    print(f"peak_allocated_gb: {peak_gb:.2f}")
    print(f"peak_reserved_gb:  {reserved_gb:.2f}")
    print(f"logits_shape: {tuple(logits.shape)}")
    print(f"loss_value: {loss.item():.4f}")

    if peak_gb < 22.0:
        print("RESULT: PASS (<22 GB), safe to launch v10_voco_l")
    elif peak_gb < 24.0:
        print("RESULT: TIGHT (22-24 GB), launch with caution")
    else:
        print("RESULT: FAIL (>=24 GB), must fall back to VoCo-B")


if __name__ == "__main__":
    main()
