"""M7 smoke test - dynamic PFESA++.

Verifies:
  1. Forward returns feature of matching (B, C, H, W) shape.
  2. Kernels differ when organ_id differs (hypernet actually conditions).
  3. Gradient flows through hypernet AND through input features.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.dynamic_pfesa import DynamicPFESA


def test_shape_and_grad() -> None:
    torch.manual_seed(0)
    mod = DynamicPFESA(in_ch=64, out_ch=64, n_organs=15, text_dim=0)
    B, C, H, W = 2, 64, 32, 32
    feat = torch.randn(B, C, H, W, requires_grad=True)
    organ_id = torch.tensor([3, 9])
    out = mod(feat, organ_id=organ_id, text_emb=None)
    assert out.shape == (B, C, H, W)
    out.sum().backward()
    assert feat.grad is not None and feat.grad.abs().sum() > 0
    hyper_grads = [p.grad for p in mod.hyper.parameters() if p.grad is not None]
    assert any(g.abs().sum() > 0 for g in hyper_grads), "hypernet got no grad"
    print("[M7] shape + grad OK")


def test_organ_conditioning_changes_output() -> None:
    torch.manual_seed(0)
    mod = DynamicPFESA(in_ch=32, out_ch=32, n_organs=15, text_dim=0).eval()
    # Kick hypernet away from zero-init so outputs can diverge.
    with torch.no_grad():
        for p in mod.hyper.parameters():
            p.add_(0.01 * torch.randn_like(p))
    feat = torch.randn(1, 32, 16, 16)
    with torch.no_grad():
        a = mod(feat, organ_id=torch.tensor([0]), text_emb=None)
        b = mod(feat, organ_id=torch.tensor([7]), text_emb=None)
    assert not torch.allclose(a, b, atol=1e-5)
    print("[M7] organ conditioning affects output")


def test_with_text_embedding() -> None:
    torch.manual_seed(0)
    mod = DynamicPFESA(in_ch=32, out_ch=32, n_organs=15, text_dim=128).eval()
    with torch.no_grad():
        for p in mod.hyper.parameters():
            p.add_(0.01 * torch.randn_like(p))
    feat = torch.randn(1, 32, 16, 16)
    text_a = torch.randn(1, 128)
    text_b = torch.randn(1, 128)
    with torch.no_grad():
        a = mod(feat, organ_id=torch.tensor([0]), text_emb=text_a)
        b = mod(feat, organ_id=torch.tensor([0]), text_emb=text_b)
    assert not torch.allclose(a, b, atol=1e-5)
    print("[M7] text embedding affects output")


def main() -> None:
    test_shape_and_grad()
    test_organ_conditioning_changes_output()
    test_with_text_embedding()
    print("[M7] ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
