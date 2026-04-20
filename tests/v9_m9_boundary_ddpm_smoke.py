"""M9 smoke test - boundary DDPM refiner.

Verifies:
  1. Training-mode forward returns a denoising MSE loss with grad.
  2. Inference-mode sampling returns a refined mask of matching shape,
     and only pixels in the uncertainty band change relative to the input.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.boundary_ddpm import BoundaryDDPM


def test_training_loss() -> None:
    torch.manual_seed(0)
    m = BoundaryDDPM(n_organs=15, hidden=32, n_timesteps=50)
    B, K, H, W = 2, 15, 64, 64
    img = torch.randn(B, 3, H, W)
    coarse_mask = torch.rand(B, K, H, W, requires_grad=True)
    gt_mask = (torch.rand(B, K, H, W) > 0.5).float()
    loss = m.training_loss(img, coarse_mask, gt_mask)
    assert loss.requires_grad and torch.isfinite(loss)
    loss.backward()
    assert coarse_mask.grad is not None and coarse_mask.grad.abs().sum() > 0
    print(f"[M9] training loss OK - {loss.item():.4f}")


def test_sampling_only_changes_boundary() -> None:
    torch.manual_seed(0)
    m = BoundaryDDPM(n_organs=15, hidden=32, n_timesteps=10).eval()
    B, K, H, W = 1, 15, 32, 32
    img = torch.randn(B, 3, H, W)
    coarse = torch.zeros(B, K, H, W)
    coarse[:, 5, 8:24, 8:24] = 0.99
    coarse[:, 5, 7:25, 7:25] = torch.where(
        coarse[:, 5, 7:25, 7:25] > 0,
        coarse[:, 5, 7:25, 7:25],
        torch.full_like(coarse[:, 5, 7:25, 7:25], 0.55),
    )
    with torch.no_grad():
        refined = m.sample(img, coarse, n_steps=5)
    conf_mask = (coarse > 0.95) | (coarse < 0.05)
    diff = (refined - coarse).abs()
    assert diff[conf_mask].max().item() < 1e-5, "confident region changed"
    print("[M9] sampling respects boundary band")


def main() -> None:
    torch.manual_seed(0)
    test_training_loss()
    test_sampling_only_changes_boundary()
    print("[M9] ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
