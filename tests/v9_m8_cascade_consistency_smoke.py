"""M8 smoke test - cascade consistency loss.

Verifies:
  1. lam_kl = lam_dice = 0 path is a no-op returning a zero scalar.
  2. Identical inputs produce near-zero loss.
  3. Divergent inputs produce positive loss with gradient flowing to both
     proposer and refiner inputs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from losses.cascade_consistency import CascadeConsistencyLoss


def test_noop() -> None:
    mod = CascadeConsistencyLoss(lam_kl=0.0, lam_dice=0.0)
    prop = torch.rand(2, 15, 32, 32, requires_grad=True)
    ref = torch.rand(2, 15, 32, 32, requires_grad=True)
    out = mod(prop, ref)
    assert out["loss"].item() == 0.0
    print("[M8] noop OK")


def test_identical_inputs_near_zero() -> None:
    mod = CascadeConsistencyLoss(lam_kl=1.0, lam_dice=1.0)
    x = torch.rand(2, 15, 32, 32).clamp(1e-3, 1 - 1e-3)
    x.requires_grad_()
    out = mod(x, x.detach().clone())
    assert out["loss"].item() < 0.1, f"expected small loss, got {out['loss'].item()}"
    print(f"[M8] identical inputs -> loss={out['loss'].item():.4f}")


def test_divergent_inputs_have_grad() -> None:
    mod = CascadeConsistencyLoss(lam_kl=1.0, lam_dice=1.0)
    prop = torch.rand(2, 15, 32, 32).clamp(1e-3, 1 - 1e-3).requires_grad_()
    ref = (1.0 - prop.detach()).clone().requires_grad_()
    out = mod(prop, ref)
    assert out["loss"].item() > 0.1
    out["loss"].backward()
    assert prop.grad is not None and prop.grad.abs().sum() > 0
    assert ref.grad is not None and ref.grad.abs().sum() > 0
    print(f"[M8] divergent inputs -> loss={out['loss'].item():.4f}, grad flows")


def main() -> None:
    torch.manual_seed(0)
    test_noop()
    test_identical_inputs_near_zero()
    test_divergent_inputs_have_grad()
    print("[M8] ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
