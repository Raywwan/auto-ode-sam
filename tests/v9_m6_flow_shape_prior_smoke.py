"""M6 smoke test — flow-matched shape prior loss.

Verifies:
  1. lam = 0 path short-circuits and returns 0 with no ODE integration.
  2. lam > 0 path produces a finite scalar with grad flowing back to
     the predicted mask AND to the flow parameters.
  3. A degenerate "prior == target" case produces ~0 loss.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from losses.flow_shape_prior import FlowShapePriorLoss


def _make_module(lam: float):
    return FlowShapePriorLoss(
        n_organs=15,
        latent_dim=32,
        n_steps=4,
        lam=lam,
    )


def test_zero_lambda_noop() -> None:
    mod = _make_module(lam=0.0)
    B = 2
    pred_mask = torch.randn(B, 15, 64, 64, requires_grad=True)
    organ_id = torch.tensor([0, 3])
    out = mod(pred_mask, organ_id)
    assert out["loss"].item() == 0.0
    assert not out["loss"].requires_grad
    print("[M6] lam=0 noop OK - short-circuits without running ODE")


def test_active_path_has_grad() -> None:
    torch.manual_seed(0)
    mod = _make_module(lam=0.5)
    B = 2
    pred_mask = torch.randn(B, 15, 64, 64, requires_grad=True)
    organ_id = torch.tensor([4, 7])
    out = mod(pred_mask, organ_id)
    l = out["loss"]
    assert torch.isfinite(l)
    assert l.requires_grad
    l.backward()
    assert pred_mask.grad is not None and pred_mask.grad.abs().sum() > 0
    any_param_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in mod.parameters()
    )
    assert any_param_grad, "flow params should receive gradient"
    print(f"[M6] active path OK - loss={l.item():.4f} grad flows")


def test_prior_equals_target_is_small() -> None:
    """When override_prior == sigmoid(pred_mask), soft-dice should collapse to ~0
    for near-binary predictions. We use very high/low logits so sigmoid is ~0/1."""
    torch.manual_seed(0)
    mod = _make_module(lam=1.0)
    B = 1
    mask = torch.full((B, 15, 64, 64), -20.0)       # background logit = -20 -> sigmoid ~ 0
    mask[:, 5, 20:40, 20:40] = 20.0                  # foreground logit = 20 -> sigmoid ~ 1
    organ_id = torch.tensor([5])
    mod.override_prior = mask.sigmoid().detach().clone()
    pred = mask.clone().requires_grad_()
    out = mod(pred, organ_id)
    assert out["loss"].item() < 0.05, f"expected small loss, got {out['loss'].item()}"
    print(f"[M6] identity prior OK - loss={out['loss'].item():.5f}")


def main() -> None:
    torch.manual_seed(0)
    test_zero_lambda_noop()
    test_active_path_has_grad()
    test_prior_equals_target_is_small()
    print("[M6] ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
