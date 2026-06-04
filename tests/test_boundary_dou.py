"""Smoke tests for BoundaryDoULoss (Sun et al., MICCAI 2023; arXiv:2308.00220).

Validates:
  1. Exact match against the paper's Fig. 2 numerical examples (S_I = 0.2/0.5/0.8
     with alpha = 0.8, both prediction and ground-truth area = 1).
  2. Gradient flow through the prediction logits.
  3. Empty-GT edge case (alpha falls back to 0; loss is finite and well-defined).
  4. Perfect-match edge case (loss approaches 0).
  5. Adaptive alpha falls in [0, alpha_max] and scales with target geometry.
  6. CombinedLoss integration: zero weight is a no-op; positive weight adds a
     'boundary_dou' breakdown entry.

Run:
    python -m pytest tests/test_boundary_dou.py -v
or directly:
    python tests/test_boundary_dou.py
"""
import math
import torch

from training.losses import BoundaryDoULoss, CombinedLoss


# ---------------------------------------------------------------------------
# 1. Paper Fig. 2 numerical matches
# ---------------------------------------------------------------------------

def _paper_fig2_case(s_i: float, alpha: float = 0.8) -> float:
    """Reference Boundary-DoU value from paper Fig 2.

    With both prediction and ground truth area normalised to 1 and intersection
    area S_I, the symmetric difference is S_D = 2 - 2*S_I and the loss is
        L = S_D / (S_D + (1 - alpha) * S_I).
    """
    s_d = 2.0 - 2.0 * s_i
    return s_d / (s_d + (1.0 - alpha) * s_i)


def test_paper_fig2_exact():
    """Construct toy 4x4 GT/pred so that S_P = S_G = 8 (area 1 after norm) and
    S_I varies, then verify the loss matches the closed-form reference values.

    Because the loss class normalises alpha from the GT geometry, here we use
    a manual fixed-alpha construction by forcing the GT to be a flat
    foreground block whose entire support is boundary (so C/S = 1 → alpha = 0).
    Instead we test the closed-form against direct probe through the analytic
    formula path: we feed soft predictions and verify that for known S_I we
    reproduce the Fig 2 numbers when alpha is patched to 0.8.
    """
    # Test directly against the closed-form formula in the loss class with a
    # patched alpha. We bypass the GT-geometry alpha estimator and instead
    # build inputs where alpha is fixed externally.
    fn = BoundaryDoULoss(alpha_max=0.8)

    for s_i_target, paper_value in [(0.2, 0.975), (0.5, 0.909), (0.8, 0.714)]:
        # Build a (B=1, H=10, W=10) prediction and GT with exact overlap.
        # Use H*W = 100 voxels split so that prediction and GT each cover 50
        # voxels and their intersection is `s_i_target * 50` voxels.
        H = W = 10
        n_voxels = H * W
        n_pos = n_voxels // 2  # 50 voxels FG in each (gives area = 0.5 each)
        n_intersection = int(round(s_i_target * n_pos))

        gt = torch.zeros(1, H, W)
        gt[0, : n_pos // W, :] = 1.0  # first rows are GT-foreground
        # Build prediction such that intersection equals n_intersection
        pred_logit = torch.full((1, H, W), -10.0)  # almost-zero probability
        # Place the first n_intersection FG voxels inside GT
        flat_pred = pred_logit.view(-1)
        gt_idx = (gt.view(-1) > 0.5).nonzero().view(-1)
        bg_idx = (gt.view(-1) <= 0.5).nonzero().view(-1)
        flat_pred[gt_idx[:n_intersection]] = 10.0           # intersection
        flat_pred[bg_idx[: (n_pos - n_intersection)]] = 10.0  # remaining FP
        # Now sigmoid(pred_logit) is ~1 inside the pred FG and ~0 outside.

        # Build a GT whose boundary length C and area S give alpha = 0.8.
        # Trick: replicate the GT to make a stripe whose only boundary is the
        # top and bottom row → C/S = 2/n_pos. So alpha = 1 - 2 * 2 / 50 = 0.92.
        # That doesn't give 0.8. Instead, we monkey-patch the loss to use
        # alpha = 0.8 directly for this test.
        with torch.no_grad():
            original_estimator = fn._gt_boundary_and_area
            fn._gt_boundary_and_area = staticmethod(
                lambda t: (
                    torch.full((t.shape[0],), 0.1 * n_pos),  # C
                    torch.full((t.shape[0],), n_pos),         # S → alpha=1-0.2=0.8
                )
            )
            try:
                loss = fn(pred_logit, gt).item()
            finally:
                fn._gt_boundary_and_area = original_estimator

        # Use the closed-form reference (Fig 2 says 0.975/0.909/0.714 for
        # alpha = 0.8 and BOTH areas = 1; here both areas = 0.5, but the
        # formula is scale-invariant because S_D and S_I scale linearly).
        expected = _paper_fig2_case(s_i_target, alpha=0.8)
        assert abs(loss - expected) < 1e-3, (
            f"S_I={s_i_target}: loss={loss:.4f}, expected={expected:.4f} "
            f"(paper Fig 2 value={paper_value})"
        )


# ---------------------------------------------------------------------------
# 2. Gradient flow
# ---------------------------------------------------------------------------

def test_gradient_flow():
    fn = BoundaryDoULoss()
    pred = torch.randn(2, 16, 16, requires_grad=True)
    gt = torch.zeros(2, 16, 16)
    gt[0, 4:12, 4:12] = 1.0
    gt[1, 6:10, 6:10] = 1.0

    loss = fn(pred, gt)
    loss.backward()

    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all(), "Gradient must be finite"
    assert pred.grad.abs().sum() > 0, "Gradient must be non-zero somewhere"


# ---------------------------------------------------------------------------
# 3. Empty-GT edge case
# ---------------------------------------------------------------------------

def test_empty_gt_is_finite():
    fn = BoundaryDoULoss()
    pred = torch.randn(2, 16, 16)
    gt = torch.zeros(2, 16, 16)  # all background

    loss = fn(pred, gt)
    assert torch.isfinite(loss), "Loss must be finite when GT is empty"
    # With empty GT, alpha falls back to 0; loss = S_D / (S_D + S_I) = IoU loss
    # S_I = 0 (gt is all zero), so loss = 1 (FP-only case) unless pred is also
    # all-zero. Just require finite + non-negative + <=1+smooth.
    assert 0.0 <= loss.item() <= 1.0 + 1e-3


# ---------------------------------------------------------------------------
# 4. Perfect-match edge case
# ---------------------------------------------------------------------------

def test_perfect_match_loss_small():
    fn = BoundaryDoULoss()
    gt = torch.zeros(1, 16, 16)
    gt[0, 4:12, 4:12] = 1.0
    # Prediction logits that produce ~exact match after sigmoid (>>0 → ~1)
    pred = torch.where(gt > 0.5, torch.full_like(gt, 10.0), torch.full_like(gt, -10.0))

    loss = fn(pred, gt)
    assert loss.item() < 1e-3, f"Perfect-match loss should be ~0, got {loss.item():.4f}"


# ---------------------------------------------------------------------------
# 5. Adaptive alpha scaling
# ---------------------------------------------------------------------------

def test_alpha_is_in_range_and_geometry_aware():
    fn = BoundaryDoULoss(alpha_max=0.8)

    # Large solid block: low C/S → alpha close to 1, clamped to 0.8
    gt_large = torch.zeros(1, 32, 32)
    gt_large[0, 4:28, 4:28] = 1.0  # 24x24 = 576 voxels, perimeter ~96
    C_large, S_large = fn._gt_boundary_and_area(gt_large)
    alpha_large = (1.0 - 2.0 * C_large / S_large.clamp_min(1.0)).clamp(0.0, 0.8)

    # Small thin block: high C/S → alpha close to 0
    gt_thin = torch.zeros(1, 32, 32)
    gt_thin[0, 14:18, 4:28] = 1.0  # 4x24 = 96 voxels, perimeter ~56
    C_thin, S_thin = fn._gt_boundary_and_area(gt_thin)
    alpha_thin = (1.0 - 2.0 * C_thin / S_thin.clamp_min(1.0)).clamp(0.0, 0.8)

    assert 0.0 <= alpha_large.item() <= 0.8
    assert 0.0 <= alpha_thin.item() <= 0.8
    assert alpha_large.item() > alpha_thin.item(), (
        f"Large solid block should have larger alpha than thin block. "
        f"Got large={alpha_large.item():.3f}, thin={alpha_thin.item():.3f}"
    )


# ---------------------------------------------------------------------------
# 6. CombinedLoss integration
# ---------------------------------------------------------------------------

def test_combined_loss_zero_weight_no_op():
    """CombinedLoss with w_boundary_dou=0 must not invoke the loss or add a
    breakdown key, preserving backward compatibility with existing configs."""
    loss = CombinedLoss(
        cfg_or_w_dice=0.5, w_focal=0.5, w_iou=0.0, w_modality=0.0,
        w_boundary=0.0, w_dice_topk=0.0, w_deepsup=0.0, w_hd_loss=0.0,
        w_boundary_dou=0.0,
    )
    pred = torch.randn(2, 16, 16)
    target = torch.zeros(2, 16, 16)
    target[0, 4:12, 4:12] = 1.0
    total, breakdown = loss(pred, target)
    assert "boundary_dou" not in breakdown
    assert torch.isfinite(total)


def test_combined_loss_positive_weight_adds_component():
    loss = CombinedLoss(
        cfg_or_w_dice=0.5, w_focal=0.5, w_iou=0.0, w_modality=0.0,
        w_boundary=0.0, w_dice_topk=0.0, w_deepsup=0.0, w_hd_loss=0.0,
        w_boundary_dou=0.1,
    )
    pred = torch.randn(2, 16, 16, requires_grad=True)
    target = torch.zeros(2, 16, 16)
    target[0, 4:12, 4:12] = 1.0
    total, breakdown = loss(pred, target)
    assert "boundary_dou" in breakdown
    assert 0.0 <= breakdown["boundary_dou"] <= 1.0 + 1e-3
    total.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


# ---------------------------------------------------------------------------
# Allow direct execution as a script
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for name in [
        "test_paper_fig2_exact",
        "test_gradient_flow",
        "test_empty_gt_is_finite",
        "test_perfect_match_loss_small",
        "test_alpha_is_in_range_and_geometry_aware",
        "test_combined_loss_zero_weight_no_op",
        "test_combined_loss_positive_weight_adds_component",
    ]:
        fn = globals()[name]
        try:
            fn()
            print(f"[PASS] {name}")
        except AssertionError as e:
            print(f"[FAIL] {name}: {e}")
            raise
