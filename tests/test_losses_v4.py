import torch
from training.losses_v4 import V4MultiOrganLoss


def test_mask_loss_ignores_absent_organs():
    loss_fn = V4MultiOrganLoss(lambda_flow=0.0, lambda_deepsup=0.0, lambda_anatomy=0.0)
    pred = torch.randn(2, 15, 64, 64)
    gt = torch.zeros(2, 15, 64, 64, dtype=torch.uint8)
    gt[0, 6, 16:32, 16:32] = 1
    gt[1, 3, 20:40, 20:40] = 1
    present = torch.zeros(2, 15, dtype=torch.bool)
    present[0, 6] = True
    present[1, 3] = True
    out = loss_fn(pred, gt, present, flow_targets=None, deepsup_logits=None, anatomy_adjacency=None, anatomy_adjacency_init=None)
    assert "mask" in out["components"]
    assert torch.isfinite(out["loss"])


def test_flow_loss_is_zero_when_targets_match():
    loss_fn = V4MultiOrganLoss(lambda_flow=1.0, lambda_deepsup=0.0, lambda_anatomy=0.0)
    pred = torch.zeros(1, 15, 8, 8)
    gt = torch.zeros(1, 15, 8, 8, dtype=torch.uint8)
    present = torch.zeros(1, 15, dtype=torch.bool); present[0, 0] = True
    flow_targets = {
        "fwd_pairs": {"pred": torch.randn(4, 3, 16), "target": torch.randn(4, 3, 16)},
        "bwd_pairs": {"pred": torch.randn(4, 3, 16), "target": torch.randn(4, 3, 16)},
    }
    for d in ("fwd_pairs", "bwd_pairs"):
        flow_targets[d]["pred"] = flow_targets[d]["target"].clone()
    out = loss_fn(pred, gt, present, flow_targets=flow_targets,
                  deepsup_logits=None, anatomy_adjacency=None, anatomy_adjacency_init=None)
    assert abs(out["components"]["flow"]) < 1e-6
