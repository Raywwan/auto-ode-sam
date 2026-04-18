import torch
from models.flow_cross_slice import FlowMatchedOrganConditionedODE


def test_flow_ode_output_shape():
    m = FlowMatchedOrganConditionedODE(dim=256, n_organs=15, ode_hidden=128, n_freqs=6, substeps=4)
    x = torch.randn(2, 8, 256, 16, 16)
    organ_id = torch.tensor([3, 7])
    y, flow_targets = m(x, organ_id, return_flow_targets=True)
    assert y.shape == x.shape
    # flow_targets should let us compute L_flow
    assert "fwd_pairs" in flow_targets and "bwd_pairs" in flow_targets


def test_flow_ode_identity_at_init():
    torch.manual_seed(0)
    m = FlowMatchedOrganConditionedODE(dim=64, n_organs=15, ode_hidden=16, n_freqs=4, substeps=2)
    x = torch.randn(1, 4, 64, 4, 4)
    organ_id = torch.tensor([1])
    y, _ = m(x, organ_id, return_flow_targets=False)
    assert torch.allclose(y, x, atol=1e-5), "At init, ODE must be identity (zero-init merge.weight)"


def test_flow_loss_nonzero_at_init():
    """L_flow must receive gradient from step 1 — the whole point of flow matching."""
    torch.manual_seed(0)
    m = FlowMatchedOrganConditionedODE(dim=64, n_organs=15, ode_hidden=16, n_freqs=4, substeps=2)
    x = torch.randn(1, 4, 64, 4, 4)
    organ_id = torch.tensor([1])
    y, ft = m(x, organ_id, return_flow_targets=True)
    loss = _flow_loss_reference(ft)
    assert loss.item() > 0.0
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
    assert has_grad, "Flow loss must produce nonzero gradients at init"


def _flow_loss_reference(ft):
    import torch.nn.functional as F
    return 0.5 * (F.mse_loss(ft["fwd_pairs"]["pred"], ft["fwd_pairs"]["target"])
                  + F.mse_loss(ft["bwd_pairs"]["pred"], ft["bwd_pairs"]["target"]))
