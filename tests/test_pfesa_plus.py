import torch
from models.pfesa_plus import PFESAPlus


def test_pfesa_plus_shape():
    m = PFESAPlus()
    x = torch.randn(2, 256, 16, 16)
    y = m(x)
    assert y.shape == x.shape


def test_pfesa_plus_learnable_params():
    m = PFESAPlus()
    params = {name: p for name, p in m.named_parameters() if p.requires_grad}
    assert set(params.keys()) == {"alpha", "cutoff_logit", "steepness_log"}
    assert sum(p.numel() for p in params.values()) == 3


def test_pfesa_plus_identity_at_init_with_cutoff_1():
    m = PFESAPlus()
    with torch.no_grad():
        m.cutoff_logit.fill_(10.0)  # sigmoid(10) ≈ 1 → mask ≈ 0 everywhere
        m.alpha.fill_(0.0)            # amplification 0 → identity
    x = torch.randn(1, 256, 8, 8)
    y = m(x)
    assert torch.allclose(y, x, atol=1e-5)


def test_pfesa_plus_backward():
    m = PFESAPlus()
    x = torch.randn(1, 256, 16, 16, requires_grad=True)
    y = m(x)
    y.sum().backward()
    for p in m.parameters():
        assert p.grad is not None
