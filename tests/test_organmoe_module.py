"""OrganMoE-3D unit tests.

Each test runs on CPU with a tiny shape so it stays under 1 second.
A separate VRAM smoke is in scripts/ for the GPU.
"""
import pytest
import torch
import torch.nn as nn
from models.organmoe_3d import MoELoRALinear, PresenceHead, PresenceConditionedRouter


def test_moe_lora_linear_forward_shape():
    base = nn.Linear(64, 96)
    moe = MoELoRALinear(base=base, n_experts=4, rank=8, alpha=8.0, n_organs=15, top_k=2)
    x = torch.randn(2, 10, 64)
    organ_presence = torch.rand(2, 15)
    out, gate = moe.forward_with_gate(x, organ_presence=organ_presence)
    assert out.shape == (2, 10, 96)
    assert gate.shape == (2, 4, 10) or gate.shape == (2, 10, 4)  # implementation choice


def test_moe_base_frozen():
    base = nn.Linear(32, 32)
    moe = MoELoRALinear(base=base, n_experts=4, rank=4, alpha=4.0, n_organs=15, top_k=2)
    for p in moe.base.parameters():
        assert not p.requires_grad


def test_top_k_routing_only_uses_k_experts_per_token():
    base = nn.Linear(32, 32)
    moe = MoELoRALinear(base=base, n_experts=8, rank=4, alpha=4.0, n_organs=15, top_k=2)
    x = torch.randn(1, 5, 32)
    organ_presence = torch.zeros(1, 15)
    _, gate = moe.forward_with_gate(x, organ_presence=organ_presence)
    # gate is (B, K, T) softmax over K; top-2 means only 2 nonzero per token.
    if gate.shape[1] == 8:
        nonzero = (gate > 0).sum(dim=1)  # (B, T)
    else:
        nonzero = (gate > 0).sum(dim=2)
    assert (nonzero <= 2).all()


def test_presence_head_output_shape():
    feat = torch.randn(2, 256, 6, 6, 6)  # (B, C, D, H, W) low-res encoder feature
    head = PresenceHead(in_channels=256, n_organs=15)
    out = head(feat)
    assert out.shape == (2, 15)


def test_router_conditions_on_presence():
    """Same input features but different presence vectors should give
    different routing decisions (router actually uses presence)."""
    base = nn.Linear(32, 32)
    moe = MoELoRALinear(base=base, n_experts=8, rank=4, alpha=4.0, n_organs=15, top_k=2)
    x = torch.randn(1, 4, 32)
    p_a = torch.zeros(1, 15)
    p_b = torch.ones(1, 15)
    _, g_a = moe.forward_with_gate(x, organ_presence=p_a)
    _, g_b = moe.forward_with_gate(x, organ_presence=p_b)
    assert not torch.allclose(g_a, g_b)


def test_grad_flows_through_experts():
    base = nn.Linear(32, 32)
    moe = MoELoRALinear(base=base, n_experts=4, rank=4, alpha=4.0, n_organs=15, top_k=2)
    x = torch.randn(1, 4, 32)
    p = torch.zeros(1, 15, requires_grad=True)
    y, _ = moe.forward_with_gate(x, organ_presence=p)
    y.sum().backward()
    # Some expert lora_A should have nonzero grad.
    grads = [e.lora_A.grad for e in moe.experts]
    has_grad = any(g is not None and g.abs().sum() > 0 for g in grads)
    assert has_grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU only")
def test_swin_proposer_with_organmoe_forward():
    from models.swin_unetr_3d import SwinUNETRProposer
    # patch_size=64 ensures deepest feature map is 4^3 (not 1^3), avoiding
    # the InstanceNorm3d ">1 spatial element" constraint during training mode.
    proposer = SwinUNETRProposer(
        n_organs=15, patch_size=64, feature_size=24, use_v2=True,
        pretrained_weights=None, lora_rank=0, use_organmoe=True,
        moe_n_experts=4, moe_rank=4, moe_alpha=4.0,
    ).cuda()
    x = torch.randn(1, 1, 64, 64, 64).cuda()
    out = proposer(x)
    assert out["logits"].shape == (1, 15, 64, 64, 64)
    assert torch.isfinite(out["logits"]).all()
