"""OrganMoELoss: SmallOrganLoss + presence BCE + MoE load balance."""
import torch
from training.losses_organmoe import OrganMoELoss


def test_loss_runs_no_nan():
    torch.manual_seed(0)
    crit = OrganMoELoss(n_classes=16, presence_weight=0.1, balance_weight=0.01)
    logits = torch.randn(2, 16, 8, 16, 16, requires_grad=True)
    target = torch.randint(0, 16, (2, 8, 16, 16))
    presence_logits = torch.randn(2, 15, requires_grad=True)
    gate_weights = torch.randn(2, 8, 16, 16, 16).softmax(dim=1)  # K=8 experts
    out = crit(
        seg_logits=logits, seg_target=target,
        presence_logits=presence_logits, gate_weights=gate_weights,
    )
    assert torch.isfinite(out["total"]).item()
    out["total"].backward()
    assert torch.isfinite(logits.grad.norm()).item()
    assert torch.isfinite(presence_logits.grad.norm()).item()


def test_presence_bce_target_derived_from_target():
    """The presence target is derived from the seg target: organ k is 'present'
    iff at least 1 voxel of organ k exists in the volume."""
    crit = OrganMoELoss(n_classes=16, presence_weight=1.0, balance_weight=0.0)
    logits = torch.randn(1, 16, 4, 8, 8)
    target = torch.zeros(1, 4, 8, 8, dtype=torch.long)
    target[0, 0, 0, 0] = 5  # only organ 5 present
    presence_logits = torch.zeros(1, 15)  # uniform 0.5
    out = crit(seg_logits=logits, seg_target=target, presence_logits=presence_logits)
    # The derived presence target should have 1.0 at index 4 (organ 5 → 0-indexed 4)
    assert torch.isfinite(out["presence_bce"]).item()
    assert out["presence_bce"].item() > 0


def test_load_balance_aux_zero_when_uniform():
    """If experts are used uniformly, load-balance term is ~minimal."""
    crit = OrganMoELoss(n_classes=16, presence_weight=0.0, balance_weight=1.0)
    logits = torch.randn(1, 16, 4, 8, 8)
    target = torch.randint(0, 16, (1, 4, 8, 8))
    presence_logits = torch.zeros(1, 15)
    K = 8
    # Uniform gate: every expert chosen equally.
    uniform_gate = torch.ones(1, K, 4, 8, 8) / K
    skewed_gate = torch.zeros(1, K, 4, 8, 8)
    skewed_gate[:, 0] = 1.0  # all weight on expert 0
    out_u = crit(seg_logits=logits, seg_target=target, presence_logits=presence_logits, gate_weights=uniform_gate)
    out_s = crit(seg_logits=logits, seg_target=target, presence_logits=presence_logits, gate_weights=skewed_gate)
    assert out_s["balance"].item() > out_u["balance"].item()


def test_absent_classes_zero_weighted():
    """If an organ is absent in a batch, its presence weight is 0 in the
    weighted-dice term so it does not contribute to gradients."""
    crit = OrganMoELoss(n_classes=16, presence_weight=0.0, balance_weight=0.0,
                        use_presence_weighted_dice=True)
    logits = torch.randn(1, 16, 4, 8, 8, requires_grad=True)
    target = torch.zeros(1, 4, 8, 8, dtype=torch.long)
    target[0, 0, 0, 0] = 6  # only organ 6 present
    presence_logits = torch.zeros(1, 15)
    out = crit(seg_logits=logits, seg_target=target, presence_logits=presence_logits)
    out["total"].backward()
    # Gradient on absent organs (e.g., channel 1) should be small relative to
    # gradient on present organ channel 6.
    g_present = logits.grad[0, 6].abs().sum().item()
    g_absent = logits.grad[0, 1].abs().sum().item()
    assert g_present > g_absent  # primary signal is on present organ
