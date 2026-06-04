"""PA-CODE identity-init sanity test.

At step 0:
  - γ = 1, β = 0   (FiLM identity)
  - f_θ output zero-init   →   dh/dt = 0 everywhere
  - merge.weight zero-init →   ode_context = 0
  - residual:                  output = features (exact)

This script verifies the implementation matches the design promise:
PA-CODE at init must be a pure pass-through of the encoder features.
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.ode_cross_slice import (
    PACodeBidirectionalNeuralODE,
    OrganConditionedBidirectionalNeuralODE,
)


def test_pa_code_identity():
    torch.manual_seed(0)
    B, D, C, H, W = 2, 8, 256, 16, 16
    feats = torch.randn(B, D, C, H, W)
    organ_id = torch.tensor([6, 1])  # liver, spleen

    ode = PACodeBidirectionalNeuralODE(
        dim=C, n_organs=15, organ_emb_dim=64,
        ode_hidden=64, n_freqs=4, n_pos_freqs=4, substeps=4,
    ).eval()

    with torch.no_grad():
        out = ode(feats, organ_id)

    diff = (out - feats).abs().max().item()
    print(f"[pa-code identity] |out - feats|_inf = {diff:.3e}")
    assert diff < 1e-5, f"PA-CODE identity violated: max-abs-diff={diff}"

    # Verify γ = 1, β = 0 at init for any organ_id
    e = ode.ode_fwd.organ_embed(torch.tensor([0, 1, 6, 14]))
    g = ode.ode_fwd.gamma_head(e)
    b = ode.ode_fwd.beta_head(e)
    print(f"[pa-code identity] gamma - 1 max-abs = {(g - 1.0).abs().max().item():.3e}")
    print(f"[pa-code identity] beta      max-abs = {b.abs().max().item():.3e}")
    assert (g - 1.0).abs().max() < 1e-5
    assert b.abs().max() < 1e-5

    # And the cache_trajectories=True path also works
    with torch.no_grad():
        _ = ode(feats, organ_id, cache_trajectories=True)
    tf, tb = ode.get_cached_trajectories()
    print(f"[pa-code identity] traj_fwd shape = {tuple(tf.shape)}, "
          f"traj_bwd shape = {tuple(tb.shape)}")
    assert tf.shape == (B * H * W, D, C)
    assert tb.shape == (B * H * W, D, C)

    print("[pa-code identity] PASS")


def test_param_count():
    pa  = PACodeBidirectionalNeuralODE(dim=256, n_organs=15, organ_emb_dim=64,
                                        ode_hidden=128, n_freqs=6, n_pos_freqs=4, substeps=4)
    old = OrganConditionedBidirectionalNeuralODE(dim=256, n_organs=15, organ_emb_dim=64,
                                                  ode_hidden=128, n_freqs=6, substeps=4)
    pa_n  = sum(p.numel() for p in pa.parameters())
    old_n = sum(p.numel() for p in old.parameters())
    print(f"[param count] PA-CODE  = {pa_n:,}")
    print(f"[param count] old ODE  = {old_n:,}")
    print(f"[param count] delta    = {pa_n - old_n:+,}")


if __name__ == "__main__":
    test_pa_code_identity()
    test_param_count()
