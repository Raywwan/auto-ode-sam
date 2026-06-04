"""M5 smoke test — cross-modal InfoNCE loss.

Verifies noop-when-BiomedCLIP-missing path AND a patched-CLIP path where we
inject a fake text-feature cache so we don't need the network.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from losses.cross_modal_contrastive import CrossModalInfoNCE


def test_noop_path() -> None:
    loss_mod = CrossModalInfoNCE(embed_dim=128, n_organs=15)
    # Force BiomedCLIP unavailable by marking the lazy-load as failed.
    loss_mod._available = False
    B = 4
    feat = torch.randn(B, 128)
    oid = torch.tensor([0, 1, 2, 3])
    out = loss_mod(feat, oid)
    assert out["loss"].item() == 0.0
    print("[M5] noop path OK — loss=0 when BiomedCLIP is unavailable")


def test_patched_clip_path() -> None:
    loss_mod = CrossModalInfoNCE(embed_dim=128, n_organs=15)
    # Pretend CLIP loaded by injecting fake text features + a matching img_proj.
    C_t = 64
    torch.manual_seed(0)
    fake_text = F.normalize(torch.randn(15, C_t), dim=-1)
    loss_mod._text_feats = fake_text
    loss_mod.img_proj = nn.Linear(128, C_t)
    loss_mod._available = True

    B = 8
    feat = torch.randn(B, 128, requires_grad=True)
    oid = torch.randint(0, 15, (B,))
    out = loss_mod(feat, oid)
    assert out["loss"].requires_grad
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert feat.grad is not None and feat.grad.abs().sum() > 0
    print(f"[M5] patched CLIP path OK — loss={out['loss'].item():.4f}, "
          f"acc@1={out['acc@1'].item():.3f}")

    # Test hard-negative weighting: construct a batch where target=1 (right
    # kidney) and the logits would otherwise favour 2 (left kidney). The
    # hard-negative weight should make the loss *larger* than without weighting.
    B = 1
    feat = torch.zeros(B, 128)
    # Craft feat so img_proj(feat) aligns closely with text index 2, less with 1.
    with torch.no_grad():
        target_dir = fake_text[2] * 2.0 + fake_text[1] * 1.0
        # Solve img_proj(feat) ≈ target_dir via pseudo-inverse.
        feat = torch.linalg.lstsq(
            loss_mod.img_proj.weight, (target_dir - loss_mod.img_proj.bias).unsqueeze(1)
        ).solution.squeeze(1).unsqueeze(0)
    oid = torch.tensor([1])
    l_with_hn = loss_mod(feat, oid)["loss"].item()

    saved_nw = loss_mod.neg_weight.clone()
    loss_mod.neg_weight = torch.ones_like(loss_mod.neg_weight)
    loss_mod.neg_weight.fill_diagonal_(0.0)
    l_plain = loss_mod(feat, oid)["loss"].item()
    loss_mod.neg_weight = saved_nw
    print(f"[M5] hard-neg weighting: loss_with_hn={l_with_hn:.4f} vs plain={l_plain:.4f}")
    assert l_with_hn >= l_plain - 1e-4, "HN weighting should not reduce loss for confusion pair"


def main() -> None:
    torch.manual_seed(0)
    test_noop_path()
    test_patched_clip_path()
    print("[M5] ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
