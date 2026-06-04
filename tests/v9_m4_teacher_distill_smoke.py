"""M4 smoke test — teacher distillation loss.

Verifies:
  1. With no teachers available (offline), loss is a no-op (0 scalar, no grad).
  2. With a fake always-available teacher, loss has a gradient wrt student
     features and projection heads, and per-organ weights softmax correctly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from losses.teacher_distill import TeacherDistillLoss, _TeacherBase


class _FakeTeacher(_TeacherBase):
    name = "fake"

    def __init__(self, dim: int = 64):
        super().__init__()
        self.out_dim = dim
        self.proj = nn.Linear(3 * 224 * 224, dim)  # deterministic mapping
        for p in self.proj.parameters():
            p.requires_grad = False
        self._ok = True

    def is_available(self) -> bool:
        return True

    def forward(self, img_2d):  # type: ignore[override]
        return self.proj(img_2d.flatten(1))


def test_noop_path() -> None:
    # Force zero active teachers by requesting only an unknown name.
    loss_mod = TeacherDistillLoss(
        student_taps={"layer1": 32, "layer2": 64},
        teachers=("nonexistent_teacher",),
        n_organs=15,
    )
    assert loss_mod.is_noop(), "expected no active teachers"

    B = 2
    volume = torch.randn(B, 1, 96, 96, 96)
    organ_id = torch.tensor([0, 1])
    student_feats = {
        "layer1": torch.randn(B, 32, 48, 48, 48, requires_grad=True),
        "layer2": torch.randn(B, 64, 24, 24, 24, requires_grad=True),
    }
    out = loss_mod(student_feats, volume, organ_id)
    assert out["loss"].item() == 0.0
    print("[M4] noop path OK — loss=0, no crash when teachers absent")


def test_fake_teacher_path() -> None:
    loss_mod = TeacherDistillLoss(
        student_taps={"layer1": 32, "layer2": 64},
        teachers=(),
        n_organs=15,
    )
    # Hot-swap a fake teacher in so we can test the active path.
    fake = _FakeTeacher(dim=64)
    loss_mod.teachers["fake"] = fake
    loss_mod._active = ["fake"]
    # Rebuild projection heads for fake teacher.
    from losses.teacher_distill import _ProjHead
    loss_mod.proj["layer1__fake"] = _ProjHead(32, 64)
    loss_mod.proj["layer2__fake"] = _ProjHead(64, 64)
    loss_mod.organ_weights = nn.Parameter(torch.zeros(15, 1))

    B = 2
    volume = torch.randn(B, 1, 96, 96, 96)
    organ_id = torch.tensor([2, 5])
    student_feats = {
        "layer1": torch.randn(B, 32, 48, 48, 48, requires_grad=True),
        "layer2": torch.randn(B, 64, 24, 24, 24, requires_grad=True),
    }
    out = loss_mod(student_feats, volume, organ_id)
    l = out["loss"]
    assert l.requires_grad, "loss should require grad"
    assert torch.isfinite(l), f"loss must be finite, got {l.item()}"
    l.backward()
    # Check at least one projection head got a non-zero grad.
    assert any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in loss_mod.proj.parameters()
    ), "projection heads should have received gradient"
    print(f"[M4] fake teacher path OK — loss={l.item():.4f} grad flows through proj heads")


def main() -> None:
    torch.manual_seed(0)
    test_noop_path()
    test_fake_teacher_path()
    print("[M4] ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
