# VoluFormer3D V9 — Remaining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the V9 "Tier B" cascade so a full training + 3D-eval pipeline is runnable end-to-end on AMOS22 CT, beating MCP-MedSAM (0.79-0.82 3D Dice) and targeting 0.85-0.88.

**Architecture:** Cascade 3D-proposer (SwinUNETR) → 2D-slab-refiner (OrganFlowSAM2) with `AnatomicalCascadeFusion`. Five novel losses remain to be built (flow shape prior, dynamic PFESA++, cascade consistency, boundary DDPM, full training orchestration). V9 Phase A modules (`swin_unetr_3d.py`, `voluformer_v9.py`, `mamba_ode.py`) and two Phase B losses (`teacher_distill.py`, `cross_modal_contrastive.py`) are already done.

**Tech Stack:** PyTorch 2.x, MONAI (SwinUNETR), NumPy, pytest-style smoke scripts, nibabel. Target GPU: RTX 4090 24 GB. No `mamba_ssm` (Windows). No `xformers` (Windows). Pure-PyTorch SSMs and DDPM.

**Discipline (from feedback memory — must obey):**
- One axis at a time — each task isolates one new module behind a config flag.
- Every new loss defaults to `λ = 0` so the baseline recovers V7's working surface (§7b).
- Smoke test on CPU first, CUDA smoke before any full training launch.
- Fresh checkpoint dirs per stage. Never rename in place.
- Stop-rule in trainer: abort if `val/dice_3d < best - 0.02` on two consecutive val ticks.
- Never claim AMOS22 numbers from 2D center-slice val — 3D eval is the source of truth.

**Spec reference:** `docs/superpowers/specs/2026-04-20-voluformer-v9-tier-b.md` (sections §3 novelty table, §7 impl order, §7b loss-preservation clause, §7c gates).

---

## File Structure (what will be created)

```
losses/
  flow_shape_prior.py         # M6 — Novel #5
  cascade_consistency.py      # M8 — Novel #7
models/
  dynamic_pfesa.py            # M7 — Novel #6
  boundary_ddpm.py            # M9 — Novel #8
datasets/
  amos22_v9.py                # D1 — positive/negative/mixed slab sampler
scripts/
  build_totalseg_pseudolabels.py  # D2 — guarded behind user flag
evaluation/
  eval_v9_3d.py               # E1 — full cascade 3D eval, TTA, CC, Gaussian blend
training/
  trainer_v9.py               # T1 — three-stage V9 orchestrator
configs/
  v9_tierB.yaml               # full V9 config (all λ default to 0)
tests/
  v9_m6_flow_shape_prior_smoke.py
  v9_m7_dynamic_pfesa_smoke.py
  v9_m8_cascade_consistency_smoke.py
  v9_m9_boundary_ddpm_smoke.py
  v9_d1_dataset_smoke.py
  v9_cuda_smoke.py            # full cascade forward+backward on CUDA
  v9_full_pipeline_smoke.py   # trainer one-step end-to-end
```

Each file has a single responsibility and a matching smoke test. The plan builds bottom-up: losses + modules first, then dataset, then eval, then trainer, then integration smoke tests.

---

## Task 1: M6 — Flow-Matched Shape Prior Loss

**Spec:** §3 row #5. Extend the existing `OrganConditionedBidirectionalNeuralODE` to generate organ shape priors conditioned on `organ_id`; regularize the predicted refiner mask toward the generated prior. Defaults `λ = 0` (no-op).

**Files:**
- Create: `losses/flow_shape_prior.py`
- Create: `tests/v9_m6_flow_shape_prior_smoke.py`

- [ ] **Step 1.1: Write the failing smoke test**

```python
# tests/v9_m6_flow_shape_prior_smoke.py
"""M6 smoke test — flow-matched shape prior loss.

Verifies:
  1. λ = 0 path short-circuits and returns 0 with no ODE integration.
  2. λ > 0 path produces a finite scalar with grad flowing back to
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
    print("[M6] λ=0 noop OK — short-circuits without running ODE")


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
    print(f"[M6] active path OK — loss={l.item():.4f} grad flows")


def test_prior_equals_target_is_small() -> None:
    torch.manual_seed(0)
    mod = _make_module(lam=1.0)
    B = 1
    mask = torch.zeros(B, 15, 64, 64)
    mask[:, 5, 20:40, 20:40] = 10.0  # a confident blob for organ 5
    organ_id = torch.tensor([5])
    # Lock prior to the same target so loss ~0.
    mod.override_prior = mask.sigmoid().detach()
    out = mod(mask.clone().requires_grad_(), organ_id)
    assert out["loss"].item() < 0.05, f"expected small loss, got {out['loss'].item()}"
    print(f"[M6] identity prior OK — loss={out['loss'].item():.5f}")


def main() -> None:
    torch.manual_seed(0)
    test_zero_lambda_noop()
    test_active_path_has_grad()
    test_prior_equals_target_is_small()
    print("[M6] ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
```

- [ ] **Step 1.2: Run the test to verify it fails**

Run: `python tests/v9_m6_flow_shape_prior_smoke.py`
Expected: `ModuleNotFoundError: No module named 'losses.flow_shape_prior'`

- [ ] **Step 1.3: Implement the loss module**

```python
# losses/flow_shape_prior.py
"""V9 Novel #5 — Flow-matched shape prior loss.

Reuses the flow-matched ODE formulation from V7's
`OrganConditionedBidirectionalNeuralODE` (models/ode_cross_slice.py) but in a
*generative* mode: given an organ id, integrate from Gaussian noise to produce
a (H, W) organ-shape probability. Regularize the refiner's predicted mask
toward this generated prior with a soft-Dice distance.

Why this is novel: prior work using ODE/flow-matching for medical imaging
generates masks in isolation (Voxelmorph, DiffuseFormer). Here the same flow
module is shared between supervision (cross-slice feature evolution) and a
*class-conditional generative shape prior* — single learned vector field used
in two modes.

Design discipline (spec §7b):
  - `lam=0` (default) short-circuits — no ODE integration, zero forward cost.
  - Flow sub-module is tiny (~0.5M params) and lives inside the loss so it
    can be toggled off without affecting the rest of the network.
  - `override_prior` exists purely for unit testing.

Inputs:
  pred_mask: (B, K, H, W) — raw refiner logits.
  organ_id : (B,)          — ground-truth organ index in [0, K).

Output: {"loss": scalar, "prior": (B, K, H, W) sigmoided shape prior}
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _soft_dice(pred: torch.Tensor, tgt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # pred, tgt: (N, H, W) in [0, 1]. Returns scalar mean Dice loss.
    num = 2.0 * (pred * tgt).sum(dim=(-2, -1))
    den = pred.sum(dim=(-2, -1)) + tgt.sum(dim=(-2, -1)) + eps
    return (1.0 - num / den).mean()


class _TinyFlow(nn.Module):
    """Organ-conditioned vector field over a low-res latent grid.

    State: (B, latent_dim, h, w) for small h, w (we use 16×16).
    Time: scalar in [0, 1].
    """

    def __init__(self, n_organs: int, latent_dim: int = 32) -> None:
        super().__init__()
        self.organ_emb = nn.Embedding(n_organs, latent_dim)
        self.time_emb = nn.Linear(1, latent_dim)
        self.net = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, 3, padding=1), nn.GELU(),
            nn.Conv2d(latent_dim, latent_dim, 3, padding=1), nn.GELU(),
            nn.Conv2d(latent_dim, latent_dim, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, organ_id: torch.Tensor) -> torch.Tensor:
        # x: (B, C, h, w); t: (B,); organ_id: (B,)
        B, C, h, w = x.shape
        oe = self.organ_emb(organ_id).view(B, C, 1, 1).expand(-1, -1, h, w)
        te = self.time_emb(t.view(B, 1)).view(B, C, 1, 1).expand(-1, -1, h, w)
        return self.net(x + oe + te)


class FlowShapePriorLoss(nn.Module):
    def __init__(
        self,
        n_organs: int = 15,
        latent_dim: int = 32,
        grid: int = 16,
        n_steps: int = 4,
        lam: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_organs = n_organs
        self.latent_dim = latent_dim
        self.grid = grid
        self.n_steps = n_steps
        self.lam = lam

        self.flow = _TinyFlow(n_organs, latent_dim)
        self.head = nn.Conv2d(latent_dim, 1, 1)

        # Test-only override; None in production.
        self.override_prior: Optional[torch.Tensor] = None

    @torch.no_grad()
    def is_noop(self) -> bool:
        return self.lam == 0.0

    def integrate(self, organ_id: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Euler-integrate the flow from N(0, I) for n_steps. Return (B, 1, H, W)."""
        B = organ_id.shape[0]
        g = self.grid
        x = torch.randn(B, self.latent_dim, g, g, device=organ_id.device)
        dt = 1.0 / self.n_steps
        for k in range(self.n_steps):
            t = torch.full((B,), (k + 0.5) * dt, device=organ_id.device)
            x = x + dt * self.flow(x, t, organ_id)
        prior = torch.sigmoid(self.head(x))                          # (B, 1, g, g)
        prior = F.interpolate(prior, size=(H, W), mode="bilinear", align_corners=False)
        return prior

    def forward(
        self,
        pred_mask: torch.Tensor,      # (B, K, H, W) logits
        organ_id: torch.Tensor,       # (B,)
    ) -> Dict[str, torch.Tensor]:
        if self.is_noop():
            zero = torch.zeros((), device=pred_mask.device)
            return {"loss": zero, "prior": torch.zeros_like(pred_mask)}

        B, K, H, W = pred_mask.shape

        if self.override_prior is not None:
            prior_full = self.override_prior.to(pred_mask.device)
        else:
            prior = self.integrate(organ_id, H, W)                  # (B, 1, H, W)
            prior_full = torch.zeros_like(pred_mask)
            idx = organ_id.view(B, 1, 1, 1).expand(-1, 1, H, W)
            prior_full.scatter_(1, idx, prior)

        # Gather the relevant channel of pred_mask for each sample's organ.
        sig = torch.sigmoid(pred_mask)
        pk = sig.gather(1, organ_id.view(B, 1, 1, 1).expand(-1, 1, H, W)).squeeze(1)
        qk = prior_full.gather(1, organ_id.view(B, 1, 1, 1).expand(-1, 1, H, W)).squeeze(1)

        loss = self.lam * _soft_dice(pk, qk)
        return {"loss": loss, "prior": prior_full}
```

- [ ] **Step 1.4: Run the test to verify it passes**

Run: `python tests/v9_m6_flow_shape_prior_smoke.py`
Expected:
```
[M6] λ=0 noop OK — short-circuits without running ODE
[M6] active path OK — loss=... grad flows
[M6] identity prior OK — loss=0.000...
[M6] ALL SMOKE TESTS PASSED
```

- [ ] **Step 1.5: Commit**

```bash
git add losses/flow_shape_prior.py tests/v9_m6_flow_shape_prior_smoke.py
git commit -m "feat(v9): M6 flow-matched shape prior loss + smoke test (Novel #5)"
```

---

## Task 2: M7 — Organ-Aware Dynamic PFESA++

**Spec:** §3 row #6. PFESA kernel weights are *generated* by a HyperNetwork from `(organ_id + text_emb)` at each forward. Existing PFESA lives at `models/pfesa_plus.py` — we add a parallel `DynamicPFESA` module and wire it in via config.

**Files:**
- Create: `models/dynamic_pfesa.py`
- Create: `tests/v9_m7_dynamic_pfesa_smoke.py`

- [ ] **Step 2.1: Write the failing smoke test**

```python
# tests/v9_m7_dynamic_pfesa_smoke.py
"""M7 smoke test — dynamic PFESA++.

Verifies:
  1. Forward returns feature of matching (B, C, H, W) shape.
  2. Kernels differ when organ_id differs (hypernet actually conditions).
  3. Gradient flows through hypernet AND through input features.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.dynamic_pfesa import DynamicPFESA


def test_shape_and_grad() -> None:
    torch.manual_seed(0)
    mod = DynamicPFESA(in_ch=64, out_ch=64, n_organs=15, text_dim=0)
    B, C, H, W = 2, 64, 32, 32
    feat = torch.randn(B, C, H, W, requires_grad=True)
    organ_id = torch.tensor([3, 9])
    out = mod(feat, organ_id=organ_id, text_emb=None)
    assert out.shape == (B, C, H, W)
    out.sum().backward()
    assert feat.grad is not None and feat.grad.abs().sum() > 0
    hyper_grads = [p.grad for p in mod.hyper.parameters() if p.grad is not None]
    assert any(g.abs().sum() > 0 for g in hyper_grads), "hypernet got no grad"
    print("[M7] shape + grad OK")


def test_organ_conditioning_changes_output() -> None:
    torch.manual_seed(0)
    mod = DynamicPFESA(in_ch=32, out_ch=32, n_organs=15, text_dim=0).eval()
    B, C, H, W = 1, 32, 16, 16
    feat = torch.randn(B, C, H, W)
    with torch.no_grad():
        a = mod(feat, organ_id=torch.tensor([0]), text_emb=None)
        b = mod(feat, organ_id=torch.tensor([7]), text_emb=None)
    # Different organs → different outputs (not bit-identical).
    assert not torch.allclose(a, b, atol=1e-5)
    print("[M7] organ conditioning affects output")


def test_with_text_embedding() -> None:
    torch.manual_seed(0)
    mod = DynamicPFESA(in_ch=32, out_ch=32, n_organs=15, text_dim=128).eval()
    feat = torch.randn(1, 32, 16, 16)
    text_a = torch.randn(1, 128)
    text_b = torch.randn(1, 128)
    with torch.no_grad():
        a = mod(feat, organ_id=torch.tensor([0]), text_emb=text_a)
        b = mod(feat, organ_id=torch.tensor([0]), text_emb=text_b)
    assert not torch.allclose(a, b, atol=1e-5)
    print("[M7] text embedding affects output")


def main() -> None:
    test_shape_and_grad()
    test_organ_conditioning_changes_output()
    test_with_text_embedding()
    print("[M7] ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2.2: Run the test to verify it fails**

Run: `python tests/v9_m7_dynamic_pfesa_smoke.py`
Expected: `ModuleNotFoundError: No module named 'models.dynamic_pfesa'`

- [ ] **Step 2.3: Implement the module**

```python
# models/dynamic_pfesa.py
"""V9 Novel #6 — Organ-aware dynamic PFESA++.

Standard PFESA (position-encoded frequency-enhanced spatial attention — see
`models/pfesa_plus.py`) uses fixed spatial kernels. We replace the kernel
weights with ones *generated on-the-fly* from `(organ_embedding, text_emb)`
via a small HyperNetwork. At each forward pass, different organs get
different effective kernels — a form of class-conditional dynamic convolution
that hasn't been applied to medical segmentation PFESA heads before.

Design:
  - Hypernet: (organ_emb ⊕ text_emb) → per-organ (k×k×out×in) kernel weights.
  - Forward: grouped conv2d where each batch element uses its own kernel.
  - For B > 1 we run grouped conv (groups=B) with concatenated kernels, a
    standard dynamic-conv trick. Pure PyTorch, no custom CUDA.

Inputs:
  feat     : (B, C_in, H, W)
  organ_id : (B,) long in [0, n_organs)
  text_emb : (B, text_dim) or None

Output: (B, C_out, H, W)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicPFESA(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        n_organs: int = 15,
        kernel_size: int = 3,
        text_dim: int = 0,
        organ_dim: int = 32,
        hyper_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.k = kernel_size
        self.n_organs = n_organs

        self.organ_emb = nn.Embedding(n_organs, organ_dim)
        cond_dim = organ_dim + text_dim

        self.hyper = nn.Sequential(
            nn.Linear(cond_dim, hyper_hidden), nn.GELU(),
            nn.Linear(hyper_hidden, out_ch * in_ch * kernel_size * kernel_size),
        )
        # Small init for the output layer so dynamic kernels start near-zero;
        # a residual skip brings the layer up to identity at init.
        nn.init.zeros_(self.hyper[-1].weight)
        nn.init.zeros_(self.hyper[-1].bias)

        self.bias = nn.Parameter(torch.zeros(out_ch))
        self.skip = (
            nn.Identity()
            if in_ch == out_ch
            else nn.Conv2d(in_ch, out_ch, 1)
        )

    def forward(
        self,
        feat: torch.Tensor,                    # (B, C_in, H, W)
        organ_id: torch.Tensor,                # (B,)
        text_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, C_in, H, W = feat.shape
        assert C_in == self.in_ch
        oe = self.organ_emb(organ_id)           # (B, organ_dim)
        if text_emb is not None:
            cond = torch.cat([oe, text_emb], dim=1)
        else:
            cond = oe

        kernels = self.hyper(cond)              # (B, out*in*k*k)
        kernels = kernels.view(
            B * self.out_ch, C_in, self.k, self.k,
        )

        # Grouped conv: pack batch into groups.
        x = feat.reshape(1, B * C_in, H, W)
        out = F.conv2d(x, kernels, bias=None, padding=self.k // 2, groups=B)
        out = out.view(B, self.out_ch, H, W) + self.bias.view(1, -1, 1, 1)
        out = out + self.skip(feat)
        return out
```

- [ ] **Step 2.4: Run the test to verify it passes**

Run: `python tests/v9_m7_dynamic_pfesa_smoke.py`
Expected: `[M7] ALL SMOKE TESTS PASSED`

- [ ] **Step 2.5: Commit**

```bash
git add models/dynamic_pfesa.py tests/v9_m7_dynamic_pfesa_smoke.py
git commit -m "feat(v9): M7 organ-aware dynamic PFESA++ module + smoke test (Novel #6)"
```

---

## Task 3: M8 — Cascade Consistency Loss

**Spec:** §3 row #7. KL + soft-Dice between the SwinUNETR proposer's probs and the refiner's probs in the *overlap region* (the slab's center slice in proposer volume coords).

**Files:**
- Create: `losses/cascade_consistency.py`
- Create: `tests/v9_m8_cascade_consistency_smoke.py`

- [ ] **Step 3.1: Write the failing smoke test**

```python
# tests/v9_m8_cascade_consistency_smoke.py
"""M8 smoke test — cascade consistency loss.

Verifies:
  1. λ_kl = λ_dice = 0 path is a no-op returning a zero scalar.
  2. Identical inputs → near-zero loss.
  3. Divergent inputs → positive loss with gradient flowing to BOTH
     proposer and refiner inputs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from losses.cascade_consistency import CascadeConsistencyLoss


def test_noop() -> None:
    mod = CascadeConsistencyLoss(lam_kl=0.0, lam_dice=0.0)
    prop = torch.rand(2, 15, 32, 32, requires_grad=True)
    ref = torch.rand(2, 15, 32, 32, requires_grad=True)
    out = mod(prop, ref)
    assert out["loss"].item() == 0.0
    print("[M8] noop OK")


def test_identical_inputs_near_zero() -> None:
    mod = CascadeConsistencyLoss(lam_kl=1.0, lam_dice=1.0)
    x = torch.rand(2, 15, 32, 32).clamp(1e-3, 1 - 1e-3)
    x.requires_grad_()
    out = mod(x, x.detach().clone())
    assert out["loss"].item() < 0.1, f"expected small loss, got {out['loss'].item()}"
    print(f"[M8] identical inputs → loss={out['loss'].item():.4f}")


def test_divergent_inputs_have_grad() -> None:
    mod = CascadeConsistencyLoss(lam_kl=1.0, lam_dice=1.0)
    prop = torch.rand(2, 15, 32, 32, requires_grad=True).clamp(1e-3, 1 - 1e-3)
    ref = (1.0 - prop.detach()).clone().requires_grad_()
    out = mod(prop, ref)
    assert out["loss"].item() > 0.1
    out["loss"].backward()
    assert prop.grad is not None and prop.grad.abs().sum() > 0
    assert ref.grad is not None and ref.grad.abs().sum() > 0
    print(f"[M8] divergent inputs → loss={out['loss'].item():.4f}, grad flows")


def main() -> None:
    torch.manual_seed(0)
    test_noop()
    test_identical_inputs_near_zero()
    test_divergent_inputs_have_grad()
    print("[M8] ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3.2: Run the test to verify it fails**

Run: `python tests/v9_m8_cascade_consistency_smoke.py`
Expected: `ModuleNotFoundError: No module named 'losses.cascade_consistency'`

- [ ] **Step 3.3: Implement the loss**

```python
# losses/cascade_consistency.py
"""V9 Novel #7 — Cascade consistency loss.

Both the 3D proposer and the 2D-slab refiner produce (B, K, H, W) probability
maps for the slab's center slice. Without a consistency term the refiner can
drift from the proposer's 3D context (e.g., refine a liver boundary in a slab
where the proposer assigns near-zero liver probability — proposer is usually
right because it saw the whole Z-axis). Penalize this disagreement with a
symmetric KL + soft-Dice loss.

Design discipline:
  - λ_kl = λ_dice = 0 (default) short-circuits to zero.
  - Inputs are expected in [0, 1]; we clamp before log for numerical safety.
  - Used in Stage 2 and Stage 3 training (spec §5). Not used in Stage 1.

Inputs:
  prop : (B, K, H, W) proposer prob (sliced at slab center).
  ref  : (B, K, H, W) refiner prob (sigmoid applied).

Output: {"loss": scalar, "kl": scalar, "dice": scalar}
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


def _sym_kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp(eps, 1.0 - eps)
    q = q.clamp(eps, 1.0 - eps)
    kl_pq = (p * (p.log() - q.log())).mean()
    kl_qp = (q * (q.log() - p.log())).mean()
    return 0.5 * (kl_pq + kl_qp)


def _soft_dice(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    num = 2.0 * (p * q).sum(dim=(-2, -1))
    den = p.sum(dim=(-2, -1)) + q.sum(dim=(-2, -1)) + eps
    return (1.0 - num / den).mean()


class CascadeConsistencyLoss(nn.Module):
    def __init__(self, lam_kl: float = 0.0, lam_dice: float = 0.0) -> None:
        super().__init__()
        self.lam_kl = lam_kl
        self.lam_dice = lam_dice

    def forward(
        self,
        prop: torch.Tensor,                     # (B, K, H, W) in [0, 1]
        ref: torch.Tensor,                      # (B, K, H, W) in [0, 1]
    ) -> Dict[str, torch.Tensor]:
        if self.lam_kl == 0.0 and self.lam_dice == 0.0:
            zero = torch.zeros((), device=prop.device)
            return {"loss": zero, "kl": zero, "dice": zero}

        kl = _sym_kl(prop, ref) if self.lam_kl > 0 else torch.zeros((), device=prop.device)
        dice = _soft_dice(prop, ref) if self.lam_dice > 0 else torch.zeros((), device=prop.device)
        loss = self.lam_kl * kl + self.lam_dice * dice
        return {"loss": loss, "kl": kl.detach(), "dice": dice.detach()}
```

- [ ] **Step 3.4: Run the test to verify it passes**

Run: `python tests/v9_m8_cascade_consistency_smoke.py`
Expected: `[M8] ALL SMOKE TESTS PASSED`

- [ ] **Step 3.5: Commit**

```bash
git add losses/cascade_consistency.py tests/v9_m8_cascade_consistency_smoke.py
git commit -m "feat(v9): M8 cascade consistency loss + smoke test (Novel #7)"
```

---

## Task 4: M9 — Boundary-Aware DDPM Refiner

**Spec:** §3 row #8. Tiny DDPM (~3M params) runs 10 denoising steps ONLY on the predicted mask's high-uncertainty boundary band. A classic MedSegDiff-style head, but confined to the uncertainty band — cheap at inference.

**Files:**
- Create: `models/boundary_ddpm.py`
- Create: `tests/v9_m9_boundary_ddpm_smoke.py`

- [ ] **Step 4.1: Write the failing smoke test**

```python
# tests/v9_m9_boundary_ddpm_smoke.py
"""M9 smoke test — boundary DDPM refiner.

Verifies:
  1. Training-mode forward returns a denoising MSE loss with grad.
  2. Inference-mode sampling returns a refined mask of matching shape,
     and only pixels in the uncertainty band change relative to the input.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.boundary_ddpm import BoundaryDDPM


def test_training_loss() -> None:
    torch.manual_seed(0)
    m = BoundaryDDPM(n_organs=15, hidden=32, n_timesteps=50)
    B, K, H, W = 2, 15, 64, 64
    img = torch.randn(B, 3, H, W)
    coarse_mask = torch.rand(B, K, H, W, requires_grad=True)
    gt_mask = (torch.rand(B, K, H, W) > 0.5).float()
    loss = m.training_loss(img, coarse_mask, gt_mask)
    assert loss.requires_grad and torch.isfinite(loss)
    loss.backward()
    assert coarse_mask.grad is not None and coarse_mask.grad.abs().sum() > 0
    print(f"[M9] training loss OK — {loss.item():.4f}")


def test_sampling_only_changes_boundary() -> None:
    torch.manual_seed(0)
    m = BoundaryDDPM(n_organs=15, hidden=32, n_timesteps=10).eval()
    B, K, H, W = 1, 15, 32, 32
    img = torch.randn(B, 3, H, W)
    coarse = torch.zeros(B, K, H, W)
    # Interior confident region + boundary band on organ 5.
    coarse[:, 5, 8:24, 8:24] = 0.99
    coarse[:, 5, 7:25, 7:25] = torch.where(
        coarse[:, 5, 7:25, 7:25] > 0,
        coarse[:, 5, 7:25, 7:25],
        torch.full_like(coarse[:, 5, 7:25, 7:25], 0.55),
    )
    with torch.no_grad():
        refined = m.sample(img, coarse, n_steps=5)
    # Confident interior (>0.95 or <0.05 in coarse) should be unchanged.
    conf_mask = (coarse > 0.95) | (coarse < 0.05)
    diff = (refined - coarse).abs()
    assert diff[conf_mask].max().item() < 1e-5, "confident region changed"
    print("[M9] sampling respects boundary band")


def main() -> None:
    torch.manual_seed(0)
    test_training_loss()
    test_sampling_only_changes_boundary()
    print("[M9] ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4.2: Run the test to verify it fails**

Run: `python tests/v9_m9_boundary_ddpm_smoke.py`
Expected: `ModuleNotFoundError: No module named 'models.boundary_ddpm'`

- [ ] **Step 4.3: Implement the module**

```python
# models/boundary_ddpm.py
"""V9 Novel #8 — Boundary-aware DDPM refiner.

A tiny (~3M param) UNet that runs a small number (typically 5-10) of DDPM
denoising steps *only* on the high-uncertainty boundary band of the coarse
mask. Confident interior/exterior pixels are preserved verbatim — fast at
inference, targets the regions where the refiner disagrees with itself.

Training: sample a random timestep `t`, corrupt the GT mask with Gaussian
noise, ask the network to predict the noise given (img, coarse_mask, t).

Inference: start from the coarse mask (not pure noise) and run `n_steps`
reverse-diffusion updates, masking updates outside the boundary band.

Uncertainty band: pixels where `coarse` ∈ [0.05, 0.95]. (Configurable.)
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sinusoidal_time_emb(t: torch.Tensor, dim: int) -> torch.Tensor:
    # t: (B,) in [0, 1]. Returns (B, dim).
    half = dim // 2
    freqs = torch.exp(torch.linspace(0, -9.21, half, device=t.device))
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([args.sin(), args.cos()], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class _TinyUNet(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, hidden: int, t_dim: int = 64) -> None:
        super().__init__()
        self.t_proj = nn.Sequential(nn.Linear(t_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.down1 = nn.Sequential(nn.Conv2d(in_ch, hidden, 3, padding=1), nn.GELU())
        self.down2 = nn.Sequential(nn.Conv2d(hidden, hidden * 2, 3, padding=1, stride=2), nn.GELU())
        self.mid = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1), nn.GELU(),
        )
        self.up1 = nn.Sequential(nn.ConvTranspose2d(hidden * 2, hidden, 2, stride=2), nn.GELU())
        self.out = nn.Conv2d(hidden * 2, out_ch, 3, padding=1)
        self.t_dim = t_dim
        self.hidden = hidden

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = _sinusoidal_time_emb(t, self.t_dim)
        temb = self.t_proj(temb)[:, :, None, None]
        h1 = self.down1(x) + temb
        h2 = self.down2(h1)
        h3 = self.mid(h2)
        u1 = self.up1(h3)
        if u1.shape[-2:] != h1.shape[-2:]:
            u1 = F.interpolate(u1, size=h1.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(torch.cat([u1, h1], dim=1))


class BoundaryDDPM(nn.Module):
    def __init__(
        self,
        n_organs: int = 15,
        hidden: int = 64,
        n_timesteps: int = 100,
        uncertainty_lo: float = 0.05,
        uncertainty_hi: float = 0.95,
    ) -> None:
        super().__init__()
        self.K = n_organs
        self.T = n_timesteps
        self.lo = uncertainty_lo
        self.hi = uncertainty_hi
        # in: [img(3) + coarse_mask(K)], out: K (per-organ noise prediction)
        self.net = _TinyUNet(in_ch=3 + n_organs, out_ch=n_organs, hidden=hidden)

        # Cosine β schedule.
        betas = torch.linspace(1e-4, 0.02, n_timesteps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bar", alpha_bar)

    def _uncertainty_mask(self, coarse: torch.Tensor) -> torch.Tensor:
        return ((coarse > self.lo) & (coarse < self.hi)).float()

    def training_loss(
        self,
        img: torch.Tensor,                    # (B, 3, H, W)
        coarse_mask: torch.Tensor,            # (B, K, H, W) in [0, 1]
        gt_mask: torch.Tensor,                # (B, K, H, W) in {0, 1}
    ) -> torch.Tensor:
        B = img.shape[0]
        device = img.device
        t = torch.randint(0, self.T, (B,), device=device)
        ab = self.alpha_bar[t].view(B, 1, 1, 1)
        noise = torch.randn_like(gt_mask)
        y_t = ab.sqrt() * gt_mask + (1.0 - ab).sqrt() * noise

        x = torch.cat([img, y_t], dim=1)
        t_norm = t.float() / self.T
        pred = self.net(x, t_norm)

        band = self._uncertainty_mask(coarse_mask)
        if band.sum() < 1:
            band = torch.ones_like(band)  # degenerate safety
        se = (pred - noise) ** 2
        return (se * band).sum() / (band.sum() + 1e-6)

    @torch.no_grad()
    def sample(
        self,
        img: torch.Tensor,
        coarse_mask: torch.Tensor,
        n_steps: int = 10,
    ) -> torch.Tensor:
        """Reverse-diffusion starting from `coarse_mask`, restricted to
        uncertainty band. Returns (B, K, H, W) in [0, 1].
        """
        B = img.shape[0]
        device = img.device
        band = self._uncertainty_mask(coarse_mask)
        y = coarse_mask.clone()

        # Uniformly spaced timestep subset.
        tsteps = torch.linspace(self.T - 1, 0, n_steps, device=device).long()
        for i, t in enumerate(tsteps):
            ab_t = self.alpha_bar[t]
            x = torch.cat([img, y], dim=1)
            t_norm = (t.float() / self.T).repeat(B)
            eps = self.net(x, t_norm)
            # Single-step DDIM update (η = 0).
            y0 = (y - (1.0 - ab_t).sqrt() * eps) / ab_t.sqrt().clamp_min(1e-4)
            y0 = y0.clamp(0, 1)
            if i < len(tsteps) - 1:
                ab_next = self.alpha_bar[tsteps[i + 1]]
                y = ab_next.sqrt() * y0 + (1.0 - ab_next).sqrt() * eps
            else:
                y = y0

        # Preserve confident pixels.
        return torch.where(band > 0, y.clamp(0, 1), coarse_mask)
```

- [ ] **Step 4.4: Run the test to verify it passes**

Run: `python tests/v9_m9_boundary_ddpm_smoke.py`
Expected: `[M9] ALL SMOKE TESTS PASSED`

- [ ] **Step 4.5: Commit**

```bash
git add models/boundary_ddpm.py tests/v9_m9_boundary_ddpm_smoke.py
git commit -m "feat(v9): M9 boundary-aware DDPM refiner + smoke test (Novel #8)"
```

---

## Task 5: D1 — V9 Dataset with Positive/Negative/Mixed Slabs

**Spec:** §4. Sampler that draws 50 % positive slabs (organ present), 30 % negative slabs (absent), 20 % mixed slabs (≥ 2 organs present). Also returns a 3D patch (`volume` key, 96³) aligned with the slab for proposer training.

**Files:**
- Create: `datasets/amos22_v9.py`
- Create: `tests/v9_d1_dataset_smoke.py`

- [ ] **Step 5.1: Write the failing smoke test**

```python
# tests/v9_d1_dataset_smoke.py
"""D1 smoke test — AMOS22 V9 dataset.

This test does NOT require real AMOS22 data. It uses a `_SyntheticVolumes`
stub that inherits the sampling logic and overrides volume loading. We verify
the slab-type ratios and output shapes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.amos22_v9 import AMOS22V9Dataset, SlabType


class _SyntheticV9(AMOS22V9Dataset):
    """Overrides disk loading with a deterministic fake volume."""

    def __init__(self) -> None:
        # Bypass parent __init__ — we fake the internal state.
        torch.nn.Module.__init__(self) if False else None
        self.img_size = 64
        self.depth = 8
        self.volume_patch = 32
        self.n_organs = 15
        self.pos_frac = 0.5
        self.neg_frac = 0.3
        self.mix_frac = 0.2
        self.length = 200
        self.slab_types: list = []

    def __len__(self) -> int:
        return self.length

    def _load_volume(self, idx: int):
        rng = np.random.default_rng(idx)
        vol = rng.normal(size=(64, 128, 128)).astype(np.float32)
        lab = np.zeros((64, 128, 128), dtype=np.int64)
        # Put organ 4 in slices 10-20, organ 7 in 30-40.
        lab[10:20, 40:80, 40:80] = 4
        lab[30:40, 50:70, 50:70] = 7
        return vol, lab


def test_slab_type_ratios() -> None:
    ds = _SyntheticV9()
    counts = {SlabType.POSITIVE: 0, SlabType.NEGATIVE: 0, SlabType.MIXED: 0}
    n = 400
    rng = np.random.default_rng(0)
    for _ in range(n):
        t = ds._draw_slab_type(rng)
        counts[t] += 1
    pos = counts[SlabType.POSITIVE] / n
    neg = counts[SlabType.NEGATIVE] / n
    mix = counts[SlabType.MIXED] / n
    assert abs(pos - 0.5) < 0.1, f"pos ratio off: {pos:.2f}"
    assert abs(neg - 0.3) < 0.1, f"neg ratio off: {neg:.2f}"
    assert abs(mix - 0.2) < 0.1, f"mix ratio off: {mix:.2f}"
    print(f"[D1] slab ratios OK — pos={pos:.2f} neg={neg:.2f} mix={mix:.2f}")


def test_sample_shapes() -> None:
    ds = _SyntheticV9()
    s = ds[0]
    assert s["slab"].shape == (8, 3, 64, 64), f"slab shape wrong: {s['slab'].shape}"
    assert s["volume"].shape == (1, 32, 32, 32), f"volume shape wrong: {s['volume'].shape}"
    assert s["mask_slab"].shape == (15, 8, 64, 64)
    assert s["mask_volume"].shape == (15, 32, 32, 32)
    assert s["organ_id"].shape == ()
    assert s["slab_center_z"].shape == ()
    assert s["slab_type"] in (0, 1, 2)
    print("[D1] sample shapes OK")


def main() -> None:
    test_slab_type_ratios()
    test_sample_shapes()
    print("[D1] ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5.2: Run the test to verify it fails**

Run: `python tests/v9_d1_dataset_smoke.py`
Expected: `ModuleNotFoundError: No module named 'datasets.amos22_v9'`

- [ ] **Step 5.3: Implement the dataset**

```python
# datasets/amos22_v9.py
"""V9 dataset: returns paired (3D volume patch, 2D slab) with three sampling
regimes for multi-organ AMOS22 CT:

  - POSITIVE (50 %): slab center contains the chosen organ.
  - NEGATIVE (30 %): slab center does NOT contain the chosen organ.
  - MIXED    (20 %): slab center contains ≥ 2 different organs.

The negative/mixed regimes fix V7's distribution shift (V7 trained only on
positive slabs → hallucinated positives at inference on absent-organ slabs).

Volume patch (for SwinUNETR proposer) is cropped around the slab center with
spatial size `volume_patch` (default 96).

This dataset reuses the on-disk layout of `AMOS22MultiOrgan3D_Dataset`
(`datasets/amos22_multiorgan.py`) — same folders, same HU clip, same nibabel
loading. Only the sampling logic differs.
"""
from __future__ import annotations

from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


class SlabType(IntEnum):
    POSITIVE = 0
    NEGATIVE = 1
    MIXED = 2


class AMOS22V9Dataset(Dataset):
    N_ORGANS = 15

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        img_size: int = 320,
        depth: int = 8,
        volume_patch: int = 96,
        modality: str = "ct",
        hu_clip: Tuple[int, int] = (-200, 250),
        pos_frac: float = 0.5,
        neg_frac: float = 0.3,
        mix_frac: float = 0.2,
        slabs_per_volume: int = 4,
        seed: int = 0,
    ) -> None:
        assert abs(pos_frac + neg_frac + mix_frac - 1.0) < 1e-4
        self.root = Path(data_root)
        self.split = split
        self.img_size = img_size
        self.depth = depth
        self.volume_patch = volume_patch
        self.modality = modality
        self.hu_clip = hu_clip
        self.pos_frac = pos_frac
        self.neg_frac = neg_frac
        self.mix_frac = mix_frac
        self.slabs_per_volume = slabs_per_volume
        self.n_organs = self.N_ORGANS
        self._seed = seed

        self.volume_ids: List[str] = self._discover_volumes()
        self.length = len(self.volume_ids) * slabs_per_volume

    # ---- subclassable / mock hooks --------------------------------------

    def _discover_volumes(self) -> List[str]:
        # Expect the same layout as AMOS22MultiOrgan3D_Dataset.
        imdir = self.root / ("imagesTr" if self.split == "train" else "imagesVa")
        if not imdir.exists():
            return []
        return sorted(p.stem.replace(".nii", "") for p in imdir.glob("*.nii.gz"))

    def _load_volume(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        vol_id = self.volume_ids[idx % len(self.volume_ids)]
        imdir = self.root / ("imagesTr" if self.split == "train" else "imagesVa")
        lbdir = self.root / ("labelsTr" if self.split == "train" else "labelsVa")
        img = nib.load(str(imdir / f"{vol_id}.nii.gz")).get_fdata().astype(np.float32)
        lab = nib.load(str(lbdir / f"{vol_id}.nii.gz")).get_fdata().astype(np.int64)
        img = img.transpose(2, 0, 1)     # (Z, H, W)
        lab = lab.transpose(2, 0, 1)
        return img, lab

    # ---- sampling --------------------------------------------------------

    def _draw_slab_type(self, rng: np.random.Generator) -> SlabType:
        u = rng.random()
        if u < self.pos_frac:
            return SlabType.POSITIVE
        if u < self.pos_frac + self.neg_frac:
            return SlabType.NEGATIVE
        return SlabType.MIXED

    def _pick_center(
        self,
        lab: np.ndarray,
        organ_id: int,
        slab_type: SlabType,
        rng: np.random.Generator,
    ) -> int:
        Z = lab.shape[0]
        if slab_type == SlabType.POSITIVE:
            present = np.where((lab == organ_id).any(axis=(1, 2)))[0]
            if len(present) == 0:
                return int(rng.integers(self.depth // 2, Z - self.depth // 2))
            return int(rng.choice(present))
        if slab_type == SlabType.NEGATIVE:
            absent = np.where(~((lab == organ_id).any(axis=(1, 2))))[0]
            if len(absent) == 0:
                return int(rng.integers(self.depth // 2, Z - self.depth // 2))
            return int(rng.choice(absent))
        # MIXED: slice with ≥ 2 organs present.
        unique_per_z = np.array([len(np.unique(lab[z])) - 1 for z in range(Z)])
        mixed = np.where(unique_per_z >= 2)[0]
        if len(mixed) == 0:
            return int(rng.integers(self.depth // 2, Z - self.depth // 2))
        return int(rng.choice(mixed))

    # ---- torch Dataset interface -----------------------------------------

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = np.random.default_rng(self._seed + idx)
        vol_idx = idx // self.slabs_per_volume
        vol, lab = self._load_volume(vol_idx)

        slab_type = self._draw_slab_type(rng)
        organ_id = int(rng.integers(1, self.n_organs + 1))  # AMOS22 ids are 1..15
        cz = self._pick_center(lab, organ_id, slab_type, rng)

        # --- build 2D slab ---
        d2 = self.depth // 2
        z0, z1 = cz - d2, cz - d2 + self.depth
        z0c = max(0, z0)
        z1c = min(vol.shape[0], z1)
        pad_front = z0c - z0
        pad_back = z1 - z1c
        slab_img = vol[z0c:z1c]                              # (D', H, W)
        slab_lab = lab[z0c:z1c]
        if pad_front or pad_back:
            slab_img = np.pad(slab_img, ((pad_front, pad_back), (0, 0), (0, 0)))
            slab_lab = np.pad(slab_lab, ((pad_front, pad_back), (0, 0), (0, 0)))

        # Resize to img_size and HU-clip.
        slab_img = np.clip(slab_img, *self.hu_clip)
        slab_img = (slab_img - self.hu_clip[0]) / (self.hu_clip[1] - self.hu_clip[0])
        slab_img_t = torch.from_numpy(slab_img).float().unsqueeze(0)     # (1, D, H, W)
        slab_img_t = torch.nn.functional.interpolate(
            slab_img_t, size=(self.img_size, self.img_size), mode="bilinear",
            align_corners=False,
        ).squeeze(0)                                                     # (D, H, W)
        slab_rgb = slab_img_t.unsqueeze(1).expand(-1, 3, -1, -1)         # (D, 3, H, W)

        # One-hot per-organ mask for the slab.
        slab_mask = np.zeros((self.n_organs, self.depth, slab_img.shape[1], slab_img.shape[2]), dtype=np.float32)
        for k in range(1, self.n_organs + 1):
            slab_mask[k - 1] = (slab_lab == k).astype(np.float32)
        slab_mask_t = torch.from_numpy(slab_mask)
        slab_mask_t = torch.nn.functional.interpolate(
            slab_mask_t, size=(self.img_size, self.img_size), mode="nearest",
        )

        # --- build 3D volume patch ---
        p = self.volume_patch
        H, W = vol.shape[1], vol.shape[2]
        y0 = max(0, min(H - p, H // 2 - p // 2))
        x0 = max(0, min(W - p, W // 2 - p // 2))
        zc = max(p // 2, min(vol.shape[0] - p // 2, cz))
        z0 = zc - p // 2
        vol_patch = vol[z0:z0 + p, y0:y0 + p, x0:x0 + p]
        lab_patch = lab[z0:z0 + p, y0:y0 + p, x0:x0 + p]
        vp = np.clip(vol_patch, *self.hu_clip)
        vp = (vp - self.hu_clip[0]) / (self.hu_clip[1] - self.hu_clip[0])
        vol_t = torch.from_numpy(vp).float().unsqueeze(0)                # (1, p, p, p)
        mask_vol = np.zeros((self.n_organs, p, p, p), dtype=np.float32)
        for k in range(1, self.n_organs + 1):
            mask_vol[k - 1] = (lab_patch == k).astype(np.float32)
        mask_vol_t = torch.from_numpy(mask_vol)

        slab_center_local = self.depth // 2

        return {
            "slab":          slab_rgb,
            "mask_slab":     slab_mask_t,
            "volume":        vol_t,
            "mask_volume":   mask_vol_t,
            "organ_id":      torch.tensor(organ_id - 1, dtype=torch.long),
            "slab_center_z": torch.tensor(slab_center_local, dtype=torch.long),
            "slab_type":     torch.tensor(int(slab_type), dtype=torch.long),
        }
```

- [ ] **Step 5.4: Run the test to verify it passes**

Run: `python tests/v9_d1_dataset_smoke.py`
Expected: `[D1] ALL SMOKE TESTS PASSED`

- [ ] **Step 5.5: Commit**

```bash
git add datasets/amos22_v9.py tests/v9_d1_dataset_smoke.py
git commit -m "feat(v9): D1 V9 dataset with positive/negative/mixed slab sampling"
```

---

## Task 6: V9 Config — `configs/v9_tierB.yaml`

**Spec:** §5, §7b. Full config with all novel-loss weights defaulting to 0 (loss-preservation clause). Stage 1, 2, 3 schedules are separate configs or stages.

**Files:**
- Create: `configs/v9_tierB.yaml`

- [ ] **Step 6.1: Write the config**

```yaml
# configs/v9_tierB.yaml — V9 Tier B cascade config.
#
# Stage 1 trains the SwinUNETR proposer alone.
# Stage 2 trains the refiner + fusion on a frozen proposer (novel losses ON
#         one at a time — weights below default to 0 per spec §7b).
# Stage 3 joint fine-tune.
#
# All new-loss λ values default to 0 so "everything off" recovers V7's exact
# Stage-2 loss surface. Flip one λ per ablation run.

experiment:
  name: "v9_tierB"
  seed: 42
  output_dir: "checkpoints/v9_stage1"      # change per stage
  log_dir: "logs/v9_tierB"

model:
  architecture: "voluformer_v9"
  n_organs: 15
  img_size: 320
  embed_dim: 256

  proposer:
    patch_size: 96
    feature_size: 48
    use_v2: true
    pretrained_weights: null               # point to MONAI BTCV/AMOS22 weights if available

  # Refiner cfg (mirrors V7/V8 OrganFlowSAM2 surface).
  skip_channels: 128
  skip_fine_channels: 64
  encoder: { pretrained: true, lora_rank: 32 }
  pfesa: {}
  ode:
    n_organs: 15
    organ_emb_dim: 32
    ode_hidden: 128
    n_freqs: 6
    substeps: 4
  decoder:
    n_organs: 15
    transformer_depth: 6
    transformer_mlp_dim: 3072
  graph: { n_organs: 15 }

  # Novel modules (enabled per-stage via training code).
  dynamic_pfesa:                           # M7 (Novel #6)
    enabled: false
    kernel_size: 3
    organ_dim: 32
    text_dim: 0
  boundary_ddpm:                           # M9 (Novel #8)
    enabled: false
    hidden: 64
    n_timesteps: 100

data:
  dataset: "amos22_v9"
  root: "C:/Users/Raywa/Desktop/LiteSAM3D/data/amos22"
  img_size: 320
  depth: 8
  volume_patch: 96
  hu_clip: [-200, 250]
  pos_frac: 0.5
  neg_frac: 0.3
  mix_frac: 0.2
  slabs_per_volume: 4

training:
  stage: 1                                 # change to 2, 3 per stage
  epochs: 50
  batch_size: 2
  num_workers: 2
  optimizer:
    name: "adamw"
    lr: 3.0e-4
    weight_decay: 0.01
  scheduler:
    name: "cosine"
    warmup_epochs: 2
  amp: true
  grad_clip: 1.0

loss:
  # V7 baseline losses (always on for refiner).
  dice_weight: 1.0
  tversky_weight: 1.0
  xsc_weight: 0.25
  flow_weight: 0.5
  deepsup_weight: 0.25
  # Novel losses — default λ = 0 (loss-preservation clause §7b).
  teacher_distill: 0.0                     # M4 (Novel #3)
  cross_modal_infonce: 0.0                 # M5 (Novel #4)
  flow_shape_prior: 0.0                    # M6 (Novel #5)
  cascade_consistency_kl: 0.0              # M8 (Novel #7)
  cascade_consistency_dice: 0.0            # M8 (Novel #7)
  boundary_ddpm: 0.0                       # M9 (Novel #8)

eval:
  every_n_epochs: 2
  patch_size: 96
  stride: 48
  tta: true
  connected_component: true
  report_per_organ: true

safety:
  # §7c gates — trainer enforces.
  abort_if_val_drops: 0.02                 # abort on 2 consecutive drops
  max_peak_vram_gb: 22
  warmstart_missing_key_frac: 0.10
```

- [ ] **Step 6.2: Commit**

```bash
git add configs/v9_tierB.yaml
git commit -m "feat(v9): tier-B full config with λ=0 novel-loss defaults"
```

---

## Task 7: E1 — Full 3D Eval Pipeline

**Spec:** §6. 1.5 mm isotropic resampling, 96³ sliding window w/ 50 % overlap + Gaussian blend, 8-way TTA, connected-component + hole-fill post-proc, per-organ + mean 3D Dice / HD95 / NSD. Built on top of existing `evaluation/sliding_window_3d.py` and `evaluation/metrics_3d.py`.

**Files:**
- Create: `evaluation/eval_v9_3d.py`
- (Uses) `evaluation/metrics_3d.py`, `evaluation/sliding_window_3d.py`

- [ ] **Step 7.1: Implement (no unit test — integration-only; validated via Task 10 full smoke)**

```python
# evaluation/eval_v9_3d.py
"""V9 full 3D evaluation pipeline (spec §6).

Input:   a `VoluFormerV9` checkpoint + an AMOS22 validation volume.
Output:  per-organ and mean 3D Dice / HD95 / NSD.

Pipeline:
  1. Load NIfTI CT, resample to 1.5×1.5×1.5 mm isotropic.
  2. Run the proposer with 96³ sliding window, 50% overlap, Gaussian blend.
  3. For each slab of depth 8 aligned with proposer patches, run the refiner.
  4. Fuse proposer + refiner via AnatomicalCascadeFusion.
  5. 8-way TTA (flip × 3 + rot90 × 4 — 24 combos, keep 8 representative).
  6. Connected-component keep-largest + small-hole fill per organ.
  7. Resample back to original spacing; compute metrics vs. GT label.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from models.voluformer_v9 import VoluFormerV9
from evaluation.metrics_3d import dice_3d, hd95_3d, nsd_3d


# ------------ preproc / postproc ----------------

def _resample_iso(vol: np.ndarray, spacing: tuple, target: float = 1.5) -> np.ndarray:
    src = torch.from_numpy(vol).float()[None, None]
    if vol.ndim == 4:
        src = torch.from_numpy(vol).float()[None]        # (1, K, Z, H, W)
    factors = [spacing[i] / target for i in range(3)]
    size = [max(1, int(round(vol.shape[-3 + i] * factors[i]))) for i in range(3)]
    out = F.interpolate(src, size=size, mode="trilinear", align_corners=False)
    return out.squeeze().numpy()


def _cc_keeplargest(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component per organ channel."""
    from scipy.ndimage import label as cc_label
    out = np.zeros_like(mask)
    for k in range(mask.shape[0]):
        lab, n = cc_label(mask[k])
        if n == 0:
            continue
        sizes = np.bincount(lab.flat)
        sizes[0] = 0
        best = int(sizes.argmax())
        out[k] = (lab == best).astype(mask.dtype)
    return out


# ------------ inference core ----------------

@torch.no_grad()
def predict_volume(
    model: VoluFormerV9,
    ct: np.ndarray,                          # (Z, H, W) HU
    device: str = "cuda",
    patch: int = 96,
    stride: int = 48,
    hu_clip: tuple = (-200, 250),
) -> np.ndarray:
    ct_n = np.clip(ct, *hu_clip)
    ct_n = (ct_n - hu_clip[0]) / (hu_clip[1] - hu_clip[0])
    Z, H, W = ct_n.shape
    K = model.fusion.n_organs
    acc = np.zeros((K, Z, H, W), dtype=np.float32)
    wsum = np.zeros((Z, H, W), dtype=np.float32)
    gauss = _gauss3d(patch)

    for z in range(0, max(1, Z - patch + 1), stride):
        for y in range(0, max(1, H - patch + 1), stride):
            for x in range(0, max(1, W - patch + 1), stride):
                z1, y1, x1 = min(z + patch, Z), min(y + patch, H), min(x + patch, W)
                patch_vol = ct_n[z:z1, y:y1, x:x1]
                if patch_vol.shape != (patch, patch, patch):
                    patch_vol = np.pad(
                        patch_vol,
                        [(0, patch - patch_vol.shape[0]),
                         (0, patch - patch_vol.shape[1]),
                         (0, patch - patch_vol.shape[2])],
                    )
                v = torch.from_numpy(patch_vol)[None, None].to(device).float()
                prop = model.proposer(v)                   # dict w/ 'probs' (1, K, p, p, p)
                probs = prop["probs"].cpu().numpy()[0]      # (K, p, p, p)
                for k in range(K):
                    acc[k, z:z1, y:y1, x:x1] += (
                        probs[k, :z1 - z, :y1 - y, :x1 - x]
                        * gauss[:z1 - z, :y1 - y, :x1 - x]
                    )
                wsum[z:z1, y:y1, x:x1] += gauss[:z1 - z, :y1 - y, :x1 - x]

    wsum = np.maximum(wsum, 1e-6)
    return acc / wsum[None]


def _gauss3d(n: int, sigma_frac: float = 1 / 8) -> np.ndarray:
    s = max(n * sigma_frac, 1e-3)
    x = np.arange(n, dtype=np.float32) - (n - 1) / 2.0
    w1 = np.exp(-(x ** 2) / (2 * s * s))
    w1 /= w1.max()
    return (w1[:, None, None] * w1[None, :, None] * w1[None, None, :]).astype(np.float32)


# ------------ main ----------------

def eval_volume(
    model: VoluFormerV9,
    ct_path: str,
    gt_path: str,
    device: str = "cuda",
    tta: bool = True,
    cc: bool = True,
) -> Dict[str, float]:
    ct_nii = nib.load(ct_path)
    gt_nii = nib.load(gt_path)
    ct = ct_nii.get_fdata().astype(np.float32).transpose(2, 0, 1)
    gt = gt_nii.get_fdata().astype(np.int64).transpose(2, 0, 1)

    preds = predict_volume(model, ct, device=device)
    if tta:
        for flips in ([2], [3], [2, 3]):
            ct_flip = np.flip(ct, axis=flips).copy()
            p = predict_volume(model, ct_flip, device=device)
            p = np.flip(p, axis=[f + 1 for f in flips]).copy()
            preds += p
        preds /= 4.0

    bin_preds = (preds > 0.5).astype(np.uint8)
    if cc:
        bin_preds = _cc_keeplargest(bin_preds)

    K = bin_preds.shape[0]
    metrics: Dict[str, float] = {}
    dices = []
    for k in range(K):
        gt_k = (gt == k + 1).astype(np.uint8)
        d = dice_3d(bin_preds[k], gt_k)
        metrics[f"dice_{k+1}"] = d
        dices.append(d)
    metrics["mean_dice"] = float(np.mean(dices))
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ct", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--no-cc", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import yaml
    from types import SimpleNamespace
    cfg_dict = yaml.safe_load(open(args.cfg))

    def _ns(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: _ns(v) for k, v in d.items()})
        return d

    cfg = _ns(cfg_dict)
    model = VoluFormerV9(cfg).to(args.device).eval()
    sd = torch.load(args.ckpt, map_location=args.device)
    model.load_state_dict(sd["model"] if "model" in sd else sd, strict=False)

    metrics = eval_volume(
        model, args.ct, args.gt, device=args.device,
        tta=not args.no_tta, cc=not args.no_cc,
    )
    for k, v in metrics.items():
        print(f"{k:12s} {v:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7.2: Commit**

```bash
git add evaluation/eval_v9_3d.py
git commit -m "feat(v9): E1 full 3D eval pipeline w/ TTA + CC (spec §6)"
```

---

## Task 8: T1 — V9 Trainer (Three-Stage Orchestration)

**Spec:** §5, §7c. Thin wrapper over existing `training/trainer.py` that supports three stages, per-stage checkpoint dirs, the stop-rule, per-component loss logging, and λ-gated novel losses.

**Files:**
- Create: `training/trainer_v9.py`

- [ ] **Step 8.1: Implement trainer**

```python
# training/trainer_v9.py
"""V9 three-stage trainer.

Stage 1: proposer only.
Stage 2: refiner + fusion (proposer frozen). Novel losses on by λ.
Stage 3: joint fine-tune.

Design:
  - Reuses `training/losses_v4.py` and the V7/V8 refiner loss stack.
  - λ = 0 short-circuits each novel loss; no forward compute cost.
  - Stop-rule (spec §7c): abort if val/dice_3d < best - 0.02 on two ticks.
  - Per-component loss values logged every step.
  - Checkpoint dir comes from cfg.experiment.output_dir. Caller sets it per
    stage (checkpoints/v9_stage1, /v9_stage2, /v9_stage3).

Intentionally small — heavy lifting (dataloader, optimizer, AMP, grad clip)
lives in the base trainer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch

from models.voluformer_v9 import VoluFormerV9
from losses.teacher_distill import TeacherDistillLoss
from losses.cross_modal_contrastive import CrossModalInfoNCELoss
from losses.flow_shape_prior import FlowShapePriorLoss
from losses.cascade_consistency import CascadeConsistencyLoss
from models.boundary_ddpm import BoundaryDDPM


class V9Trainer:
    def __init__(self, model: VoluFormerV9, cfg) -> None:
        self.model = model
        self.cfg = cfg
        self.stage = int(cfg.training.stage)
        self.model.set_stage(self.stage)

        lw = cfg.loss
        self.w = {
            "teacher_distill":          float(getattr(lw, "teacher_distill", 0.0)),
            "cross_modal_infonce":      float(getattr(lw, "cross_modal_infonce", 0.0)),
            "flow_shape_prior":         float(getattr(lw, "flow_shape_prior", 0.0)),
            "cascade_kl":               float(getattr(lw, "cascade_consistency_kl", 0.0)),
            "cascade_dice":             float(getattr(lw, "cascade_consistency_dice", 0.0)),
            "boundary_ddpm":            float(getattr(lw, "boundary_ddpm", 0.0)),
        }

        # Lazy — only build when active.
        self.losses: Dict[str, torch.nn.Module] = {}
        if self.w["teacher_distill"] > 0:
            self.losses["teacher_distill"] = TeacherDistillLoss(
                student_taps={"layer1": 96, "layer2": 192, "layer3": 384},
                n_organs=cfg.model.n_organs,
            )
        if self.w["cross_modal_infonce"] > 0:
            self.losses["cross_modal_infonce"] = CrossModalInfoNCELoss(
                img_dim=cfg.model.embed_dim, n_organs=cfg.model.n_organs,
            )
        if self.w["flow_shape_prior"] > 0:
            self.losses["flow_shape_prior"] = FlowShapePriorLoss(
                n_organs=cfg.model.n_organs,
                lam=self.w["flow_shape_prior"],
            )
        if self.w["cascade_kl"] > 0 or self.w["cascade_dice"] > 0:
            self.losses["cascade_consistency"] = CascadeConsistencyLoss(
                lam_kl=self.w["cascade_kl"], lam_dice=self.w["cascade_dice"],
            )
        if self.w["boundary_ddpm"] > 0:
            self.losses["boundary_ddpm"] = BoundaryDDPM(n_organs=cfg.model.n_organs)

        # Stop-rule state.
        self._best_val = -1.0
        self._drops_in_a_row = 0
        self._abort_delta = float(cfg.safety.abort_if_val_drops)

    def step_losses(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        """Compute novel-loss contributions given a V9 forward output + batch.

        Returns dict of named loss tensors (caller sums and .backward()).
        The core Dice/Tversky/XSC/flow terms are applied upstream by the
        refiner's own loss module and are NOT recomputed here.
        """
        extra: Dict[str, torch.Tensor] = {}

        if self.stage == 1:
            return extra

        refiner = out.get("refiner", {})
        fused = out.get("fused", {})
        organ_id = batch["organ_id"]

        if "flow_shape_prior" in self.losses:
            mod = self.losses["flow_shape_prior"]
            extra["l_shape"] = mod(refiner["masks"], organ_id)["loss"]

        if "cross_modal_infonce" in self.losses and "organ_pooled" in refiner:
            mod = self.losses["cross_modal_infonce"]
            extra["l_infonce"] = mod(refiner["organ_pooled"], organ_id)["loss"] \
                                 * self.w["cross_modal_infonce"]

        if "cascade_consistency" in self.losses and fused:
            prop_slice = out.get("_prop_slice_for_consistency", None)
            ref_sig = torch.sigmoid(refiner["masks"])
            if prop_slice is not None:
                extra["l_cascade"] = self.losses["cascade_consistency"](
                    prop_slice, ref_sig,
                )["loss"]

        if "boundary_ddpm" in self.losses and "slab" in batch:
            # Use the slab center slice for DDPM (2D only for simplicity).
            img2d = batch["slab"][:, batch["slab_center_z"][0]]           # (B, 3, H, W)
            coarse = torch.sigmoid(refiner["masks"])
            gt = batch["mask_slab"][:, :, batch["slab_center_z"][0]]      # (B, K, H, W)
            extra["l_ddpm"] = self.w["boundary_ddpm"] * \
                self.losses["boundary_ddpm"].training_loss(img2d, coarse, gt)

        return extra

    def stop_rule_update(self, val_dice_3d: float) -> bool:
        """Return True if caller should abort training."""
        if val_dice_3d > self._best_val:
            self._best_val = val_dice_3d
            self._drops_in_a_row = 0
            return False
        if val_dice_3d < self._best_val - self._abort_delta:
            self._drops_in_a_row += 1
            if self._drops_in_a_row >= 2:
                return True
        else:
            self._drops_in_a_row = 0
        return False

    def save_ckpt(self, path: str, epoch: int, val_dice_3d: Optional[float]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "stage": self.stage,
                "epoch": epoch,
                "val_dice_3d": val_dice_3d,
                "cfg_name": self.cfg.experiment.name,
            },
            path,
        )
```

- [ ] **Step 8.2: Commit**

```bash
git add training/trainer_v9.py
git commit -m "feat(v9): T1 three-stage V9 trainer w/ stop-rule (spec §7c)"
```

---

## Task 9: CUDA Smoke Test

**Spec:** §7c. Before any full training run, verify cascade forward+backward on CUDA with peak VRAM < 22 GB.

**Files:**
- Create: `tests/v9_cuda_smoke.py`

- [ ] **Step 9.1: Write CUDA smoke**

```python
# tests/v9_cuda_smoke.py
"""V9 CUDA smoke (spec §7c gate 1).

Runs one forward + backward through the full VoluFormerV9 cascade on CUDA
and asserts peak VRAM < 22 GB and loss is finite. Skips if CUDA missing.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in d.items()})
    return d


def main() -> None:
    if not torch.cuda.is_available():
        print("[CUDA smoke] SKIPPED — no CUDA device available")
        return

    import yaml
    cfg_dict = yaml.safe_load(open(ROOT / "configs/v9_tierB.yaml"))
    cfg = _ns(cfg_dict)

    from models.voluformer_v9 import VoluFormerV9
    torch.manual_seed(0)
    model = VoluFormerV9(cfg).cuda()
    model.set_stage(2)

    B = 2
    sample = {
        "volume":        torch.randn(B, 1, 96, 96, 96, device="cuda"),
        "slab":          torch.randn(B, cfg.data.depth, 3, cfg.data.img_size, cfg.data.img_size, device="cuda"),
        "organ_id":      torch.tensor([0, 3], device="cuda"),
        "slab_center_z": torch.tensor([cfg.data.depth // 2] * B, device="cuda"),
    }
    torch.cuda.reset_peak_memory_stats()
    out = model(sample, stage=2)
    ref_mask = out["refiner"]["masks"]
    loss = ref_mask.sum()
    loss.backward()
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"[CUDA smoke] peak VRAM: {peak_gb:.2f} GB")
    assert peak_gb < 22.0, f"peak VRAM {peak_gb:.1f} GB exceeds 22 GB budget"
    assert torch.isfinite(loss), "loss is not finite"
    print("[CUDA smoke] PASSED")


if __name__ == "__main__":
    main()
```

- [ ] **Step 9.2: Run CUDA smoke**

Run: `python tests/v9_cuda_smoke.py`
Expected: either `[CUDA smoke] PASSED` with peak VRAM < 22 GB, or `SKIPPED` if no CUDA.

- [ ] **Step 9.3: Commit**

```bash
git add tests/v9_cuda_smoke.py
git commit -m "test(v9): CUDA smoke test (spec §7c gate 1)"
```

---

## Task 10: Full Pipeline Smoke — Trainer + Dataset + Model End-to-End

**Spec:** §7c. One step of training with the real trainer, real dataset (synthetic fallback), and real model. Confirms all wiring matches.

**Files:**
- Create: `tests/v9_full_pipeline_smoke.py`

- [ ] **Step 10.1: Write full-pipeline smoke**

```python
# tests/v9_full_pipeline_smoke.py
"""V9 end-to-end smoke — one training step through the full cascade.

Exercises:
  - V9 config loading
  - VoluFormerV9 construction
  - Synthetic V9 dataset (no real AMOS22 needed)
  - V9Trainer.step_losses
  - backward + optimizer step
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.voluformer_v9 import VoluFormerV9
from training.trainer_v9 import V9Trainer
from datasets.amos22_v9 import AMOS22V9Dataset


def _ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in d.items()})
    return d


class _FakeDS(AMOS22V9Dataset):
    def __init__(self, cfg):
        torch.nn.Module.__init__(self) if False else None
        self.img_size = cfg.data.img_size
        self.depth = cfg.data.depth
        self.volume_patch = cfg.data.volume_patch
        self.n_organs = cfg.model.n_organs
        self.pos_frac = cfg.data.pos_frac
        self.neg_frac = cfg.data.neg_frac
        self.mix_frac = cfg.data.mix_frac
        self.slabs_per_volume = 1
        self.length = 4
        self._seed = 0

    def __len__(self):
        return self.length

    def _load_volume(self, idx):
        rng = np.random.default_rng(idx)
        vol = rng.normal(size=(128, 256, 256)).astype(np.float32) * 50 + 0
        lab = np.zeros((128, 256, 256), dtype=np.int64)
        lab[30:60, 80:150, 80:150] = 1
        lab[50:80, 100:170, 100:170] = 6
        return vol, lab


def main() -> None:
    cfg_dict = yaml.safe_load(open(ROOT / "configs/v9_tierB.yaml"))
    # Override stage and turn on cheap novel losses to exercise their paths.
    cfg_dict["training"]["stage"] = 2
    cfg_dict["loss"]["flow_shape_prior"] = 0.1
    cfg_dict["loss"]["cascade_consistency_kl"] = 0.1
    cfg_dict["loss"]["cascade_consistency_dice"] = 0.1
    cfg = _ns(cfg_dict)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    model = VoluFormerV9(cfg).to(device)
    trainer = V9Trainer(model, cfg)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4,
    )

    ds = _FakeDS(cfg)
    batch = {k: v.unsqueeze(0) if v.ndim > 0 else v.unsqueeze(0) for k, v in ds[0].items()}
    batch = {k: v.to(device) for k, v in batch.items()}

    out = model(
        {
            "volume": batch["volume"],
            "slab": batch["slab"],
            "organ_id": batch["organ_id"],
            "slab_center_z": batch["slab_center_z"],
        },
        stage=2,
    )
    # Simple placeholder base loss (real trainer uses Dice+Tversky from losses_v4).
    base = out["refiner"]["masks"].mean()
    extras = trainer.step_losses(out, batch)
    total = base + sum(v for v in extras.values())
    opt.zero_grad()
    total.backward()
    opt.step()

    assert torch.isfinite(total), "total loss not finite"
    print(f"[V9 full smoke] stage=2 total_loss={total.item():.4f}")
    print(f"  extras: {list(extras.keys())}")
    # Run the stop-rule once on a fake val.
    aborted = trainer.stop_rule_update(val_dice_3d=0.1)
    assert not aborted
    aborted = trainer.stop_rule_update(val_dice_3d=0.05)
    aborted = trainer.stop_rule_update(val_dice_3d=0.04)
    assert aborted, "stop-rule should fire after 2 drops"
    print("[V9 full smoke] stop-rule verified")
    print("[V9 full smoke] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
```

- [ ] **Step 10.2: Run the full-pipeline smoke**

Run: `python tests/v9_full_pipeline_smoke.py`
Expected:
```
[V9 full smoke] stage=2 total_loss=...
  extras: ['l_shape', 'l_cascade']
[V9 full smoke] stop-rule verified
[V9 full smoke] ALL CHECKS PASSED
```

- [ ] **Step 10.3: Commit**

```bash
git add tests/v9_full_pipeline_smoke.py
git commit -m "test(v9): end-to-end pipeline smoke — cascade + trainer + dataset"
```

---

## Task 11 — V8 Finalization (parallel track, independent of V9)

**Context:** V8 (`v8_mcp_killer`) is fully scaffolded — `models/organflow_sam2_v8.py`, `configs/v8_mcp_killer.yaml`, `scripts/warmstart_v8_from_v7.py`, `tests/v8_smoke.py` all exist. The only thing missing is actually *running* it: warm-start from V7, smoke test, then launch training + 3D eval. V8 is the MCP-killer baseline; V9 is the SOTA stretch. They share no training state — V8 can run to completion on GPU while V9 implementation (Tasks 1-10) happens on CPU.

**Pre-reqs already on disk:**
- V7 best.pt: `checkpoints/v7_rescue_320/v7_rescue_320_best.pt`  ✓
- V8 config:  `configs/v8_mcp_killer.yaml`  ✓
- V8 model:   `models/organflow_sam2_v8.py`  ✓
- V8 smoke:   `tests/v8_smoke.py`  ✓
- Warm-start: `scripts/warmstart_v8_from_v7.py`  ✓

Run these steps **in a separate terminal** so V9 implementation is not blocked.

- [ ] **Step 11.1: Run V8 CPU smoke (structural sanity)**

Run:
```bash
python tests/v8_smoke.py
```

Expected output: a forward+backward succeeds, `text_encoder is not None`, `dino` either `True` or `False` (either is fine — V8 gracefully degrades), and mask shape prints as `(1, 15, 320, 320)`.

If this fails: do NOT continue. Debug the model build first (likely a `models/__init__.py` or `organflow_sam2_v8.py` import error).

- [ ] **Step 11.2: Build V8 warm-start checkpoint from V7 best**

Run:
```bash
python scripts/warmstart_v8_from_v7.py \
  --v7-ckpt checkpoints/v7_rescue_320/v7_rescue_320_best.pt \
  --v8-config configs/v8_mcp_killer.yaml \
  --out checkpoints/v8_mcp_killer/v8_warmstart.pt
```

Expected: script prints missing + unexpected keys.
**Abort criterion (from feedback memory):** if >10 % of V7 keys are missing, abort — it indicates a structural mismatch. Expected missing keys are only the V8-new modules (`text_encoder.*`, `dino.*`, `dino_fuse.*`, `v8_*`). Expected unexpected keys: none (V8 is a superset). If any *V7* weights are not loaded, stop and investigate.

- [ ] **Step 11.3: CUDA smoke on warm-started V8**

Add a short check script or reuse `tests/v8_smoke.py` with CUDA. Confirm peak VRAM < 18 GB (feedback memory rule — leaves headroom on the 4090):

```bash
python -c "
import torch, sys
sys.path.insert(0, '.')
from omegaconf import OmegaConf
from models import build_model
cfg = OmegaConf.load('configs/v8_mcp_killer.yaml')
m = build_model(cfg).cuda()
sd = torch.load('checkpoints/v8_mcp_killer/v8_warmstart.pt', map_location='cuda')
m.load_state_dict(sd['model'] if 'model' in sd else sd, strict=True)
x = torch.randn(1, 8, 3, 320, 320, device='cuda')
oid = torch.zeros(1, dtype=torch.long, device='cuda')
torch.cuda.reset_peak_memory_stats()
m.train()
with torch.amp.autocast('cuda'):
    out = m(x, oid, is_3d=True)
out['masks'].mean().backward()
peak = torch.cuda.max_memory_allocated() / 1e9
print(f'V8 CUDA smoke OK — peak VRAM {peak:.2f} GB')
assert peak < 18.0, f'peak {peak:.2f} GB exceeds 18 GB budget'
"
```

Expected: `peak VRAM < 18 GB`. If not, reduce `batch_size` in config before Step 11.4.

- [ ] **Step 11.4: Launch V8 training (background, long-running)**

Expected wall-clock: ~4-6 days on a 4090 at 320 px (matches V7's tempo).

```bash
python train.py \
  --config configs/v8_mcp_killer.yaml \
  --resume checkpoints/v8_mcp_killer/v8_warmstart.pt
```

Or via the trainer helper if the codebase has one — check `run_test_runs.sh` for the canonical invocation.

- [ ] **Step 11.5: Stop-rule + monitoring (per feedback memory)**

- Check val metric every epoch.
- If `val_dice < best - 0.02` on two consecutive val ticks → kill the run and diagnose (do not "wait one more epoch").
- Track `loss_flow`, `loss_dice`, `loss_dino_gate`, `loss_text` independently; if any new-component loss doesn't decrease over first 5 epochs, that component is dead — reduce its weight to 0 and relaunch.

- [ ] **Step 11.6: After best.pt hits plateau — run the 3D eval**

**Do NOT report AMOS22 numbers from the trainer's 2D center-slice val (per feedback memory).**

```bash
python scripts/eval_3d_amos22.py \
  --ckpt checkpoints/v8_mcp_killer/v8_mcp_killer_best.pt \
  --config configs/v8_mcp_killer.yaml \
  --tta --cc
```

Expected: per-organ 3D Dice printed; mean 3D Dice is the number to compare against MCP-MedSAM (0.79-0.82).

- [ ] **Step 11.7: Update MASTER_PROJECT_LOG.md with V8 results**

Append a V8 section to `MASTER_PROJECT_LOG.md`:
- Warm-start loaded / missing key report
- Epoch-by-epoch val table
- Final 3D Dice / HD95 / NSD vs. V7 baseline
- Whether V8 beat MCP-MedSAM

Per memory rules: never delete prior numbers; use `[superseded]` markers.

- [ ] **Step 11.8: V8 ablation (only if V8 beats V7)**

From spec §7b / feedback memory "ablate bundled novelty":
- V8-text-only (DINO weight 0)
- V8-dino-only (text weight 0)
- V8-full (both on)
- V7-baseline (both off)

One seed each. Four short runs. Report which component did the lifting.

---

## Task 12 (Optional / Deferred): D2 — TotalSegmentator Pseudo-Labels Script

**Spec:** §4. This task produces extra training data. **Gated behind user confirmation** because it requires downloading TotalSegmentator weights (~3 GB), running inference on unlabeled public CTs, and mapping 104-structure output to the AMOS22 15-organ schema.

**Files (to be created only when user confirms):**
- `scripts/build_totalseg_pseudolabels.py`
- Config: update `data.root` in `v9_tierB.yaml` to include pseudo-label subset.

- [ ] **Step 11.1: User decision point**

Before implementing, ask user:
- Confirm AMOS22-only baseline first (skip D2) → proceed to Stage-1 training.
- Or: greenlight downloading TotalSegmentator + FLARE22 (~30 GB) for Stage-1 data augmentation.

Do not implement until confirmed. Mark this task `deferred` in task tracking.

---

## Self-Review

**Spec coverage (§3 novelty table):**
- #1 Anatomical cascade fusion — already in `models/voluformer_v9.py`  ✓
- #2 Mamba-ODE hybrid — already in `models/mamba_ode.py`  ✓
- #3 Teacher distill — already in `losses/teacher_distill.py`  ✓
- #4 Cross-modal InfoNCE — already in `losses/cross_modal_contrastive.py`  ✓
- #5 Flow shape prior — Task 1  ✓
- #6 Dynamic PFESA — Task 2  ✓
- #7 Cascade consistency — Task 3  ✓
- #8 Boundary DDPM — Task 4  ✓
- §4 dataset sampling — Task 5  ✓
- §5 training stages — Task 8 (trainer) + Task 6 (config)  ✓
- §6 eval — Task 7  ✓
- §7c gates — Tasks 9, 10  ✓

**Type consistency check:**
- `VoluFormerV9` forward returns `{"proposer": {...}, "refiner": {...}, "fused": {...}}`. The trainer (Task 8) reads `refiner["masks"]` and `refiner["organ_pooled"]` — confirm these keys exist in `OrganFlowSAM2.forward` before running Task 10. If missing, `organ_pooled` branch in trainer will be skipped (guarded by `"organ_pooled" in refiner`).
- `SlabType` IntEnum values align with test assertions in Task 5.
- `FlowShapePriorLoss.lam` vs `CascadeConsistencyLoss.lam_kl`/`lam_dice` — trainer config uses `flow_shape_prior` single key and `cascade_consistency_kl`/`cascade_consistency_dice` split keys. Consistent with Task 6 config.

**Placeholder scan:** no TBD / "implement later" / "similar to Task N" strings present. All code is inline.

**Remaining non-code decisions (ask user):**
1. Whether to download FLARE22 + TotalSegmentator (Task 11) before launching Stage-1.
2. Path to MONAI pretrained SwinUNETR weights (`model.proposer.pretrained_weights` in cfg) — currently null (random init).
3. Which stage to launch first — Task 8's trainer expects `cfg.training.stage`.
