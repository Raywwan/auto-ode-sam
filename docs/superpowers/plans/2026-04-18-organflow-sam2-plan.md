# OrganFlow-SAM2 (VoluFormer3D V4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and train OrganFlow-SAM2 — a MedSAM2-based 15-organ abdominal CT segmentation model with flow-matched cross-slice ODE, anatomy-graph DETR decoder, PFESA++ — and launch the first training run to the ep5 go-gate (val_dice ≥ 0.15).

**Architecture:** MedSAM2 Hiera-Tiny encoder (frozen + LoRA rank-16) → PFESA++ → Flow-Matched Organ-Conditioned Bidirectional ODE → Anatomy-Graph Query Decoder (TwoWayTransformer depth 4, shared mask head, learnable 15×15 organ adjacency) → 15 masks + IoU + deepsup. ~59M trainable / ~97M total.

**Tech Stack:** PyTorch 2.x, CUDA 12.x, timm, nibabel, OmegaConf, tqdm, mixed precision (autocast + GradScaler). Environment: `C:\Users\Raywa\Desktop\LiteSAM3D\.venv`. MedSAM2 checkpoint via Hugging Face.

**Spec reference:** `C:\Users\Raywa\Desktop\VoluFormer3D_V4\docs\superpowers\specs\2026-04-18-organflow-sam2-design.md`

**Working directory for ALL commands:** `C:\Users\Raywa\Desktop\VoluFormer3D_V4`

**Python command (use verbatim everywhere):** `C:\Users\Raywa\Desktop\LiteSAM3D\.venv\Scripts\python`

---

## File Structure

### New files
| Path | Purpose |
|---|---|
| `models/medsam2_encoder.py` | MedSAM2 Hiera-Tiny loader with LoRA rank-16 wrapper, stage-1 + stage-2 feature tap |
| `models/pfesa_plus.py` | PFESA++ with learnable α, cutoff, steepness |
| `models/flow_cross_slice.py` | Flow-matched organ-conditioned bidirectional ODE |
| `models/anatomy_graph_decoder.py` | AnatomyGraphAttention + AnatomyGraphDecoder with shared mask head |
| `models/organflow_sam2.py` | Top-level `OrganFlowSAM2` model |
| `datasets/amos22_multiorgan.py` | Per-volume 15-channel dataset |
| `training/losses_v4.py` | Multi-organ mask + flow + deepsup + anatomy regularizer |
| `configs/v4_organflow_sam2_256px.yaml` | V4 config |
| `scripts/download_medsam2.py` | Download MedSAM2 Hiera-Tiny checkpoint |
| `scripts/smoke_test_v4.py` | End-to-end 2-volume forward+backward sanity check |
| `tests/test_pfesa_plus.py` | PFESA++ shape + param-count test |
| `tests/test_medsam2_encoder.py` | Encoder shape + LoRA trainable-param test |
| `tests/test_flow_cross_slice.py` | Flow ODE shape + gradient test |
| `tests/test_anatomy_graph_decoder.py` | Decoder shape + graph attention test |
| `tests/test_organflow_sam2.py` | Full forward pass shape test |
| `tests/test_amos22_multiorgan.py` | Dataset returns correct shape/dtype |
| `tests/test_losses_v4.py` | Loss math correctness |

### Modified files
| Path | Change |
|---|---|
| `models/__init__.py` | Register `OrganFlowSAM2`; remove V3 registrations at the end |
| `training/trainer.py` | Add V4 branch; V3 path left intact for rollback |

### Unchanged (reused from V3)
- `models/mask_decoder.py::TwoWayTransformer, MLP`
- `models/ode_cross_slice.py::ODEFunction` (used as the `f_θ` core inside flow module)
- `datasets/amos22.py` single-organ classes (kept for historical runs, V4 never calls them)
- `utils/`, `evaluation/`, `inference/`
- `train.py` top-level entrypoint

### To delete when V4 is working (NOT during initial plan execution)
- `models/auto_ode_sam.py`
- `models/organ_query_decoder.py`

---

## Task 0: Prerequisites and MedSAM2 checkpoint download

**Files:**
- Create: `scripts/download_medsam2.py`

- [ ] **Step 1: Verify environment**

Run:
```
C:\Users\Raywa\Desktop\LiteSAM3D\.venv\Scripts\python -c "import torch, timm; print(torch.__version__, torch.cuda.is_available(), timm.__version__)"
```
Expected: prints a torch version ≥ 2.0, `True`, a timm version ≥ 0.9. If not True, stop — GPU not visible.

- [ ] **Step 2: Create checkpoint download script**

Create `scripts/download_medsam2.py`:

```python
"""Download MedSAM2 Hiera-Tiny checkpoint to checkpoints/medsam2/."""
from pathlib import Path
from urllib.request import urlretrieve

MEDSAM2_URL = "https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_latest.pt"
OUT_DIR = Path(__file__).resolve().parents[1] / "checkpoints" / "medsam2"
OUT_PATH = OUT_DIR / "MedSAM2_hiera_tiny.pt"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUT_PATH.exists():
        print(f"Already exists: {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"Downloading MedSAM2 to {OUT_PATH} ...")
    urlretrieve(MEDSAM2_URL, OUT_PATH)
    print(f"Done: {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the download**

Run: `C:\Users\Raywa\Desktop\LiteSAM3D\.venv\Scripts\python scripts/download_medsam2.py`
Expected: file at `checkpoints/medsam2/MedSAM2_hiera_tiny.pt`, size between 150 MB and 600 MB.

- [ ] **Step 4: Load-smoke-test the checkpoint**

Run:
```
C:\Users\Raywa\Desktop\LiteSAM3D\.venv\Scripts\python -c "import torch; sd = torch.load('checkpoints/medsam2/MedSAM2_hiera_tiny.pt', map_location='cpu'); k = sd.get('model', sd); print(len(k), list(k.keys())[:5])"
```
Expected: prints a number > 100 and 5 keys starting with `image_encoder.` or similar. If it fails with unpickling error, inspect the file and switch `MEDSAM2_URL` to an alternative mirror (Hugging Face, GitHub release).

- [ ] **Step 5: Commit**

```
git init  # if not already a repo
git add scripts/download_medsam2.py
git commit -m "feat(v4): add MedSAM2 checkpoint downloader"
```

---

## Task 1: PFESA++ module

**Files:**
- Create: `models/pfesa_plus.py`
- Test: `tests/test_pfesa_plus.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_pfesa_plus.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\Raywa\Desktop\LiteSAM3D\.venv\Scripts\python -m pytest tests/test_pfesa_plus.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'models.pfesa_plus'`.

- [ ] **Step 3: Implement PFESA++**

Create `models/pfesa_plus.py`:

```python
"""PFESA++ — parameterized extension of MICCAI 2025 PFESA.

Learnable amplification α and soft radial high-frequency cutoff.
At init (alpha=1.0, cutoff_logit=0.0, steepness_log=log(10)) the behavior
matches the original PFESA α=1.0 cutoff=0.5 up to the sigmoid soft edge.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class PFESAPlus(nn.Module):
    """FFT-based high-frequency amplifier with 3 learnable scalars."""

    def __init__(self) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.cutoff_logit = nn.Parameter(torch.tensor(0.0))  # σ(0)=0.5
        self.steepness_log = nn.Parameter(torch.tensor(math.log(10.0)))

    def _radial_grid(self, H: int, W: int, device, dtype) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        r = torch.sqrt(xx * xx + yy * yy)
        return r.clamp_max(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → same shape."""
        B, C, H, W = x.shape
        r = self._radial_grid(H, W, x.device, torch.float32)
        cutoff = torch.sigmoid(self.cutoff_logit)
        steep = torch.exp(self.steepness_log)
        mask = torch.sigmoid(steep * (r - cutoff))  # soft high-pass
        x32 = x.float()
        F = torch.fft.fft2(x32, norm="ortho")
        F_enh = F * (1.0 + self.alpha * mask.unsqueeze(0).unsqueeze(0))
        y = torch.fft.ifft2(F_enh, norm="ortho").real
        return y.to(x.dtype)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\Users\Raywa\Desktop\LiteSAM3D\.venv\Scripts\python -m pytest tests/test_pfesa_plus.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```
git add models/pfesa_plus.py tests/test_pfesa_plus.py
git commit -m "feat(v4): add PFESA++ learnable FFT amplifier"
```

---

## Task 2: MedSAM2 encoder with LoRA wrapper

**Files:**
- Create: `models/medsam2_encoder.py`
- Test: `tests/test_medsam2_encoder.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_medsam2_encoder.py`:

```python
import torch
from models.medsam2_encoder import MedSAM2Encoder


def test_encoder_output_shape():
    enc = MedSAM2Encoder(embed_dim=256, lora_rank=16, pretrained=False)
    x = torch.randn(2, 3, 256, 256)
    main, skip = enc(x)
    assert main.shape == (2, 256, 16, 16), f"main {main.shape}"
    assert skip.shape == (2, 128, 32, 32), f"skip {skip.shape}"


def test_encoder_lora_only_trainable():
    enc = MedSAM2Encoder(embed_dim=256, lora_rank=16, pretrained=False)
    trainable = [n for n, p in enc.named_parameters() if p.requires_grad]
    # LoRA A/B, projection heads, skip projection must be trainable
    assert any("lora_A" in n for n in trainable)
    assert any("proj_main" in n for n in trainable)
    assert any("proj_skip" in n for n in trainable)
    # Backbone blocks must be frozen
    frozen = [n for n, p in enc.named_parameters() if not p.requires_grad]
    assert len(frozen) > 0


def test_encoder_trainable_param_count_reasonable():
    enc = MedSAM2Encoder(embed_dim=256, lora_rank=16, pretrained=False)
    n_trainable = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    # LoRA deltas (~4M) + projection heads (~0.2M) < total (~38M)
    assert 1e6 < n_trainable < 10e6, f"Trainable {n_trainable} outside expected band"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\Raywa\Desktop\LiteSAM3D\.venv\Scripts\python -m pytest tests/test_medsam2_encoder.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the encoder**

Create `models/medsam2_encoder.py`:

```python
"""MedSAM2 Hiera-Tiny encoder with LoRA rank-16 adaptation.

Loads MedSAM2 pretrained weights (if present), freezes backbone, injects LoRA
deltas on attention qkv + projection layers of every block. Taps stride-8
(stage 1, 192 ch) and stride-16 (stage 2, 384 ch) feature maps for decoder
consumption.

Implementation uses timm's `hiera_tiny` (matches SAM2/MedSAM2 backbone) via
`features_only=True` to keep multi-scale outputs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError as e:
    raise ImportError("timm is required for MedSAM2Encoder. Install: pip install timm") from e


class LoRALinear(nn.Module):
    """y = W x + (B A) x · (α / r). Wraps an existing nn.Linear (frozen)."""

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = rank
        self.scale = alpha / rank
        in_f = base.in_features
        out_f = base.out_features
        self.lora_A = nn.Parameter(torch.zeros(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        # lora_B stays zero → LoRA contribution starts at zero

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        delta = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale
        return y + delta


def _inject_lora(module: nn.Module, rank: int) -> int:
    """Replace every nn.Linear whose name ends in {'qkv','proj'} with LoRALinear. Returns count."""
    n_replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in ("qkv", "proj"):
            setattr(module, name, LoRALinear(child, rank=rank))
            n_replaced += 1
        else:
            n_replaced += _inject_lora(child, rank)
    return n_replaced


class MedSAM2Encoder(nn.Module):
    """Hiera-Tiny backbone with LoRA + stage-1/stage-2 taps.

    Args:
        embed_dim: target channels for main tap (projects 384 → embed_dim).
        skip_channels: target channels for skip tap (projects 192 → skip_channels).
        lora_rank: LoRA rank (default 16).
        pretrained: if True, try to load `checkpoints/medsam2/MedSAM2_hiera_tiny.pt`.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        skip_channels: int = 128,
        lora_rank: int = 16,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "hiera_tiny_224",
            pretrained=False,
            features_only=True,
            out_indices=[1, 2],  # stride-8 (192ch) and stride-16 (384ch)
        )
        fi = self.backbone.feature_info
        skip_in_ch = fi[0]["num_chs"]
        main_in_ch = fi[1]["num_chs"]

        if pretrained:
            ckpt_path = Path(__file__).resolve().parents[1] / "checkpoints" / "medsam2" / "MedSAM2_hiera_tiny.pt"
            if ckpt_path.exists():
                state = torch.load(str(ckpt_path), map_location="cpu")
                sd = state.get("model", state)
                # Extract just image_encoder keys if the ckpt is a full SAM2 state
                sd_enc = {}
                for k, v in sd.items():
                    if k.startswith("image_encoder.trunk."):
                        sd_enc[k.replace("image_encoder.trunk.", "")] = v
                    elif k.startswith("trunk."):
                        sd_enc[k.replace("trunk.", "")] = v
                if sd_enc:
                    missing, unexpected = self.backbone.load_state_dict(sd_enc, strict=False)
                    print(f"[MedSAM2Encoder] loaded {len(sd_enc)} keys, missing={len(missing)}, unexpected={len(unexpected)}")
                else:
                    print(f"[MedSAM2Encoder] warning: no 'image_encoder.trunk.' keys in checkpoint — using random init.")
            else:
                print(f"[MedSAM2Encoder] warning: checkpoint not found at {ckpt_path} — using random init.")

        # Freeze all backbone params, then inject LoRA (which un-freezes its own params)
        for p in self.backbone.parameters():
            p.requires_grad = False
        n_lora = _inject_lora(self.backbone, rank=lora_rank)
        print(f"[MedSAM2Encoder] injected LoRA into {n_lora} Linear layers (rank={lora_rank})")

        # Projection heads (trainable)
        self.proj_main = nn.Conv2d(main_in_ch, embed_dim, kernel_size=1)
        self.proj_skip = nn.Conv2d(skip_in_ch, skip_channels, kernel_size=1)
        self.skip_channels = skip_channels
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (B, 3, H, W) → (main (B, embed, H/16, W/16), skip (B, skip_ch, H/8, W/8))."""
        feats = self.backbone(x)
        skip_raw, main_raw = feats[0], feats[1]  # (B, 192, H/8, W/8), (B, 384, H/16, W/16)
        # Hiera may return NHWC in some timm versions; normalize to NCHW
        if skip_raw.dim() == 4 and skip_raw.shape[-1] in (192, 96, 384, 768):
            skip_raw = skip_raw.permute(0, 3, 1, 2).contiguous()
        if main_raw.dim() == 4 and main_raw.shape[-1] in (192, 96, 384, 768):
            main_raw = main_raw.permute(0, 3, 1, 2).contiguous()
        main = self.proj_main(main_raw)
        skip = self.proj_skip(skip_raw)
        return main, skip

    def get_output_size(self) -> Tuple[int, int]:
        """Main-tap spatial size for input 256×256 = 16×16."""
        return 16, 16
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\Users\Raywa\Desktop\LiteSAM3D\.venv\Scripts\python -m pytest tests/test_medsam2_encoder.py -v`
Expected: 3 PASS. If `timm.create_model("hiera_tiny_224", ...)` fails with "Unknown model", upgrade timm: `C:\Users\Raywa\Desktop\LiteSAM3D\.venv\Scripts\pip install -U timm` — Hiera lands in timm ≥ 1.0.

- [ ] **Step 5: Commit**

```
git add models/medsam2_encoder.py tests/test_medsam2_encoder.py
git commit -m "feat(v4): MedSAM2 Hiera-Tiny encoder with LoRA rank-16"
```

---

## Task 3: Flow-matched cross-slice ODE module

**Files:**
- Create: `models/flow_cross_slice.py`
- Test: `tests/test_flow_cross_slice.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_flow_cross_slice.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\Raywa\Desktop\LiteSAM3D\.venv\Scripts\python -m pytest tests/test_flow_cross_slice.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the flow-matched ODE**

Create `models/flow_cross_slice.py`:

```python
"""Flow-matched organ-conditioned bidirectional ODE for cross-slice feature evolution.

The module does three things simultaneously on the forward pass:
  1. Integrates the ODE forward from slice 0 and backward from slice D-1 using Heun's method.
  2. Merges the two trajectories with a zero-init Linear (residual add → identity at init).
  3. Collects (v_θ(h_i, t_i), finite-difference target) pairs so the trainer can compute L_flow.

L_flow is NOT computed here — it is computed in training/losses_v4.py using the pairs this
module returns. This keeps the module stateless wrt training schedules.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from models.ode_cross_slice import ODEFunction


class OrganConditionedFlowFunction(nn.Module):
    """dh/dt = base_ode(h, t) + MLP(organ_embed[organ_id])."""

    def __init__(self, dim: int, n_organs: int = 15, organ_emb_dim: int = 32,
                 hidden: Optional[int] = None, n_freqs: int = 4) -> None:
        super().__init__()
        self.dim = dim
        self.n_organs = n_organs
        self.base = ODEFunction(dim=dim, hidden=hidden, n_freqs=n_freqs)
        self.organ_embed = nn.Embedding(n_organs + 1, organ_emb_dim)
        self.bias_mlp = nn.Linear(organ_emb_dim, dim)
        nn.init.zeros_(self.bias_mlp.weight)
        nn.init.zeros_(self.bias_mlp.bias)

    def forward(self, h: torch.Tensor, t: float, organ_id: torch.Tensor) -> torch.Tensor:
        N = h.shape[0]
        B = organ_id.shape[0]
        HW = N // B
        ids = organ_id.repeat_interleave(HW)
        bias = self.bias_mlp(self.organ_embed(ids))
        return self.base(h, t) + bias


class FlowMatchedOrganConditionedODE(nn.Module):
    """Bidirectional organ-conditioned ODE + flow-matching target collection."""

    def __init__(self, dim: int, n_organs: int = 15, organ_emb_dim: int = 32,
                 ode_hidden: Optional[int] = None, n_freqs: int = 4, substeps: int = 4) -> None:
        super().__init__()
        self.substeps = substeps
        self.ode_fwd = OrganConditionedFlowFunction(dim, n_organs, organ_emb_dim, ode_hidden, n_freqs)
        self.ode_bwd = OrganConditionedFlowFunction(dim, n_organs, organ_emb_dim, ode_hidden, n_freqs)
        self.merge = nn.Linear(dim * 2, dim, bias=False)
        nn.init.zeros_(self.merge.weight)
        self.norm = nn.LayerNorm(dim)

    def _heun(self, func, h0, D, organ_id):
        t_pts = [d / max(D - 1, 1) for d in range(D)]
        h, states = h0, [h0]
        for i in range(1, D):
            t0, t1 = t_pts[i - 1], t_pts[i]
            dt = (t1 - t0) / self.substeps
            for j in range(self.substeps):
                tc = t0 + j * dt
                k1 = func(h, tc, organ_id)
                k2 = func(h + dt * k1, tc + dt, organ_id)
                h = h + 0.5 * dt * (k1 + k2)
            states.append(h)
        return torch.stack(states, dim=1)

    def _flow_pairs(self, func, x, D, organ_id, direction: str) -> Dict[str, torch.Tensor]:
        """Collect (v_θ(h_i, t_i), (h_{i±1} - h_i) · (D-1)) pairs at slice timestamps."""
        t_pts = [d / max(D - 1, 1) for d in range(D)]
        preds, targets = [], []
        scale = float(D - 1)
        if direction == "fwd":
            for i in range(D - 1):
                preds.append(func(x[:, i], t_pts[i], organ_id))
                targets.append((x[:, i + 1] - x[:, i]) * scale)
        else:  # bwd
            for i in range(1, D):
                preds.append(func(x[:, i], t_pts[i], organ_id))
                targets.append((x[:, i - 1] - x[:, i]) * scale)
        return {
            "pred": torch.stack(preds, dim=1),
            "target": torch.stack(targets, dim=1),
        }

    def forward(
        self,
        features: torch.Tensor,       # (B, D, C, H, W)
        organ_id: torch.Tensor,       # (B,) long
        return_flow_targets: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Dict[str, torch.Tensor]]]]:
        assert organ_id.shape[0] == features.shape[0]
        B, D, C, H, W = features.shape
        x = features.permute(0, 3, 4, 1, 2).reshape(B * H * W, D, C)

        traj_fwd = self._heun(self.ode_fwd, x[:, 0], D, organ_id)
        traj_bwd = self._heun(self.ode_bwd, x[:, -1], D, organ_id)
        traj_bwd = torch.flip(traj_bwd, dims=[1])

        ode_ctx = self.merge(torch.cat([traj_fwd, traj_bwd], dim=-1))
        ode_ctx = ode_ctx.reshape(B, H, W, D, C).permute(0, 3, 4, 1, 2)
        ode_ctx = ode_ctx.permute(0, 1, 3, 4, 2)
        ode_ctx = self.norm(ode_ctx)
        ode_ctx = ode_ctx.permute(0, 1, 4, 2, 3)
        out = features + ode_ctx

        flow_targets: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
        if return_flow_targets:
            flow_targets = {
                "fwd_pairs": self._flow_pairs(self.ode_fwd, x, D, organ_id, "fwd"),
                "bwd_pairs": self._flow_pairs(self.ode_bwd, x, D, organ_id, "bwd"),
            }
        return out, flow_targets
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\Users\Raywa\Desktop\LiteSAM3D\.venv\Scripts\python -m pytest tests/test_flow_cross_slice.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```
git add models/flow_cross_slice.py tests/test_flow_cross_slice.py
git commit -m "feat(v4): flow-matched organ-conditioned cross-slice ODE"
```

---

## Task 4: Anatomy-Graph Decoder

**Files:**
- Create: `models/anatomy_graph_decoder.py`
- Test: `tests/test_anatomy_graph_decoder.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_anatomy_graph_decoder.py`:

```python
import torch
from models.anatomy_graph_decoder import AnatomyGraphAttention, AnatomyGraphDecoder


def test_graph_attention_shape():
    ga = AnatomyGraphAttention(n_organs=15, embed_dim=256)
    q = torch.randn(15, 256)
    q_out = ga(q)
    assert q_out.shape == q.shape


def test_graph_attention_identity_if_adjacency_zero():
    ga = AnatomyGraphAttention(n_organs=15, embed_dim=256)
    # sigmoid(-100) ≈ 0 → message is ≈ 0 → output ≈ q (residual)
    with torch.no_grad():
        ga.adjacency.fill_(-100.0)
    q = torch.randn(15, 256)
    q_out = ga(q)
    assert torch.allclose(q_out, q, atol=1e-3)


def test_decoder_output_shape():
    dec = AnatomyGraphDecoder(
        embed_dim=256, n_organs=15, skip_channels=128,
        transformer_depth=4, transformer_mlp_dim=2048,
    )
    B, C, H, W = 2, 256, 16, 16
    image_emb = torch.randn(B, C, H, W)
    dense_pe = torch.randn(B, C, H, W)
    skip = torch.randn(B, 128, 32, 32)
    masks, iou = dec(image_emb, dense_pe, skip)
    assert masks.shape == (B, 15, 64, 64)
    assert iou.shape == (B, 15)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\Raywa\Desktop\LiteSAM3D\.venv\Scripts\python -m pytest tests/test_anatomy_graph_decoder.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the decoder**

Create `models/anatomy_graph_decoder.py`:

```python
"""Anatomy-Graph Query Decoder.

Differences from V3 `OrganQueryDecoder`:
  - Anatomy-graph message pass on organ queries before the TwoWayTransformer
  - SINGLE shared mask MLP (V3 had 15 separate, which collapsed to identical weights)
  - SINGLE shared IoU MLP
  - HQ token + hq_skip_proj DELETED (dead in every V3 checkpoint)
  - Stage-1 skip + Haar edge enhancement kept (V3 feature that worked)
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mask_decoder import TwoWayTransformer, MLP


# AMOS22 anatomical adjacency prior (1-indexed organ IDs).
# Pairs are symmetric. (organ_a, organ_b).
AMOS22_ADJACENCY = [
    (6, 2), (6, 7), (6, 4),           # liver — right kidney, stomach, gallbladder
    (1, 3), (3, 12), (2, 11),         # spleen — left kidney, kidneys — adrenals
    (8, 9),                           # aorta — IVC
    (10, 7), (10, 13), (7, 13),       # pancreas — stomach, duodenum; stomach — duodenum
    (14, 15),                         # bladder — prostate/uterus
    (5, 7),                           # esophagus — stomach
]


def _build_adjacency_init(n_organs: int) -> torch.Tensor:
    """15×15 tensor of adjacency logits. Init ~2.0 for adjacent, ~-2.0 otherwise.

    sigmoid(2) ≈ 0.88, sigmoid(-2) ≈ 0.12 — nonzero everywhere so gradients flow.
    """
    A = torch.full((n_organs, n_organs), -2.0)
    for a, b in AMOS22_ADJACENCY:
        A[a - 1, b - 1] = 2.0
        A[b - 1, a - 1] = 2.0
    A.fill_diagonal_(0.0)  # no self-loops in message pass
    return A


class AnatomyGraphAttention(nn.Module):
    """One GAT-style message pass with learnable 15×15 adjacency."""

    def __init__(self, n_organs: int = 15, embed_dim: int = 256) -> None:
        super().__init__()
        self.n_organs = n_organs
        self.adjacency = nn.Parameter(_build_adjacency_init(n_organs))
        self.w_msg = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.zeros_(self.w_msg.weight)  # start as identity residual

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """q: (n_organs, embed_dim) → (n_organs, embed_dim)."""
        A_soft = torch.sigmoid(self.adjacency)
        m = A_soft @ q
        return q + self.w_msg(m)


class AnatomyGraphDecoder(nn.Module):
    """TwoWayTransformer + anatomy-graph queries + shared mask head."""

    def __init__(
        self,
        embed_dim: int = 256,
        n_organs: int = 15,
        skip_channels: int = 128,
        transformer_depth: int = 4,
        transformer_mlp_dim: int = 2048,
    ) -> None:
        super().__init__()
        self.n_organs = n_organs
        self.organ_queries = nn.Embedding(n_organs, embed_dim)
        nn.init.normal_(self.organ_queries.weight, std=0.02)
        self.graph = AnatomyGraphAttention(n_organs, embed_dim)
        self.transformer = TwoWayTransformer(
            depth=transformer_depth, embedding_dim=embed_dim,
            num_heads=8, mlp_dim=transformer_mlp_dim,
        )
        self.upsample_conv1 = nn.ConvTranspose2d(embed_dim, embed_dim // 4, kernel_size=2, stride=2)
        self.upsample_ln = nn.LayerNorm(embed_dim // 4)
        self.upsample_conv2 = nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=2, stride=2)
        self.skip_gate = nn.Parameter(torch.full((1,), 0.05))
        self.skip_ln = nn.LayerNorm(skip_channels)
        self.mask_mlp = MLP(embed_dim, embed_dim, embed_dim // 8, depth=3)
        self.iou_mlp = MLP(embed_dim, 256, 1, depth=3)

    @staticmethod
    def _haar_edge(x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2], x.shape[-1]
        pad_h, pad_w = h % 2, w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        x00 = x[:, :, 0::2, 0::2]; x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]; x11 = x[:, :, 1::2, 1::2]
        LH = (x00 - x01 + x10 - x11) * 0.25
        HL = (x00 + x01 - x10 - x11) * 0.25
        HH = (x00 - x01 - x10 + x11) * 0.25
        hf = LH.abs() + HL.abs() + HH.abs()
        up = F.interpolate(hf, size=(h + pad_h, w + pad_w), mode="bilinear", align_corners=False)
        return (x + 0.5 * up)[:, :, :h, :w]

    def forward(
        self,
        image_emb: torch.Tensor,   # (B, C, H, W)
        dense_pe: torch.Tensor,    # (B, C, H, W)
        skip: torch.Tensor,        # (B, skip_ch, 2H, 2W)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = image_emb.shape
        q = self.organ_queries.weight          # (15, C)
        q = self.graph(q)                       # anatomy-graph message pass
        queries = q.unsqueeze(0).expand(B, -1, -1).contiguous()
        q_pe = torch.zeros_like(queries)
        src = image_emb.flatten(2).permute(0, 2, 1)
        pe = dense_pe.flatten(2).permute(0, 2, 1)
        hs, src_updated = self.transformer(src, pe, queries, q_pe)
        src_spatial = src_updated.permute(0, 2, 1).reshape(B, C, H, W)
        up = self.upsample_conv1(src_spatial)
        up = up.permute(0, 2, 3, 1)
        up = self.upsample_ln(up)
        up = up.permute(0, 3, 1, 2)
        up = F.gelu(up)
        if skip.shape[-2:] != up.shape[-2:]:
            skip = F.interpolate(skip.float(), size=up.shape[-2:], mode="bilinear", align_corners=False).to(up.dtype)
        sf_flat = skip.permute(0, 2, 3, 1)
        sf_flat = self.skip_ln(sf_flat).permute(0, 3, 1, 2)
        sf = self._haar_edge(sf_flat)
        # project skip to up.channels via 1×1 conv on the fly (stateless)
        if sf.shape[1] != up.shape[1]:
            proj = nn.functional.conv2d(sf, torch.zeros(up.shape[1], sf.shape[1], 1, 1, device=sf.device, dtype=sf.dtype))
            # ^ zero projection → no-op; effectively skip disabled if channel mismatch.
            # Proper sized projection must be registered in __init__ if skip_channels ≠ embed_dim//4.
            sf = proj
        up = up + self.skip_gate * sf
        up = F.gelu(self.upsample_conv2(up))
        b, c, h, w = up.shape
        weights = self.mask_mlp(hs[:, :self.n_organs, :])          # (B, 15, C//8)
        masks = (weights @ up.view(b, c, h * w)).view(b, self.n_organs, h, w)
        iou = self.iou_mlp(hs[:, :self.n_organs, :]).squeeze(-1)   # (B, 15)
        return masks, iou
```

- [ ] **Step 4: Fix skip-channel projection (design cleanup)**

Edit `models/anatomy_graph_decoder.py` — replace the ad-hoc skip channel handling with a real projection.

Replace this block in `__init__`:
```python
        self.skip_gate = nn.Parameter(torch.full((1,), 0.05))
        self.skip_ln = nn.LayerNorm(skip_channels)
```
with:
```python
        self.skip_gate = nn.Parameter(torch.full((1,), 0.05))
        self.skip_ln = nn.LayerNorm(skip_channels)
        self.skip_proj = nn.Conv2d(skip_channels, embed_dim // 4, kernel_size=1)
```

Replace this block in `forward` (the `if sf.shape[1] != up.shape[1]:` block):
```python
        if sf.shape[1] != up.shape[1]:
            proj = nn.functional.conv2d(sf, torch.zeros(up.shape[1], sf.shape[1], 1, 1, device=sf.device, dtype=sf.dtype))
            sf = proj
```
with:
```python
        sf = self.skip_proj(sf)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `C:\Users\Raywa\Desktop\LiteSAM3D\.venv\Scripts\python -m pytest tests/test_anatomy_graph_decoder.py -v`
Expected: 3 PASS.

- [ ] **Step 6: Commit**

```
git add models/anatomy_graph_decoder.py tests/test_anatomy_graph_decoder.py
git commit -m "feat(v4): anatomy-graph decoder with shared mask head"
```

---

## Task 5: Top-level OrganFlowSAM2 model

**Files:**
- Create: `models/organflow_sam2.py`
- Modify: `models/__init__.py`
- Test: `tests/test_organflow_sam2.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_organflow_sam2.py`:

```python
import torch
from omegaconf import OmegaConf
from models.organflow_sam2 import OrganFlowSAM2


def _cfg():
    return OmegaConf.create({
        "model": {
            "architecture": "organflow_sam2",
            "img_size": 256,
            "embed_dim": 256,
            "skip_channels": 128,
            "encoder": {"pretrained": False, "lora_rank": 16},
            "pfesa": {},
            "ode": {"n_organs": 15, "organ_emb_dim": 32, "ode_hidden": 128, "n_freqs": 6, "substeps": 4},
            "decoder": {"n_organs": 15, "transformer_depth": 4, "transformer_mlp_dim": 2048},
            "graph": {"n_organs": 15},
        }
    })


def test_forward_shape():
    m = OrganFlowSAM2(_cfg())
    images = torch.randn(1, 4, 3, 256, 256)  # (B, D, C, H, W) with D=4 for speed
    organ_id = torch.tensor([6])
    out = m(images, organ_id, is_3d=True)
    assert out["masks"].shape == (1, 15, 64, 64)
    assert out["iou_pred"].shape == (1, 15)
    assert out["deepsup_logits"].shape[:2] == (1, 15)
    assert out["flow_targets"] is not None


def test_param_budget():
    m = OrganFlowSAM2(_cfg())
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in m.parameters())
    print(f"trainable={n_train/1e6:.1f}M total={n_total/1e6:.1f}M")
    assert 40e6 < n_train < 80e6, "trainable params outside expected 40–80M band"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\Raywa\Desktop\LiteSAM3D\.venv\Scripts\python -m pytest tests/test_organflow_sam2.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the top-level model**

Create `models/organflow_sam2.py`:

```python
"""OrganFlow-SAM2 — V4 top-level model.

Wires MedSAM2Encoder → PFESAPlus → FlowMatchedOrganConditionedODE →
AnatomyGraphDecoder plus a DeepSupervisionHead at ODE midpoint.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.medsam2_encoder import MedSAM2Encoder
from models.pfesa_plus import PFESAPlus
from models.flow_cross_slice import FlowMatchedOrganConditionedODE
from models.anatomy_graph_decoder import AnatomyGraphDecoder


class DeepSupervisionHead(nn.Module):
    def __init__(self, embed_dim: int, n_organs: int = 15) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 4, 1),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, 2, 2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 8, n_organs, 2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class OrganFlowSAM2(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        mcfg = cfg.model
        self.cfg = cfg
        self.embed_dim = mcfg.embed_dim
        self.n_organs = mcfg.ode.n_organs

        self.encoder = MedSAM2Encoder(
            embed_dim=mcfg.embed_dim,
            skip_channels=mcfg.skip_channels,
            lora_rank=mcfg.encoder.lora_rank,
            pretrained=mcfg.encoder.pretrained,
        )
        self.pfesa = PFESAPlus()
        self.ode = FlowMatchedOrganConditionedODE(
            dim=mcfg.embed_dim,
            n_organs=mcfg.ode.n_organs,
            organ_emb_dim=mcfg.ode.organ_emb_dim,
            ode_hidden=mcfg.ode.ode_hidden,
            n_freqs=mcfg.ode.n_freqs,
            substeps=mcfg.ode.substeps,
        )
        self.deep_sup = DeepSupervisionHead(mcfg.embed_dim, n_organs=mcfg.ode.n_organs)
        self.decoder = AnatomyGraphDecoder(
            embed_dim=mcfg.embed_dim,
            n_organs=mcfg.decoder.n_organs,
            skip_channels=mcfg.skip_channels,
            transformer_depth=mcfg.decoder.transformer_depth,
            transformer_mlp_dim=mcfg.decoder.transformer_mlp_dim,
        )
        feat_size = self.encoder.get_output_size()[0]
        self.register_buffer("dense_pe", self._make_dense_pe(mcfg.embed_dim, feat_size))

    @staticmethod
    def _make_dense_pe(embed_dim: int, feat_size: int) -> torch.Tensor:
        half = embed_dim // 2
        grid = torch.arange(feat_size, dtype=torch.float32) / feat_size
        x, y = torch.meshgrid(grid, grid, indexing="ij")
        dim_t = torch.arange(half, dtype=torch.float32)
        dim_t = 10000 ** (2 * dim_t / half)
        pe_x = torch.sin(x.unsqueeze(-1) / dim_t.unsqueeze(0).unsqueeze(0))
        pe_y = torch.cos(y.unsqueeze(-1) / dim_t.unsqueeze(0).unsqueeze(0))
        pe = torch.cat([pe_x, pe_y], dim=-1)
        return pe.permute(2, 0, 1).unsqueeze(0)

    def forward(
        self,
        images: torch.Tensor,         # (B, D, 3, H, W)  (or (B*D, 3, H, W))
        organ_id: torch.Tensor,       # (B,) long
        is_3d: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if images.dim() == 5:
            B, D, C, H, W = images.shape
            images_2d = images.reshape(B * D, C, H, W)
        else:
            images_2d = images
            B = organ_id.shape[0]
            D = images_2d.shape[0] // B

        feat2d, skip2d = self.encoder(images_2d)        # (B*D, 256, 16, 16), (B*D, 128, 32, 32)
        feat2d = self.pfesa(feat2d)
        feat3d = feat2d.reshape(B, D, self.embed_dim, feat2d.shape[-2], feat2d.shape[-1])

        feat3d_out, flow_targets = self.ode(feat3d, organ_id, return_flow_targets=self.training)

        mid = D // 2
        center_feat = feat3d_out[:, mid]
        center_skip = skip2d.reshape(B, D, skip2d.shape[1], skip2d.shape[2], skip2d.shape[3])[:, mid]

        dense_pe = self.dense_pe.to(center_feat.dtype).expand(B, -1, -1, -1)
        if dense_pe.shape[-2:] != center_feat.shape[-2:]:
            dense_pe = F.interpolate(dense_pe.float(), size=center_feat.shape[-2:],
                                      mode="bilinear", align_corners=False).to(center_feat.dtype)

        masks, iou = self.decoder(center_feat, dense_pe, center_skip)
        deepsup = self.deep_sup(feat3d_out[:, mid].contiguous())
        if deepsup.shape[-2:] != masks.shape[-2:]:
            deepsup = F.interpolate(deepsup, size=masks.shape[-2:], mode="bilinear", align_corners=False)

        return {
            "masks": masks,
            "iou_pred": iou,
            "deepsup_logits": deepsup,
            "flow_targets": flow_targets,
        }
```

- [ ] **Step 4: Register model in models/__init__.py**

Open `models/__init__.py` and add at the end:

```python
from models.organflow_sam2 import OrganFlowSAM2  # noqa: F401
```

- [ ] **Step 5: Run test to verify it passes**

Run: `C:\Users\Raywa\Desktop\LiteSAM3D\.venv\Scripts\python -m pytest tests/test_organflow_sam2.py -v -s`
Expected: 2 PASS. The `-s` flag prints the param-count line so you can record it.

- [ ] **Step 6: Commit**

```
git add models/organflow_sam2.py models/__init__.py tests/test_organflow_sam2.py
git commit -m "feat(v4): top-level OrganFlowSAM2 model"
```

---

## Task 6: Multi-organ volume dataset

**Files:**
- Create: `datasets/amos22_multiorgan.py`
- Test: `tests/test_amos22_multiorgan.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_amos22_multiorgan.py`:

```python
import numpy as np
import pytest
import torch
from datasets.amos22_multiorgan import AMOS22MultiOrgan3D_Dataset

DATA_ROOT = r"C:\Users\Raywa\Desktop\LiteSAM3D\data\amos22"


@pytest.mark.skipif(not __import__("pathlib").Path(DATA_ROOT).exists(),
                    reason="AMOS22 data not present on this machine")
def test_dataset_shape_and_dtype():
    ds = AMOS22MultiOrgan3D_Dataset(
        data_root=DATA_ROOT, split="train",
        img_size=256, depth=8, modality="ct",
    )
    assert len(ds) > 0
    sample = ds[0]
    assert sample["image"].shape == (8, 1, 256, 256)
    assert sample["masks"].shape == (15, 8, 256, 256)
    assert sample["present_mask"].shape == (15,)
    assert sample["image"].dtype == torch.float32
    assert sample["masks"].dtype == torch.uint8
    # at least ONE organ must be present for a training sample
    assert sample["present_mask"].sum() >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\Raywa\Desktop\LiteSAM3D\.venv\Scripts\python -m pytest tests/test_amos22_multiorgan.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the dataset**

Create `datasets/amos22_multiorgan.py`:

```python
"""AMOS22 per-volume 15-organ dataset.

Returns a depth-D stack of axial slices centered on a slice that has the most
organs present (heuristic: pick centers that contain ≥5 organs with ≥50 voxels).

Each sample provides all 15 binary organ masks, making multi-organ supervision
one-forward-pass.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


class AMOS22MultiOrgan3D_Dataset(Dataset):
    N_ORGANS = 15

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        img_size: int = 256,
        depth: int = 8,
        modality: str = "ct",
        hu_clip: tuple = (-200, 250),
    ) -> None:
        self.root = Path(data_root)
        self.split = split
        self.img_size = img_size
        self.depth = depth
        self.modality = modality
        self.hu_clip = hu_clip
        self.samples: List[Dict] = self._build_index()

    def _build_index(self) -> List[Dict]:
        out: List[Dict] = []
        img_dir = self.root / ("imagesTr" if self.split == "train" else "imagesVa")
        lbl_dir = self.root / ("labelsTr" if self.split == "train" else "labelsVa")
        half = self.depth // 2
        if not img_dir.exists() or not lbl_dir.exists():
            raise FileNotFoundError(f"AMOS22 split dirs not found under {self.root}")
        for img_path in sorted(img_dir.glob("*.nii.gz")):
            # Modality filter: CT IDs start with 'amos_' and lie in 0001..0500 (CT), 0501..0600 (MRI)
            stem = img_path.name.replace(".nii.gz", "")
            try:
                case_id = int(stem.split("_")[-1])
            except ValueError:
                continue
            if self.modality == "ct" and not (1 <= case_id <= 500):
                continue
            if self.modality == "mri" and not (501 <= case_id <= 600):
                continue
            lbl_path = lbl_dir / img_path.name
            if not lbl_path.exists():
                continue
            try:
                lbl = nib.load(str(lbl_path))
                n_slices = lbl.shape[2]
                if n_slices < self.depth:
                    continue
                out.append({
                    "image_path": str(img_path),
                    "label_path": str(lbl_path),
                    "n_slices": n_slices,
                })
            except Exception as e:  # noqa: BLE001
                print(f"  skip {img_path.name}: {e}")
                continue
        return out

    def __len__(self) -> int:
        return len(self.samples)

    def _normalize_image(self, x: np.ndarray) -> np.ndarray:
        lo, hi = self.hu_clip
        x = np.clip(x, lo, hi)
        x = (x - lo) / max(hi - lo, 1)
        return x.astype(np.float32)

    def _resize_2d(self, arr: np.ndarray, order: int) -> np.ndarray:
        from scipy.ndimage import zoom
        h, w = arr.shape
        zh = self.img_size / h
        zw = self.img_size / w
        return zoom(arr, (zh, zw), order=order)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        img_nib = nib.load(s["image_path"])
        lbl_nib = nib.load(s["label_path"])
        img_vol = np.asarray(img_nib.dataobj, dtype=np.float32)    # (H, W, Z)
        lbl_vol = np.asarray(lbl_nib.dataobj, dtype=np.uint8)      # (H, W, Z)

        H, W, Z = img_vol.shape
        half = self.depth // 2

        # Pick center slice that maximizes organs-present (clamped to valid range)
        counts_per_slice = np.zeros(Z, dtype=np.int32)
        for k in range(1, self.N_ORGANS + 1):
            has_k = (lbl_vol == k).reshape(H * W, Z).sum(axis=0) >= 50
            counts_per_slice += has_k.astype(np.int32)
        valid = np.arange(half, Z - half)
        if valid.size == 0:
            center = Z // 2
        else:
            best = valid[np.argmax(counts_per_slice[valid])]
            center = int(best)

        z0, z1 = center - half, center + half  # [z0, z1) is depth slice range
        img_stack = img_vol[..., z0:z1]                 # (H, W, D)
        lbl_stack = lbl_vol[..., z0:z1]                 # (H, W, D)

        # Resize each slice
        img_resized = np.stack([self._resize_2d(img_stack[..., d], order=1) for d in range(self.depth)], axis=-1)
        lbl_resized = np.stack([self._resize_2d(lbl_stack[..., d], order=0) for d in range(self.depth)], axis=-1)

        img_norm = self._normalize_image(img_resized)   # (H, W, D)
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(1)  # (D, 1, H, W)

        # Build per-organ binary masks (15, D, H, W)
        masks = np.zeros((self.N_ORGANS, self.depth, self.img_size, self.img_size), dtype=np.uint8)
        for k in range(1, self.N_ORGANS + 1):
            masks[k - 1] = (lbl_resized == k).transpose(2, 0, 1).astype(np.uint8)
        masks_tensor = torch.from_numpy(masks)

        present_mask = torch.tensor(
            [masks_tensor[k].sum() >= 50 for k in range(self.N_ORGANS)],
            dtype=torch.bool,
        )

        return {
            "image": img_tensor,
            "masks": masks_tensor,
            "present_mask": present_mask,
            "case_id": Path(s["image_path"]).name.replace(".nii.gz", ""),
            "center_slice": center,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\Users\Raywa\Desktop\LiteSAM3D\.venv\Scripts\python -m pytest tests/test_amos22_multiorgan.py -v`
Expected: 1 PASS (or SKIP if data not on the machine — that's fine, we'll confirm end-to-end in Task 10).

- [ ] **Step 5: Commit**

```
git add datasets/amos22_multiorgan.py tests/test_amos22_multiorgan.py
git commit -m "feat(v4): per-volume multi-organ AMOS22 dataset"
```

---

## Task 7: V4 losses

**Files:**
- Create: `training/losses_v4.py`
- Test: `tests/test_losses_v4.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_losses_v4.py`:

```python
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
    # Set pred = target → flow loss = 0
    for d in ("fwd_pairs", "bwd_pairs"):
        flow_targets[d]["pred"] = flow_targets[d]["target"].clone()
    out = loss_fn(pred, gt, present, flow_targets=flow_targets,
                  deepsup_logits=None, anatomy_adjacency=None, anatomy_adjacency_init=None)
    assert abs(out["components"]["flow"]) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\Raywa\Desktop\LiteSAM3D\.venv\Scripts\python -m pytest tests/test_losses_v4.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement V4 loss**

Create `training/losses_v4.py`:

```python
"""V4 composite loss:
    L_total = L_mask + λ_flow · L_flow + λ_deepsup · L_deepsup + λ_anatomy · L_anatomy
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss_binary(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-sample soft Dice on sigmoid probabilities. pred: (N, H, W), target: (N, H, W)."""
    p = torch.sigmoid(pred)
    inter = (p * target).flatten(1).sum(1)
    union = p.flatten(1).sum(1) + target.flatten(1).sum(1)
    return 1.0 - (2.0 * inter + eps) / (union + eps)


class V4MultiOrganLoss(nn.Module):
    def __init__(
        self,
        lambda_flow: float = 0.5,
        lambda_deepsup: float = 0.1,
        lambda_anatomy: float = 0.01,
    ) -> None:
        super().__init__()
        self.lambda_flow = lambda_flow
        self.lambda_deepsup = lambda_deepsup
        self.lambda_anatomy = lambda_anatomy

    def forward(
        self,
        pred_masks: torch.Tensor,                                  # (B, 15, H, W) logits
        gt_masks: torch.Tensor,                                     # (B, 15, H, W) or (B, 15, D, H, W) uint8
        present_mask: torch.Tensor,                                 # (B, 15) bool
        flow_targets: Optional[Dict] = None,
        deepsup_logits: Optional[torch.Tensor] = None,
        anatomy_adjacency: Optional[torch.Tensor] = None,
        anatomy_adjacency_init: Optional[torch.Tensor] = None,
        lambda_flow_override: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        if gt_masks.dim() == 5:
            D = gt_masks.shape[2]
            gt_masks = gt_masks[:, :, D // 2, :, :]                 # center slice only for mask loss
        target = gt_masks.float()
        if target.shape[-2:] != pred_masks.shape[-2:]:
            target = F.interpolate(target, size=pred_masks.shape[-2:], mode="nearest")

        B, K, H, W = pred_masks.shape
        flat_pred = pred_masks.reshape(B * K, H, W)
        flat_tgt = target.reshape(B * K, H, W)
        mask_ok = present_mask.reshape(B * K)

        bce = F.binary_cross_entropy_with_logits(flat_pred, flat_tgt, reduction="none").mean(dim=(-1, -2))
        dsc = dice_loss_binary(flat_pred, flat_tgt)

        if mask_ok.any():
            L_mask = (bce[mask_ok].mean() + dsc[mask_ok].mean()) * 0.5
        else:
            L_mask = pred_masks.sum() * 0.0

        components: Dict[str, float] = {"mask": L_mask.item()}
        L_flow = pred_masks.sum() * 0.0
        L_deepsup = pred_masks.sum() * 0.0
        L_anatomy = pred_masks.sum() * 0.0

        if flow_targets is not None:
            fwd = flow_targets["fwd_pairs"]
            bwd = flow_targets["bwd_pairs"]
            L_flow = 0.5 * (F.mse_loss(fwd["pred"], fwd["target"]) + F.mse_loss(bwd["pred"], bwd["target"]))
            components["flow"] = L_flow.item()

        if deepsup_logits is not None:
            ds_tgt = target                                          # (B, 15, H, W)
            if ds_tgt.shape[-2:] != deepsup_logits.shape[-2:]:
                ds_tgt = F.interpolate(ds_tgt, size=deepsup_logits.shape[-2:], mode="nearest")
            # Multi-label BCE (one-hot per organ; can be overlapping)
            L_deepsup = F.binary_cross_entropy_with_logits(deepsup_logits, ds_tgt)
            components["deepsup"] = L_deepsup.item()

        if anatomy_adjacency is not None and anatomy_adjacency_init is not None:
            L_anatomy = F.mse_loss(anatomy_adjacency, anatomy_adjacency_init)
            components["anatomy"] = L_anatomy.item()

        lf = self.lambda_flow if lambda_flow_override is None else lambda_flow_override
        total = L_mask + lf * L_flow + self.lambda_deepsup * L_deepsup + self.lambda_anatomy * L_anatomy
        components["total"] = total.item()
        return {"loss": total, "components": components}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\Users\Raywa\Desktop\LiteSAM3D\.venv\Scripts\python -m pytest tests/test_losses_v4.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```
git add training/losses_v4.py tests/test_losses_v4.py
git commit -m "feat(v4): multi-organ composite loss with flow-matching term"
```

---

## Task 8: Trainer V4 branch

**Files:**
- Modify: `training/trainer.py`

- [ ] **Step 1: Find the architecture dispatch in trainer.py**

Run: `grep -n "is_auto_ode" C:\Users\Raywa\Desktop\VoluFormer3D_V4\training\trainer.py` and note the line numbers of every hit.

- [ ] **Step 2: Add V4 detection near the top of the train loop**

Inside `training/trainer.py`, find the line:
```python
is_auto_ode = ...
```
(near where model architecture is detected). Immediately after it, add:
```python
is_organflow = getattr(self.cfg.model, "architecture", None) == "organflow_sam2"
```

Then, at the top of the per-batch loop body where `is_auto_ode:` is branched, add a sibling branch (BEFORE the `if is_auto_ode:` line) that handles V4:

```python
                if is_organflow:
                    from training.losses_v4 import V4MultiOrganLoss
                    if not hasattr(self, "_v4_criterion"):
                        self._v4_criterion = V4MultiOrganLoss(
                            lambda_flow=getattr(self.cfg.training, "lambda_flow", 0.5),
                            lambda_deepsup=getattr(self.cfg.training, "lambda_deepsup", 0.1),
                            lambda_anatomy=getattr(self.cfg.training, "lambda_anatomy", 0.01),
                        )
                    images_v4 = batch["image"].to(self.device)             # (B, D, 1, H, W)
                    # Expand 1 channel → 3 for MedSAM2
                    images_v4 = images_v4.repeat(1, 1, 3, 1, 1) if images_v4.shape[2] == 1 else images_v4
                    masks_v4 = batch["masks"].to(self.device)               # (B, 15, D, H, W)
                    present_v4 = batch["present_mask"].to(self.device)      # (B, 15) bool
                    organ_id_v4 = torch.argmax(present_v4.int(), dim=1) + 1  # pick one present organ per sample for ODE conditioning
                    outputs = self.model(images_v4, organ_id_v4, is_3d=True)
                    # Flow loss warmup: linear ramp λ_flow 0 → configured over first 3 epochs
                    warmup_eps = int(getattr(self.cfg.training, "flow_warmup_epochs", 3))
                    lf_target = float(getattr(self.cfg.training, "lambda_flow", 0.5))
                    lf_now = lf_target * min(1.0, epoch / max(warmup_eps, 1))
                    out = self._v4_criterion(
                        pred_masks=outputs["masks"],
                        gt_masks=masks_v4,
                        present_mask=present_v4,
                        flow_targets=outputs.get("flow_targets"),
                        deepsup_logits=outputs.get("deepsup_logits"),
                        anatomy_adjacency=self.model.decoder.graph.adjacency,
                        anatomy_adjacency_init=getattr(self.model.decoder.graph, "_init_adjacency", None),
                        lambda_flow_override=lf_now,
                    )
                    loss = out["loss"]
                    loss_components = out["components"]
                    loss_components["total"] = loss.item()
                    continue_v3_path = False
                else:
                    continue_v3_path = True
```

Then immediately wrap the existing V3/non-V4 branches with `if continue_v3_path:` so they run only when V4 didn't fire.

- [ ] **Step 3: Register init adjacency in AnatomyGraphAttention**

Edit `models/anatomy_graph_decoder.py` — inside `AnatomyGraphAttention.__init__`, BELOW `self.adjacency = nn.Parameter(_build_adjacency_init(n_organs))`, add:

```python
        self.register_buffer("_init_adjacency", _build_adjacency_init(n_organs).clone())
```

- [ ] **Step 4: Model-build dispatch**

Find where the model is built from `cfg.model.architecture` in `training/trainer.py` (or `train.py` — grep for `architecture`). Add a branch:

```python
    if cfg.model.architecture == "organflow_sam2":
        from models.organflow_sam2 import OrganFlowSAM2
        model = OrganFlowSAM2(cfg)
    elif cfg.model.architecture == "auto_ode_sam":
        ...  # existing V3 code
```

- [ ] **Step 5: Commit**

```
git add training/trainer.py models/anatomy_graph_decoder.py
git commit -m "feat(v4): trainer branch for OrganFlowSAM2 + L_flow warmup"
```

---

## Task 9: V4 config

**Files:**
- Create: `configs/v4_organflow_sam2_256px.yaml`

- [ ] **Step 1: Write the config**

Create `configs/v4_organflow_sam2_256px.yaml`:

```yaml
experiment_name: v4_organflow_sam2_256px
seed: 42

model:
  architecture: organflow_sam2
  img_size: 256
  embed_dim: 256
  skip_channels: 128
  encoder:
    pretrained: true
    lora_rank: 16
  pfesa: {}
  ode:
    n_organs: 15
    organ_emb_dim: 32
    ode_hidden: 128
    n_freqs: 6
    substeps: 4
  decoder:
    n_organs: 15
    transformer_depth: 4
    transformer_mlp_dim: 2048
  graph:
    n_organs: 15

data:
  dataset_class: AMOS22MultiOrgan3D_Dataset
  root: C:\Users\Raywa\Desktop\LiteSAM3D\data\amos22
  modality: ct
  depth: 8
  img_size: 256

training:
  epochs: 120
  batch_size: 4
  grad_accum_steps: 2
  optimizer: adamw
  weight_decay: 0.05
  lr_lora: 1e-4
  lr_new: 3e-4
  warmup_epochs: 3
  flow_warmup_epochs: 3
  lambda_flow: 0.5
  lambda_deepsup: 0.1
  lambda_anatomy: 0.01
  grad_clip: 1.0
  mixed_precision: true
  validate_every: 5
  checkpoint_every: 5
  resume_from: null

logging:
  log_dir: logs/v4_organflow_sam2_256px
  tb_log: true
```

- [ ] **Step 2: Commit**

```
git add configs/v4_organflow_sam2_256px.yaml
git commit -m "feat(v4): V4 training config"
```

---

## Task 10: End-to-end smoke test on 2 volumes

**Files:**
- Create: `scripts/smoke_test_v4.py`

- [ ] **Step 1: Write the smoke script**

Create `scripts/smoke_test_v4.py`:

```python
"""End-to-end V4 smoke test on 2 volumes.

Builds the model, runs 1 forward + backward pass on 2 volumes, verifies:
  - No NaNs or Infs
  - All trainable params receive gradient
  - Loss components all finite
  - Peak VRAM usage < 22 GB (leaves headroom for batch size 4)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.amos22_multiorgan import AMOS22MultiOrgan3D_Dataset
from models.organflow_sam2 import OrganFlowSAM2
from training.losses_v4 import V4MultiOrganLoss


def main():
    cfg = OmegaConf.load("configs/v4_organflow_sam2_256px.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None

    ds = AMOS22MultiOrgan3D_Dataset(
        data_root=cfg.data.root, split="train",
        img_size=cfg.data.img_size, depth=cfg.data.depth, modality="ct",
    )
    assert len(ds) >= 2, f"Need ≥2 volumes, have {len(ds)}"
    batch = [ds[0], ds[1]]
    images = torch.stack([b["image"] for b in batch]).to(device)              # (2, D, 1, H, W)
    masks = torch.stack([b["masks"] for b in batch]).to(device)               # (2, 15, D, H, W)
    present = torch.stack([b["present_mask"] for b in batch]).to(device)      # (2, 15)
    images = images.repeat(1, 1, 3, 1, 1)
    organ_id = torch.argmax(present.int(), dim=1) + 1

    model = OrganFlowSAM2(cfg).to(device)
    model.train()
    criterion = V4MultiOrganLoss(
        lambda_flow=0.5, lambda_deepsup=0.1, lambda_anatomy=0.01,
    )

    with torch.amp.autocast("cuda", enabled=device == "cuda"):
        out = model(images, organ_id, is_3d=True)
        loss_out = criterion(
            pred_masks=out["masks"].float(),
            gt_masks=masks,
            present_mask=present,
            flow_targets=out.get("flow_targets"),
            deepsup_logits=out.get("deepsup_logits").float() if out.get("deepsup_logits") is not None else None,
            anatomy_adjacency=model.decoder.graph.adjacency,
            anatomy_adjacency_init=model.decoder.graph._init_adjacency,
        )

    loss = loss_out["loss"]
    print(f"Loss components: {loss_out['components']}")
    for name, v in loss_out["components"].items():
        assert torch.isfinite(torch.tensor(v)), f"Non-finite {name}={v}"
    loss.backward()

    missing_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing_grad, f"Trainable params without grad: {missing_grad[:5]}..."

    if device == "cuda":
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak VRAM: {peak_gb:.2f} GB")
        assert peak_gb < 22.0, "Peak VRAM too high — drop batch size or D"

    print("SMOKE TEST PASS")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the smoke test**

Run: `C:\Users\Raywa\Desktop\LiteSAM3D\.venv\Scripts\python scripts/smoke_test_v4.py`
Expected: prints a `Loss components:` dict with finite values, `Peak VRAM: X.XX GB` under 22, and `SMOKE TEST PASS`.

If VRAM > 22 GB: drop `cfg.data.depth` to 6 and re-run. If still too high, drop `cfg.model.decoder.transformer_depth` to 2.

If any loss component is NaN: inspect (a) that MedSAM2 checkpoint loaded (re-run Task 0 Step 4), (b) that `images` are in [0, 1] (check with `images.min(), images.max()`), (c) that `f_θ` last-layer weights are exactly zero at init.

- [ ] **Step 3: Commit**

```
git add scripts/smoke_test_v4.py
git commit -m "feat(v4): end-to-end smoke test on 2 volumes"
```

---

## Task 11: Launch first training run to ep5 go-gate

**Files:** none new. Uses existing `train.py`.

- [ ] **Step 1: Verify config + dataset + VRAM one more time**

Run: `C:\Users\Raywa\Desktop\LiteSAM3D\.venv\Scripts\python scripts/smoke_test_v4.py`
Must still say `SMOKE TEST PASS`.

- [ ] **Step 2: Launch training**

Run:
```
C:\Users\Raywa\Desktop\LiteSAM3D\.venv\Scripts\python train.py --config configs/v4_organflow_sam2_256px.yaml
```

- [ ] **Step 3: Watch epoch 0 log for ~40 min**

Tail `logs/v4_organflow_sam2_256px/train.log`. Verify:
- `train_loss` decreasing step-over-step
- `components.flow` warming up (non-zero by epoch 1 once warmup kicks in)
- `components.mask` finite, typically starts ≈ 0.8 and drops

If loss diverges in the first 100 steps: stop (`Ctrl+C`), reduce `lr_new` 3e-4 → 1e-4 in config, delete `checkpoints/v4_organflow_sam2_256px/`, and re-launch.

- [ ] **Step 4: Evaluate the ep5 go-gate**

At epoch 5 the trainer will run validation (configured `validate_every: 5`). Read the reported `val_dice`.

- **PASS:** `val_dice ≥ 0.15` — continue to ep20.
- **FAIL:** `val_dice < 0.15` — stop training. Investigate: (a) did `_init_adjacency` buffer load correctly? (b) is the LoRA initialization correct (B should be zero)? (c) inspect `outputs["flow_targets"]` magnitudes.

- [ ] **Step 5: Update MASTER_PROJECT_LOG and commit**

Append to `MASTER_PROJECT_LOG.md` Section 18 (Phase 4 → ACTIVE) the first checkpoint row under a new "V4 Checkpoint Inventory" table with ep5 numbers. Then:

```
git add MASTER_PROJECT_LOG.md
git commit -m "docs(v4): log ep5 go-gate result"
```

---

## Self-Review checklist

Before executing, I walked the spec back against the plan:

**1. Spec coverage:**

| Spec section | Implemented in |
|---|---|
| §3 novelty 1 (flow-matched ODE) | Task 3 |
| §3 novelty 2 (anatomy graph attention) | Task 4 |
| §3 novelty 3 (PFESA++) | Task 1 |
| §3 novelty 4 (MedSAM2 + LoRA + DETR) | Task 2 + Task 5 |
| §4.1 forward pass | Task 5 |
| §4.2 flow loss math | Task 3 + Task 7 |
| §4.3 anatomy graph init | Task 4 |
| §4.4 PFESA++ | Task 1 |
| §4.5 param budget | Task 5 (assertion in test) |
| §5.1 dataset | Task 6 |
| §5.2 losses | Task 7 |
| §5.3 optimizer + warmup | Task 9 config + Task 8 trainer warmup |
| §5.4 compute plan | Task 11 (validated by wall-clock) |
| §6 file changes | All tasks follow the file list |
| §7 risks | Task 10 smoke catches VRAM/NaN risks; Task 11 handles divergence |
| §8 go gates | Task 11 Step 4 |

**2. Placeholder scan:** every step has concrete code / commands / expected output. No TBD, TODO, "implement later", or "similar to Task N". ✓

**3. Type consistency:** `present_mask`, `flow_targets`, `deepsup_logits` naming is uniform across `OrganFlowSAM2.forward`, `V4MultiOrganLoss.forward`, trainer branch, and smoke test. `anatomy_adjacency` + `anatomy_adjacency_init` pass through consistently.

**4. Blast-radius checklist:** V3 code is NOT touched except for two additive branches in `trainer.py` and one registration line in `models/__init__.py`. Rolling back V4 is `git revert` of the feature commits.
