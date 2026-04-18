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
        # lora_B stays zero → delta=0 and lora_A.grad=0 at step 0 (standard LoRA init);
        # both unlock after the first lora_B update.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        delta = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale
        return y + delta


def _inject_lora(module: nn.Module, rank: int) -> int:
    """Replace every nn.Linear whose name is in {'qkv','proj','fc1','fc2'} with LoRALinear. Returns count."""
    n_replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in ("qkv", "proj", "fc1", "fc2"):
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
            out_indices=[1, 2],  # tap stride-8 (192ch) and stride-16 (384ch); stage-3 excluded to avoid dead LoRA params
            img_size=256,
        )
        fi = self.backbone.feature_info
        # feature_info lists all 4 stages (0–3); out_indices=[1,2] returns feats[0..1]
        # feats[0] ← fi[1]: stage-1, 192ch, stride-8  (skip tap)
        # feats[1] ← fi[2]: stage-2, 384ch, stride-16 (main tap)
        skip_in_ch = fi[1]["num_chs"]   # 192ch
        main_in_ch = fi[2]["num_chs"]   # 384ch

        if pretrained:
            ckpt_path = Path(__file__).resolve().parents[1] / "checkpoints" / "medsam2" / "MedSAM2_hiera_tiny.pt"
            if ckpt_path.exists():
                state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
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
        # timm 1.0.22 returns NCHW from features_only; assert it so any future
        # layout change fails loudly instead of silently mis-shaping downstream.
        assert skip_raw.shape[1] == 192, f"Expected C=192 at dim 1 (NCHW), got {skip_raw.shape}"
        assert main_raw.shape[1] == 384, f"Expected C=384 at dim 1 (NCHW), got {main_raw.shape}"
        main = self.proj_main(main_raw)
        skip = self.proj_skip(skip_raw)
        return main, skip

    def get_output_size(self) -> Tuple[int, int]:
        """Main-tap spatial size for input 256×256 = 16×16."""
        return 16, 16
