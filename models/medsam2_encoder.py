"""MedSAM2 Hiera-Tiny encoder with LoRA adaptation.

V6: 3-scale skip taps — stride-4 (stage 0, 96ch), stride-8 (stage 1, 192ch),
stride-16 (stage 2, 384ch). Feeds an FPN-style multi-scale decoder.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

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


def _inject_lora(module: nn.Module, rank: int, alpha: float = None) -> int:
    """Replace every nn.Linear whose name is in {'qkv','proj','fc1','fc2'} with LoRALinear. Returns count."""
    if rank is None or int(rank) <= 0:
        return 0
    if alpha is None:
        alpha = float(rank)
    n_replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in ("qkv", "proj", "fc1", "fc2"):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha))
            n_replaced += 1
        else:
            n_replaced += _inject_lora(child, rank, alpha)
    return n_replaced


class MedSAM2Encoder(nn.Module):
    """Hiera-Tiny backbone with LoRA + stage-0/1/2 taps.

    V6 adds a stride-4 tap (stage 0) to support FPN-style multi-scale decoder
    fusion. Still excludes stage-3 (stride-32, only 10×10 at 320²) which would
    collapse small-organ detail without benefit.

    Args:
        embed_dim: target channels for main tap (projects 384 → embed_dim).
        skip_channels: target channels for stride-8 skip (projects 192 → skip_channels).
        skip_fine_channels: target channels for stride-4 skip (projects 96 → this).
        lora_rank: LoRA rank (default 16).
        pretrained: if True, try to load `checkpoints/medsam2/MedSAM2_hiera_tiny.pt`.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        skip_channels: int = 128,
        skip_fine_channels: int = 64,
        lora_rank: int = 16,
        lora_alpha: float = None,
        pretrained: bool = True,
        img_size: int = 320,
    ) -> None:
        super().__init__()
        if lora_alpha is None:
            lora_alpha = float(lora_rank)
        self.img_size = img_size
        self.backbone = timm.create_model(
            "hiera_tiny_224",
            pretrained=False,
            features_only=True,
            out_indices=[0, 1, 2],  # stride-4, stride-8, stride-16
            img_size=img_size,
        )
        fi = self.backbone.feature_info
        fine_in_ch = fi[0]["num_chs"]   # 96ch  (stride-4)
        skip_in_ch = fi[1]["num_chs"]   # 192ch (stride-8)
        main_in_ch = fi[2]["num_chs"]   # 384ch (stride-16)

        if pretrained:
            ckpt_path = Path(__file__).resolve().parents[1] / "checkpoints" / "medsam2" / "MedSAM2_hiera_tiny.pt"
            if ckpt_path.exists():
                state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
                sd = state.get("model", state)
                # SAM2 names image-encoder keys `image_encoder.trunk.<X>`; timm's
                # features_only wrapper prefixes every key with `model.`. Also, SAM2's
                # Hiera uses `mlp.layers.{0,1}` while timm uses `mlp.{fc1,fc2}`.
                # `pos_embed` is a shape mismatch (SAM2 4D windowed vs timm 3D tokens)
                # and is dropped — it gets re-initialized and learns during fine-tune.
                bb_sd = self.backbone.state_dict()
                sd_enc = {}
                dropped_shape = 0
                for k, v in sd.items():
                    if not k.startswith("image_encoder.trunk."):
                        continue
                    stripped = k.replace("image_encoder.trunk.", "")
                    stripped = re.sub(r"mlp\.layers\.0", "mlp.fc1", stripped)
                    stripped = re.sub(r"mlp\.layers\.1", "mlp.fc2", stripped)
                    target = "model." + stripped
                    if target in bb_sd and bb_sd[target].shape == v.shape:
                        sd_enc[target] = v
                    else:
                        dropped_shape += 1
                if sd_enc:
                    missing, unexpected = self.backbone.load_state_dict(sd_enc, strict=False)
                    print(f"[MedSAM2Encoder] loaded {len(sd_enc)} keys "
                          f"(dropped {dropped_shape} due to name/shape mismatch), "
                          f"missing={len(missing)}, unexpected={len(unexpected)}")
                    if len(sd_enc) < 100:
                        print(f"[MedSAM2Encoder] WARNING: only {len(sd_enc)} keys loaded — expected >= 120.")
                else:
                    print(f"[MedSAM2Encoder] warning: no 'image_encoder.trunk.' keys in checkpoint — using random init.")
            else:
                print(f"[MedSAM2Encoder] warning: checkpoint not found at {ckpt_path} — using random init.")

        # Freeze all backbone params, then inject LoRA (which un-freezes its own params)
        for p in self.backbone.parameters():
            p.requires_grad = False
        n_lora = _inject_lora(self.backbone, rank=lora_rank, alpha=lora_alpha)
        if lora_rank and int(lora_rank) > 0:
            print(f"[MedSAM2Encoder] injected LoRA into {n_lora} Linear layers "
                  f"(rank={lora_rank}, alpha={lora_alpha}, scale={lora_alpha/lora_rank:.3f})")
        else:
            print(f"[MedSAM2Encoder] LoRA disabled (rank={lora_rank}); encoder fully frozen.")

        # Projection heads (trainable)
        self.proj_fine = nn.Conv2d(fine_in_ch, skip_fine_channels, kernel_size=1)
        self.proj_skip = nn.Conv2d(skip_in_ch, skip_channels, kernel_size=1)
        self.proj_main = nn.Conv2d(main_in_ch, embed_dim, kernel_size=1)
        self.skip_fine_channels = skip_fine_channels
        self.skip_channels = skip_channels
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """x: (B, 3, H, W) → {main (B, embed, H/16, W/16),
        skip (B, skip_ch, H/8, W/8), skip_fine (B, fine_ch, H/4, W/4)}."""
        feats = self.backbone(x)
        fine_raw, skip_raw, main_raw = feats[0], feats[1], feats[2]
        assert fine_raw.shape[1] == 96, f"Expected C=96 at dim 1, got {fine_raw.shape}"
        assert skip_raw.shape[1] == 192, f"Expected C=192 at dim 1, got {skip_raw.shape}"
        assert main_raw.shape[1] == 384, f"Expected C=384 at dim 1, got {main_raw.shape}"
        return {
            "main": self.proj_main(main_raw),
            "skip": self.proj_skip(skip_raw),
            "skip_fine": self.proj_fine(fine_raw),
        }

    def get_output_size(self):
        s = self.img_size // 16
        return s, s
