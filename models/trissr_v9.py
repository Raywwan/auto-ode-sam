"""TriSSR-V9 — Tri-Scan Selective-State-Space Refiner for the V9 cascade.

Novel cascade component (2026-04-23): a small *logit-refinement* head that
takes (proposer_logits ⊕ CT_image) and outputs refined per-organ logits.
Unlike SegMamba (MICCAI 2024), which uses tri-orient Mamba in the encoder,
this module sits AFTER the proposer and refines its 16-way softmax logits
using a compact 3D CNN stem + 3 tri-direction Mamba blocks at 24³ feature
resolution. Trained with frozen proposer, Dice + BCE losses.

Why it fits the V9 architecture:
  - Our cascade already has a proposer → refiner → fusion pattern. This
    module can drop-in as a *third* refinement path (or replace the
    plateaued ODE refiner entirely for inference).
  - The tri-direction scan gives the model global 3D context missing from
    SwinUNETR's windowed attention — exactly the failure mode our 0.17
    full-volume eval exposed (model can't localise lateralized organs).
  - Pure-PyTorch Mamba (no mamba-ssm Windows dep required).

VRAM budget at training (B=2, 96³ patches, 32-48 feat channels):
  ~2-3 GB activations + gradients — genuine GPU-heavy, fits on a 4090
  alongside a running fine-tune if timed sequentially.

Interface: forward(proposer_logits, image) -> refined_logits, both shaped
  (B, K+1, 96, 96, 96) where K=15 (bg channel prepended).

Init is residual-zero — at epoch 0 this module is an identity, so loading
it into inference without training cannot regress the proposer output.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.trimamba_sam import TriDirectionalMambaModule


class _ConvBNGELU(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.conv = nn.Conv3d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn   = nn.BatchNorm3d(out_c)
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class _DeconvBNGELU(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.up   = nn.ConvTranspose3d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn   = nn.BatchNorm3d(out_c)
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.up(x)))


class TriSSRV9(nn.Module):
    """Tri-Scan SSM logit-refinement head for the V9 cascade.

    Args:
        n_organs: 15 for AMOS22 (K=15, output channels = K+1 = 16 with bg).
        embed_dim: SSM feature dimension at the 24³ bottleneck. 48 keeps
            VRAM reasonable; bump to 64-96 if lift warrants it.
        n_blocks: number of stacked TriDirectionalMambaModule blocks.
        d_state: recurrent state slots per SSM (paper uses 16; we use 8).
    """

    def __init__(
        self,
        n_organs: int = 15,
        embed_dim: int = 48,
        n_blocks: int = 3,
        d_state: int = 8,
    ) -> None:
        super().__init__()
        self.n_organs  = n_organs
        self.embed_dim = embed_dim
        in_c = (n_organs + 1) + 1   # proposer logits (K+1) + image (1)

        # --- Encoder stem: 96³ -> 48³ -> 24³ ---
        self.stem1 = _ConvBNGELU(in_c,       embed_dim // 2, k=3, s=2, p=1)
        self.stem2 = _ConvBNGELU(embed_dim // 2, embed_dim, k=3, s=2, p=1)

        # --- Tri-scan bottleneck ---
        self.blocks = nn.ModuleList([
            TriDirectionalMambaModule(d_model=embed_dim, d_state=d_state)
            for _ in range(n_blocks)
        ])
        self.block_norm = nn.LayerNorm(embed_dim)

        # --- Decoder: 24³ -> 48³ -> 96³ ---
        self.up1 = _DeconvBNGELU(embed_dim,     embed_dim // 2)
        self.up2 = _DeconvBNGELU(embed_dim // 2, embed_dim // 4)

        # --- Refinement head (1x1x1 conv to K+1 logits) ---
        self.head = nn.Conv3d(embed_dim // 4, n_organs + 1, kernel_size=1, bias=True)

        # --- Residual-zero init: output adds to proposer_logits, start at 0 ---
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        proposer_logits: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """Refine proposer logits via tri-scan SSM.

        Args:
            proposer_logits: (B, K+1, D, H, W) pre-softmax or post-softmax logits.
                Expected raw logits for best conditioning, but post-softmax
                probabilities also work (will be re-offset by the residual).
            image: (B, 1, D, H, W) normalised CT intensity in [0, 1].

        Returns:
            refined_logits: (B, K+1, D, H, W) — proposer_logits + residual.
        """
        assert proposer_logits.dim() == 5, proposer_logits.shape
        assert image.dim() == 5, image.shape
        assert proposer_logits.shape[2:] == image.shape[2:], (
            f"spatial mismatch: proposer={proposer_logits.shape} image={image.shape}"
        )
        x = torch.cat([proposer_logits, image], dim=1)   # (B, K+2, 96, 96, 96)

        # Encoder stem
        x = self.stem1(x)                                 # (B, C/2, 48, 48, 48)
        x = self.stem2(x)                                 # (B, C,   24, 24, 24)

        # Tri-scan bottleneck — expects (B, D, C, H, W)
        B, C, D, H, W = x.shape
        x_tri = x.permute(0, 2, 1, 3, 4).contiguous()     # (B, D, C, H, W)
        for blk in self.blocks:
            x_tri = blk(x_tri)
        x = x_tri.permute(0, 2, 1, 3, 4).contiguous()     # (B, C, D, H, W)

        # LayerNorm on channel dim for stability (flatten spatial)
        x_flat = x.flatten(2).transpose(1, 2)             # (B, D*H*W, C)
        x_flat = self.block_norm(x_flat)
        x = x_flat.transpose(1, 2).view(B, C, D, H, W)

        # Decoder upsample
        x = self.up1(x)                                   # (B, C/2, 48, 48, 48)
        x = self.up2(x)                                   # (B, C/4, 96, 96, 96)

        # Residual logit refinement
        residual = self.head(x)                           # (B, K+1, 96, 96, 96)
        return proposer_logits + residual

    @torch.no_grad()
    def smoke(self, device: str = "cpu") -> None:
        """Quick shape/init check. Verifies residual-zero identity behaviour."""
        self.eval().to(device)
        pl = torch.randn(1, self.n_organs + 1, 96, 96, 96, device=device)
        im = torch.rand (1, 1,                 96, 96, 96, device=device)
        out = self.forward(pl, im)
        assert out.shape == pl.shape, out.shape
        # With zero-init head, output should match input exactly at epoch 0.
        max_diff = (out - pl).abs().max().item()
        assert max_diff < 1e-5, f"non-identity at init: max_diff={max_diff}"
        print(
            f"[TriSSRV9.smoke] OK  shape={tuple(out.shape)}  "
            f"identity_init_diff={max_diff:.2e}"
        )


if __name__ == "__main__":
    m = TriSSRV9(n_organs=15, embed_dim=48, n_blocks=3)
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"TriSSRV9: {n_params / 1e6:.2f}M trainable parameters")
    m.smoke(device="cpu")
