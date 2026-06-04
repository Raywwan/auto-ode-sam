# =============================================================================
# models/fca_sam.py — FCA-SAM: Frequency Cross-slice Adapter SAM
#
# Cross-slice context via FFT along the depth dimension:
#   1. Reshape feature volume to (B*H*W, D, C) — depth as sequence
#   2. FFT along D: complex spectrum (B*H*W, D//2+1, C)
#   3. Low-pass filter: zero out high-frequency bins (D//4 and above)
#   4. Learnable complex linear mixer on kept frequencies
#   5. iFFT → fused features
#   6. Residual + LayerNorm
#
# Why: Low-frequency depth components = gradual anatomical change across
# slices (smooth liver boundary, etc.). High-frequency = slice-specific
# noise. O(D log D) complexity. No external deps: torch.fft is native PyTorch.
#
# Distinct from SAM2Rad: that uses FFT adapters for domain adaptation
# (natural→medical), NOT for cross-slice inter-slice context sharing.
# No published work applies FFT specifically along the depth/slice
# dimension of CT feature maps for inter-slice information sharing.
# =============================================================================

from typing import Dict

import torch
import torch.nn as nn

from models.multiscale_encoder import MultiScaleEncoder
from models.prompt_encoder import PromptEncoder
from models.mask_decoder import MaskDecoder


class FrequencyCrossSliceAdapter(nn.Module):
    """
    FFT-based inter-slice feature fusion module.

    For each spatial position (i,j) in the feature map, the depth sequence
    (across all D slices) is FFT'd, low-pass filtered, mixed via a learnable
    complex linear layer, and iFFT'd back. Residual + LayerNorm applied.

    Args:
        dim: Feature channels (embed_dim after TinyViT projection).
        n_keep: Low-frequency bins to retain. For D=8: n_keep=3 keeps
                DC + 2 harmonics (smooth depth changes). For D=16: use 5.
    """

    def __init__(self, dim: int, n_keep: int = 3):
        super().__init__()
        self.dim = dim
        self.n_keep = n_keep

        # Learnable complex linear mixer on the kept frequency bins.
        # Implemented as two real Linear layers (real + imaginary parts).
        # Applied on the frequency axis (n_keep → n_keep), shared across
        # all spatial positions and channels: only 2 * n_keep^2 + 2*n_keep params.
        self.mixer_real = nn.Linear(n_keep, n_keep, bias=True)
        self.mixer_imag = nn.Linear(n_keep, n_keep, bias=True)

        # Initialize: identity real part + zero imaginary → pure low-pass at start
        nn.init.eye_(self.mixer_real.weight)
        nn.init.zeros_(self.mixer_real.bias)
        nn.init.zeros_(self.mixer_imag.weight)
        nn.init.zeros_(self.mixer_imag.bias)

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (D, B, C, H, W) — volume feature maps, same format as DA-ISA.
        Returns:
            (D, B, C, H, W) — depth-fused features with FFT cross-slice context.
        """
        D, B, C, H, W = x.shape
        orig_dtype = x.dtype

        # (D, B, C, H, W) → (B, H, W, D, C) → (B*H*W, D, C)
        x_perm = x.permute(1, 3, 4, 0, 2).reshape(B * H * W, D, C)

        # ---- FFT along depth (must be float32 — ComplexHalf not supported) ----
        x_freq = torch.fft.rfft(x_perm.float(), dim=1)  # (B*H*W, D//2+1, C) complex64
        n_freq = x_freq.shape[1]                          # = D//2 + 1
        n_keep = min(self.n_keep, n_freq)

        # ---- Complex linear mixing on kept bins only ----
        # No inplace ops — build output tensor from scratch to keep autograd happy.
        kept = x_freq[:, :n_keep, :]                     # (B*H*W, n_keep, C) complex
        kept_r = kept.real.permute(0, 2, 1)              # (B*H*W, C, n_keep)
        kept_i = kept.imag.permute(0, 2, 1)

        # Complex multiplication: (W_r + i W_i)(x_r + i x_i)
        #   = (W_r x_r - W_i x_i) + i(W_r x_i + W_i x_r)
        out_r = self.mixer_real(kept_r) - self.mixer_imag(kept_i)  # (B*H*W, C, n_keep)
        out_i = self.mixer_real(kept_i) + self.mixer_imag(kept_r)

        mixed_kept = torch.complex(
            out_r.permute(0, 2, 1),                      # (B*H*W, n_keep, C)
            out_i.permute(0, 2, 1),
        )

        # Low-pass: concatenate mixed kept bins with complex zeros (no inplace)
        if n_keep < n_freq:
            zeros = torch.zeros(
                B * H * W, n_freq - n_keep, C,
                dtype=torch.complex64, device=x.device,
            )
            x_mixed = torch.cat([mixed_kept, zeros], dim=1)  # (B*H*W, n_freq, C)
        else:
            x_mixed = mixed_kept

        # ---- iFFT → cast back to input dtype ----
        x_out = torch.fft.irfft(x_mixed, n=D, dim=1)    # (B*H*W, D, C) float32
        x_out = x_out.to(orig_dtype)

        # (B*H*W, D, C) → (B, H, W, D, C) → (D, B, C, H, W)
        x_out = x_out.reshape(B, H, W, D, C).permute(3, 0, 4, 1, 2)

        # ---- Residual + LayerNorm ----
        x_resid = x + x_out                              # (D, B, C, H, W)
        out_perm = x_resid.permute(0, 1, 3, 4, 2)       # (D, B, H, W, C)
        out_perm = self.norm(out_perm)
        return out_perm.permute(0, 1, 4, 2, 3)          # (D, B, C, H, W)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, n_keep={self.n_keep}"


class FCASAM(nn.Module):
    """
    FCA-SAM: Frequency Cross-slice Adapter SAM.

    Identical architecture to VoluFormer3D (MultiScaleISA) except DA-ISA
    is replaced with FrequencyCrossSliceAdapter. Shares the same
    encoder/decoder/trainer — only the cross-slice module changes.

    Args:
        cfg: OmegaConf config. Uses same keys as multiscale_isa.
             Optional key: model.fca.n_keep (default 3, designed for D=8).
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg.model

        stage_index = getattr(
            getattr(model_cfg, "multiscale_isa", None), "stage_index", 2
        )

        # ---- Image Encoder (same Stage 2 as MultiScaleISA) ----
        self.encoder = MultiScaleEncoder(
            model_name=model_cfg.encoder_name,
            img_size=model_cfg.img_size,
            embed_dim=model_cfg.embed_dim,
            pretrained=model_cfg.encoder_pretrained,
            stage_index=stage_index,
        )

        feat_size = self.encoder.get_output_size()[0]

        # ---- Frequency Cross-Slice Adapter ----
        fca_cfg = getattr(model_cfg, "fca", None)
        n_keep = getattr(fca_cfg, "n_keep", 3) if fca_cfg is not None else 3
        self.fca = FrequencyCrossSliceAdapter(dim=model_cfg.embed_dim, n_keep=n_keep)

        # ---- Prompt Encoder ----
        self.prompt_encoder = PromptEncoder(
            embed_dim=model_cfg.embed_dim,
            image_size=model_cfg.img_size,
            image_embedding_size=feat_size,
            num_modalities=model_cfg.prompt_encoder.num_modalities,
            modality_embed_dim=model_cfg.prompt_encoder.modality_embed_dim,
            content_embed_dim=model_cfg.prompt_encoder.content_embed_dim,
        )

        # ---- Mask Decoder ----
        self.mask_decoder = MaskDecoder(
            embed_dim=model_cfg.embed_dim,
            num_multimask_outputs=model_cfg.mask_decoder.num_multimask_outputs,
            iou_head_depth=model_cfg.mask_decoder.iou_head_depth,
            iou_head_hidden_dim=model_cfg.mask_decoder.iou_head_hidden_dim,
            num_modalities=model_cfg.prompt_encoder.num_modalities,
        )

    def encode_image(
        self,
        images: torch.Tensor,
        is_3d: bool = False,
        return_all_slices: bool = False,
        organ_id: torch.Tensor = None,
    ) -> torch.Tensor:
        if is_3d:
            B, D, C, H, W = images.shape
            images_2d = images.reshape(B * D, C, H, W)
            features_2d = self.encoder(images_2d)              # (B*D, embed_dim, Hf, Wf)

            _, embed_dim, Hf, Wf = features_2d.shape
            features_3d = features_2d.reshape(B, D, embed_dim, Hf, Wf)
            features_3d = features_3d.permute(1, 0, 2, 3, 4)  # (D, B, C, Hf, Wf)
            features_3d = self.fca(features_3d)                # FFT cross-slice fusion
            features_3d = features_3d.permute(1, 0, 2, 3, 4)  # (B, D, C, Hf, Wf)
            features_2d = features_3d.reshape(B * D, embed_dim, Hf, Wf)

            if return_all_slices:
                return features_2d
            else:
                features_vol = features_2d.reshape(B, D, embed_dim, Hf, Wf)
                return features_vol[:, D // 2]
        else:
            return self.encoder(images)

    def forward(
        self,
        images: torch.Tensor,
        boxes: torch.Tensor,
        modality_ids: torch.Tensor,
        is_3d: bool = False,
        multimask_output: bool = True,
        organ_id: torch.Tensor = None,   # accepted for API compatibility; unused
    ) -> Dict[str, torch.Tensor]:
        features = self.encode_image(images, is_3d=is_3d)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            boxes=boxes,
            modality_ids=modality_ids,
            image_features=features,
        )

        masks, iou_pred, modality_logits = self.mask_decoder(
            image_embeddings=features,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        return {"masks": masks, "iou_pred": iou_pred, "modality_logits": modality_logits}

    def predict(
        self,
        images: torch.Tensor,
        boxes: torch.Tensor,
        modality_ids: torch.Tensor,
        is_3d: bool = False,
        organ_id: torch.Tensor = None,
    ) -> torch.Tensor:
        if is_3d and modality_ids.dim() == 2:
            modality_ids = modality_ids[:, modality_ids.shape[1] // 2]
        with torch.no_grad():
            out = self.forward(images, boxes, modality_ids, is_3d=is_3d,
                               multimask_output=True, organ_id=organ_id)
        best_idx = out["iou_pred"].argmax(dim=1)
        best_masks = out["masks"][
            torch.arange(out["masks"].shape[0], device=out["masks"].device), best_idx
        ].unsqueeze(1)
        return torch.sigmoid(best_masks)

    def predict_all_slices(
        self,
        images: torch.Tensor,
        boxes: torch.Tensor,
        modality_ids: torch.Tensor,
        organ_id: torch.Tensor = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            features = self.encode_image(images, is_3d=True, return_all_slices=True)
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                boxes=boxes, modality_ids=modality_ids, image_features=features,
            )
            masks, iou_pred, _ = self.mask_decoder(
                image_embeddings=features,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
            )
        best_idx = iou_pred.argmax(dim=1)
        best_masks = masks[
            torch.arange(masks.shape[0], device=masks.device), best_idx
        ].unsqueeze(1)
        return torch.sigmoid(best_masks)

    def count_parameters(self) -> Dict[str, int]:
        def count(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {
            "encoder": count(self.encoder),
            "fca": count(self.fca),
            "prompt_encoder": count(self.prompt_encoder),
            "mask_decoder": count(self.mask_decoder),
            "total": count(self),
        }
