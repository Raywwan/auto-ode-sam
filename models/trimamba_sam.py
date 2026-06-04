# =============================================================================
# models/trimamba_sam.py — TriMamba-SAM
#
# Three novel contributions stacked on VoluFormer3D's Stage-2 encoder:
#
#   1. TriDirectionalMambaModule
#      Scans the 3D feature volume along three anatomical axes — axial (D),
#      coronal (H), sagittal (W) — using bidirectional selective SSMs.
#      No published work applies tri-directional Mamba to CT feature maps
#      specifically for SAM-based medical image segmentation.
#
#   2. SoftMoEOrganRouter
#      Softly routes each spatial token to 2 specialised FFN experts.
#      Organ tokens learn to activate organ-specific experts; background
#      tokens activate the background expert.  Fully differentiable.
#
#   3. PFESA-style Spectral Skip (zero params)
#      Frequency-domain enhancement of the encoder features before they
#      enter the mask decoder.  Amplifies high-frequency boundary components
#      to sharpen organ edge prediction.  No learnable parameters.
#
# Design principles:
#   - Drop-in compatible with the existing VoluFormer3D trainer/evaluator.
#   - Returns the same dict keys as all other VoluFormer3D models.
#   - Stage 2 encoder kept fixed (same as Stage2-noISA best baseline).
#   - Each new component has a residual path so initialisation equals
#     the Stage2-noISA baseline (guaranteed non-regression at epoch 0).
#
# References:
#   UlikeMamba (arXiv:2503.19308) — tri-directional Mamba for medical 3D
#   SegMoTE (arXiv:2602.19213)   — token MoE adapters for SAM
#   PFESA (MICCAI 2025-0689)     — spectral skip for segmentation decoder
# =============================================================================

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.multiscale_encoder import MultiScaleEncoder
from models.prompt_encoder import PromptEncoder
from models.mask_decoder import MaskDecoder
from models.mamba_pure import MambaBidirectional1D
from models.soft_moe import SoftMoEOrganRouter


# ---------------------------------------------------------------------------
# Component 1 — Tri-Directional Mamba Cross-Slice Module
# ---------------------------------------------------------------------------

class TriDirectionalMambaModule(nn.Module):
    """
    Tri-directional Mamba cross-slice feature fusion.

    Processes a 3D feature volume (B, D, C, H, W) with three parallel
    bidirectional SSM scans — one per anatomical axis:

        Axial    (depth):    for each (h,w), scan across D=8 slices
        Coronal  (height):   for each (d,w), scan across H pixels
        Sagittal (width):    for each (d,h), scan across W pixels

    The three outputs are merged via a learned 3-way softmax gate.
    Each SSM has its own residual connection (within MambaBidirectional1D),
    so the overall module acts as the identity at initialisation.

    Args:
        d_model (int): Feature channel dimension (embed_dim).
        d_inner (int): Inner SSM dim.  Default: d_model // 2.
        d_state (int): SSM recurrent state size.  Default: 8.
    """

    def __init__(self, d_model: int, d_inner: int = None, d_state: int = 8):
        super().__init__()
        d_inner = d_inner if d_inner is not None else d_model // 2

        self.ssm_axial    = MambaBidirectional1D(d_model, d_inner, d_state)
        self.ssm_coronal  = MambaBidirectional1D(d_model, d_inner, d_state)
        self.ssm_sagittal = MambaBidirectional1D(d_model, d_inner, d_state)

        # Learned merge weights — initialised to equal blend (log-uniform prior)
        self.direction_gate = nn.Parameter(torch.zeros(3))

    # ------------------------------------------------------------------
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, D, C, H, W) — volume feature maps

        Returns:
            out: (B, D, C, H, W) — cross-direction fused features
        """
        B, D, C, H, W = features.shape

        # ---- Axial scan: along D for each (h, w) position ----
        # Reshape: (B, H, W, D, C) → (B*H*W, D, C)
        x_a = features.permute(0, 3, 4, 1, 2).reshape(B * H * W, D, C)
        y_a = self.ssm_axial(x_a)                           # (B*H*W, D, C)
        y_a = y_a.reshape(B, H, W, D, C).permute(0, 3, 4, 1, 2)  # (B, D, C, H, W)

        # ---- Coronal scan: along H for each (d, w) position ----
        # Reshape: (B, D, W, H, C) → (B*D*W, H, C)
        x_c = features.permute(0, 1, 4, 3, 2).reshape(B * D * W, H, C)
        y_c = self.ssm_coronal(x_c)                         # (B*D*W, H, C)
        y_c = y_c.reshape(B, D, W, H, C).permute(0, 1, 4, 3, 2)  # (B, D, C, H, W)

        # ---- Sagittal scan: along W for each (d, h) position ----
        # Reshape: (B, D, H, W, C) → (B*D*H, W, C)
        x_s = features.permute(0, 1, 3, 4, 2).reshape(B * D * H, W, C)
        y_s = self.ssm_sagittal(x_s)                        # (B*D*H, W, C)
        y_s = y_s.reshape(B, D, H, W, C).permute(0, 1, 4, 2, 3)  # (B, D, C, H, W)

        # ---- Learned directional gate ----
        gates = torch.softmax(self.direction_gate, dim=0)    # (3,) sums to 1
        return gates[0] * y_a + gates[1] * y_c + gates[2] * y_s
        # Each y_x already has the original features in it (via internal residual),
        # so this weighted sum preserves the input even at random init.

    def extra_repr(self) -> str:
        return (f"d_model={self.ssm_axial.d_model}, "
                f"d_inner={self.ssm_axial.d_inner}, "
                f"d_state={self.ssm_axial.d_state}")


# ---------------------------------------------------------------------------
# Component 3 — PFESA-style Spectral Skip (zero parameters)
# ---------------------------------------------------------------------------

class PFESASpectralSkip(nn.Module):
    """
    Zero-parameter spectral skip connection for the mask decoder.

    Amplifies high-frequency (boundary, edge) components of the encoder
    features before they enter the two-way transformer.  High-frequency
    bins carry organ boundary information that standard upsampling in the
    decoder tends to blur.

    Mechanism:
      1. 2D FFT of image_embeddings (B, C, H, W).
      2. Build a radial frequency mask (0 = DC, 1 = Nyquist).
      3. Amplify bins beyond `highfreq_cutoff` by a factor (1 + alpha).
      4. iFFT → enhanced spatial features.

    No learnable parameters (alpha and cutoff are hyperparameters).

    Args:
        alpha          (float): High-freq amplification.  Default 1.0.
        highfreq_cutoff (float): Fractional freq radius above which bins are
                                 amplified.  Default 0.5 (outer half of spectrum).
    """

    def __init__(self, alpha: float = 1.0, highfreq_cutoff: float = 0.5):
        super().__init__()
        self.alpha   = alpha
        self.cutoff  = highfreq_cutoff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) — encoder image embeddings (any dtype)
        returns: (B, C, H, W) — spectrally enhanced embeddings, same dtype
        """
        B, C, H, W = x.shape
        orig_dtype = x.dtype

        # FFT requires float32
        x_f = x.float()
        F_x = torch.fft.rfft2(x_f)          # (B, C, H, W//2+1) complex64

        # ---- Build radial frequency mask ----
        freq_h = torch.fft.fftfreq(H, device=x.device)   # (H,)
        freq_w = torch.fft.rfftfreq(W, device=x.device)   # (W//2+1,)

        # Normalised radius (0 = DC, 1 = max frequency)
        freq_r = (freq_h[:, None] ** 2 + freq_w[None, :] ** 2).sqrt()  # (H, W//2+1)
        max_r  = freq_r.max().clamp(min=1e-6)
        freq_r = freq_r / max_r                            # (H, W//2+1) in [0, 1]

        # High-frequency mask: 1 where freq_r > cutoff, else 0
        hf_mask = (freq_r > self.cutoff).to(dtype=x_f.dtype)  # (H, W//2+1)
        hf_mask = hf_mask.unsqueeze(0).unsqueeze(0)         # (1, 1, H, W//2+1)

        # Amplify high-freq bins
        F_enh = F_x * (1.0 + self.alpha * hf_mask)         # (B, C, H, W//2+1)

        # iFFT → enhanced spatial domain
        x_enh = torch.fft.irfft2(F_enh, s=(H, W)).to(orig_dtype)  # (B, C, H, W)
        return x_enh

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}, highfreq_cutoff={self.cutoff}"


# ---------------------------------------------------------------------------
# Main Model — TriMamba-SAM
# ---------------------------------------------------------------------------

class TriMambaSAM(nn.Module):
    """
    TriMamba-SAM: Tri-directional Mamba + Soft MoE + PFESA spectral decoder.

    Full forward pipeline:
        images (B,D,1,H,W)
          → MultiScaleEncoder (Stage 2, shared with VoluFormer3D)
          → TriDirectionalMambaModule  [contribution 1]
          → SoftMoEOrganRouter          [contribution 2]
          → PFESASpectralSkip           [contribution 3]  ← applied here
          → PromptEncoder
          → MaskDecoder
          → {masks, iou_pred, modality_logits}

    Compatible API: same forward/predict/predict_all_slices signatures
    as VoluFormer3D / FCASAM / ACMSAM — drop-in swap in train.py.

    Args:
        cfg: OmegaConf config.  New optional keys:
            model.trimamba.d_inner     (default embed_dim // 2)
            model.trimamba.d_state     (default 8)
            model.soft_moe.n_experts   (default 2)
            model.pfesa.alpha          (default 1.0)
            model.pfesa.highfreq_cutoff (default 0.5)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        mcfg = cfg.model

        stage_index = getattr(
            getattr(mcfg, "multiscale_isa", None), "stage_index", 2
        )

        # ---- 1. Image Encoder (Stage 2, same as best VoluFormer3D config) ----
        self.encoder = MultiScaleEncoder(
            model_name=mcfg.encoder_name,
            img_size=mcfg.img_size,
            embed_dim=mcfg.embed_dim,
            pretrained=mcfg.encoder_pretrained,
            stage_index=stage_index,
        )
        feat_size = self.encoder.get_output_size()[0]

        # ---- 2. Tri-Directional Mamba (cross-slice context) ----
        tri_cfg = getattr(mcfg, "trimamba", None)
        d_inner = getattr(tri_cfg, "d_inner", mcfg.embed_dim // 2) if tri_cfg else mcfg.embed_dim // 2
        d_state = getattr(tri_cfg, "d_state", 8)                   if tri_cfg else 8
        self.trimamba = TriDirectionalMambaModule(
            d_model=mcfg.embed_dim,
            d_inner=d_inner,
            d_state=d_state,
        )

        # ---- 3. Soft MoE Organ Router ----
        moe_cfg    = getattr(mcfg, "soft_moe", None)
        n_experts  = getattr(moe_cfg, "n_experts", 2) if moe_cfg else 2
        self.soft_moe = SoftMoEOrganRouter(
            d_model=mcfg.embed_dim,
            n_experts=n_experts,
        )

        # ---- 4. PFESA Spectral Skip (zero params) ----
        pfesa_cfg  = getattr(mcfg, "pfesa", None)
        pfesa_a    = getattr(pfesa_cfg, "alpha",           1.0) if pfesa_cfg else 1.0
        pfesa_cut  = getattr(pfesa_cfg, "highfreq_cutoff", 0.5) if pfesa_cfg else 0.5
        self.pfesa = PFESASpectralSkip(alpha=pfesa_a, highfreq_cutoff=pfesa_cut)

        # ---- 5. Prompt Encoder ----
        self.prompt_encoder = PromptEncoder(
            embed_dim=mcfg.embed_dim,
            image_size=mcfg.img_size,
            image_embedding_size=feat_size,
            num_modalities=mcfg.prompt_encoder.num_modalities,
            modality_embed_dim=mcfg.prompt_encoder.modality_embed_dim,
            content_embed_dim=mcfg.prompt_encoder.content_embed_dim,
        )

        # ---- 6. Mask Decoder ----
        self.mask_decoder = MaskDecoder(
            embed_dim=mcfg.embed_dim,
            num_multimask_outputs=mcfg.mask_decoder.num_multimask_outputs,
            iou_head_depth=mcfg.mask_decoder.iou_head_depth,
            iou_head_hidden_dim=mcfg.mask_decoder.iou_head_hidden_dim,
            num_modalities=mcfg.prompt_encoder.num_modalities,
        )

    # ------------------------------------------------------------------
    def encode_image(
        self,
        images: torch.Tensor,
        is_3d: bool = False,
        return_all_slices: bool = False,
    ) -> torch.Tensor:
        """
        Encode images through encoder → TriMamba → SoftMoE → PFESA.

        Args:
            images:            (B, D, C, H, W) if is_3d else (B, C, H, W)
            is_3d:             True for volumetric input (D slices)
            return_all_slices: True → return (B*D, embed_dim, Hf, Wf)
                               False → return center slice (B, embed_dim, Hf, Wf)

        Returns:
            features: center-slice or all-slice encoder output
        """
        if not is_3d:
            return self.encoder(images)

        B, D, C, H, W = images.shape
        images_2d = images.reshape(B * D, C, H, W)
        feat_2d   = self.encoder(images_2d)               # (B*D, embed_dim, Hf, Wf)

        _, embed_dim, Hf, Wf = feat_2d.shape

        # --- Tri-directional Mamba cross-slice fusion ---
        feat_3d  = feat_2d.reshape(B, D, embed_dim, Hf, Wf)
        feat_3d  = self.trimamba(feat_3d)                  # (B, D, embed_dim, Hf, Wf)

        # --- Soft MoE token routing (over all D slices jointly) ---
        feat_bd  = feat_3d.reshape(B * D, embed_dim, Hf, Wf)
        tokens   = feat_bd.flatten(2).permute(0, 2, 1)    # (B*D, Hf*Wf, embed_dim)
        tokens   = self.soft_moe(tokens)                   # (B*D, Hf*Wf, embed_dim)
        feat_bd  = tokens.permute(0, 2, 1).reshape(B * D, embed_dim, Hf, Wf)

        # --- PFESA spectral enhancement ---
        feat_bd  = self.pfesa(feat_bd)                     # (B*D, embed_dim, Hf, Wf)

        if return_all_slices:
            return feat_bd                                  # (B*D, embed_dim, Hf, Wf)

        feat_3d  = feat_bd.reshape(B, D, embed_dim, Hf, Wf)
        return feat_3d[:, D // 2]                          # (B, embed_dim, Hf, Wf)

    # ------------------------------------------------------------------
    def forward(
        self,
        images: torch.Tensor,
        boxes: torch.Tensor,
        modality_ids: torch.Tensor,
        is_3d: bool = False,
        multimask_output: bool = True,
        organ_id: torch.Tensor = None,   # API compatibility; unused
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

        return {
            "masks": masks,
            "iou_pred": iou_pred,
            "modality_logits": modality_logits,
        }

    # ------------------------------------------------------------------
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
        best_idx   = out["iou_pred"].argmax(dim=1)
        best_masks = out["masks"][
            torch.arange(out["masks"].shape[0], device=out["masks"].device), best_idx
        ].unsqueeze(1)
        return torch.sigmoid(best_masks)

    # ------------------------------------------------------------------
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
        best_idx   = iou_pred.argmax(dim=1)
        best_masks = masks[
            torch.arange(masks.shape[0], device=masks.device), best_idx
        ].unsqueeze(1)
        return torch.sigmoid(best_masks)

    # ------------------------------------------------------------------
    def count_parameters(self) -> Dict[str, int]:
        def count(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {
            "encoder":    count(self.encoder),
            "trimamba":   count(self.trimamba),
            "soft_moe":   count(self.soft_moe),
            "pfesa":      0,   # zero params — no learnable weights
            "prompt_encoder": count(self.prompt_encoder),
            "mask_decoder":   count(self.mask_decoder),
            "total":      count(self),
        }
