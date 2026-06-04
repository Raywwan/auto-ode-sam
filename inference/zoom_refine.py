# =============================================================================
# inference/zoom_refine.py — Two-stage zoom-in refinement for small organs
#
# Small organs (adrenal glands, esophagus, duodenum, gallbladder) are ~3–40px
# wide at 256px resolution. This module implements a two-pass inference
# strategy:
#
#   Pass 1: Run model at full 256px → coarse masks for all 15 organs
#   Pass 2: For each small organ with detected foreground in Pass 1, extract
#           the ROI bounding box, resize that crop to 256px (4× effective
#           resolution), re-run inference, then paste the refined mask back.
#
# Reference: PRNet 2025, RAPS-3D — +3–8% DSC on small organs.
#
# Constraints:
#   - All operations under torch.no_grad()
#   - Works ONLY with AutoODESAM (has .decoder, no .prompt_encoder)
#   - Non-AutoODESAM models are silently skipped (coarse masks returned)
#   - No extra dependencies beyond torch
# =============================================================================

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Organ IDs for small structures that benefit from zoom-in refinement.
# Indices match the AMOS22 15-organ class convention used across VoluFormer3D:
#   0=spleen, 1=right_kidney, 2=left_kidney, 3=gallbladder, 4=esophagus,
#   5=liver, 6=stomach, 7=aorta, 8=inferior_vena_cava, 9=pancreas,
#   10=right_adrenal, 11=left_adrenal, 12=duodenum, 13=bladder, 14=prostate
#
# IDs selected here: gallbladder(3), esophagus(4), right adrenal(10),
# left adrenal(11), duodenum(12)
# ---------------------------------------------------------------------------
SMALL_ORGAN_IDS: List[int] = [3, 4, 10, 11, 12]
# 3=gallbladder, 4=esophagus, 10=right_adrenal, 11=left_adrenal, 12=duodenum


def _is_auto_ode_sam(model: torch.nn.Module) -> bool:
    """
    Return True iff model is AutoODESAM.
    Detection rule: has .decoder attribute AND does NOT have .prompt_encoder.
    This avoids importing AutoODESAM here (would create a circular dep risk).
    """
    return hasattr(model, "decoder") and not hasattr(model, "prompt_encoder")


def extract_roi_bbox(
    mask: torch.Tensor,
    padding: int = 32,
    min_size: int = 48,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Find the bounding box of foreground pixels in a 2-D binary mask.

    Args:
        mask     : (H, W) float tensor — binary foreground mask (0/1 or logit).
                   Any value > 0 is treated as foreground.
        padding  : Pixels to add on each side of the tight bounding box.
        min_size : Minimum side length for the returned crop region.

    Returns:
        (y1, x1, y2, x2) with padding applied and clamped to image bounds,
        or None if no foreground pixels are found.
    """
    H, W = mask.shape[-2], mask.shape[-1]

    # Flatten to 2-D just in case extra leading dims sneak in
    if mask.dim() > 2:
        mask = mask.squeeze()
    assert mask.dim() == 2, f"extract_roi_bbox expects 2-D mask, got {mask.shape}"

    fg = (mask > 0).nonzero(as_tuple=False)  # (N, 2) — rows are (y, x) coords
    if fg.numel() == 0:
        return None

    y_min = int(fg[:, 0].min().item())
    y_max = int(fg[:, 0].max().item())
    x_min = int(fg[:, 1].min().item())
    x_max = int(fg[:, 1].max().item())

    # Add padding
    y1 = max(0, y_min - padding)
    x1 = max(0, x_min - padding)
    y2 = min(H, y_max + padding + 1)
    x2 = min(W, x_max + padding + 1)

    # Enforce minimum crop size — expand symmetrically if too small
    if (y2 - y1) < min_size:
        deficit = min_size - (y2 - y1)
        expand_top = deficit // 2
        expand_bot = deficit - expand_top
        y1 = max(0, y1 - expand_top)
        y2 = min(H, y2 + expand_bot)
        # If still short (near boundary), push the other side
        if (y2 - y1) < min_size:
            if y1 == 0:
                y2 = min(H, min_size)
            else:
                y1 = max(0, y2 - min_size)

    if (x2 - x1) < min_size:
        deficit = min_size - (x2 - x1)
        expand_left  = deficit // 2
        expand_right = deficit - expand_left
        x1 = max(0, x1 - expand_left)
        x2 = min(W, x2 + expand_right)
        if (x2 - x1) < min_size:
            if x1 == 0:
                x2 = min(W, min_size)
            else:
                x1 = max(0, x2 - min_size)

    return (y1, x1, y2, x2)


def zoom_refine_organ(
    model: torch.nn.Module,
    image_3d: torch.Tensor,
    coarse_mask: torch.Tensor,
    organ_id: int,
    img_size: int = 256,
    padding: int = 32,
) -> torch.Tensor:
    """
    Zoom-in refinement for a single organ.

    Crops the volume around the coarse detection, resizes to img_size for
    4× effective resolution, runs a second forward pass, then pastes the
    refined mask back into the original spatial grid.

    Args:
        model       : AutoODESAM instance (checked by caller).
        image_3d    : (D, 3, H, W) float tensor — one CT volume.
        coarse_mask : (H, W) float tensor — binary mask from Pass 1.
        organ_id    : AMOS22 organ class index (0-based).
        img_size    : Target spatial size for the zoomed crop (default 256).
        padding     : Bounding-box padding passed to extract_roi_bbox.

    Returns:
        refined_mask: (H, W) float tensor — refined binary mask in original
                      coordinate space.  Returns coarse_mask unchanged if no
                      foreground bbox is found.
    """
    H, W = coarse_mask.shape[-2], coarse_mask.shape[-1]

    bbox = extract_roi_bbox(coarse_mask, padding=padding)
    if bbox is None:
        return coarse_mask

    y1, x1, y2, x2 = bbox

    # Crop the volume to the ROI across all depth slices
    # image_3d: (D, 3, H, W)  → crop: (D, 3, crop_h, crop_w)
    crop = image_3d[:, :, y1:y2, x1:x2]
    D, C, crop_h, crop_w = crop.shape

    # Resize crop to (D, 3, img_size, img_size) for higher effective resolution
    crop_resized = F.interpolate(
        crop,                        # (D, 3, crop_h, crop_w)
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False,
    )  # (D, 3, img_size, img_size)

    # Build inputs expected by AutoODESAM.forward:
    #   images   : (1, D, 3, img_size, img_size)  (batch=1)
    #   organ_id : (1,) long tensor
    images_batch = crop_resized.unsqueeze(0)  # (1, D, 3, img_size, img_size)
    organ_id_t   = torch.tensor([organ_id], dtype=torch.long,
                                device=image_3d.device)

    with torch.no_grad():
        out = model(images_batch, organ_id_t,
                    target_organ_ids=[organ_id], is_3d=True)

    # out["masks"]: (1, 1, img_size, img_size) for single organ query
    logits_zoomed = out["masks"][0, 0]          # (img_size, img_size)
    refined_zoomed = torch.sigmoid(logits_zoomed)  # (img_size, img_size)

    # Resize refined mask back to the original crop dimensions
    refined_crop = F.interpolate(
        refined_zoomed.unsqueeze(0).unsqueeze(0),  # (1, 1, img_size, img_size)
        size=(crop_h, crop_w),
        mode="bilinear",
        align_corners=False,
    )[0, 0]  # (crop_h, crop_w)

    # Paste into a zeros canvas of the original image dimensions
    refined_mask = torch.zeros(H, W,
                               dtype=refined_crop.dtype,
                               device=refined_crop.device)
    refined_mask[y1:y2, x1:x2] = refined_crop

    return refined_mask


def two_stage_inference(
    model: torch.nn.Module,
    image_3d: torch.Tensor,
    img_size: int = 256,
    small_organs: Optional[List[int]] = None,
    refine_threshold: float = 0.1,
) -> Dict[int, torch.Tensor]:
    """
    Two-stage inference: coarse pass over all organs, then zoom-in refinement
    for small organs that have detectable foreground in the coarse pass.

    Requires AutoODESAM (model.decoder present, model.prompt_encoder absent).
    Non-AutoODESAM models raise TypeError immediately (before any forward call).

    Args:
        model             : AutoODESAM segmentation model.
        image_3d          : (D, 3, H, W) float tensor — one CT volume,
                            no batch dimension.
        img_size          : Spatial resolution for model inference (default 256).
        small_organs      : List of organ IDs to zoom-refine.
                            Defaults to SMALL_ORGAN_IDS.
        refine_threshold  : Minimum mean foreground fraction in the coarse mask
                            for a zoom-refinement pass to be triggered.

    Returns:
        Dict mapping each organ_id (0–14) to its (H, W) float mask tensor.
        Masks for large organs come from Pass 1; masks for small organs with
        sufficient foreground come from Pass 2 (zoom-refined).
    """
    if small_organs is None:
        small_organs = SMALL_ORGAN_IDS

    # Guard must be at the TOP — before any forward call — so non-AutoODESAM
    # models fail fast with a clear error rather than crashing inside forward().
    if not _is_auto_ode_sam(model):
        raise TypeError(
            "two_stage_inference requires AutoODESAM (model.decoder present, "
            "model.prompt_encoder absent). Use the model's own predict() method instead."
        )

    D, C, H, W = image_3d.shape
    n_organs = 15

    # ------------------------------------------------------------------
    # Pass 1 — full-volume coarse inference for all 15 organs
    # ------------------------------------------------------------------
    # Use organ_id=1 (spleen) as a neutral ODE anchor for Pass 1.
    # The OrganQueryDecoder always produces all 15 masks regardless of the
    # organ_id fed to the ODE, so the choice here only affects the ODE
    # trajectory — spleen (id=1) gives a meaningful embedding vs. a zero.
    organ_id_all = torch.ones(1, dtype=torch.long, device=image_3d.device)  # organ 1 (spleen) as neutral anchor
    images_batch  = image_3d.unsqueeze(0)  # (1, D, 3, H, W)

    with torch.no_grad():
        out = model(images_batch, organ_id_all,
                    target_organ_ids=list(range(n_organs)), is_3d=True)

    # out["masks"]: (1, n_organs, H_out, W_out)
    coarse_logits = out["masks"][0]              # (n_organs, H_out, W_out)
    coarse_masks  = torch.sigmoid(coarse_logits) # (n_organs, H_out, W_out)

    # Resize coarse masks to original spatial dims if decoder upsampled to
    # a different size (e.g. 256 vs. 1024)
    H_out, W_out = coarse_masks.shape[-2], coarse_masks.shape[-1]
    if H_out != H or W_out != W:
        coarse_masks = F.interpolate(
            coarse_masks.unsqueeze(0),   # (1, n_organs, H_out, W_out)
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )[0]  # (n_organs, H, W)

    # Build result dict from coarse pass
    result: Dict[int, torch.Tensor] = {
        oid: coarse_masks[oid] for oid in range(n_organs)
    }

    # ------------------------------------------------------------------
    # Pass 2 — zoom-in refinement
    # ------------------------------------------------------------------
    for organ_id in small_organs:
        if organ_id >= n_organs:
            continue
        coarse_mask = result[organ_id]  # (H, W)

        # Only refine if enough foreground was detected in Pass 1
        if coarse_mask.mean().item() <= refine_threshold:
            continue

        refined = zoom_refine_organ(
            model=model,
            image_3d=image_3d,
            coarse_mask=coarse_mask,
            organ_id=organ_id,
            img_size=img_size,
            padding=32,
        )
        result[organ_id] = refined

    return result
