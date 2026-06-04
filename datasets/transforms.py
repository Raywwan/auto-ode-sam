# =============================================================================
# datasets/transforms.py — Data Augmentation Pipeline
#
# Medical image augmentation is different from natural image augmentation:
#   - Elastic deformation is critical (organs deform differently in each scan)
#   - Colour jitter is less important (CT/MRI have fixed modality characteristics)
#   - Spatial augmentations must be applied consistently to image AND mask
#   - 3D augmentations (z-axis flip, slice dropout) improve 3D generalisation
#
# This module provides:
#   1. get_train_transforms() — heavy augmentation for training
#   2. get_val_transforms()   — minimal augmentation for validation
#   3. get_3d_transforms()    — additional 3D-specific augmentations
# =============================================================================

import random
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# =============================================================================
# Helper Functions
# =============================================================================

def resize_and_pad(
    image: np.ndarray,
    mask: np.ndarray,
    target_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resize image and mask to target_size x target_size, maintaining aspect ratio
    by padding with zeros.

    Args:
        image: (H, W) or (H, W, C) numpy array
        mask: (H, W) numpy array
        target_size: Target square size

    Returns:
        image: (target_size, target_size) or (target_size, target_size, C)
        mask: (target_size, target_size)
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)

    new_h = int(h * scale)
    new_w = int(w * scale)

    # Resize
    image_tensor = torch.from_numpy(image.copy()).float()
    mask_tensor = torch.from_numpy(mask.copy()).float()

    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    else:
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)

    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    image_resized = F.interpolate(
        image_tensor, size=(new_h, new_w), mode="bilinear", align_corners=False
    ).squeeze()
    mask_resized = F.interpolate(
        mask_tensor, size=(new_h, new_w), mode="nearest"
    ).squeeze()

    # Pad to target_size x target_size
    pad_h = target_size - new_h
    pad_w = target_size - new_w

    if image_resized.dim() == 2:
        image_resized = F.pad(image_resized, (0, pad_w, 0, pad_h), value=0)
    else:
        image_resized = F.pad(image_resized, (0, pad_w, 0, pad_h), value=0)

    mask_resized = F.pad(mask_resized, (0, pad_w, 0, pad_h), value=0)

    # Permute back to (H, W, C) so transform classes can do .permute(2, 0, 1) → (C, H, W)
    if image_resized.dim() == 3:
        image_resized = image_resized.permute(1, 2, 0)  # (C, H, W) → (H, W, C)

    return image_resized.numpy(), mask_resized.numpy()


def normalise_intensity(
    image: np.ndarray,
    clip_range: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Normalise image intensity to [0, 1].

    For CT: clip HU values first (e.g., -175 to 250 for abdominal soft tissue).
    For other modalities: just normalise to [0, 1].

    Args:
        image: Input image array
        clip_range: (min, max) HU clipping range (None = no clipping)

    Returns:
        Normalised image in [0, 1]
    """
    if clip_range is not None:
        image = np.clip(image, clip_range[0], clip_range[1])
        vmin, vmax = clip_range
    else:
        vmin = image.min()
        vmax = image.max()

    if vmax - vmin < 1e-8:
        return np.zeros_like(image, dtype=np.float32)

    return ((image - vmin) / (vmax - vmin)).astype(np.float32)


def to_3channel(image: np.ndarray) -> np.ndarray:
    """
    Convert grayscale (H, W) image to 3-channel (H, W, 3) for TinyViT.
    TinyViT was pretrained on ImageNet (3-channel RGB).

    Args:
        image: (H, W) float32 in [0, 1]

    Returns:
        image: (H, W, 3) float32 in [0, 1]
    """
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)
    elif image.shape[-1] == 1:
        return np.concatenate([image, image, image], axis=-1)
    return image  # Already 3-channel


def get_bounding_box(
    mask: np.ndarray,
    padding: int = 0,
) -> np.ndarray:
    """
    Extract the bounding box of the foreground in a binary mask.

    Returns [x1, y1, x2, y2] in pixel coordinates.
    If mask is empty, returns [0, 0, W-1, H-1] (full image).

    Args:
        mask: (H, W) binary mask
        padding: Extra pixels to expand the box (for robustness)

    Returns:
        box: [x1, y1, x2, y2] numpy array
    """
    H, W = mask.shape

    if not mask.any():
        # No foreground — return full image bounding box
        return np.array([0, 0, W - 1, H - 1], dtype=np.float32)

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]

    # Apply padding and clip to image bounds
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(W - 1, x2 + padding)
    y2 = min(H - 1, y2 + padding)

    return np.array([x1, y1, x2, y2], dtype=np.float32)


# =============================================================================
# Augmentation Functions
# =============================================================================

def random_flip(
    image: np.ndarray,
    mask: np.ndarray,
    prob: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Random horizontal and vertical flip."""
    if random.random() < prob:
        image = np.fliplr(image)
        mask = np.fliplr(mask)
    if random.random() < prob:
        image = np.flipud(image)
        mask = np.flipud(mask)
    return image, mask


def random_rotate(
    image: np.ndarray,
    mask: np.ndarray,
    max_angle: float = 15.0,
    prob: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Random rotation by up to max_angle degrees.
    Uses bilinear interpolation for image, nearest-neighbour for mask.
    """
    if random.random() > prob:
        return image, mask

    try:
        from scipy.ndimage import rotate
        angle = random.uniform(-max_angle, max_angle)
        image = rotate(image, angle, reshape=False, order=1, mode="nearest")
        mask = rotate(mask, angle, reshape=False, order=0, mode="nearest")
    except ImportError:
        pass  # Skip if scipy not available

    return image, mask


def random_windowing(
    lo: float,
    hi: float,
    center_jitter_frac: float = 0.10,
    width_jitter_frac:  float = 0.15,
    prob: float = 0.5,
) -> Tuple[float, float]:
    """Random HU windowing jitter for training (Eilertsen et al., MIDL 2025).

    Returns a jittered (lo', hi') pair to use INSTEAD of the canonical clip
    range, simulating different display-window / acquisition-protocol settings.

    Conceptually equivalent to randomising the window-center / window-width
    used for HU normalisation: the network sees the same anatomy under
    slightly different intensity contracts, which helps generalisation to
    CT volumes acquired with a different reconstruction kernel or rescaled
    by a different reviewer.

    Implementation: convert (lo, hi) → (center, width), jitter both
    multiplicatively, convert back. The default jitter is conservative
    (±10% center, ±15% width) so the median sample sees a near-canonical
    windowing and only the tails of the distribution see strong jitter.

    With prob < 1.0, the function passes through the canonical (lo, hi) on
    a fraction of calls so the network sees the canonical windowing too.

    Args:
        lo: Canonical lower clip (e.g. -175 for abdominal CT).
        hi: Canonical upper clip (e.g. 250 for abdominal CT).
        center_jitter_frac: Multiplicative jitter on window CENTER, sampled
            uniformly from [1 - cjf, 1 + cjf].
        width_jitter_frac: Multiplicative jitter on window WIDTH, sampled
            uniformly from [1 - wjf, 1 + wjf].
        prob: Probability of applying jitter (1.0 = always).

    Returns:
        (lo', hi') tuple, with lo' < hi' guaranteed.
    """
    if random.random() >= prob:
        return float(lo), float(hi)

    center = 0.5 * (lo + hi)
    width  = hi - lo
    c_jitter = 1.0 + random.uniform(-center_jitter_frac, center_jitter_frac)
    w_jitter = 1.0 + random.uniform(-width_jitter_frac,  width_jitter_frac)
    new_center = center * c_jitter
    new_width  = max(width * w_jitter, 1e-3)
    new_lo = new_center - 0.5 * new_width
    new_hi = new_center + 0.5 * new_width
    return float(new_lo), float(new_hi)


def random_brightness_contrast(
    image: np.ndarray,
    brightness_range: float = 0.2,
    contrast_range: float = 0.2,
    prob: float = 0.3,
) -> np.ndarray:
    """
    Random brightness and contrast adjustment.
    Applied to image only (not mask).

    Note: Only applied to grayscale medical images.
    Do NOT apply to CT images after HU clipping (may introduce artefacts).
    """
    if random.random() > prob:
        return image

    # Brightness: add random offset
    if random.random() < 0.5:
        brightness = random.uniform(-brightness_range, brightness_range)
        image = image + brightness

    # Contrast: scale around mean
    if random.random() < 0.5:
        contrast = random.uniform(1 - contrast_range, 1 + contrast_range)
        mean_val = image.mean()
        image = (image - mean_val) * contrast + mean_val

    return np.clip(image, 0, 1).astype(np.float32)


def random_elastic_deformation(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 50.0,
    sigma: float = 10.0,
    prob: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Elastic deformation — critical for medical image augmentation.
    Simulates the natural deformation of organs between scans.

    Algorithm (Simard et al., 2003):
      1. Generate random displacement fields dx, dy
      2. Smooth with Gaussian filter (sigma controls smoothness)
      3. Scale by alpha (controls magnitude)
      4. Apply to image and mask

    Args:
        alpha: Displacement magnitude (larger = more deformation)
        sigma: Gaussian smoothness (smaller = rougher/less realistic)
    """
    if random.random() > prob:
        return image, mask

    try:
        from scipy.ndimage import gaussian_filter, map_coordinates

        h, w = image.shape[:2]

        # Random displacement fields
        dx = gaussian_filter(
            (np.random.rand(h, w) * 2 - 1), sigma, mode="constant", cval=0
        ) * alpha
        dy = gaussian_filter(
            (np.random.rand(h, w) * 2 - 1), sigma, mode="constant", cval=0
        ) * alpha

        # Create meshgrid and apply displacement
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        indices = (
            np.reshape(y + dy, (-1, 1)),
            np.reshape(x + dx, (-1, 1)),
        )

        # Apply deformation
        if image.ndim == 2:
            image_def = map_coordinates(image, indices, order=1, mode="nearest")
            image_def = image_def.reshape(h, w)
        else:
            # Apply channel by channel for 3-channel images
            image_def = np.stack([
                map_coordinates(image[:, :, c], indices, order=1, mode="nearest").reshape(h, w)
                for c in range(image.shape[2])
            ], axis=-1)

        mask_def = map_coordinates(mask, indices, order=0, mode="nearest").reshape(h, w)

        return image_def.astype(np.float32), mask_def.astype(mask.dtype)

    except ImportError:
        return image, mask


def random_gaussian_noise(
    image: np.ndarray,
    sigma_range: Tuple[float, float] = (0.0, 0.05),
    prob: float = 0.2,
) -> np.ndarray:
    """Add random Gaussian noise to the image."""
    if random.random() > prob:
        return image

    sigma = random.uniform(*sigma_range)
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    return np.clip(image + noise, 0, 1)


def random_box_jitter(
    box: np.ndarray,
    jitter_fraction: float = 0.1,
    img_size: int = 512,
) -> np.ndarray:
    """
    Randomly jitter the bounding box to simulate imperfect annotations.
    Critical for robustness: at test time, boxes come from human annotators
    or other detectors and may not be perfectly tight.

    Args:
        box: [x1, y1, x2, y2] in pixel space
        jitter_fraction: Max jitter as fraction of box size
        img_size: Image size (for clamping)
    """
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1

    jitter_x = random.uniform(-jitter_fraction * bw, jitter_fraction * bw)
    jitter_y = random.uniform(-jitter_fraction * bh, jitter_fraction * bh)

    x1 = np.clip(x1 + jitter_x, 0, img_size - 1)
    y1 = np.clip(y1 + jitter_y, 0, img_size - 1)
    x2 = np.clip(x2 + jitter_x, 0, img_size - 1)
    y2 = np.clip(y2 + jitter_y, 0, img_size - 1)

    return np.array([x1, y1, x2, y2], dtype=np.float32)


# =============================================================================
# Transform Pipelines
# =============================================================================

class TrainTransforms:
    """
    Full augmentation pipeline for training.

    Order matters:
      1. Spatial transforms (flip, rotate, elastic) — applied to image + mask
      2. Intensity transforms (brightness, noise) — applied to image only
      3. Box jitter — applied to bounding box
      4. Normalisation and resizing — applied last

    Args:
        img_size (int): Output image size (square)
        clip_range: HU range for CT images (None for other modalities)
    """

    def __init__(
        self,
        img_size: int = 512,
        clip_range: Optional[Tuple[float, float]] = None,
    ):
        self.img_size = img_size
        self.clip_range = clip_range

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        box: np.ndarray = None,
    ) -> Dict:
        # ---- Spatial augmentations ----
        image, mask = random_flip(image, mask, prob=0.5)
        image, mask = random_rotate(image, mask, max_angle=15.0, prob=0.3)
        image, mask = random_elastic_deformation(image, mask, prob=0.2)

        # ---- Intensity augmentations (image only) ----
        image = random_brightness_contrast(image, prob=0.3)
        image = random_gaussian_noise(image, prob=0.2)

        # ---- Normalise intensity ----
        image = normalise_intensity(image, self.clip_range)

        # ---- Convert to 3-channel for TinyViT ----
        image = to_3channel(image)

        # ---- Resize and pad ----
        image, mask = resize_and_pad(image, mask, self.img_size)

        # ---- Bounding box ----
        if box is None:
            box = get_bounding_box(mask, padding=10)
        else:
            box = random_box_jitter(box, jitter_fraction=0.1, img_size=self.img_size)

        # ---- Convert to tensors ----
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()  # (3, H, W)
        mask_tensor = torch.from_numpy(mask).long()                       # (H, W)
        box_tensor = torch.from_numpy(box).float()                        # (4,)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "box": box_tensor,
        }


class ValTransforms:
    """
    Minimal preprocessing for validation/test — no augmentation.

    Just normalise and resize.
    """

    def __init__(
        self,
        img_size: int = 512,
        clip_range: Optional[Tuple[float, float]] = None,
    ):
        self.img_size = img_size
        self.clip_range = clip_range

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        box: np.ndarray = None,
    ) -> Dict:
        # Normalise
        image = normalise_intensity(image, self.clip_range)

        # 3-channel
        image = to_3channel(image)

        # Resize
        image, mask = resize_and_pad(image, mask, self.img_size)

        # Bounding box
        if box is None:
            box = get_bounding_box(mask, padding=10)

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask).long()
        box_tensor = torch.from_numpy(box).float()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "box": box_tensor,
        }
