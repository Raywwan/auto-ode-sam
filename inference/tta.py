"""Test-time augmentation wrapper.

Runs the model under a set of spatial transforms (identity + H-flip + V-flip +
rot90), inverts the transforms on predictions, averages probability maps.
Applied once per sliding-window step — composes naturally with
sliding_window_predict_3d.

Usage:
    probs = tta_predict(model, image_window, organ_id, transforms=("id","hflip","vflip"))
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F


_TRANSFORMS_ALL = ("id", "hflip", "vflip", "rot90", "rot180", "rot270")


def _apply(x: torch.Tensor, tf: str) -> torch.Tensor:
    """x: (..., H, W). Return transformed tensor."""
    if tf == "id":
        return x
    if tf == "hflip":
        return torch.flip(x, dims=(-1,))
    if tf == "vflip":
        return torch.flip(x, dims=(-2,))
    if tf == "rot90":
        return torch.rot90(x, k=1, dims=(-2, -1))
    if tf == "rot180":
        return torch.rot90(x, k=2, dims=(-2, -1))
    if tf == "rot270":
        return torch.rot90(x, k=3, dims=(-2, -1))
    raise ValueError(f"unknown transform {tf}")


def _invert(x: torch.Tensor, tf: str) -> torch.Tensor:
    if tf in ("id", "hflip", "vflip", "rot180"):
        return _apply(x, tf)  # self-inverse
    if tf == "rot90":
        return torch.rot90(x, k=3, dims=(-2, -1))
    if tf == "rot270":
        return torch.rot90(x, k=1, dims=(-2, -1))
    raise ValueError(f"unknown transform {tf}")


@torch.no_grad()
def tta_predict(
    model: torch.nn.Module,
    images: torch.Tensor,                  # (B, D, 3, H, W)
    organ_id: torch.Tensor,                # (B,) long
    transforms: Sequence[str] = ("id", "hflip", "vflip"),
    amp: bool = True,
) -> torch.Tensor:
    """Return sigmoid-averaged masks (B, K, H, W) at model native output size."""
    probs_sum = None
    for tf in transforms:
        img_tf = _apply(images, tf)
        with torch.amp.autocast("cuda", enabled=amp):
            out = model(img_tf, organ_id, is_3d=True)
        masks = out["masks"].float()                   # (B, K, h, w)
        masks = _invert(masks, tf)
        p = torch.sigmoid(masks)
        probs_sum = p if probs_sum is None else probs_sum + p
    return probs_sum / float(len(transforms))
