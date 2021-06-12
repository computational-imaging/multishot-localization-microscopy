# Copyright (c) 2021, The Board of Trustees of the Leland Stanford Junior University

"""Helper functions."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def resize2d(x: torch.Tensor, scale_factor: int):
    """2D Downsample input with a scale factor."""
    return scale_factor ** 2 * F.avg_pool2d(x, scale_factor)


def scale_image(x):
    """Normalize iamge for visualization purpose."""
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x


def adjust_axial_sz_px(n: int, base_multiplier: int = 16) -> int:
    """Increase a input integer to be a multiple of mase_multiplier."""
    remainder = n % base_multiplier
    if remainder == 0:
        return n
    else:
        return n + base_multiplier - remainder
