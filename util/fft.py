# Copyright (c) 2021, The Board of Trustees of the Leland Stanford Junior University

"""FFT-related functions."""

from __future__ import annotations

from typing import Union, Iterable

import torch


def fftshift(x, dims: Iterable[int]):
    """Apply fftshift over specified dimentions."""
    shifts = [(x.size(dim)) // 2 for dim in dims]
    x = torch.roll(x, shifts=shifts, dims=dims)
    return x


def ifftshift(x, dims: Iterable[int]):
    """Apply ifftshift over specified dimentions."""
    shifts = [(x.size(dim) + 1) // 2 for dim in dims]
    x = torch.roll(x, shifts=shifts, dims=dims)
    return x


def crop_psf(x, sz: Union[int, Iterable[int]]):
    """Crop PSF.

    This function crop a PSF which has a peak at the corners (not at the center).

    Args:
        x (torch.tensor): psf without applying fftshift (the center is upper left)
            shape (S x D x H x W)
        sz : size after cropping

    Returns:
        cropped psf
            shape (S x D x n x n)
    """
    device = x.device
    if isinstance(sz, int):
        sz = (sz, sz)
    p0 = (sz[0] - 1) // 2 + 1
    p1 = (sz[1] - 1) // 2 + 1
    q0 = sz[0] - p0
    q1 = sz[1] - p1
    x_0 = torch.index_select(
        x,
        dim=-2,
        index=torch.cat(
            [
                torch.arange(p0, device=device),
                torch.arange(x.shape[-2] - q0, x.shape[-2], device=device),
            ],
            dim=0,
        ),
    )
    x_1 = torch.index_select(
        x_0,
        dim=-1,
        index=torch.cat(
            [
                torch.arange(p1, device=device),
                torch.arange(x.shape[-1] - q1, x.shape[-1], device=device),
            ],
            dim=0,
        ),
    )
    return x_1
