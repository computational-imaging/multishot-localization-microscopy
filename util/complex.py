# Copyright (c) 2021, The Board of Trustees of the Leland Stanford Junior University

"""Helper functions for complex data.

NOTE:
This repo is developed before Pytorch implemented the complex data type.
"""
from __future__ import annotations

import torch


def pack(real: torch.Tensor, imag: torch.Tensor):
    """Stack real and imaginary parts at the last dimention."""
    return torch.stack([real, imag], dim=-1)


def unpack(x: torch.Tensor):
    """Unpack the real and imaginary parts."""
    return x[..., 0], x[..., 1]


def conj(x):
    """Apply conjuagate on complex tensor."""
    return torch.stack([x[..., 0], -x[..., 1]], dim=-1)


def abs2(x: torch.Tensor):
    """Return the squared absolute value of a tensor in complex form."""
    return x[..., -1] ** 2 + x[..., -2] ** 2


def multiply(x: torch.Tensor, y: torch.Tensor):
    """Multiply each elements in complex form."""
    x_real, x_imag = unpack(x)
    y_real, y_imag = unpack(y)
    return torch.stack(
        [x_real * y_real - x_imag * y_imag, x_imag * y_real + x_real * y_imag], dim=-1
    )


def multiply_conj(x, y):
    """Return x * conj(y) in complex form."""
    x_real, x_imag = unpack(x)
    y_real, y_imag = unpack(y)
    yconj_real = y_real
    yconj_imag = -y_imag
    return torch.stack(
        [
            x_real * yconj_real - x_imag * yconj_imag,
            x_imag * yconj_real + x_real * yconj_imag,
        ],
        dim=-1,
    )
