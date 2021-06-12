# Copyright (c) 2021, The Board of Trustees of the Leland Stanford Junior University

"""Implementation of finite sized beads.

This layer convolves a superresolution sparse volume with a finite sized beads.
"""

from __future__ import annotations

from argparse import ArgumentParser
from typing import Iterable

import torch
import torch.nn.functional as F
from util import complex
from util.fft import fftshift


class Bead(torch.nn.Module):
    """A layer which blurs a superresolution volume with a bead size."""

    def __init__(self, voxel_sz_nm: Iterable[float], sphere_diameter_nm: float = 175):
        """__init__ for Bead layer."""
        super().__init__()
        assert len(voxel_sz_nm) == 2

        self.voxel_sz_nm = voxel_sz_nm
        self.sphere_diameter_nm = sphere_diameter_nm
        self.input_sz = None

        self.pad = [
            int(self.voxel_sz_nm[i] / self.sphere_diameter_nm) for i in range(2)
        ]

    def set_input_sz(self, input_sz: torch.Size, device):
        """Adjust the bead grid with the input size."""
        if not self.input_sz == input_sz:
            y = (
                torch.arange(input_sz[-2], device=device).float() - input_sz[-2] / 2
            ) * self.voxel_sz_nm[-2]
            x = (
                torch.arange(input_sz[-1], device=device).float() - input_sz[-1] / 2
            ) * self.voxel_sz_nm[-1]
            yy, xx = torch.meshgrid(y, x)
            rr = torch.sqrt(yy ** 2 + xx ** 2)
            sphere = (rr < self.sphere_diameter_nm / 2).float()
            if torch.sum(sphere) > 0:
                sphere /= torch.sum(sphere)
            else:
                sphere[input_sz[-2] // 2, input_sz[-1] // 2] = 1
            sphere = fftshift(sphere, (-1, -2))
            self.rfft_sphere = torch.rfft(sphere, 2)

    def conv(self, x):
        """Convolve the input with the bead size."""
        input_shape = x.shape
        x = F.pad(x, (0, self.pad[-1], 0, self.pad[-2]), mode="replicate")
        self.set_input_sz(x.shape[-2:], x.device)
        rfft_x = torch.rfft(x, 2)
        rfft_y = complex.multiply(rfft_x, self.rfft_sphere)

        y = torch.irfft(rfft_y, 2, onesided=True, signal_sizes=x.shape[-2:])
        y = y[..., : input_shape[-2], : input_shape[-1]]
        return y

    def forward(self, x):
        """Compute forward step."""
        return self.conv(x)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Add the bead diamter as a flag."""
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--sphere_diameter_nm", type=float, default=175.0)
        return parser
