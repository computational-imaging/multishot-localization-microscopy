# Copyright (c) 2021, The Board of Trustees of the Leland Stanford Junior University

"""A layer for 2D Gaussian blur."""

from __future__ import annotations

import torch
import torch.nn
import torch.nn.functional as F

from util import complex, fft


class GaussianBlur2D(torch.nn.Module):
    """A 2D Gaussian layer."""

    def __init__(self, sigma: float, requires_grad=False):
        """__init__ for GaussianBlur2D."""
        super().__init__()
        self.input_sz = None
        self.sigma_ = torch.nn.Parameter(
            torch.tensor(sigma).float(), requires_grad=requires_grad
        )

    def set_input_sz(self, input_sz: torch.Size, device):
        """Change the size of Gaussian with the input size."""
        if not self.input_sz == input_sz:
            y = torch.arange(input_sz[-2], device=device).float() - input_sz[-2] / 2
            x = torch.arange(input_sz[-1], device=device).float() - input_sz[-1] / 2
            y, x = torch.meshgrid(y, x)
            self.x = x[None, None, ...]
            self.y = y[None, None, ...]

    def sigma(self):
        """Return the current standard deviation."""
        return F.relu(self.sigma_) + 1e-6

    def gaussian(self):
        """Compute a Gaussian function."""
        sigma = self.sigma().reshape(-1, 1, 1, 1)  # S x D x H x W
        g = torch.exp(-(self.x ** 2 + self.y ** 2) / (2 * sigma ** 2))
        g = g / g.sum(dim=(-1, -2), keepdim=True)
        return fft.fftshift(g, (-1, -2))

    def conv2d(self, x):
        """Convolve a Gaussian function."""
        self.set_input_sz(x.shape, x.device)
        g = self.gaussian()
        rfft_x = torch.rfft(x, 2)
        rfft_g = torch.rfft(g, 2)
        rfft_y = complex.multiply(rfft_x, rfft_g)
        y = torch.irfft(rfft_y, 2, onesided=True, signal_sizes=x.shape[-2:])
        return y

    def forward(self, x):
        """Compute forward step."""
        return self.conv2d(x)
