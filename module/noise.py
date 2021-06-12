# Copyright (c) 2021, The Board of Trustees of the Leland Stanford Junior University

"""A differentiable layer for Poisson/Gaussian noise."""

from __future__ import annotations

import torch
import torch.nn as nn


class Noise(nn.Module):
    """Poisson-Gaussian noise layer.

    Noise layer based on the reparameterization trick
    https://stats.stackexchange.com/questions/199605/how-does-the-reparameterization-trick-for-vaes-work-and-why-is-it-important
    """

    def __init__(self, gain, readnoise_std, bg_min, bg_max, num_shots):
        """__init__ for Noise."""
        super().__init__()
        if isinstance(gain, float):
            gain = [gain for _ in range(num_shots)]
        if isinstance(readnoise_std, float):
            readnoise_std = [readnoise_std for _ in range(num_shots)]
        assert len(gain) == num_shots
        assert len(readnoise_std) == num_shots
        self.bg_min = bg_min
        self.bg_range = bg_max - bg_min

        self.register_buffer(
            "gain", torch.tensor(gain).reshape(1, -1, 1, 1), persistent=False
        )
        self.register_buffer(
            "sigma", torch.tensor(readnoise_std).reshape(1, -1, 1, 1), persistent=False
        )

    def forward(self, x):
        """Compute the forward step."""
        batch_sz = x.shape[0]
        bg = (
            self.bg_min + self.bg_range * torch.rand(batch_sz, device=self.gain.device)
        ).reshape(batch_sz, 1, 1, 1)
        x_plus_bg = x + bg
        unit_noise = torch.randn_like(x)
        noise_sigma = torch.sqrt(x_plus_bg) + self.sigma / self.gain
        noisy_img = self.gain * (x_plus_bg + noise_sigma * unit_noise)
        return noisy_img
