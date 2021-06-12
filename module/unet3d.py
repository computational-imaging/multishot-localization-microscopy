# Copyright (c) 2021, The Board of Trustees of the Leland Stanford Junior University

"""Implementaion of Unet3D.

NOTE:
Our Unet3D implementaion uses LeakyReLU and replication padding.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from module.nnutils import Conv3dWithReplication, Upsample


class Unet3d(nn.Module):
    """A main Unet3D class."""

    def __init__(self, n_shots: int, base_ch: int, leaky_relu_a: float):
        """__init__ for Unet3d.

        Args:
            n_shots: A number of cameras.
            base_ch: A number of channels defining the whole Unet3D arch.
            leaky_relu_a: A slope for LeakyReLU.
        """
        super().__init__()
        self.conv_in = Inconv3d(n_shots, base_ch, 2 * base_ch, leaky_relu_a)
        self.down1 = Downconv3d(2 * base_ch, 4 * base_ch, leaky_relu_a)
        self.down2 = Downconv3d(4 * base_ch, 8 * base_ch, leaky_relu_a)
        self.down3 = Downconv3d(8 * base_ch, 16 * base_ch, leaky_relu_a)
        self.down4 = Downconv3d(16 * base_ch, 32 * base_ch, leaky_relu_a)
        self.up4 = Upconv3d(32 * base_ch + 16 * base_ch, 16 * base_ch, leaky_relu_a)
        self.up3 = Upconv3d(16 * base_ch + 8 * base_ch, 8 * base_ch, leaky_relu_a)
        self.up2 = Upconv3d(8 * base_ch + 4 * base_ch, 4 * base_ch, leaky_relu_a)
        self.up1 = Upconv3d(4 * base_ch + 2 * base_ch, 2 * base_ch, leaky_relu_a)
        self.conv_out = Conv3dWithReplication(2 * base_ch, 1, 1, padding=0, bias=False)

    def forward(self, x):
        """Compute CNN output."""
        assert torch.isnan(x).sum() == 0
        x1 = self.conv_in(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.conv_out(x).squeeze(1)
        return x


class Inconv3d(nn.Module):
    """Two layers of conv with replication padding and leaky relu."""

    def __init__(self, in_ch, mid_ch, out_ch, a):
        """__init__ for Inconv3d."""
        super().__init__()
        self.conv = nn.Sequential(
            Conv3dWithReplication(in_ch, mid_ch, 3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=a, inplace=True),
            Conv3dWithReplication(mid_ch, out_ch, 3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=a, inplace=True),
        )

    def forward(self, x):
        """Compute forward step."""
        x = self.conv(x)
        return x


class Downconv3d(nn.Module):
    """Downsampling and two layers of conv with replication padding and leaky relu."""

    def __init__(self, in_ch, out_ch, a):
        """__init__ for Downconv3d."""
        super().__init__()
        self.conv = nn.Sequential(
            nn.AvgPool3d(2),
            Conv3dWithReplication(in_ch, in_ch, 3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=a, inplace=True),
            Conv3dWithReplication(in_ch, out_ch, 3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=a, inplace=True),
        )

    def forward(self, x):
        """Compute forward step."""
        x = self.conv(x)
        return x


class Upconv3d(nn.Module):
    """Upsampling and two layers of conv with replication padding and leaky relu."""

    def __init__(self, combined_ch, out_ch, a):
        """__init__ for Upconv3d."""
        super().__init__()
        self.up = Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Sequential(
            Conv3dWithReplication(combined_ch, out_ch, 3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=a, inplace=True),
            Conv3dWithReplication(out_ch, out_ch, 3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=a, inplace=True),
        )

    def forward(self, x1, x2):
        """Compute forward step."""
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
