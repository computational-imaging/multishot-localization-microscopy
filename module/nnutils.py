# Copyright (c) 2021, The Board of Trustees of the Leland Stanford Junior University

"""Helper layers for CNN."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class Upsample(nn.Module):
    """Upsampling layer."""

    def __init__(self, scale_factor, mode):
        """__init__ for Upsample class."""
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        """Compute forward step."""
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class Conv3dWithReplication(nn.Module):
    """3D conv layer with replication padding."""

    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        """__init__ for Conv3dWithReplication class."""
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReplicationPad3d(padding),
            nn.Conv3d(
                in_ch,
                out_ch,
                kernel_size,
                stride=stride,
                padding=0,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
        )

    def forward(self, x):
        """Compute forward step."""
        return self.conv(x)
