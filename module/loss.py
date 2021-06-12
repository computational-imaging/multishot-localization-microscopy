# Copyright (c) 2021, The Board of Trustees of the Leland Stanford Junior University

"""Implementaion of loss functions."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from pytorch_lightning.metrics import Metric
from util import complex


class DeepStormLoss(Metric):
    """A loss function layer.

    This module computes the Huber loss between the target and the input after
    convovlving a Gaussian function in 3D.
    """

    def __init__(
        self,
        lateral_sz: int,
        axial_sz: int,
        sigma_xy: float,
        sigma_z: float,
        offset_sppx: int,
        **kwargs
    ):
        """__init__ for DeepStormLoss.

        Args:
            lateral_sz: Lateral size of input in pixels.
            axial_sz: Axial size of input in pixels.
            sigma_xy: Lateral std of Gaussian in pixels
            sigma_z: Axial std of Gaussian in pixels.
            offset_sppx: A number of pixels at the edge where we exclude from loss
                computation.
            kwargs: Parameters for Metric class.
        """
        super().__init__(**kwargs)
        self.xy_pad = int(sigma_xy * 6)
        self.z_pad = int(sigma_z * 6) if int(sigma_z * 6) > 0 else 1
        filter_lateral_sz = lateral_sz - 2 * offset_sppx + self.xy_pad * 2
        filter_axial_sz = axial_sz + self.z_pad * 2
        gauss_lateral = gaussian(filter_lateral_sz, sigma_xy)
        gauss_axial = gaussian(filter_axial_sz, sigma_z)
        gauss3d = (
            gauss_axial.reshape(-1, 1, 1)
            * gauss_lateral.reshape(1, -1, 1)
            * gauss_lateral.reshape(1, 1, -1)
        )
        gauss3d = np.fft.fftshift(gauss3d)
        filter = torch.from_numpy(gauss3d)  # D x H x W  [None, None, ...]
        self.register_buffer("Ffilter3d", torch.rfft(filter, 3), persistent=False)
        self.p = offset_sppx

        self.add_state("huber_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def padconv(self, x: torch.Tensor) -> torch.Tensor:
        """Convolve a Gaussian filter with padding."""
        if self.p > 0:
            x = x[..., self.p : -self.p, self.p : -self.p]
        self.Ffilter3d = self.Ffilter3d.to(x.device)
        x_pad = F.pad(
            x,
            (
                self.xy_pad,
                self.xy_pad,
                self.xy_pad,
                self.xy_pad,
                self.z_pad,
                self.z_pad,
            ),
            mode="constant",
            value=0,
        )
        X = torch.rfft(x_pad, 3)
        convX = complex.multiply(X, self.Ffilter3d)
        convx = torch.irfft(convX, 3, signal_sizes=x_pad.shape[-3:])
        return convx

    def huber_loss(self, diff: torch.Tensor, delta: float) -> torch.Tensor:
        """Compute Huber loss.

        This implementation is slightly different from toch.nn.functional.smooth_l1_loss.
        torch.nn.SmoothL1Loss scales the error before computing the mean.
        Either way is fine, but we stick to this implementation as we tuned the
        regularization parameter based on this implementation.

        PyTorch's implementation
        https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html#torch.nn.SmoothL1Loss
        Our implementation
        https://en.wikipedia.org/wiki/Huber_loss
        """
        abs_diff = torch.abs(diff)
        error = torch.where(
            torch.le(abs_diff, delta),
            0.5 * (diff ** 2),
            delta * (abs_diff - 0.5 * delta),
        )
        return error.sum()

    def train_loss(self, input: torch.Tensor, gt: torch.Tensor):
        """Compute a train loss."""
        conv_input = self.padconv(input)
        conv_gt = self.padconv(gt)
        return self.huber_loss(conv_input - conv_gt, 1000.0) / conv_gt.numel()

    def update(self, input: torch.Tensor, gt: torch.Tensor):
        """Update Metric. This is implicitely used for computing validation loss."""
        conv_input = self.padconv(input)
        conv_gt = self.padconv(gt)
        self.huber_error += self.huber_loss(conv_input - conv_gt, 1000.0).cpu()
        self.total += conv_gt.numel()

    def compute(self) -> torch.Tensor:
        """Compute the validation loss. This is called in foward by Metric class."""
        return self.huber_error / self.total


def gaussian(N: int, sigma: float):
    """Compute a 1D Gaussian function."""
    assert N % 2 == 0, "Signal dimension has to be even."
    x = np.arange(0, N, dtype=np.float32) - N / 2
    return np.exp(-((x / sigma) ** 2) / 2)
