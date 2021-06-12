# Copyright (c) 2021, The Board of Trustees of the Leland Stanford Junior University

"""Tests for fft routines."""

import itertools

import numpy as np
import pytest
import torch

from util.fft import fftshift, ifftshift, crop_psf

size = ((3,), (4,), (4, 4), (3, 4), (4, 3), (4, 5, 6))


@pytest.mark.parametrize("size", size)
def test_fftshift(size):
    """Test fftshift."""
    ndims = len(size)
    x = torch.rand(size)
    x_np = x.numpy()
    for d in range(ndims):
        for axes in itertools.combinations(range(ndims), d + 1):
            y = fftshift(x, axes)
            y_np = np.fft.fftshift(x_np, axes)
            print(axes, size)
            print(x, "\n", x_np)
            print(y, "\n", y_np)
            torch.testing.assert_allclose(y, y_np)


@pytest.mark.parametrize("size", size)
def test_ifftshift(size):
    """Test ifftshift."""
    ndims = len(size)
    x = torch.rand(size)
    x_np = x.numpy()
    for d in range(ndims):
        for axes in itertools.combinations(range(ndims), d + 1):
            y = ifftshift(x, axes)
            y_np = np.fft.ifftshift(x_np, axes)
            print(axes, size)
            print(x, "\n", x_np)
            print(y, "\n", y_np)
            torch.testing.assert_allclose(y, y_np)


def test_crop_psf():
    """Test crop_psf."""
    x = torch.tensor(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15],
        ]
    )
    y2_expected = torch.tensor(
        [
            [0, 3],
            [12, 15],
        ]
    )
    y2 = crop_psf(x, 2)
    y3_expected = torch.tensor(
        [
            [0, 1, 3],
            [4, 5, 7],
            [12, 13, 15],
        ]
    )
    y3 = crop_psf(x, 3)
    y4_expected = x.clone()
    y4 = crop_psf(x, 4)
    torch.testing.assert_allclose(y2, y2_expected)
    torch.testing.assert_allclose(y3, y3_expected)
    torch.testing.assert_allclose(y4, y4_expected)
