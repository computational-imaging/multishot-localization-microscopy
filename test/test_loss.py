# Copyright (c) 2021, The Board of Trustees of the Leland Stanford Junior University

"""Tests for loss functions."""

import numpy as np
import scipy.signal as signal

from module.loss import gaussian


def test_gaussian():
    """Test Gaussian."""
    g_scipy = signal.gaussian(10, 3, sym=False)
    g = gaussian(10, 3)
    assert np.allclose(g, g_scipy)
    assert g[5] == 1
