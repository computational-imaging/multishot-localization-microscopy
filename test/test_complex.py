# Copyright (c) 2021, The Board of Trustees of the Leland Stanford Junior University

"""Test functions for complex.py."""

import numpy as np
import torch
from util import complex


class TestComplex:
    """A test cllass for complex.py."""

    def setup(self):
        """Set up input arrays."""
        n = 3
        self.input1_np = np.random.randn(n) + 1j * np.random.randn(n)
        self.input2_np = np.random.randn(n) + 1j * np.random.randn(n)
        self.input1_torch = complex.pack(
            torch.from_numpy(np.real(self.input1_np)),
            torch.from_numpy(np.imag(self.input1_np)),
        )
        self.input2_torch = complex.pack(
            torch.from_numpy(np.real(self.input2_np)),
            torch.from_numpy(np.imag(self.input2_np)),
        )

    def to_numpy(self, x: torch.Tensor) -> np.ndarray:
        """Tensor in complex form to numpy."""
        real, imag = complex.unpack(x)
        return real.numpy() + 1j * imag.numpy()

    def test_conj(self):
        """Test conj."""
        self.setup()
        output1_np = np.conj(self.input1_np)
        output1_torch = complex.conj(self.input1_torch)
        np.testing.assert_equal(output1_np, self.to_numpy(output1_torch))

    def test_abs2(self):
        """Test abs2."""
        self.setup()
        output_np = np.abs(self.input1_np) ** 2
        output_torch = complex.abs2(self.input1_torch)
        np.testing.assert_almost_equal(output_np, output_torch.numpy())

    def test_multiply(self):
        """Test multiply."""
        self.setup()
        output_np = self.input1_np * self.input2_np
        output_torch = complex.multiply(self.input1_torch, self.input2_torch)
        np.testing.assert_almost_equal(output_np, self.to_numpy(output_torch))

    def test_multiply_conj(self):
        """Test multiply conj."""
        self.setup()
        output_np = self.input1_np * np.conj(self.input2_np)
        output_torch = complex.multiply_conj(self.input1_torch, self.input2_torch)
        np.testing.assert_almost_equal(output_np, self.to_numpy(output_torch))
