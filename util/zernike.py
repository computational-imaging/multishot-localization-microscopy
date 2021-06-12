# Copyright (c) 2021, The Board of Trustees of the Leland Stanford Junior University

"""Functions for computing Zernike polynomials."""

from __future__ import annotations

import math
import torch


def noll2mn(j):
    """Convert Noll's index to (m,n)."""
    n = int(math.ceil((-3 + math.sqrt(1 + 8 * j)) / 2))
    jr = j - int(n * (n + 1) / 2)
    p = n % 4
    if p == 0 or p == 1:
        m1 = jr
        m2 = -(jr - 1)
        if (n - m1) % 2 == 0:
            m = m1
        else:
            m = m2
    else:
        m1 = jr - 1
        m2 = -jr
        if (n - m1) % 2 == 0:
            m = m1
        else:
            m = m2
    return n, m


def radial(n, m, r):
    """Compute a radial function of Zernike polynomial."""
    assert m >= 0
    output = torch.zeros_like(r).double()
    for k in range(0, int((n - m) / 2) + 1):
        output += (
            (((-1) ** k) * math.factorial(n - k))
            / (
                math.factorial(k)
                * math.factorial(int((n + m) / 2 - k))
                * math.factorial(int((n - m) / 2 - k))
            )
            * r ** (n - 2 * k)
        )
    return output


def zernike_nm(n, m, r, theta):
    """Compute zernike polynomials from m,n index."""
    r = r.cpu()
    theta = theta.cpu()
    binary_mask = r <= 1.0
    if m == 0:
        zern = radial(n, 0, r)
    else:
        if m > 0:
            zern = radial(n, m, r) * torch.cos(m * theta)
        else:
            m = abs(m)
            zern = radial(n, m, r) * torch.sin(m * theta)

    zern = zern * binary_mask

    return zern.float()


def zernike_array(j, r, theta):
    """Compute zernike polynomials from Noll's index.

    Args:
        j: Noll's index
        r: Normalized radial coordinates
        theta: Angle in polar coordinates

    Returns:
        Zernike polynomial
    """
    n, m = noll2mn(j)
    return zernike_nm(n, m, r, theta)
