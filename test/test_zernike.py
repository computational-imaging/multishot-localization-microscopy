# Copyright (c) 2021, Stanford niversity

"""Test module for Zernike polynomial."""

import pytest
from util import zernike

index_pair = (
    (1, 0, 0),
    (2, 1, 1),
    (3, 1, -1),
    (4, 2, 0),
    (5, 2, -2),
    (6, 2, 2),
    (7, 3, -1),
    (8, 3, 1),
    (9, 3, -3),
    (10, 3, 3),
    (11, 4, 0),
    (12, 4, 2),
    (13, 4, -2),
    (14, 4, 4),
    (15, 4, -4),
    (16, 5, 1),
    (17, 5, -1),
    (18, 5, 3),
    (19, 5, -3),
    (20, 5, 5),
)


@pytest.mark.parametrize("index_pair", index_pair)
def test_noll(index_pair):
    """Test Noll's index conversion."""
    j = index_pair[0]
    n, m = zernike.noll2mn(j)
    assert index_pair == (j, n, m)


# import numpy.testing
# import aotools
# import torch
# @pytest.mark.parametrize("index", range(50))
# def test_zernike_R(index):
#     r = torch.linspace(0.0, 1.0).double()
#     n, m = zernike.noll2mn(index)
#     m = abs(m)
#     numpy.testing.assert_allclose(
#         aotools.functions.zernikeRadialFunc(n, m, r.numpy()),
#         zernike.radial(n, m, r).numpy(),
#     )
