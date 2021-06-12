# Copyright (c) 2021, The Board of Trustees of the Leland Stanford Junior University

"""Tests for helper functions."""

from util.helper import adjust_axial_sz_px


def test_adjust_axial_sz_px():
    """Test adjust_axial_sz_px."""
    assert adjust_axial_sz_px(1, 16) == 16
    assert adjust_axial_sz_px(2, 16) == 16
    assert adjust_axial_sz_px(16, 16) == 16
    assert adjust_axial_sz_px(17, 16) == 32
    assert adjust_axial_sz_px(32, 16) == 32
