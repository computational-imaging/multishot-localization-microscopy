# Copyright (c) 2021, Stanford niversity

"""Tests for peak finder."""

import numpy as np
import pandas as pd
from util import peak_finder
from dataclasses import dataclass
from skimage.filters import gaussian


@dataclass()
class Point:
    """A class for defining a point in 3D."""

    x: int
    y: int
    z: int
    w: float


def test_peak_finder():
    """Test peak finder."""
    x = np.zeros((20, 30, 30))
    points = [
        Point(5, 7, 4, 3000),
        Point(10, 12, 10, 2000),
    ]
    gt_df = pd.DataFrame(columns=["z", "y", "x"])
    for i, p in enumerate(points):
        x[p.z, p.y, p.x] = p.w
        gt_df.loc[i] = [float(p.z), float(p.y), float(p.x)]
    x = np.expand_dims(
        gaussian(
            x,
            sigma=1.0,
        ),
        axis=0,
    )
    est_df = peak_finder.peak_finder(
        x, diameter=3, threshold=0, n_processes=1, preprocess=False
    )
    est_df = est_df.drop(
        ["mass", "size", "ecc", "signal", "ep", "frame", "raw_mass"], axis=1
    ).sort_values(by=["x"])
    pd.testing.assert_frame_equal(gt_df, est_df, check_exact=False)
