# Copyright (c) 2021, The Board of Trustees of the Leland Stanford Junior University

"""Helper functions for saving images viewable in ImageJ."""

from __future__ import annotations

from typing import Union

import numpy as np
import skimage.io
import torch


def imsave(fname, x: Union[np.ndarray, torch.Tensor]):
    """Save input images (C, D, H, W) as an ImageJ viewerble tiff."""
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    if x.ndim == 4:
        x = x.transpose((1, 0, 2, 3))
    elif x.ndim == 3:
        x = x[:, None, ...]
    elif x.ndim == 2:
        pass
    else:
        raise ValueError("Input has to be 2D, 3D or 4D.")
    skimage.io.imsave(fname, x, check_contrast=False, plugin="tifffile", imagej=True)


def imread(fname: str) -> np.ndarray:
    """Read an image file."""
    return skimage.io.imread(fname)
