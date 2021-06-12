# Copyright (c) 2021, The Board of Trustees of the Leland Stanford Junior University

"""Dataset class for loading captured images.

Our camera has a gain of 2.0 and an offset of 100. Based on these parameters, this class
converts the captured image to photoelectron numbers.

"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.utils.data
import util.io
from skimage.util import view_as_windows


class CapturedImageStack(torch.utils.data.Dataset):
    def __init__(
        self,
        img_path: str,
        patch_sz: int,
        overlap_sz: int = 20,
        z_start: Optional[int] = None,
        z_end: Optional[int] = None,
    ):
        img = util.io.imread(img_path).astype(np.float32)  # D x S x H x W
        if img.ndim == 3:
            img = img[None, ...]
        if img.shape[-1] == 3 or img.shape[-1] == 4:
            img = img.transpose((0, 3, 1, 2))

        if z_start is not None and z_end is not None:
            img = img[z_start : z_end + 1]
        elif z_start is not None and z_end is None:
            img = img[z_start:]
        elif z_start is None and z_end is not None:
            img = img[: z_end + 1]
        img = img[:, :2]

        assert img.ndim == 4
        gain = 2.0
        offset = 100
        # view_as_windows discards the right and the bottom edges.
        self.patches = view_as_windows(
            (img - offset) / gain,
            window_shape=(1, img.shape[-3], patch_sz, patch_sz),
            step=(1, img.shape[-3], patch_sz - overlap_sz, patch_sz - overlap_sz),
        )
        self.n_tiles = (
            self.patches.shape[0],
            self.patches.shape[2],
            self.patches.shape[3],
        )
        self.patches = self.patches.reshape(-1, img.shape[-3], patch_sz, patch_sz)
        self.overlap_sz = overlap_sz
        self.full_img_shape = img.shape

    def __len__(self):
        return self.patches.shape[0]

    def __getitem__(self, idx):
        patch = torch.from_numpy(self.patches[idx]).float()
        return {"captimg": patch, "idx": idx}
