# Copyright (c) 2021, The Board of Trustees of the Leland Stanford Junior University

"""Function for finding peaks in a sparse volume."""

from __future__ import annotations

from typing import Union

import numpy as np
import torch
import trackpy as tp

tp.quiet(suppress=True)


def peak_finder(
    input: Union[np.ndarray, torch.Tensor],
    diameter: float = 7,
    threshold: float = 100,
    preprocess: bool = True,
    n_processes: int = 20,
):
    """Find peaks with Trackpy."""
    assert input.ndim == 4

    if isinstance(input, torch.Tensor):
        input = input.detach().cpu().numpy()

    # Trackpy doesn't handle a floating-pint image with high dynamic range.
    # It performs better if it is converted to an integer image.
    input_max = input.max()
    if input_max < 2 ** 8:
        input = input
    elif input_max < 2 ** 16:
        input = input.astype(np.uint16)
    elif input_max < 2 ** 32:
        input = input.astype(np.uint32)
    elif input_max < 2 ** 64:
        input = input.astype(np.uint64)
    else:
        raise ValueError("Max of the input has to be less than 2 ** 64.")

    tp_output = tp.batch(
        input,
        diameter,
        noise_size=0,
        preprocess=preprocess,
        threshold=threshold,
        processes=n_processes,
    )

    return tp_output
