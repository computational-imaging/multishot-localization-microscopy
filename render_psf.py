"""Render the trained PSF.

The simulated PSF will be saved at "result" directory.
Download and expand the dataset in "data" directory.

Usage:
    python render_psf.py
"""
import os
from argparse import ArgumentParser

import torch

import util.io
from module.bead import Bead
from module.microscope import Microscope
from util.fft import crop_psf, fftshift


def main(hparams):
    """Render PSF."""
    save_dir = "result"
    os.makedirs(save_dir, exist_ok=True)
    phase_path = os.path.join("data", "fitted_psf", "fixed_cell", "designed_phase.tif")

    axial_sampling_nm = 200
    depth_range_nm = 12000
    focus_offset_nm = 6000
    output_psf_sz = 128

    hparams.axial_sampling_nm = axial_sampling_nm
    hparams.focus_offset_nm = focus_offset_nm
    hparams.depth_range_nm = depth_range_nm
    axial_sz_px = int(depth_range_nm // hparams.axial_sampling_nm)

    phase = util.io.imread(phase_path)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    microscope = Microscope(
        hparams,
        init_phase=phase,
    ).to(device)
    psf = microscope.psf(torch.arange(axial_sz_px))
    psf = fftshift(crop_psf(psf, output_psf_sz), (-1, -2))
    psf /= psf.max()
    psf = torch.cat([psf[0], psf[1]], dim=-1)

    util.io.imsave(os.path.join(save_dir, "psf.tif"), psf)


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser = Microscope.add_model_specific_args(parser)
    parser = Bead.add_model_specific_args(parser)
    hparams = parser.parse_args()
    main(hparams)
