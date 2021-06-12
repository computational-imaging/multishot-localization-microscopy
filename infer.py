# Copyright (c) 2021, The Board of Trustees of the Leland Stanford Junior University

"""
An inference script for the fixed cell dataset.

This script saves the image in result directory.

Usage:
    python infer.py \
    --img_path data/captured_data/fixed_cell.tif \
    --ckpt_path data/trained_model/fixed_cell.ckpt \
    --batch_sz 10 --save_dir result
"""
import os
from argparse import ArgumentParser, Namespace

import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from dataset.captured_image import CapturedImageStack
from localizer import Localizer


def main(args: Namespace):
    """Main fucntion for the inference."""
    os.makedirs(args.save_dir, exist_ok=True)

    dataset = CapturedImageStack(
        args.img_path,
        args.patch_sz,
        overlap_sz=args.overlap_sz,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_sz,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    # Override the parameters used for retraining.
    # The gain and aberrations are estimated separately from a captured bead.
    hparams = vars(args)
    hparams["depth_range_nm"] = 8000
    hparams["axial_sampling_nm"] = 250.0
    hparams["focus_offset_nm"] = 3875.0
    hparams["gain"] = [0.72, 1.0]
    hparams["init_phase"] = "data/fitted_psf/fixed_cell/designed_phase.tif"
    hparams[
        "depth_independent_aberration_path"
    ] = "data/fitted_psf/fixed_cell/depth_independent_aberration_w_tip_tilt_defocus.tif"

    hparams[
        "depth_dependent_aberration_path"
    ] = "data/fitted_psf/fixed_cell/depth_dependent_aberration.tif"
    hparams["patch_sz"] = args.patch_sz
    hparams["n_tiles"] = dataset.n_tiles
    hparams["full_img_shape"] = dataset.full_img_shape
    hparams["overlap_sz"] = dataset.overlap_sz

    model = Localizer(hparams=hparams)
    model.load_state_dict(torch.load(args.ckpt_path))

    trainer = Trainer.from_argparse_args(hparams)
    trainer.test(model, test_dataloaders=dataloader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--img_path", type=str)
    parser.add_argument("--patch_sz", type=int, default=32)
    parser.add_argument("--overlap_sz", type=int, default=20)
    parser = Trainer.add_argparse_args(parser)
    parser = Localizer.add_model_specific_args(parser)

    args = parser.parse_args()
    main(args)
