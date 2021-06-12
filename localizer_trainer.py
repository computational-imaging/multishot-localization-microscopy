# Copyright (c) 2021, The Board of Trustees of the Leland Stanford Junior University

"""A launcher of localizer.py.

This script is the entry point of our training pipeline. It trains a CNN-based
localization model and a set of 3D PSFs parametrized with their corresponding phase
masks.
"""

from __future__ import annotations

import os
from argparse import ArgumentParser, Namespace
from typing import Tuple

import torch
import torch.utils.data
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from dataset import emitter
from localizer import Localizer


def prepare_dataset(
    hparams,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, str]:
    """Prepare datasets.

    Args:
        hparams: A set of hyperparameters.

    Returns:
        Dataloaders for training and validation, and a path to a csv file saving the
        coordinates and intensity of validation set.
        The training dataloader uses dataset.emitter.RandomEmitterDataset, and the
        validataion dataloader uses dataset.emitter.EmitterDatasetFromCSV.
    """
    offset_px = 0
    # As the number of training images is infinite, this doesn't matter.
    # It defines the number of frames per epoch for the visualization/logging purepose.
    train_n_frames = 100  # 50000
    # This defines the number of frames for validation.
    # The coordinates will be saved as csv file.
    val_n_frames = 300

    photon_sampler_params = emitter.UniformParams(
        low=hparams.uniform_low, high=hparams.uniform_high
    )
    pparam = f"low{hparams.uniform_low}_high{hparams.uniform_high}"
    hparams.mean_photons = (hparams.uniform_low + hparams.uniform_high) / 2

    mol_density = hparams.mol_density
    filename = (
        f"Uniform_{pparam}_capt{hparams.capt_sz_px}px"
        f"_depthrange{hparams.depth_range_nm * 1e-3:.1f}um_"
        f"zstep{args.axial_sampling_nm}nm_density{mol_density}_{val_n_frames}frames.csv"
    )

    os.makedirs(hparams.dataset_dir, exist_ok=True)
    dataset_path = os.path.join(hparams.dataset_dir, filename)
    if hparams.regenerate_val_dataset or not os.path.exists(dataset_path):
        print("Generating dataset...")
        emitter.generate_validation_dataset(
            dataset_path,
            hparams,
            hparams.capt_sz_px,
            offset_px,
            mol_density,
            mol_density,
            photon_sampler_params,
            num_frames=val_n_frames,
        )
    else:
        print(f"Reusing the validation set: {dataset_path} \n")

    train_dataset = emitter.RandomEmitterDataset(
        hparams,
        hparams.capt_sz_px,
        offset_px,
        mol_density,
        mol_density,
        photon_sampler_params,
        num_frames=train_n_frames,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,  # the dataset itself is random.
        batch_size=hparams.batch_sz,
        num_workers=hparams.num_workers,
    )

    validation_dataset = emitter.EmitterDatasetFromCSV(
        dataset_path, hparams, hparams.capt_sz_px
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        shuffle=False,
        batch_size=hparams.batch_sz * 2,
        num_workers=hparams.num_workers,
    )

    # UNet takes the image size of n * 16.
    # The dataset class automatically changes the depth range to be a multiple of 16.
    # So, overwriting the depth_range_nm with the adjusted axial_sz_px.
    hparams.init_depth_range_nm = hparams.depth_range_nm
    hparams.depth_range_nm = train_dataset.depth_range_nm

    return train_loader, validation_loader, dataset_path


def main(hparams: Namespace):
    """Run the training."""
    seed_everything(123)

    train_dataloader, val_dataloader, val_dataset_path = prepare_dataset(
        hparams=hparams
    )

    if hparams.optimize_optics:
        strategy = "end2end_" + hparams.init_phase
    else:
        strategy = "fix_" + os.path.basename(hparams.init_phase).split(".")[0]
    version = (
        f"{strategy}_{hparams.depth_range_nm * 1e-3:.1e}um_{hparams.num_shots}shots"
        f"_cnnlr{hparams.cnn_lr:.1e}_opticslr{hparams.optics_lr:.1e}"
        f"_bs{hparams.batch_sz}_bg{hparams.bg_min}-{hparams.bg_max}"
        f"_reg{hparams.reg:.1e}_axsamp{hparams.axial_sampling_nm}nm_"
        f"midx{hparams.medium_index}_"
        f"decayg{hparams.decaygaussian}_beads{hparams.with_beads}"
        f"_psfjitter{hparams.psf_jitter}"
        f"focusshift{hparams.focus_offset_nm}nm"
    )
    if hparams.note is not None:
        version += "_" + hparams.note

    logger = TensorBoardLogger(
        hparams.default_root_dir, name=hparams.experiment_name, version=version
    )

    checkpoint_callback = ModelCheckpoint(
        verbose=True,
        monitor="val_loss",
        dirpath=logger.log_dir,
        filename="{epoch}-{val_loss:.4f}",
        save_top_k=1,
        period=1,
        mode="min",
    )

    model = Localizer(hparams, log_dir=logger.log_dir)

    trainer = Trainer.from_argparse_args(
        hparams,
        logger=logger,
        callbacks=[checkpoint_callback],
        benchmark=True,
        gradient_clip_val=1,
    )
    trainer.fit(
        model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader
    )


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)

    parser.add_argument("--experiment_name", type=str, default="DeepPSF")
    parser.add_argument("--dataset_dir", type=str, default="data/dataset")
    parser.add_argument(
        "--uniform_low", type=float, default=5000
    )  # # of photoelectrons
    parser.add_argument(
        "--uniform_high", type=float, default=80000
    )  # # of photoelectrons
    parser.add_argument(
        "--mol_density", type=float, default=0.3
    )  # in [# emitters / um^-2]

    parser.add_argument(
        "--regenerate_val_dataset", dest="regenerate_val_dataset", action="store_true"
    )
    parser.set_defaults(regenerate_val_dataset=False)

    parser = Trainer.add_argparse_args(parser)
    parser = Localizer.add_model_specific_args(parser)

    parser.set_defaults(
        gpus=0,
        resume_from_checkpoint=None,
        pretrain_ckpt_path=None,
        default_root_dir="data/logs",
        max_epochs=10000,
        check_val_every_n_epoch=1,
        fast_dev_run=False,
    )

    args = parser.parse_args()

    main(args)
