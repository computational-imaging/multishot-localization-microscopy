# Copyright (c) 2021, The Board of Trustees of the Leland Stanford Junior University

"""Dataset class for generating random emitters."""

from __future__ import annotations

import random
from argparse import Namespace
from dataclasses import dataclass
from typing import Union

import torch.utils.data
import pandas as pd
import torch.distributions
import torch.nn.functional as F
from tqdm import tqdm

from module.microscope import OpticalRecipe
from util.helper import adjust_axial_sz_px


@dataclass(frozen=True)
class UniformParams:
    """A dataclass defining the low and high params for uniform distribution."""

    low: float
    high: float


class RandomEmitterDataset(torch.utils.data.Dataset):
    """A dataset class which generages a random number of emitters at random locations."""

    def __init__(
        self,
        recipe: Union[OpticalRecipe, Namespace],
        capt_sz_px: int,
        offset_px: int,
        emitter_density_min: float,
        emitter_density_max: float,
        photon_sampler_params: UniformParams,
        num_frames: int,
    ):
        """__init__ for RandomEmitterDataset.

        Args:
            recipe: A recipe defining the lateral upsampling factor and the axial pixel
                size.
            capt_sz_px: Lateral resolution of superresolution volume.
            offset_px: A number of pixels at the edges where the emitters are not
                generated.
            emitter_density_min: Minimum of emitter density in [# emitters / um^-2].
            emitter_density_max: Maximum of emitter density in [# emitters / um^-2].
            photon_sampler_params: Parameters defining the uniform distribution of
                photon numbers.
            num_frames: A number of frames.
        """
        super().__init__()
        assert capt_sz_px % 32 == 0, "capt_sz_px has to be multiple of 32."
        sampling_nm = recipe.camera_pixel_sz_nm / recipe.upsampling_factor
        axial_sampling_nm = recipe.axial_sampling_nm
        spcamera_sz_px = capt_sz_px * recipe.upsampling_factor

        target_axial_sz_px = int(recipe.depth_range_nm / recipe.axial_sampling_nm)
        axial_sz_px = adjust_axial_sz_px(target_axial_sz_px, base_multiplier=2 ** 4)

        self.sp_shape = (axial_sz_px, spcamera_sz_px, spcamera_sz_px)
        self.offset_px = offset_px
        self.ax_offset = axial_sz_px - target_axial_sz_px
        self.mol_prob_min = (sampling_nm * sampling_nm * axial_sampling_nm * 1e-9) * (
            emitter_density_min / (recipe.depth_range_nm * 1e-3)
        )  # in [# of emitters / voxel]
        self.mol_prob_max = (sampling_nm * sampling_nm * axial_sampling_nm * 1e-9) * (
            emitter_density_max / (recipe.depth_range_nm * 1e-3)
        )  # in [# of emitters / voxel]
        self.num_frames = num_frames
        self.depth_range_nm = axial_sz_px * recipe.axial_sampling_nm

        self.photon_sampler = torch.distributions.Uniform(
            photon_sampler_params.low, photon_sampler_params.high, validate_args=True
        )
        print(
            f"Photon sampler is set with \n"
            f"Uniform: (high) {self.photon_sampler.high:.1f}, (low) {self.photon_sampler.low:.1f}"
        )
        assert self.mol_prob_max < 1.0, "Molecule density is too high."

    def __len__(self):
        """Retrun a number of frames.

        This is a meaningless parameter. It is used for defining the training epoch
        for tensorboard visualization purpose.
        """
        return self.num_frames

    def __getitem__(self, idx):
        """Return a single superresolution volume with random emitters."""
        mol_prob = (
            random.random() * (self.mol_prob_max - self.mol_prob_min)
            + self.mol_prob_min
        )
        mol_prob_vol = torch.empty(self.sp_shape).fill_(mol_prob)
        superresimg = torch.bernoulli(mol_prob_vol)

        if self.offset_px != 0:
            superresimg[:, : self.offset_px, :] = 0
            superresimg[:, -self.offset_px :, :] = 0
            superresimg[:, :, : self.offset_px] = 0
            superresimg[:, :, -self.offset_px :] = 0
        if self.ax_offset != 0:
            superresimg[-self.ax_offset :, :, :] = 0

        xyz_idx = superresimg.nonzero(as_tuple=False)
        num_mols = xyz_idx.shape[0]
        if num_mols > 0:
            w = F.relu(self.photon_sampler.sample((num_mols,)))
            superresimg[superresimg == 1.0] = w
        sample = {"superresimg": superresimg, "frame": idx}
        return sample


class EmitterDatasetFromCSV(torch.utils.data.Dataset):
    """A dataset class which reads a CSV file."""

    def __init__(
        self, csv_path: str, recipe: Union[OpticalRecipe, Namespace], capt_sz_px: int
    ):
        """__init__ for EmitterDatasetFromCSV."""
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.n_frames = self.df.frame.max() + 1
        spcamera_sz_px = capt_sz_px * recipe.upsampling_factor

        target_axial_sz_px = int(recipe.depth_range_nm / recipe.axial_sampling_nm)
        axial_sz_px = adjust_axial_sz_px(target_axial_sz_px, base_multiplier=2 ** 4)

        self.depth_range_nm = axial_sz_px * recipe.axial_sampling_nm
        self.sp_shape = (axial_sz_px, spcamera_sz_px, spcamera_sz_px)

    def __len__(self):
        """Return the number of frames in the csv file."""
        return self.n_frames

    def __getitem__(self, idx):
        """Return a superresolution volume with a given index."""
        stars = self.df[self.df.frame == idx]
        image = torch.zeros(self.sp_shape)
        if stars.shape[0] > 0:
            x = torch.from_numpy(stars.x.values).long()
            y = torch.from_numpy(stars.y.values).long()
            z = torch.from_numpy(stars.z.values).long()
            w = torch.from_numpy(stars.w.values).float()
            image[z, y, x] += w
        sample = {"superresimg": image, "frame": idx}
        return sample


def generate_validation_dataset(
    save_csvpath: str,
    recipe: Union[OpticalRecipe, Namespace],
    capt_sz_px: int,
    offset_px: int,
    emitter_density_min: float,
    emitter_density_max: float,
    photon_sampler_params: UniformParams,
    num_frames: int,
):
    """Generage a csv file containing a random number of 3D coordinates and intensity.

    Each emitter lies on the super-resolution grid.

    Args:
        save_csvpath: A path where the output csv file is saved.
        recipe: A recipe defining the lateral upsampling factor and the axial pixel
            size.
        capt_sz_px: Lateral resolution of superresolution volume.
        offset_px: A number of pixels at the edges where the emitters are not generated.
        emitter_density_min: Minimum of emitter density in [# emitters / um^-2].
        emitter_density_max: Maximum of emitter density in [# emitters / um^-2].
        num_frames: A number of frames.
    """
    assert capt_sz_px % 32 == 0, "capt_sz_px has to be multiple of 32."
    dataset = RandomEmitterDataset(
        recipe,
        capt_sz_px,
        offset_px,
        emitter_density_min,
        emitter_density_max,
        photon_sampler_params,
        num_frames,
    )
    df_list = []
    for i in tqdm(range(len(dataset))):
        image = dataset[i]["superresimg"]
        nnz_idx = image.nonzero(as_tuple=True)
        n_stars = nnz_idx[0].shape[0]
        if n_stars > 0:
            z = nnz_idx[0]
            y = nnz_idx[1]
            x = nnz_idx[2]
            w = image[nnz_idx]
            frame = torch.empty(n_stars).fill_(i).long()
            df_list.append(
                pd.DataFrame({"frame": frame, "x": x, "y": y, "z": z, "w": w})
            )
    df = pd.concat(df_list)
    df.to_csv(save_csvpath)
