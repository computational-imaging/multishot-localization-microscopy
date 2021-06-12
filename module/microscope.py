# Copyright (c) 2021, The Board of Trustees of the Leland Stanford Junior University

"""Implementation of an optical microscope based on Fourier optics."""

from __future__ import annotations

import math
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import complex
from util.fft import crop_psf, fftshift, ifftshift
from util.helper import resize2d
from util.zernike import zernike_array

from module.bead import Bead
from module.gaussian import GaussianBlur2D
from module.noise import Noise


@dataclass(frozen=True)
class OpticalRecipe:
    """A dataclass defining optics parameters for simulated optical microscope."""

    depth_range_nm: int
    camera_pixel_sz_nm: float
    mask_sz_px: int
    axial_sampling_nm: float
    numerical_aperture: float
    wavelength_nm: float
    medium_index: float
    designed_index: float
    num_shots: int
    upsampling_factor: int
    mask_padding_factor: int
    # The objective lens's focal plane from a coverslip
    # Zero means that the obejective lens is focused at the coverslip.
    # It has to be nonnegative.
    focus_offset_nm: float = 0.0
    nominal_focal_depth_nm: float = 0.0


@dataclass(frozen=True)
class MicroscopeOutputs:
    """A dataclass defining the output of forward step of microscope simulator."""

    backproj_vol: torch.Tensor
    noisy_img: torch.Tensor
    noiseless_img: torch.Tensor
    psfimg: torch.Tensor


class BaseMicroscope(nn.Module):
    """Base microscope class."""

    def __init__(self, recipe: Union[OpticalRecipe, Namespace]):
        """__init__ for BaseMicroscope class."""
        super().__init__()
        assert recipe.mask_padding_factor >= 2
        assert recipe.nominal_focal_depth_nm >= 0
        assert recipe.focus_offset_nm >= 0

        # Enforce the total internal reflection
        if recipe.numerical_aperture > recipe.medium_index:
            self.numerical_aperture = recipe.medium_index
        else:
            self.numerical_aperture = recipe.numerical_aperture

        # Set basic parameters
        self.recipe = recipe
        self.camera_pixel_sz_nm = recipe.camera_pixel_sz_nm
        self.medium_index = recipe.medium_index
        self.wavelength_nm = recipe.wavelength_nm
        self.coherent_freqcut = self.numerical_aperture / recipe.wavelength_nm
        self.incoherent_freqcut = 2 * self.numerical_aperture / recipe.wavelength_nm
        self.mask_sz_px = recipe.mask_sz_px
        self.axial_sz_px = int(recipe.depth_range_nm / recipe.axial_sampling_nm)
        self.upsampling_factor = recipe.upsampling_factor
        self.axial_sampling_nm = recipe.axial_sampling_nm

        # Find the sampling rate in nm for PSF simulation and superresolution image
        self.psf_sampling_nm = (
            1 / (self.coherent_freqcut * recipe.mask_padding_factor) / 2
        )
        self.sp_pixel_sz_nm = recipe.camera_pixel_sz_nm / recipe.upsampling_factor
        self.psf_upsampling_factor = recipe.camera_pixel_sz_nm / self.psf_sampling_nm

        # interpolation factor from psf simulation to super resolution image
        self.interpolation_factor = self.psf_sampling_nm / self.sp_pixel_sz_nm

        # PSF simulation size in pixel
        self.sim_sz_px = recipe.mask_sz_px * recipe.mask_padding_factor
        self.num_shots = recipe.num_shots
        p = self.sim_sz_px - self.mask_sz_px
        self.psf_pad = (0, 0, 0, p, 0, p)

        # Find depth sampling in nm
        # Previous depth range (focused at the middle of the target depth range
        depths = (
            torch.arange(self.axial_sz_px).float() * recipe.axial_sampling_nm
            - recipe.focus_offset_nm
        )

        # Initialize optical frequency coordinate meshgrid
        freq_grid = torch.from_numpy(
            np.fft.fftshift(
                np.fft.fftfreq(
                    recipe.mask_sz_px, self.psf_sampling_nm * recipe.mask_padding_factor
                )
            )
        ).float()
        kx = freq_grid.repeat(recipe.mask_sz_px, 1)
        ky = freq_grid.view(-1, 1).repeat(1, recipe.mask_sz_px)
        krsq = kx ** 2 + ky ** 2
        kr = torch.sqrt(krsq)
        binary_mask = (kr <= self.coherent_freqcut).float()
        byte_mask = kr <= self.coherent_freqcut

        # Set defocus phase factor
        # The objective plane is assumed to focus at the middle plane of the target
        # depth range.
        def propagation_factor(n):
            d_max = 2 * math.pi * n / recipe.wavelength_nm
            d_min = (
                2
                * math.pi
                * math.sqrt(
                    (n / recipe.wavelength_nm) ** 2 - self.coherent_freqcut ** 2
                )
            )
            defocus = (
                2
                * math.pi
                * torch.sqrt((n / recipe.wavelength_nm) ** 2 - binary_mask * krsq)
            )
            # Removing piston from a defocus factor for visualization purpose
            # After this operation, the absolute values of max and min are the same.
            return binary_mask * (defocus - ((d_max - d_min) / 2 + d_min))

        defocus_factor = propagation_factor(self.medium_index)

        # If the medium index is the same as the designed refractive index, the factor
        # due to the refractiv index mismatch is zero.
        mismatch_factor = propagation_factor(recipe.medium_index) - propagation_factor(
            recipe.designed_index
        )

        # Set apodization factor
        # Scale the amplitute to set the PSF intensity's sum to be em_photons
        # (excluding apodization)
        omega = self.numerical_aperture / recipe.medium_index
        alpha = torch.asin(
            torch.min(
                omega * (kr / self.coherent_freqcut) * binary_mask, torch.tensor(1.0)
            )
        )  # mask_sz_px x mask_sz_px
        apodization = (
            1 / torch.sqrt(torch.max(torch.cos(alpha), torch.tensor(0.0))) * binary_mask
        )  # mask_sz_px x mask_sz_px
        scaling_factor = 1 / (self.sim_sz_px * torch.sqrt(binary_mask.sum()))
        apodization = scaling_factor * apodization

        self.noise = Noise(
            recipe.gain,
            recipe.readnoise_std,
            recipe.bg_min / recipe.num_shots,
            recipe.bg_max / recipe.num_shots,
            recipe.num_shots,
        )

        if recipe.with_beads:
            self.bead = Bead(
                voxel_sz_nm=[self.sp_pixel_sz_nm, self.sp_pixel_sz_nm],
                sphere_diameter_nm=recipe.sphere_diameter_nm,
            )

        if recipe.with_gaussian:
            sigma = [s for s in self.recipe.gaussian_sigma]
            self.gaussian_blur = GaussianBlur2D(sigma=sigma, requires_grad=False)

        self.register_buffer("mismatch_factor", mismatch_factor, persistent=False)
        self.register_buffer("depths", depths, persistent=False)
        self.register_buffer("kx", kx, persistent=False)
        self.register_buffer("ky", ky, persistent=False)
        self.register_buffer("binary_mask", binary_mask, persistent=False)
        self.register_buffer("byte_mask", byte_mask, persistent=False)
        self.register_buffer("defocus_factor", defocus_factor, persistent=False)
        self.register_buffer("apodization", apodization, persistent=False)

    def defocus(self, psf_jitter=torch.tensor(False)):
        # Find depth sampling in nm
        # Previous depth range (focused at the middle of the target depth range
        device = self.defocus_factor.device
        depths = (
            torch.arange(self.axial_sz_px, device=device).float()
            * self.recipe.axial_sampling_nm
            - self.recipe.focus_offset_nm
        )
        # Experimental dataset may have a fluorescence object outside of the designed depth range.
        # To robustify the model, we randomly change the depth at the first and final plane.
        if psf_jitter:
            # depths[0] -= torch.rand(1, device=device)[0] * 1000.0
            depths[-1] += torch.rand(1, device=device)[0] * 3000.0
        defocus = self.defocus_factor * depths.reshape(-1, 1, 1)
        defocus = defocus + self.mismatch_factor * self.recipe.focus_offset_nm

        return defocus

    def fisher(self, depths_idx, em_photons, bg_photons):
        # compute phase function (pytorch doesn't have complex number)
        pupil_real, pupil_imag = self.pupil(depths_idx)
        pupil_real = pupil_real * math.sqrt(em_photons / self.num_shots)
        pupil_imag = pupil_imag * math.sqrt(em_photons / self.num_shots)
        pupil = torch.stack(
            (pupil_real, pupil_imag), dim=4
        )  # num_shots x axial_sz_px x mask_sz_px x mask_sz_px x 2

        # zero-pad the pupil function to simulation size
        p = self.sim_sz_px - self.mask_sz_px
        pad = (0, 0, 0, p, 0, p)
        pupil = F.pad(
            pupil, pad, mode="constant", value=0
        )  # num_shots x axial_sz_px x sim_sz_px x sim_sz_px x 2

        # compute APSF
        apsf = torch.fft(pupil, 2)

        # compute PSF with background
        psf_im = (
            apsf[..., 0] ** 2
            + apsf[..., 1] ** 2
            + bg_photons / self.num_shots / self.psf_upsampling_factor ** 2
        )

        # (-1im * 2pi) * kx * pupil
        pupil_dx = (
            2
            * math.pi
            * torch.stack((pupil_imag * self.kx, -pupil_real * self.kx), dim=4)
        )
        # (-1im * 2pi) * ky * pupil
        pupil_dy = (
            2
            * math.pi
            * torch.stack((pupil_imag * self.ky, -pupil_real * self.ky), dim=4)
        )
        # -1im * defocus_factor * pupil
        pupil_dz = torch.stack(
            (pupil_imag * self.defocus_factor, -pupil_real * self.defocus_factor), dim=4
        )

        # zero-pad the pupil derivatives to simulation size
        pupil_dx = F.pad(pupil_dx, pad, mode="constant", value=0)
        pupil_dy = F.pad(pupil_dy, pad, mode="constant", value=0)
        pupil_dz = F.pad(pupil_dz, pad, mode="constant", value=0)

        # compute the APSF devivatives
        apsf_dx = torch.fft(pupil_dx, 2)  # S x D x H x W x 2
        apsf_dy = torch.fft(pupil_dy, 2)  # S x D x H x W x 2
        apsf_dz = torch.fft(pupil_dz, 2)  # S x D x H x W x 2

        # 2 * Real( conj(apsf) * apsf_dx )
        psf_dx = 2 * (
            apsf[..., 0] * apsf_dx[..., 0] + apsf[..., 1] * apsf_dx[..., 1]
        )  # S x D x H x W
        psf_dy = 2 * (
            apsf[..., 0] * apsf_dy[..., 0] + apsf[..., 1] * apsf_dy[..., 1]
        )  # S x D x H x W
        psf_dz = 2 * (
            apsf[..., 0] * apsf_dz[..., 0] + apsf[..., 1] * apsf_dz[..., 1]
        )  # S x D x H x W

        f_xx = torch.sum(psf_dx ** 2 / psf_im, dim=(0, 2, 3))
        f_yy = torch.sum(psf_dy ** 2 / psf_im, dim=(0, 2, 3))
        f_zz = torch.sum(psf_dz ** 2 / psf_im, dim=(0, 2, 3))

        f_xy = torch.sum(psf_dx * psf_dy / psf_im, dim=(0, 2, 3))
        f_yz = torch.sum(psf_dy * psf_dz / psf_im, dim=(0, 2, 3))
        f_zx = torch.sum(psf_dz * psf_dx / psf_im, dim=(0, 2, 3))

        f_x = torch.stack((f_xx, f_xy, f_zx), dim=1)
        f_y = torch.stack((f_xy, f_yy, f_yz), dim=1)
        f_z = torch.stack((f_zx, f_yz, f_zz), dim=1)

        fisher = torch.stack((f_x, f_y, f_z), dim=2)  # axial_sz_px x 3 x 3

        return fisher

    def crb(self, depths_idx, em_photons, bg_photons):
        """Returns Cramer-Rao bound for single-emitter localization."""
        D = len(depths_idx)
        fisher_mat = self.fisher(depths_idx, em_photons, bg_photons)
        inv_fisher = torch.stack([torch.inverse(fisher_mat[i, :, :]) for i in range(D)])
        crb_x = inv_fisher[:, 0, 0]
        crb_y = inv_fisher[:, 1, 1]
        crb_z = inv_fisher[:, 2, 2]
        return crb_x, crb_y, crb_z

    def sqcrb(self, depths_idx, em_photons, bg_photons):
        """Returns the square root of Cramer-Rao bound for single-emitter localization."""
        crb_x, crb_y, crb_z = self.crb(depths_idx, em_photons, bg_photons)
        return torch.sqrt(crb_x), torch.sqrt(crb_y), torch.sqrt(crb_z)

    def a_optimality(self, depths_idx, em_photons, bg_photons):
        D = len(depths_idx)
        fisher_mat = self.fisher(depths_idx, em_photons, bg_photons)
        a_opt = torch.stack(
            [torch.trace(torch.inverse(fisher_mat[i, :, :])) for i in range(D)]
        ).mean()
        return a_opt

    def adjust_psf_sz(
        self, psfimg: torch.Tensor, width: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        N = psfimg.numel()

        orig_psfenergy = torch.sum(psfimg)

        psfimg = crop_psf(psfimg, math.ceil(width / self.interpolation_factor))

        cropped_psfenergy = (
            (orig_psfenergy - torch.sum(psfimg))
            / (N - psfimg.numel())
            * self.interpolation_factor ** 2
        )

        psfimg = fftshift(psfimg, dims=(-1, -2))

        # bilinear is too bad for this upsampling.
        psfimg = F.interpolate(
            psfimg,
            scale_factor=self.interpolation_factor,
            mode="bicubic",
            align_corners=False,
            recompute_scale_factor=False,
        )
        psfimg = ifftshift(psfimg, dims=(-1, -2))

        # Scale the psf
        scaler = psfimg.sum(dim=(-1, -2), keepdim=True).max(dim=-3, keepdim=True)[
            0
        ]  # S x 1 x 1 x 1
        psfimg = psfimg / scaler

        psfimg = crop_psf(psfimg, width)

        # Ensure the nonnegativity of PSF
        # Bicubic interpolation sometimes introduces negative values.
        psfimg = F.relu(psfimg)

        return psfimg, cropped_psfenergy

    def remove_defocus_from_phase(self, phase) -> Tuple[torch.Tensor, torch.Tensor]:
        defocus_factor = self.defocus_factor[None, None, ...]
        a = torch.sum(phase * defocus_factor, dim=(-1, -2), keepdims=True) / torch.sum(
            defocus_factor ** 2, dim=(-1, -2), keepdims=True
        )
        defocus = a * defocus_factor
        return phase - defocus, defocus

    def phase_wo_defocus(self) -> torch.Tensor:
        phase = self.phase()
        return self.remove_defocus_from_phase(phase)[0]

    def pupil(
        self, depths_idx: int, psf_jitter=torch.tensor(False)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        defocus = self.defocus(psf_jitter)[depths_idx].unsqueeze(0)
        phase = self.phase()  # num_shots x axial_sz_px (or 1) x mask_sz_px x mask_sz_px
        if phase.shape[1] > 1:
            phase = phase[:, depths_idx]
        phase_z = phase + defocus  # num_shots x axial_sz_px x mask_sz_px x mask_sz_px
        # compute the real and imaginary part of the pupil function
        pupil_real = torch.cos(phase_z) * self.apodization
        pupil_imag = torch.sin(phase_z) * self.apodization
        return pupil_real, pupil_imag

    def psf_img(self, depths_idx: int, img_sz_px: int) -> torch.Tensor:
        psf_sim = self.psf(depths_idx)
        psf_sim = fftshift(self.adjust_psf_sz(psf_sim, img_sz_px)[0], axes=(-1, -2))
        psf_sim = resize2d(psf_sim, self.upsampling_factor)
        return psf_sim

    def psf(self, depths_idx, psf_jitter=torch.tensor(False)) -> torch.Tensor:
        apsf = self.apsf(depths_idx, psf_jitter=psf_jitter)
        return complex.abs2(apsf)

    def apsf(self, depths_idx, psf_jitter=torch.tensor(False)) -> torch.Tensor:
        # compute phase function (pytorch doesn't have complex number)
        pupil_real, pupil_imag = self.pupil(depths_idx, psf_jitter)
        pupil = torch.stack(
            (pupil_real, pupil_imag), dim=-1
        )  # num_shots x axial_sz_px x mask_sz_px x mask_sz_px x 2

        # zero-pad the pupil function to simulation size
        pupil = F.pad(
            pupil, self.psf_pad, mode="constant", value=0
        )  # num_shots x axial_sz_px x sim_sz_px x sim_sz_px x 2

        # compute APSF
        return torch.fft(pupil, 2)

    def psf_at_spcamera(
        self, n: int, psf_jitter=torch.tensor(False)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # PSF is normalized by adjust_psf_sz.
        psfimg = self.psf(torch.arange(self.axial_sz_px), psf_jitter=psf_jitter)
        psfimg, cropped_psfenergy = self.adjust_psf_sz(psfimg, n)
        if self.recipe.with_gaussian:
            psfimg = self.gaussian_blur(psfimg)
        return psfimg, cropped_psfenergy

    def capture_with_rfft_psf(
        self, img: torch.Tensor, rfft_psfimg: torch.Tensor
    ) -> torch.Tensor:
        rfft_psfimg = rfft_psfimg.unsqueeze(0)  # 1 x S x D x H x W

        # Convolve PSF with super resolution image
        # Split the emission photons to multi shots before convolution
        img = img.unsqueeze(1) / self.num_shots  # B x 1 x D x H x W
        rfft_img = torch.rfft(img, 2)
        rfft_convimg = complex.multiply(rfft_img, rfft_psfimg)

        conv_img3d = torch.irfft(
            rfft_convimg, 2, onesided=True, signal_sizes=img.shape[-2:]
        )

        # Sum along the axial direction
        capt_spimg = torch.sum(conv_img3d, dim=-3)  # B x S x H x W

        if self.upsampling_factor > 1:
            # Sum over a pixel
            noiseless_img = resize2d(capt_spimg, self.upsampling_factor)
        else:
            noiseless_img = capt_spimg

        noiseless_img = F.relu(noiseless_img)

        return noiseless_img

    def backprojection_with_rfft_psf(self, img, rfft_psfimg) -> torch.Tensor:
        img = F.interpolate(img, scale_factor=self.upsampling_factor, mode="nearest")
        img_shape = img.shape

        # pad the image to reduce the edge artifact
        img = F.pad(
            img,
            (
                0,
                img_shape[-1] - self.upsampling_factor,
                0,
                img_shape[-2] - self.upsampling_factor,
            ),
            mode="reflect",
        )
        img = img.unsqueeze(-3)

        # Backpropagate
        rfft_img = torch.rfft(img, 2)

        rfft_bpimg = complex.multiply_conj(rfft_img, rfft_psfimg)
        backprop_img = torch.irfft(
            rfft_bpimg, 2, onesided=True, signal_sizes=img.shape[-2:]
        )

        # crop the edge artifacts
        backprop_img = backprop_img[..., : img_shape[-2], : img_shape[-1]]

        return backprop_img

    def forward(
        self, img: torch.Tensor, psf_jitter=torch.tensor(False)
    ) -> Tuple[MicroscopeOutputs, torch.Tensor]:
        """Compute forward step.

        Input img has to be square.
        """
        img_shape = img.shape
        img = F.pad(
            img,
            (
                0,
                img_shape[-1] - self.upsampling_factor,
                0,
                img_shape[-2] - self.upsampling_factor,
            ),
            mode="reflect",
        )

        if self.recipe.with_beads:
            img = self.bead.conv(img)

        # psf simulator still doesn't support rectangular input.
        psfimg, cropped_psfenergy = self.psf_at_spcamera(
            img.shape[-1], psf_jitter=psf_jitter
        )

        rfft_psfimg = torch.rfft(psfimg, 2)
        noiseless_img = self.capture_with_rfft_psf(img, rfft_psfimg)

        # crop the edge artifacts
        noiseless_img = noiseless_img[
            ...,
            : img_shape[-2] // self.upsampling_factor,
            : img_shape[-1] // self.upsampling_factor,
        ]

        noisy_img = self.noise(noiseless_img)

        if psf_jitter:
            psfimg, _ = self.psf_at_spcamera(
                img.shape[-1], psf_jitter=torch.tensor(False)
            )
            rfft_psfimg = torch.rfft(psfimg, 2)

        backproj_vol = self.backprojection_with_rfft_psf(noisy_img, rfft_psfimg)

        outputs = MicroscopeOutputs(
            backproj_vol=backproj_vol,
            psfimg=psfimg,
            noisy_img=noisy_img,
            noiseless_img=noiseless_img,
        )

        return outputs, cropped_psfenergy

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        """Add hyperparameters of BaseMicroscope class."""
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--depth_range_nm", type=float, default=5000)
        parser.add_argument("--camera_pixel_sz_nm", type=float, default=108.0)
        parser.add_argument("--mask_sz_px", type=int, default=256)
        parser.add_argument("--axial_sampling_nm", type=float, default=108.0)
        parser.add_argument("--numerical_aperture", type=float, default=1.25)
        parser.add_argument("--wavelength_nm", type=float, default=660)
        parser.add_argument("--medium_index", type=float, default=1.518)
        parser.add_argument("--designed_index", type=float, default=1.518)
        parser.add_argument("--num_shots", type=int, default=2)
        parser.add_argument("--upsampling_factor", type=int, default=4)
        parser.add_argument("--mask_padding_factor", type=int, default=3)
        parser.add_argument("--focus_offset_nm", type=float, default=0.0)
        parser.add_argument("--nominal_focal_depth_nm", type=float, default=0.0)

        parser.add_argument("--with_beads", dest="with_beads", action="store_true")
        parser.add_argument("--without_beads", dest="with_beads", action="store_false")
        parser.set_defaults(with_beads=True)

        parser.add_argument(
            "--gaussian_sigma",
            type=lambda s: [float(item) for item in s.split(",")],
            default=[1.0, 1.0],
        )
        parser.add_argument(
            "--with_gaussian", dest="with_gaussian", action="store_true"
        )
        parser.add_argument(
            "--without_gaussian", dest="with_gaussian", action="store_false"
        )
        parser.set_defaults(with_gaussian=False)

        # Camera noise parameters
        parser.add_argument("--bg_min", type=float, default=1000)
        parser.add_argument("--bg_max", type=float, default=1000)
        parser.add_argument(
            "--gain",
            type=lambda s: [float(item) for item in s.split(",")],
            default=[1.0, 1.0],
        )
        parser.add_argument("--readnoise_std", type=float, default=2.3)
        return parser


class Microscope(BaseMicroscope):
    def __init__(
        self,
        recipe: OpticalRecipe,
        init_phase: Union[str, torch.Tensor, np.ndarray],
        init_abbe_coeff: Optional[np.ndarray] = None,
        depth_independent_aberration: Optional[np.ndarray] = None,
        depth_dependent_aberration: Optional[np.ndarray] = None,
        defocus_aberration_nm: Optional[float] = None,
        requires_grad: bool = False,
        requires_aberration_grad: bool = False,
    ):
        super().__init__(recipe)

        self.init_depth_range_nm = recipe.init_depth_range_nm

        init_depth_independent_aberration = (
            torch.from_numpy(depth_independent_aberration)
            .float()
            .reshape(-1, 1, recipe.mask_sz_px, recipe.mask_sz_px)
            if depth_independent_aberration is not None
            else torch.zeros((1, 1, recipe.mask_sz_px, recipe.mask_sz_px))
        )
        init_depth_dependent_aberration = (
            torch.from_numpy(depth_dependent_aberration)
            .float()
            .reshape(-1, 1, recipe.mask_sz_px, recipe.mask_sz_px)
            if depth_dependent_aberration is not None
            else torch.zeros((1, 1, recipe.mask_sz_px, recipe.mask_sz_px))
        )

        # Initialize phase
        num_zernike = 21
        max_zernike = 56
        self.phi = nn.Parameter(
            torch.zeros((recipe.num_shots, recipe.mask_sz_px, recipe.mask_sz_px)),
            requires_grad=requires_grad,
        )
        if requires_aberration_grad:
            init_abbe_coeff = (
                torch.from_numpy(init_abbe_coeff)
                .float()
                .reshape(recipe.num_shots, max_zernike * 2)
                if init_abbe_coeff is not None
                else torch.zeros(recipe.num_shots, max_zernike * 2)
            )
            self.abbe_coeffs = nn.Parameter(
                init_abbe_coeff, requires_grad=requires_aberration_grad
            )
        else:
            self.abbe_coeffs = 0.0
        self.set_phase(init_phase)

        self.requires_aberration_grad = requires_aberration_grad
        if requires_aberration_grad:
            x = self.kx / self.coherent_freqcut
            y = self.ky / self.coherent_freqcut
            r = torch.sqrt(x ** 2 + y ** 2)
            theta = torch.atan2(y, x)
            zernike = (
                self.binary_mask
                * torch.stack(
                    [zernike_array(j, r, theta) for j in range(1, max_zernike + 1)]
                )
            )[None, ...]
            self.register_buffer("zernike", zernike, persistent=False)
            self.num_zernike = num_zernike
            self.max_zernike = max_zernike

        self.defocus_aberration_nm = defocus_aberration_nm
        self.register_buffer(
            "init_depth_independent_aberration",
            init_depth_independent_aberration,
            persistent=False,
        )
        self.register_buffer(
            "init_depth_dependent_aberration",
            init_depth_dependent_aberration,
            persistent=False,
        )

    def set_phase(self, init_phase: Union[str, torch.Tensor, np.ndarray]):
        if isinstance(init_phase, str):
            if init_phase == "wfm":
                self.phi.data.fill_(0)
            elif init_phase == "multiplane":
                n = self.phi.data.shape[0]
                step = (self.depths.max() - self.depths.min()) / n
                print(
                    f"Initializing multiplane ({n} shots, depth: {self.init_depth_range_nm}[nm])..."
                )
                for i in range(n):
                    d = (
                        step / 2 + i * step + self.depths.min()
                    )  # - self.recipe.focus_offset_nm
                    self.phi.data[i] = -d * self.defocus_factor
                    print(f"{i + 1}-th plane is focusing at {d}[nm].")
            else:
                raise ValueError("This phase initialization is not supported.")
        elif isinstance(init_phase, (torch.Tensor, np.ndarray)):
            if isinstance(init_phase, np.ndarray):
                init_phase = torch.from_numpy(init_phase).float().squeeze().cpu()
            if init_phase.dim() == 2:
                init_phase = init_phase.unsqueeze(0)
            assert init_phase.dim() == 3, "Phase dimension has to be 2 or 3."
            assert init_phase.shape[1] == init_phase.shape[2], "Phase has to be square."
            assert (
                init_phase.shape[1] == self.mask_sz_px
            ), "The phase mask size doesn't match!"
            self.phi.data = init_phase
        else:
            raise ValueError(
                "init_phase has to be string or torch.tensor or numpy.ndarray."
            )
        mask = (self.byte_mask == 0).repeat(self.phi.shape[0], 1, 1).cpu()
        self.phi.data.masked_fill_(mask, float("nan"))

    def depth_independent_aberration(
        self, with_tip_tilt=torch.tensor(True)
    ) -> torch.Tensor:
        aberration = self.init_depth_independent_aberration
        if self.defocus_aberration_nm is not None:
            aberration = aberration + self.defocus_aberration_nm * self.defocus_factor
        if self.requires_aberration_grad:
            abbe_coeffs = self.abbe_coeffs[..., None, None]
            if with_tip_tilt:
                aberration = (
                    aberration
                    + (
                        self.zernike[:, 1 : self.num_zernike]
                        * abbe_coeffs[:, 1 : self.num_zernike]
                    ).sum(dim=1)[None, ...]
                )
            else:
                aberration = (
                    aberration
                    + (
                        self.zernike[:, 3 : self.num_zernike]
                        * abbe_coeffs[:, 3 : self.num_zernike]
                    ).sum(dim=1)[None, ...]
                )
        return aberration

    def depth_dependent_aberration(self) -> torch.Tensor:
        aberration = self.init_depth_dependent_aberration
        if self.requires_aberration_grad:
            aberration = (
                aberration
                + (
                    self.zernike[:, 2 - 1]
                    * self.abbe_coeffs[:, self.max_zernike + 2 - 1]
                )[None, ...]
                + (
                    self.zernike[:, 3 - 1]
                    * self.abbe_coeffs[:, self.max_zernike + 3 - 1]
                )[None, ...]
                + (
                    self.zernike[:, 4 - 1]
                    * self.abbe_coeffs[:, self.max_zernike + 4 - 1]
                )[None, ...]
                + (
                    self.zernike[:, 11 - 1]
                    * self.abbe_coeffs[:, self.max_zernike + 11 - 1]
                )[None, ...]
                + (
                    self.zernike[:, 22 - 1]
                    * self.abbe_coeffs[:, self.max_zernike + 22 - 1]
                )[None, ...]
                + (
                    self.zernike[:, 37 - 1]
                    * self.abbe_coeffs[:, self.max_zernike + 37 - 1]
                )[None, ...]
                + (
                    self.zernike[:, 56 - 1]
                    * self.abbe_coeffs[:, self.max_zernike + 56 - 1]
                )[None, ...]
            )
        return aberration

    def phase(self) -> torch.Tensor:
        phase = self.phi.clone().unsqueeze(1)
        phase = phase + self.aberration()
        phase.masked_fill_(torch.isnan(phase), 0)
        return phase

    def aberration(self):
        return (
            self.depth_dependent_aberration() * self.depths.reshape(1, -1, 1, 1)
            + self.depth_independent_aberration()
        )

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = BaseMicroscope.add_model_specific_args(parent_parser)
        parser.add_argument(
            "--init_phase", choices=["multiplane", "wfm"], default="multiplane"
        )
        parser.add_argument("--init_depth_range_nm", type=float, default=None)
        parser.add_argument("--defocus_aberration_nm", type=float, default=0)
        return parser
