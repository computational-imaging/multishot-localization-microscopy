# Copyright (c) 2021, The Board of Trustees of the Leland Stanford Junior University

"""Main file defining the lightning module for training and evaluation."""

from __future__ import annotations

import os
import shutil
from argparse import ArgumentParser, Namespace
from typing import Callable, List, Mapping, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed
import torch.nn.functional as F
import torch.optim
import torchvision.utils
from tqdm import tqdm

import util.io
from module.bead import Bead
from module.loss import DeepStormLoss
from module.microscope import Microscope, MicroscopeOutputs
from module.unet3d import Unet3d
from util.fft import fftshift
from util.helper import resize2d, scale_image


class Localizer(pl.LightningModule):
    """A main module which performs training and evaluation."""

    def __init__(
        self,
        hparams: Union[Namespace, Mapping],
        log_dir: str = "data/logs",
    ):
        """__init__ for Localizer Lightning Module.

        Args:
            hparams: A set of hyperparameters. See the flags of this Localizer class
                and pl.Lightning.Trainer.
            log_dir: A path to th directory for tensorboard log and other files.
        """
        super().__init__()
        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)

        self.logdir = log_dir
        self.hparams: Namespace = hparams
        self.save_hyperparameters(hparams)

        # Load designed phase
        if os.path.isfile(self.hparams.init_phase):
            phase = util.io.imread(self.hparams.init_phase).reshape(
                (hparams.num_shots, hparams.mask_sz_px, hparams.mask_sz_px)
            )
        else:
            phase = self.hparams.init_phase

        if self.hparams.depth_independent_aberration_path is not None:
            depth_independent_aberration = util.io.imread(
                self.hparams.depth_independent_aberration_path
            )
        else:
            depth_independent_aberration = None
        if self.hparams.depth_dependent_aberration_path is not None:
            depth_dependent_aberration = util.io.imread(
                self.hparams.depth_dependent_aberration_path
            )
        else:
            depth_dependent_aberration = None

        # Set up a Microscope module
        self.microscope = Microscope(
            hparams,
            init_phase=phase,
            depth_independent_aberration=depth_independent_aberration,
            depth_dependent_aberration=depth_dependent_aberration,
            requires_grad=hparams.optimize_optics,
            requires_aberration_grad=False,
        )

        # Set up a CNN.
        self.net = Unet3d(hparams.num_shots, hparams.unet_base_ch, leaky_relu_a=0.01)

        # Set up the loss function
        loss_sigma_nm = hparams.loss_sigma_nm
        self.init_loss_sigma_nm = loss_sigma_nm
        self.loss_sigma_nm = loss_sigma_nm
        self.min_sigma_nm = hparams.loss_min_sigma_nm

        self.output_axial_sz = self.microscope.axial_sz_px

        self.offset_sppx = 16
        self.sp_lateral_sz = hparams.capt_sz_px * self.microscope.upsampling_factor

        # Loss function for training which decays the std of Gaussian function over
        # the course of training.
        self.lossfn = DeepStormLoss(
            self.sp_lateral_sz,
            self.output_axial_sz,
            sigma_xy=loss_sigma_nm / self.microscope.sp_pixel_sz_nm,
            sigma_z=loss_sigma_nm / self.microscope.axial_sampling_nm,
            offset_sppx=self.offset_sppx,
        )

        # Loss function for validation which has the std of Gaussian to be the minimum.
        self.val_lossfn = DeepStormLoss(
            self.sp_lateral_sz,
            self.output_axial_sz,
            sigma_xy=self.min_sigma_nm / self.microscope.sp_pixel_sz_nm,
            sigma_z=self.min_sigma_nm / self.microscope.axial_sampling_nm,
            offset_sppx=self.offset_sppx,
        )

        # Set up some paths for storing the results.
        self.psf_path = os.path.join(log_dir, "psf.tif")
        self.phase_path = os.path.join(log_dir, "phase.tif")

        self.example_input_array = torch.ones(
            (
                hparams.batch_sz,
                self.microscope.axial_sz_px,
                self.sp_lateral_sz,
                self.sp_lateral_sz,
            )
        )

    def _update_lossfn(self, iter: int):
        """Update the loss function.

        Our loss function involves the convolution with a Gaussian function.
        The standard deviation of the Gaussian function is decayed over the course of
        training.


        Args:
            iter: The current number of iterations in training.
        """
        # Decay sigma for loss function
        if self.loss_sigma_nm > self.min_sigma_nm:
            loss_sigma_nm = self.init_loss_sigma_nm * 0.9999 ** iter
            self.loss_sigma_nm = max(self.min_sigma_nm, loss_sigma_nm)
            self.lossfn = DeepStormLoss(
                self.sp_lateral_sz,
                self.output_axial_sz,
                sigma_xy=self.loss_sigma_nm / self.microscope.sp_pixel_sz_nm,
                sigma_z=self.loss_sigma_nm / self.microscope.axial_sampling_nm,
                offset_sppx=16,
            )

    # learning rate warm-up
    def optimizer_step(
        self,
        epoch: int = None,
        batch_idx: int = None,
        optimizer: torch.optim.Optimizer = None,
        optimizer_idx: int = None,
        optimizer_closure: Optional[Callable] = None,
        on_tpu: bool = None,
        using_native_amp: bool = None,
        using_lbfgs: bool = None,
    ) -> None:
        """Optimizer step with warm start."""
        # warm up lr
        if self.trainer.global_step < 4000:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / 4000.0)
            optimizer.param_groups[0]["lr"] = lr_scale * self.hparams.optics_lr
            optimizer.param_groups[1]["lr"] = lr_scale * self.hparams.cnn_lr
        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def configure_optimizers(self):
        """Configure an optimizer.

        The Adam optimizer is set up with a different learning rates for the phase mask
        and CNN parameters.
        """
        params = [
            {"params": self.microscope.parameters(), "lr": self.hparams.optics_lr},
            {"params": self.net.parameters(), "lr": self.hparams.cnn_lr},
        ]
        optimizer = torch.optim.Adam(params)
        return optimizer

    def forward(self, inputs, psf_jitter=torch.tensor(False)):
        """Simulate the captured image and feed it to a CNN."""
        microscope_outputs, cropped_psfenergy = self.microscope(
            inputs, psf_jitter=psf_jitter
        )
        est = self.net(microscope_outputs.backproj_vol)
        return est, microscope_outputs, cropped_psfenergy

    def training_step(self, sample: Mapping, batch_idx: int):
        """Compute a training loss.

        Args:
            sample: A training sample. A superresolution volume which has a random
                number of emitters at random locations and its frame number. See
                dataset.emitter.RandomEmitterDataset.
            batch_idx: A batch index.

        Returns:
            A training loss.
        """
        if torch.tensor(self.hparams.decaygaussian):
            if self.global_step % 100 == 0:
                self._update_lossfn(self.global_step)

        superresimg: torch.Tensor = sample["superresimg"]
        est, microscope_outputs, cropped_psfenergy = self.forward(
            superresimg, psf_jitter=torch.tensor(self.hparams.psf_jitter)
        )

        # When the psf is jittered, the prediction of the final plane is unreliable.
        # Therefore, we are dropping the two last planes.
        if self.hparams.psf_jitter:
            superresimg[..., -2:, :, :] = 0

        data_loss = self.lossfn.train_loss(est, superresimg)
        reg_loss = cropped_psfenergy

        loss = data_loss + self.hparams.reg * reg_loss

        self.log("train/data_loss", data_loss)
        self.log("train/reg_loss", reg_loss)
        self.log("train/loss_sigma_nm", self.loss_sigma_nm)
        self.log("train/total_loss", loss)

        if torch.tensor(self.global_step % 1000 == 0):
            self._visualize_sample(microscope_outputs, est, superresimg, "train")

        return loss

    def training_epoch_end(self, outputs) -> None:
        """Reset pl.metrics.Metric."""
        self.lossfn.reset()

    def validation_step(self, sample, batch_idx) -> None:
        """Compute validation loss.

        Args:
            sample: A validation sample. A superresolution volume which has a random
                number of emitters at random locations and its frame number. See
                dataset.emitter.EmitterDatasetFromCSV.
            batch_idx: A batch index.
        """
        superresimg = sample["superresimg"]
        est, microscope_outputs, cropped_psfenergy = self.forward(
            superresimg, psf_jitter=torch.tensor(False)
        )
        if self.hparams.psf_jitter:
            superresimg[..., -2:, :, :] = 0

        self.val_lossfn(est, superresimg)
        if batch_idx == 0:
            self._visualize_sample(microscope_outputs, est, superresimg, "val")
        self.log("val_loss", self.val_lossfn, on_step=False, on_epoch=True)

    def validation_epoch_end(self, outputs) -> None:
        """Reset pl.metrics.Metric."""
        self.val_lossfn.reset()

    @torch.no_grad()
    def _visualize_sample(
        self,
        microscope_outputs: MicroscopeOutputs,
        est: torch.Tensor,
        gt: torch.Tensor,
        tag: str,
    ):
        """Visualize the current training state at tensorboard and export some results."""
        est = F.relu(est)
        global_step = self.global_step
        logger = self.logger.experiment
        phase = self.microscope.phi.detach().unsqueeze(1)  # S x 1 x H x W
        phase = torch.where(torch.isnan(phase), torch.zeros_like(phase), phase)
        self.log_dict(
            {
                f"{tag}/maximum_of_groundtruth": gt.max(),
                f"{tag}/minimum_of_groundtruth": gt.min(),
                f"{tag}/maximum_of_estimation": est.max(),
                f"{tag}/minimum_of_estimation": est.min(),
                f"{tag}/maximum_of_captured_image": microscope_outputs.noisy_img.max(),
                f"{tag}/minimum_of_captured_image": microscope_outputs.noisy_img.min(),
                "Phase_maximum": phase.max(),
                "Phase_minimum": phase.min(),
            }
        )
        batch_idx = 0
        util.io.imsave(os.path.join(self.logdir, tag + "_gt.tif"), gt[batch_idx])
        util.io.imsave(
            os.path.join(self.logdir, tag + "_modeloutput.tif"), est[batch_idx]
        )
        util.io.imsave(
            os.path.join(self.logdir, tag + "_captimg.tif"),
            microscope_outputs.noisy_img[batch_idx],
        )

        capt_img = F.interpolate(
            microscope_outputs.noisy_img,
            scale_factor=self.microscope.upsampling_factor,
            mode="nearest",
        )  # B x S x H x W

        gt = scale_image(gt)
        est = scale_image(est)
        capt_img = scale_image(capt_img)

        gt_xy = gt.max(dim=-3)[0]
        gt_yz = gt.max(dim=-2)[0]
        est_xy = est.max(dim=-3)[0]
        est_yz = est.max(dim=-2)[0]

        concat_capt_img = torch.cat(
            [capt_img[:, i] for i in range(capt_img.shape[1])], dim=-2
        )

        margin_px = 5
        margin = torch.ones(
            (capt_img.shape[0], margin_px, capt_img.shape[2]), device=capt_img.device
        )

        summary_image = torch.cat(
            [
                concat_capt_img,
                margin,
                gt_xy,
                margin,
                gt_yz,
                margin,
                est_xy,
                margin,
                est_yz,
            ],
            dim=-2,
        )
        summary_image = summary_image.unsqueeze(1)  # add color dim
        if summary_image.shape[0] > self.hparams.summary_max_images:
            summary_image = summary_image[: self.hparams.summary_max_images]
        summary_image_grid = torchvision.utils.make_grid(
            summary_image, nrow=summary_image.shape[0], pad_value=1.0, normalize=False
        )

        logger.add_image(f"{tag}/summary", summary_image_grid, global_step)
        logger.add_histogram(f"{tag}/gt_output", gt, global_step)
        logger.add_histogram(f"{tag}/est_output", est, global_step)

        if tag == "val":
            # Visualization of phase
            concat_phase = torchvision.utils.make_grid(
                scale_image(phase),
                nrow=phase.shape[0],
                pad_value=1.0,
            )
            logger.add_image("phase", concat_phase, global_step)
            util.io.imsave(self.phase_path, phase)

            # Visualization of PSF
            psf_sz_px = 128
            if self.hparams.with_beads:
                psf_spcamera = self.microscope.psf_at_spcamera(
                    psf_sz_px * self.microscope.upsampling_factor
                )[0]
                psf_spcamera = self.microscope.bead.conv(
                    fftshift(psf_spcamera, (-1, -2))
                )
                psf = resize2d(psf_spcamera, self.microscope.upsampling_factor)
                psf = (
                    psf * self.hparams.mean_photons
                    + (self.hparams.bg_max + self.hparams.bg_min) / 2
                ) / self.hparams.num_shots
                psf /= psf.max()
            else:
                psfimg_orig = self.microscope.psf_at_spcamera(psf_sz_px)[0]
                psf = fftshift(psfimg_orig, dims=(-1, -2))
            concat_psf = torch.cat(
                [psf[i] for i in range(psf.shape[0])], dim=-1
            ).unsqueeze(
                -3
            )  # D x 1 x H x 2W
            concat_psf_grid = torchvision.utils.make_grid(
                scale_image(concat_psf), nrow=8, pad_value=1.0, normalize=False
            )
            logger.add_image("psf", concat_psf_grid, global_step)
            util.io.imsave(self.psf_path, concat_psf)

            depths = self.microscope.depths
            sqcrb_x = torch.zeros(self.microscope.axial_sz_px, device=depths.device)
            sqcrb_y = torch.zeros(self.microscope.axial_sz_px, device=depths.device)
            sqcrb_z = torch.zeros(self.microscope.axial_sz_px, device=depths.device)
            em_photon = self.hparams.mean_photons
            bg_photon = (self.hparams.bg_max + self.hparams.bg_min) / 2
            for i in range(self.microscope.axial_sz_px):
                sqcrb_x[i], sqcrb_y[i], sqcrb_z[i] = self.microscope.sqcrb(
                    torch.tensor([i], device=depths.device), em_photon, bg_photon
                )
            depths = depths.detach().cpu().numpy()
            figcrb = plt.figure(figsize=(10, 8))
            plt.plot(depths * 1e-3, sqcrb_x.detach().cpu().numpy(), "o:")
            plt.plot(depths * 1e-3, sqcrb_y.detach().cpu().numpy(), "o:")
            plt.plot(depths * 1e-3, sqcrb_z.detach().cpu().numpy(), "o:")
            plt.legend(["x", "y", "z"])
            plt.ylabel("sqrt(crb) [nm")
            plt.xlabel("depths [um]")
            plt.ylim([0, 200])
            plt.title(f"Emission: {em_photon:.1f}  Background: {bg_photon:.1f}")
            logger.add_figure("sqrt_crb", figcrb, global_step)
            plt.close(figcrb)

    def on_test_epoch_start(self):
        """Make a directory for exporting results and simulate PSF for test step."""
        patch_sz = self.hparams.patch_sz
        psfimg, _ = self.microscope.psf_at_spcamera(
            (2 * patch_sz - 1) * self.hparams.upsampling_factor,
            psf_jitter=torch.tensor(False),
        )
        self.rfft_psfimg = torch.rfft(psfimg, 2)

        save_dir = self.hparams.save_dir
        self.tmp_dir = os.path.join(save_dir, "tmp")
        os.makedirs(self.tmp_dir, exist_ok=True)

    def test_step(self, samples, batch_idx):
        """Infer and save results."""
        captimgs = samples["captimg"]
        idx = samples["idx"].cpu()
        backproj_vol = self.microscope.backprojection_with_rfft_psf(
            captimgs, self.rfft_psfimg
        )
        est = F.relu_(self.net(backproj_vol))

        est[..., :, : self.offset_sppx] = 0
        est[..., : self.offset_sppx, :] = 0
        est[..., :, -self.offset_sppx :] = 0
        est[..., -self.offset_sppx :, :] = 0

        n_tiles = self.hparams.n_tiles

        for est_i, idx_i, captimg_i in zip(est, idx, captimgs):
            t_i, y_i, x_i = np.unravel_index(idx_i, n_tiles)
            util.io.imsave(
                os.path.join(self.tmp_dir, f"est-{t_i}_{y_i}_{x_i}.tif"), est_i
            )

    def test_epoch_end(self, output_list: List):
        """Concatenate the batched images to a volume."""
        est_vol_dir = os.path.join(self.hparams.save_dir, "est")
        os.makedirs(est_vol_dir, exist_ok=True)

        if self.global_rank == 0:
            full_img_shape = self.hparams.full_img_shape
            up_factor = self.microscope.upsampling_factor
            o = self.hparams.overlap_sz
            p = (self.hparams.patch_sz - o) * up_factor
            est_volume = np.zeros(
                (
                    full_img_shape[0],
                    self.microscope.axial_sz_px,
                    full_img_shape[-2] * up_factor,
                    full_img_shape[-1] * up_factor,
                ),
                dtype=np.float32,
            )
            est_weight = np.zeros(
                (
                    full_img_shape[0],
                    self.microscope.axial_sz_px,
                    full_img_shape[-2] * up_factor,
                    full_img_shape[-1] * up_factor,
                ),
                dtype=np.float32,
            )
            n_tiles = self.hparams.n_tiles
            for t_i in tqdm(range(n_tiles[0]), desc="Reconstructing from patches"):
                for y_i in range(n_tiles[1]):
                    for x_i in range(n_tiles[2]):
                        est_i = util.io.imread(
                            os.path.join(self.tmp_dir, f"est-{t_i}_{y_i}_{x_i}.tif")
                        )
                        est_volume[
                            t_i,
                            :,
                            y_i * p : (y_i + 1) * p + o * up_factor,
                            x_i * p : (x_i + 1) * p + o * up_factor,
                        ] += est_i
                        est_weight[
                            t_i,
                            :,
                            y_i * p
                            + self.offset_sppx : (y_i + 1) * p
                            + o * up_factor
                            - self.offset_sppx,
                            x_i * p
                            + self.offset_sppx : (x_i + 1) * p
                            + o * up_factor
                            - self.offset_sppx,
                        ] += 1.0

            shutil.rmtree(self.tmp_dir)
            est_volume[est_weight > 0] /= est_weight[est_weight > 0]
            for t_i in tqdm(range(n_tiles[0]), desc="Saving volumes"):
                util.io.imsave(
                    os.path.join(est_vol_dir, f"est_{t_i:06d}.tif"), est_volume[t_i]
                )

    @staticmethod
    def add_model_specific_args(parent_parser):
        """Add flags of Localizer module."""
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser = Microscope.add_model_specific_args(parser)
        parser = Bead.add_model_specific_args(parser)

        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--summary_max_images", type=int, default=4)
        parser.add_argument("--note", type=str, default=None)

        # For retraining/inference on captured images
        parser.add_argument(
            "--depth_independent_aberration_path", type=str, default=None
        )
        parser.add_argument("--depth_dependent_aberration_path", type=str, default=None)

        # Learning
        parser.add_argument("--cnn_lr", type=float, default=1e-4)
        parser.add_argument("--optics_lr", type=float, default=1e-4)
        parser.add_argument("--batch_sz", type=int, default=2)

        # Loss function
        parser.add_argument("--reg", type=float, default=1e4)
        parser.add_argument("--loss_sigma_nm", type=float, default=180.0)
        parser.add_argument("--loss_min_sigma_nm", type=float, default=25.0)
        parser.add_argument("--updatelossfn_every", type=int, default=100)
        parser.add_argument(
            "--decaygaussian", dest="decaygaussian", action="store_true"
        )
        parser.add_argument(
            "--no-decaygaussian", dest="decaygaussian", action="store_false"
        )
        parser.set_defaults(decaygaussian=True)

        # Model
        parser.add_argument("--capt_sz_px", type=int, default=32)
        parser.add_argument("--unet_base_ch", type=int, default=4)

        # Optics
        parser.add_argument(
            "--optimize_optics", dest="optimize_optics", action="store_true"
        )
        parser.add_argument(
            "--no-optimize_optics", dest="optimize_optics", action="store_false"
        )
        parser.set_defaults(optimize_optics=False)

        parser.add_argument("--psf_jitter", dest="psf_jitter", action="store_true")
        parser.add_argument("--no-psf_jitter", dest="psf_jitter", action="store_false")
        parser.set_defaults(psf_jitter=False)

        # For inference.
        parser.add_argument("--ckpt_path", type=str, default=None)
        parser.add_argument("--save_dir", type=str, default="inference_results")

        return parser
