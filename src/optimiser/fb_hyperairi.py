"""
HyperAIRI, built on forward-backward algorithm
"""

import os
from typing import Union, Optional
import torch
import torch.distributed as dist
import numpy as np
from astropy.io import fits

from .forward_backward import ForwardBackward
from ..regulariser import DenHyperAIRI
from ..ri_measurement_operator import MeasOpHyperspectral
from ..interface import fit_clean_beam, get_spec_idx


class FBHyperAIRI(ForwardBackward):
    """
    FBHyperAIRI implements the HyperAIRI algorithm based on Forward-Backward.

    It handles data fidelity in the forward step and uses AIRI denoisers
    in the backward step for regularization.
    """

    def __init__(
        self,
        meas: torch.Tensor,
        ch_freq: torch.Tensor,
        meas_op: MeasOpHyperspectral,
        regulariser: DenHyperAIRI,
        rank_ch_start: int = 0,
        im_min_itr: int = 100,
        im_max_itr: int = 2000,
        im_var_tol: float = 1e-4,
        im_peak_est: Optional[torch.Tensor] = None,
        heu_noise_scale: float = 1.0,
        adapt_net_select: bool = True,
        peak_tol_min: float = 1e-3,
        peak_tol_max: float = 0.1,
        peak_tol_step: float = 0.1,
        spec_update: bool = True,
        spec_idx_init: Union[torch.Tensor, str] = "dirty",
        spec_max_update_count: int = 50,
        spec_max_update_interval: int = 60,
        spec_update_tol_max: float = 1.6e-3,
        spec_update_tol_min: float = 1e-5,
        spec_update_tol_step: float = 0.5,
        verbose: bool = True,
        save_pth: str = "results",
        file_prefix: str = "",
        iter_save: int = 1000,
    ) -> None:
        """
        Initializes the AIRI optimiser, which is built on forward-backward algorithm.

        Args:
            meas (torch.Tensor): The measurement tensor.
            meas_op (MeasOpHyperspectral): The measurement operator.
            regulariser (DenHyperAIRI): The denoising operator.
            im_min_itr (int, optional): The minimum number of iterations. Defaults to 100.
            im_max_itr (int, optional): The maximum number of iterations. Defaults to 2000.
            im_var_tol (float, optional): The tolerance for image variation. Defaults to 1e-4.
            im_peak_est (Union[torch.Tensor, None], optional): The estimated peak of the image. Defaults to None.
            heu_noise_scale (float, optional): The heuristic noise scale. Defaults to 1.0.
            adapt_net_select (bool, optional): Whether to adapt network selection. Defaults to True.
            peak_tol_min (float, optional): The minimum tolerance for peak. Defaults to 1e-3.
            peak_tol_max (float, optional): The maximum tolerance for peak. Defaults to 0.1.
            peak_tol_step (float, optional): The step size for peak tolerance. Defaults to 0.1.
            spec_update (bool, optional): Whether to update spectral index map. Defaults to True.
            spec_idx_init (Union[torch.Tensor, str], optional): The initialization method for spectral index map.
                It can be 'zero', 'dirty', or a tensor. Defaults to 'zero'.
            spec_max_update_count (int, optional): The maximum number of spectral index map updates.
                Defaults to 50.
            spec_max_update_interval (int, optional): The maximum interval between spectral index map updates.
                Defaults to 60.
            spec_update_tol_max (float, optional): The maximum tolerance for spectral index map update. Defaults to 8e-3.
            spec_update_tol_min (float, optional): The minimum tolerance for spectral index map update. Defaults to 1e-5.
            spec_update_tol_step (float, optional): The step size for spectral index map update tolerance. Defaults to 0.5.
            verbose (bool, optional): Whether to print verbose logs. Defaults to True.
            save_pth (str, optional): The path where results will be saved. Defaults to "results".
            file_prefix (str, optional): The prefix of the saving files. Defaults to "".
            iter_save (int, optional): The number of iterations after which to save results.
                Defaults to 1000.
        """
        super().__init__(
            meas,
            ch_freq,
            meas_op,
            regulariser,
            heu_noise_scale=heu_noise_scale,
            im_max_itr=im_max_itr,
            save_pth=save_pth,
            file_prefix=file_prefix,
            rank_ch_start=rank_ch_start,
        )

        self._im_min_itr = im_min_itr
        self._im_var_tol = im_var_tol
        self._im_peak_est = im_peak_est
        if self._im_peak_est is not None:
            self._im_peak_est = self._im_peak_est.to(
                device=self._meas_op.get_device(), dtype=self._meas_op.get_data_type()
            ).view(1, -1, 1, 1)

        # network adaptation parameters
        self._adapt_net_select = adapt_net_select
        self._peak_tol_min = peak_tol_min
        self._peak_tol = peak_tol_max
        self._peak_tol_step = peak_tol_step
        self._peak_val_range = []

        # spectral index map update parameters
        self._spec_idx_init = spec_idx_init
        self._spec_update = spec_update
        self._spec_max_update_count = spec_max_update_count
        self._spec_max_update_interval = spec_max_update_interval
        self._spec_update_tol = spec_update_tol_max
        self._spec_update_tol_min = spec_update_tol_min
        self._spec_update_tol_step = spec_update_tol_step
        self._spec_update_count = 0
        self._spec_update_iter = 0
        self._clean_beam = None

        self._verbose = verbose
        self._iter_save = iter_save

        self._prev_peak_val = None
        self._im_rel_var = 1.0

    @torch.no_grad()
    def initialisation(self) -> None:
        """
        Initialises specific parameters of AIRI.

        This method initialises the step size, estimates the peak value, sets the
        heuristic noise level, initialises the AIRI proximal operator, and prints
        some information if verbose is True.
        """
        if self._is_distributed:
            dist.barrier()
        if self._rank == 0 and self._verbose:
            print("\nINFO: Initialisation HyperAIRI optimiser...", flush=True)

        # step size
        self._gd_step_size = 1.98 / self._meas_op.get_op_norm()

        # estimate peak value
        self._estimate_channel_wise_peak_val()

        # heuristic noise level of each channel
        self._calculate_channel_wise_heuristic()

        # fit clean beam to the PSF of the last channel
        # then broadcast the clean beam to all ranks
        self._fit_and_broadcast_clean_beam()

        # get the precision threshold for spectral index map fitting
        self._spec_idx_prec = self._heuristic.min()
        if self._is_distributed:
            dist.all_reduce(self._spec_idx_prec, op=dist.ReduceOp.MIN)
        self._spec_idx_prec = self._spec_idx_prec.item()

        # initialise spectral index map
        self._initialise_spectral_index_map()

        # initialise HyperAIRI prox
        if self._verbose:
            if self._is_distributed:
                dist.barrier()
            if self._rank == 0:
                print("\nINFO: Initialising HyperAIRI proximal operator...", flush=True)
        self._peak_val_range = self._regulariser.update(
            ch_heuristic=self._heuristic,
            ch_peak_est=self._prev_peak_val,
            spec_idx=self._spec_idx_map,
        )

        # logging
        if self._verbose:
            if self._is_distributed:
                dist.barrier()
            if self._rank == 0:
                print(
                    f"INFO: forward-backward step size: {self._gd_step_size}",
                    flush=True,
                )
                print("\n*************************************************", flush=True)
                print("********* STARTING ALGORITHM: HyperAIRI *********", flush=True)
                print("*************************************************", flush=True)

    def _estimate_channel_wise_peak_val(self) -> None:
        """
        Estimates the peak value for each channel from the dirty image if the user does not
        specify it.
        """
        if self._im_peak_est is None:
            self._prev_peak_val = (
                self._meas_bp.amax(dim=(2, 3), keepdim=True) / self._psf_peak
            )
            if self._verbose:
                if self._rank == 0:
                    print("\nINFO: use normalised dirty peak as estimated", flush=True)
                if self._is_distributed:
                    dist.barrier()
                print(
                    f"INFO: channel {self._rank_ch_start + 1} - {self._rank_ch_start + self._rank_ch_num}, "
                    f"estimated image peak value {self._prev_peak_val.squeeze().cpu().tolist()}",
                    flush=True,
                )
        else:
            self._prev_peak_val = self._im_peak_est
            if self._verbose:
                if self._rank == 0:
                    print(
                        "INFO: user specified the estimated image peak value",
                        flush=True,
                    )
                if self._is_distributed:
                    dist.barrier()
                print(
                    f"INFO: channel {self._rank_ch_start + 1} - {self._rank_ch_start + self._rank_ch_num}, "
                    f"specified image peak value {self._prev_peak_val.squeeze().cpu().tolist()}",
                    flush=True,
                )

    def _fit_and_broadcast_clean_beam(self) -> None:
        """
        Fits the clean beam from the PSF of the last channel and broadcasts it to all ranks.

        This method is used when the spectral index map update is enabled, as the clean
        beam is needed for the spectral index map fitting.

        The result clean beam is saved in the attribute `_clean_beam`, which is a tensor
        of shape (1, 1, H, W).
        """
        if self._rank == self._world_size - 1:
            psf_ch_end = self._psf[0, -1, :, :].cpu().numpy()
            self._clean_beam, _, _, _ = fit_clean_beam(
                psf_ch_end, verbose=self._verbose
            )
            self._clean_beam = torch.tensor(
                self._clean_beam,
                device=self._meas_op.get_device(),
                dtype=self._meas_op.get_data_type(),
            ).reshape(1, 1, *self._clean_beam.shape)
        if self._is_distributed:
            if self._rank == self._world_size - 1:
                clean_beam_shape = torch.tensor(
                    self._clean_beam.shape,
                    device=self._meas_op.get_device(),
                    dtype=torch.int,
                )
            else:
                clean_beam_shape = torch.empty(
                    4,
                    device=self._meas_op.get_device(),
                    dtype=torch.int,
                )
            dist.broadcast(clean_beam_shape, src=self._world_size - 1)
            if self._rank < self._world_size - 1:
                self._clean_beam = torch.empty(
                    *clean_beam_shape.cpu().tolist(),
                    device=self._meas_op.get_device(),
                    dtype=self._meas_op.get_data_type(),
                )
            dist.broadcast(self._clean_beam, src=self._world_size - 1)

    def _initialise_spectral_index_map(self) -> None:
        """
        Initializes the spectral index map based on the specified method.

        The spectral index map is saved in the attribute `_spec_idx_map`,
        which is a tensor of shape (1, 1, H, W),
        """
        if isinstance(self._spec_idx_init, torch.Tensor):
            self._spec_idx_map = self._spec_idx_init.to(
                device=self._meas_op.get_device(),
                dtype=self._meas_op.get_data_type(),
            ).reshape(1, 1, *self._model.shape[2:])
        elif self._spec_idx_init == "dirty":
            dirty_normalised_np = self.get_dirty_image(normalise=self._psf_peak, dest=0)
            if self._rank == 0:
                self._spec_idx_map = get_spec_idx(
                    img_cube=dirty_normalised_np,
                    freqs=self._ch_freq.cpu().numpy().squeeze(),
                    kernel=self._clean_beam.cpu().numpy().squeeze(),
                    threshold=self._spec_idx_prec,
                )
                self._spec_idx_map = np.nan_to_num(
                    self._spec_idx_map, nan=0.0, posinf=0.0, neginf=0.0
                )
                self._spec_idx_map = torch.tensor(
                    self._spec_idx_map,
                    device=self._meas_op.get_device(),
                    dtype=self._meas_op.get_data_type(),
                ).reshape(1, 1, *self._model.shape[2:])
            else:
                self._spec_idx_map = torch.empty(
                    (1, 1, self._model.shape[2], self._model.shape[3]),
                    device=self._meas_op.get_device(),
                    dtype=self._meas_op.get_data_type(),
                )
            if self._is_distributed:
                dist.broadcast(self._spec_idx_map, src=0)
        else:
            self._spec_idx_map = torch.zeros(
                1,
                1,
                self._model.shape[2],
                self._model.shape[3],
                device=self._meas_op.get_device(),
                dtype=self._meas_op.get_data_type(),
            )

    def _each_iter_begin(self):
        """
        Method to be executed at the beginning of each iteration.

        This method is empty for AIRI.
        """

    @torch.no_grad()
    def _stop_criteria(self) -> bool:
        """
        Checks the stop criteria for the algorithm.

        This method calculates the image relative variation, logs some information if
        `verbose` is True, and checks if the stop criteria for the algorithm are met.

        Returns:
            bool: True if the stop criteria are met, False otherwise.
        """
        # img relative variation
        model_norm = torch.sum(self._model**2)
        model_diff_norm = torch.sum((self._model - self._model_prev) ** 2)
        # gather norms from all ranks
        if self._is_distributed:
            dist.all_reduce(model_norm, op=dist.ReduceOp.SUM)
            dist.all_reduce(model_diff_norm, op=dist.ReduceOp.SUM)
        self._im_rel_var = torch.sqrt(model_diff_norm / model_norm).item()

        # log
        if self._verbose and self._rank == 0:
            print(
                f"\nIter {self._iter+1}: relative variation {self._im_rel_var}",
                f"\ntimings: gradient step {self._t_forward} sec,",
                f"denoising step {self._t_backward} sec,"
                f"iteration {self._t_iter} sec.",
                flush=True,
            )

        # stop criteria
        if self._iter + 1 >= self._im_min_itr and self._im_rel_var < self._im_var_tol:
            return True

        return False

    @torch.no_grad()
    def _each_iter_end(self) -> None:
        """
        Method to be executed at the end of each iteration.

        This method saves intermediate results, and updates the HyperAIRI denoiser selection
        if `adapt_net_select` is True.
        """
        # save intermediate results
        if (self._iter + 1) % self._iter_save == 0:
            model_full_np = self.get_model_image(dtype=torch.float32)
            residual_full_np = self.get_residual_image(dtype=torch.float32)
            if self._rank == 0:
                fits.writeto(
                    os.path.join(
                        self._save_pth,
                        self._file_prefix
                        + "tmp_model_itr_"
                        + str(self._iter + 1)
                        + ".fits",
                    ),
                    model_full_np,
                    overwrite=True,
                )
                fits.writeto(
                    os.path.join(
                        self._save_pth,
                        self._file_prefix
                        + "tmp_residual_itr_"
                        + str(self._iter + 1)
                        + ".fits",
                    ),
                    residual_full_np / self._psf_peak,
                    overwrite=True,
                )

        # HyperAIRI denoiser selection update
        if self._adapt_net_select:
            curr_peak_val = self._model.amax(dim=(2, 3), keepdim=True)
            peak_var = (
                torch.abs(curr_peak_val - self._prev_peak_val) / self._prev_peak_val
            )
            peak_var_max = peak_var.max()
            if self._is_distributed:
                dist.all_reduce(peak_var_max, op=dist.ReduceOp.MAX)
            peak_var_max = peak_var_max.item()
            if self._verbose:
                curr_peak_val_full = self._gather_ch_img_to_tensor(curr_peak_val)
                if self._rank == 0:
                    curr_peak_val_ch_1 = curr_peak_val_full[0, 0, 0, 0].item()
                    curr_peak_val_ch_end = curr_peak_val_full[0, -1, 0, 0].item()
                    print(
                        f"  Model image peak value: ch 1 {curr_peak_val_ch_1:.3f}, "
                        f"ch end {curr_peak_val_ch_end:.3f}, "
                        f"max relative variation = {peak_var_max:.3e}",
                        flush=True,
                    )

            peak_out_of_range = torch.logical_or(
                curr_peak_val < self._peak_val_range[0],
                curr_peak_val > self._peak_val_range[1],
            )

            self._prev_peak_val = curr_peak_val
            if peak_var_max < self._peak_tol and torch.any(peak_out_of_range):
                self._peak_tol = max(
                    self._peak_tol * self._peak_tol_step, self._peak_tol_min
                )
                self._peak_val_range = self._regulariser.update(
                    ch_heuristic=self._heuristic,
                    ch_peak_est=self._prev_peak_val,
                )

        # spectral index map update
        if self._spec_update:
            if (
                self._im_rel_var < self._spec_update_tol
                or self._spec_update_iter >= self._spec_max_update_interval
            ):
                self._spec_update_iter = 0
                self._spec_update_count += 1
                self._spec_update_tol = max(
                    self._spec_update_tol * self._spec_update_tol_step,
                    self._spec_update_tol_min,
                )
                spec_idx_map = self._update_spec_idx_map()
                _ = self._regulariser.update(spec_idx=spec_idx_map)
            else:
                self._spec_update_iter += 1

            if self._spec_update_count >= self._spec_max_update_count:
                self._spec_update = False  # stop further updates if interval exceeded

    @torch.no_grad()
    def finalisation(self) -> None:
        """
        Finalises the FBAIRI optimiser.

        This method prints final log information if `verbose` is True, and saves
        the result model and residual images.
        """
        if self._is_distributed:
            dist.barrier()

        if self._verbose and self._rank == 0:
            print("\n**************************************", flush=True)
            print("********** END OF ALGORITHM **********", flush=True)
            print("**************************************\n", flush=True)
            print(
                f"Imaging finished in {self._t_total} sec,",
                f"total number of iterations {self._iter+1}",
                flush=True,
            )

        # save final results
        model_full_np = self.get_model_image()
        residual_full_np = self.get_residual_image()
        if self._rank == 0:
            fits.writeto(
                os.path.join(self._save_pth, self._file_prefix + "model_image.fits"),
                model_full_np,
                overwrite=True,
            )
            fits.writeto(
                os.path.join(
                    self._save_pth, self._file_prefix + "residual_dirty_image.fits"
                ),
                residual_full_np,
                overwrite=True,
            )
            result_spec_idx = get_spec_idx(
                model_full_np,
                self._ch_freq.cpu().numpy().squeeze(),
                threshold=self._spec_idx_prec,
            )
            fits.writeto(
                os.path.join(
                    self._save_pth, self._file_prefix + "spectral_index_map.fits"
                ),
                result_spec_idx,
                overwrite=True,
            )

    @torch.no_grad()
    def _update_spec_idx_map(self) -> torch.Tensor:
        """
        Updates the spectral index map based on the current model image.
        Returns:
            torch.Tensor: The updated spectral index map.
        """
        # convolve with clean beam
        img_blur = self._model.view(self._rank_ch_num, 1, *self._model.shape[2:])
        img_blur = torch.nn.functional.conv2d(
            img_blur, self._clean_beam, padding="same"
        )
        img_blur = img_blur.view(1, self._rank_ch_num, *self._model.shape[2:])
        img_blur = torch.clamp(img_blur, min=0.0)

        # build valid mask
        img_blur_valid_count = torch.sum(
            (img_blur > self._spec_idx_prec).to(torch.float32), dim=1, keepdim=True
        )
        if self._is_distributed:
            dist.all_reduce(img_blur_valid_count, op=dist.ReduceOp.SUM)
        img_blur_valid_mask = img_blur_valid_count > self._ch_freq.shape[1] * 0.5

        # log frequency, x
        prec_mask = img_blur < self._spec_idx_prec
        log_ch_freq = torch.log(
            self._ch_freq[:, self._ch_start : self._ch_start + self._rank_ch_num, :, :]
        )
        log_ch_freq = log_ch_freq.expand_as(img_blur)
        log_ch_freq = log_ch_freq.masked_fill(prec_mask, 0.0)
        # log blurred image, y
        log_img_blur = torch.log(torch.clamp(img_blur, min=self._spec_idx_prec))
        log_img_blur = log_img_blur.masked_fill(prec_mask, 0.0)

        # least squares fitting y = a * x + b
        # a=\frac{n\sum_i x_i y_i - (\sum_i x_i)(\sum_i y_i)}{n\sum_i x_i^2-(\sum_i x_i)^2}
        sum_x = torch.sum(log_ch_freq, dim=1, keepdim=True)
        sum_y = torch.sum(log_img_blur, dim=1, keepdim=True)
        sum_xy = torch.sum(log_ch_freq * log_img_blur, dim=1, keepdim=True)
        sum_x2 = torch.sum(log_ch_freq**2, dim=1, keepdim=True)
        if self._is_distributed:
            dist.all_reduce(sum_x, op=dist.ReduceOp.SUM)
            dist.all_reduce(sum_y, op=dist.ReduceOp.SUM)
            dist.all_reduce(sum_xy, op=dist.ReduceOp.SUM)
            dist.all_reduce(sum_x2, op=dist.ReduceOp.SUM)
        numerator = img_blur_valid_count * sum_xy - sum_x * sum_y
        denominator = img_blur_valid_count * sum_x2 - sum_x**2
        img_blur_valid_mask = torch.logical_and(img_blur_valid_mask, denominator > 1e-8)
        spec_idx_map = torch.where(
            img_blur_valid_mask,
            numerator / torch.clamp(denominator, min=1e-8),
            torch.zeros_like(numerator),
        )
        spec_idx_map = spec_idx_map.masked_fill(spec_idx_map > -1e-8, 0.0)

        # post processing
        img_blur_avg = torch.sum(img_blur, dim=1, keepdim=True)
        if self._is_distributed:
            dist.all_reduce(img_blur_avg, op=dist.ReduceOp.SUM)
        img_blur_avg = img_blur_avg / self._ch_freq.shape[1]

        img_blur_avg = img_blur_avg.masked_fill(img_blur_avg < self._spec_idx_prec, 0.0)
        mask_alpha_0 = (spec_idx_map == 0).expand_as(self._model)
        self._model = torch.where(
            mask_alpha_0, img_blur_avg.expand_as(self._model), self._model
        )

        mask_alpha_neg = (spec_idx_map < 0).expand_as(self._model)
        low_val_mask = torch.logical_and(
            mask_alpha_neg, self._model < self._heuristic.view(1, -1, 1, 1)
        )
        low_val_mask = torch.logical_and(low_val_mask, self._model < img_blur)
        self._model = torch.where(low_val_mask, img_blur, self._model)

        return spec_idx_map
