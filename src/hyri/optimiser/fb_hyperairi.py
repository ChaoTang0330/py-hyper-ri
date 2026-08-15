"""Define the hyperspectral AIRI optimiser.

This module provides the HyperAIRI optimiser built on the shared
forward-backward backbone with adaptive denoiser selection and spectral-index
updates.
"""

import os

import numpy as np
import torch
import torch.distributed as dist
from astropy.io import fits

from ..interface import fit_clean_beam, get_spec_idx
from ..measurement_operator import MeasOpHyperspectral
from ..regulariser import DenHyperAIRI
from .forward_backward import ForwardBackward


class FBHyperAIRI(ForwardBackward):
    """Represent the HyperAIRI forward-backward optimiser.

    This class combines the shared forward-backward optimisation loop with a
    hyperspectral AIRI denoiser. Besides the image update, it can adapt the
    denoiser selection from channel-wise peak estimates and refine an internal
    spectral-index map used by the HyperAIRI regulariser.

    Shape:
        Image tensor: ``(1, C_rank, H, W)``.
        Visibility tensor: tuple of ``C_rank`` tensors, each with shape
            ``(1, 1, N_c)``.
        Frequency tensor: ``(1, C_total, 1, 1)`` after reshaping in the base
            optimiser.
        Spectral-index tensor: ``(1, 1, H, W)``.

    Attributes:
        _im_min_itr (int): Minimum number of iterations before convergence is
            checked.
        _im_var_tol (float): Relative-variation tolerance for the image model.
        _im_peak_est (torch.Tensor | None): User-provided channel-wise image
            peak estimates, if any.
        _adapt_net_select (bool): Whether HyperAIRI denoiser selection is
            adapted during optimisation.
        _peak_tol_min (float): Minimum allowed relative peak-variation
            threshold.
        _peak_tol (float): Current relative peak-variation threshold.
        _peak_tol_step (float): Multiplicative decay factor applied to the
            peak-variation threshold.
        _peak_val_range (list): Current valid peak-value range returned by the
            denoiser selector.
        _spec_idx_init (torch.Tensor | str): Initial spectral-index map or its
            initialisation mode.
        _spec_update (bool): Whether spectral-index updates remain enabled.
        _spec_max_update_count (int): Maximum number of spectral-index updates.
        _spec_max_update_interval (int): Maximum iteration gap between
            spectral-index updates.
        _spec_update_tol (float): Current relative-variation threshold used to
            trigger a spectral-index update.
        _spec_update_tol_min (float): Lower bound for the spectral-index
            update threshold.
        _spec_update_tol_step (float): Multiplicative decay factor for the
            spectral-index update threshold.
        _spec_update_count (int): Number of spectral-index updates already
            applied.
        _spec_update_iter (int): Iteration count since the last
            spectral-index update.
        _clean_beam (torch.Tensor | None): Clean-beam kernel used for
            spectral-index estimation.
        _verbose (bool): Whether informational logs are printed.
        _iter_save (int): Interval between intermediate FITS snapshots.
        _prev_peak_val (torch.Tensor | None): Previous channel-wise image peak
            estimate.
        _im_rel_var (float): Current relative image variation.

    Example:
        >>> optimiser = FBHyperAIRI(
        ...     meas=meas,
        ...     ch_freq=ch_freq,
        ...     meas_op=meas_op,
        ...     regulariser=regulariser,
        ... )
        >>> optimiser.initialisation()
        >>> optimiser.run()
        >>> optimiser.finalisation()

    References:
        Tang, C., Dabbech, A., Jackson, A., & Wiaux, Y. (2026). `HyperAIRI: A
        Plug-and-play Algorithm for Precise Hyperspectral Image Reconstruction
        in Radio Interferometry <https://doi.org/10.3847/1538-4365/ae3149>`_.
        *The Astrophysical Journal Supplement Series*, 283(1), 9.
    """

    def __init__(
        self,
        meas: list[torch.Tensor],
        ch_freq: torch.Tensor,
        meas_op: MeasOpHyperspectral,
        regulariser: DenHyperAIRI,
        meas_op_approx: MeasOpHyperspectral | None = None,
        rank_ch_start: int = 0,
        im_min_itr: int = 100,
        im_max_itr: int = 2000,
        im_var_tol: float = 1e-4,
        im_peak_est: torch.Tensor | None = None,
        heu_noise_scale: float = 1.0,
        adapt_net_select: bool = True,
        peak_tol_min: float = 1e-3,
        peak_tol_max: float = 0.1,
        peak_tol_step: float = 0.1,
        spec_update: bool = True,
        spec_idx_init: torch.Tensor | str = "dirty",
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
        """Initialise the HyperAIRI optimiser.

        Args:
            meas (List[torch.Tensor]): Local measurement tensors, typically one
                tensor per local channel.
            ch_freq (torch.Tensor): Frequencies for all spectral channels.
            meas_op (MeasOpHyperspectral): Precise hyperspectral measurement
                operator used for diagnostics and final products.
            regulariser (DenHyperAIRI): HyperAIRI denoising regulariser.
            meas_op_approx (MeasOpHyperspectral | None, optional): Optional
                approximate measurement operator used during the iterative
                forward step. Default: None.
            rank_ch_start (int, optional): Global starting channel index owned
                by the current rank. Default: 0.
            im_min_itr (int, optional): Minimum number of iterations before
                convergence is checked. Default: 100.
            im_max_itr (int, optional): Maximum number of iterations.
                Default: 2000.
            im_var_tol (float, optional): Relative-variation tolerance.
                Default: 1e-4.
            im_peak_est (torch.Tensor | None, optional): User-provided
                channel-wise image peak estimates. Default: None.
            heu_noise_scale (float, optional): Multiplicative scaling applied
                to heuristic noise levels. Default: 1.0.
            adapt_net_select (bool, optional): Adapt HyperAIRI denoiser
                selection during optimisation. Default: True.
            peak_tol_min (float, optional): Minimum allowed relative
                peak-variation threshold. Default: 1e-3.
            peak_tol_max (float, optional): Initial relative peak-variation
                threshold. Default: 0.1.
            peak_tol_step (float, optional): Multiplicative decay factor for
                the peak-variation threshold. Default: 0.1.
            spec_update (bool, optional): Enable spectral-index updates during
                optimisation. Default: True.
            spec_idx_init (torch.Tensor | str, optional): Initial
                spectral-index map or initialisation mode. Supported string
                values are ``"zero"`` and ``"dirty"``. Default: ``"dirty"``.
            spec_max_update_count (int, optional): Maximum number of
                spectral-index updates. Default: 50.
            spec_max_update_interval (int, optional): Maximum number of
                iterations between spectral-index updates. Default: 60.
            spec_update_tol_max (float, optional): Initial relative-variation
                threshold used to trigger a spectral-index update.
                Default: 1.6e-3.
            spec_update_tol_min (float, optional): Lower bound for the
                spectral-index update threshold. Default: 1e-5.
            spec_update_tol_step (float, optional): Multiplicative decay
                factor for the spectral-index update threshold. Default: 0.5.
            verbose (bool, optional): Enable informational logging.
                Default: True.
            save_pth (str, optional): Directory where optimiser products are
                saved. Default: ``"results"``.
            file_prefix (str, optional): Prefix prepended to saved file names.
                Default: ``""``.
            iter_save (int, optional): Save interval for intermediate FITS
                products. Default: 1000.

        Note:
            When ``meas_op_approx`` is provided, the approximate operator is
            used in the iterative forward step while ``meas_op`` remains the
            reference operator for dirty-image, PSF, clean-beam, and heuristic
            calculations.
        """
        super().__init__(
            meas,
            ch_freq,
            meas_op_approx if meas_op_approx is not None else meas_op,
            regulariser,
            meas_op_precise=meas_op,
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

        # Configure denoiser-selection adaptation.
        self._adapt_net_select = adapt_net_select
        self._peak_tol_min = peak_tol_min
        self._peak_tol = peak_tol_max
        self._peak_tol_step = peak_tol_step
        self._peak_val_range = []

        # Configure spectral-index-map updates.
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
        """Initialise HyperAIRI-specific optimisation parameters.

        This method sets the forward-backward step size, estimates channel-wise
        image peaks, computes heuristic noise levels, fits the clean beam used
        for spectral-index estimation, initialises the spectral-index map, and
        configures the HyperAIRI denoiser selector.
        """
        if self._is_distributed:
            dist.barrier()
        if self._rank == 0 and self._verbose:
            print("\nINFO: initialise HyperAIRI optimiser...", flush=True)

        # Set the forward-backward step size from the precise operator norm.
        self._gd_step_size = 1.98 / self._meas_op_precise.get_op_norm()

        # Estimate channel-wise image peaks from the current dirty image.
        self._estimate_channel_wise_peak_val()

        # Estimate heuristic regularisation levels for local channels.
        self._calculate_channel_wise_heuristic()

        # Fit the clean beam on the last rank and broadcast it to all ranks.
        self._fit_and_broadcast_clean_beam()

        # Use the smallest heuristic level as the log-fit precision threshold.
        self._spec_idx_prec = self._heuristic.min()
        if self._is_distributed:
            dist.all_reduce(self._spec_idx_prec, op=dist.ReduceOp.MIN)
        self._spec_idx_prec = self._spec_idx_prec.item()

        # Initialise the spectral-index map before denoiser selection.
        self._initialise_spectral_index_map()

        # Initialise the HyperAIRI denoiser and cache its supported peak range.
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

        if self._verbose:
            if self._is_distributed:
                dist.barrier()
            if self._rank == 0:
                print(
                    f"INFO: forward-backward step size: {self._gd_step_size:.4e}",
                    flush=True,
                )
                print("\n*************************************************", flush=True)
                print("********* STARTING ALGORITHM: HyperAIRI *********", flush=True)
                print("*************************************************", flush=True)

    def _estimate_channel_wise_peak_val(self) -> None:
        """Estimate or set channel-wise image peak values.

        When no user estimate is provided, this method uses the channel-wise
        maxima of the normalised dirty image.
        """
        if self._im_peak_est is None:
            self._prev_peak_val = (
                self._meas_bp.amax(dim=(2, 3), keepdim=True) / self._psf_peak
            )
            if self._verbose:
                if self._rank == 0:
                    print("INFO: use normalised dirty peak as estimated", flush=True)
                if self._is_distributed:
                    dist.barrier()
                peak_val_list = self._prev_peak_val.squeeze().cpu().tolist()
                if not isinstance(peak_val_list, list):
                    peak_val_list = [peak_val_list]
                print(
                    f"INFO: channel {self._rank_ch_start + 1} - "
                    f"{self._rank_ch_start + self._rank_ch_num}, "
                    f"estimated image peak value "
                    f"[{', '.join(f'{val:#.4g}' for val in peak_val_list)}]\n",
                    flush=True,
                    end="",
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
                peak_val_list = self._prev_peak_val.squeeze().cpu().tolist()
                if not isinstance(peak_val_list, list):
                    peak_val_list = [peak_val_list]
                print(
                    f"INFO: channel {self._rank_ch_start + 1} - "
                    f"{self._rank_ch_start + self._rank_ch_num}, "
                    f"specified image peak value "
                    f"[{', '.join(f'{val:#.4g}' for val in peak_val_list)}]\n",
                    flush=True,
                    end="",
                )

    def _fit_and_broadcast_clean_beam(self) -> None:
        """Fit and broadcast the clean beam used in spectral-index updates.

        The clean beam is fitted from the PSF of the last global channel and
        stored in ``self._clean_beam`` with shape ``(1, 1, H, W)``. The last
        channel has the highest resolution.
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
        """Initialise the spectral-index map.

        The map is stored in ``self._spec_idx_map`` with shape
        ``(1, 1, H, W)``. It can be supplied directly, estimated from the
        normalised dirty image, or initialised to zero.
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

    def _each_iter_begin(self) -> None:
        """Run logic at the beginning of each iteration.

        No per-iteration pre-processing is required for HyperAIRI.
        """

    @torch.no_grad()
    def _stop_criteria(self) -> bool:
        """Evaluate the HyperAIRI stopping criterion.

        Returns:
            bool: ``True`` when the minimum iteration count has been reached
            and the relative image variation falls below the stopping
            tolerance, otherwise ``False``.
        """
        # Compute the relative variation of the image model.
        model_norm = torch.sum(self._model**2)
        model_diff_norm = torch.sum((self._model - self._model_prev) ** 2)
        # Gather the global norms across all ranks.
        if self._is_distributed:
            dist.all_reduce(model_norm, op=dist.ReduceOp.SUM)
            dist.all_reduce(model_diff_norm, op=dist.ReduceOp.SUM)
        self._im_rel_var = torch.sqrt(model_diff_norm / model_norm).item()

        if self._verbose and self._rank == 0:
            print(
                f"\nIter {self._iter+1}: relative variation {self._im_rel_var:.3e}"
                f"\ntimings: gradient step {self._t_forward:#.3g} sec, "
                f"denoising step {self._t_backward:#.3g} sec, "
                f"iteration {self._t_iter:#.3g} sec.",
                flush=True,
            )

        if self._iter + 1 >= self._im_min_itr and self._im_rel_var < self._im_var_tol:
            return True

        return False

    @torch.no_grad()
    def _each_iter_end(self) -> None:
        """Run post-processing at the end of each iteration.

        This method saves intermediate products, updates the HyperAIRI
        denoiser selection when enabled, and refreshes the spectral-index map
        according to the current relative-variation schedule.
        """
        # Save intermediate model and residual products periodically.
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

        # Update denoiser selection from the current channel-wise peak values.
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
                        f"  Model image peak value: ch first "
                        f"{curr_peak_val_ch_1:#.3g}, "
                        f"ch end {curr_peak_val_ch_end:#.3g}, "
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

        # Update the spectral-index map once the image has stabilised enough.
        if self._spec_update:
            self._spec_update_iter += 1
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

            if self._spec_update_count >= self._spec_max_update_count:
                self._spec_update = False

    @torch.no_grad()
    def finalisation(self) -> None:
        """Finalise HyperAIRI and save the reconstructed products.

        This method prints final log information, saves the reconstructed model
        and residual dirty images, and exports a spectral-index map estimated
        from the final image cube.
        """
        if self._is_distributed:
            dist.barrier()

        if self._verbose and self._rank == 0:
            print("\n**************************************", flush=True)
            print("********** END OF ALGORITHM **********", flush=True)
            print("**************************************\n", flush=True)
            print(
                f"Imaging finished in {self._t_total:.1f} sec,",
                f"total number of iterations {self._iter+1}",
                flush=True,
            )

        # Save the final model, residual, and spectral-index products.
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
        """Update the spectral-index map from the current model image.

        The update is computed by fitting a log-log linear model across the
        blurred channel images and frequencies on pixels with sufficient signal.

        Returns:
            torch.Tensor: Updated spectral-index map with shape ``(1, 1, H, W)``.
        """
        # Convolve each channel image with the clean beam before fitting.
        img_blur = self._model.view(self._rank_ch_num, 1, *self._model.shape[2:])
        img_blur = torch.nn.functional.conv2d(
            img_blur, self._clean_beam, padding="same"
        )
        img_blur = img_blur.view(1, self._rank_ch_num, *self._model.shape[2:])
        img_blur = torch.clamp(img_blur, min=0.0)

        # Require enough valid channels for a stable spectral fit.
        img_blur_valid_count = torch.sum(
            (img_blur > self._spec_idx_prec).to(torch.float32), dim=1, keepdim=True
        )
        if self._is_distributed:
            dist.all_reduce(img_blur_valid_count, op=dist.ReduceOp.SUM)
        img_blur_valid_mask = img_blur_valid_count > self._ch_freq.shape[1] * 0.5

        # Build the log-frequency design variable.
        prec_mask = img_blur < self._spec_idx_prec
        log_ch_freq = torch.log(
            self._ch_freq[
                :, self._rank_ch_start : self._rank_ch_start + self._rank_ch_num, :, :
            ]
        )
        log_ch_freq = log_ch_freq.expand_as(img_blur)
        log_ch_freq = log_ch_freq.masked_fill(prec_mask, 0.0)
        # Build the log-intensity response variable.
        log_img_blur = torch.log(torch.clamp(img_blur, min=self._spec_idx_prec))
        log_img_blur = log_img_blur.masked_fill(prec_mask, 0.0)

        # Fit the spectral slope with closed-form least squares.
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

        # Post-process the fitted map and use it to stabilise weak pixels.
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
