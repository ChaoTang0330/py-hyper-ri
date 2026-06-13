"""Define the unconstrained AIRI optimiser.

This module provides the AIRI optimiser built on the shared forward-backward
backbone with adaptive denoiser selection.
"""

import os

import torch
import torch.distributed as dist
from astropy.io import fits

from ..interface import get_spec_idx
from ..measurement_operator import MeasOpHyperspectral
from ..regulariser import DenAIRI
from .forward_backward import ForwardBackward


class FBAIRI(ForwardBackward):
    """Represent the AIRI forward-backward optimiser.

    This class combines the shared forward-backward optimisation loop with a
    denoising AIRI prior. It optionally uses an approximate measurement
    operator during the iterative forward step and can adapt the denoiser
    selection from channel-wise peak estimates as the reconstruction evolves.

    Shape:
        Image tensor: ``(1, C_rank, H, W)``.
        Visibility tensor: tuple of ``C_rank`` tensors, each with shape
            ``(1, 1, N_c)``.
        Frequency tensor: ``(1, C_rank, 1, 1)`` after reshaping in the base
            optimiser.

    Attributes:
        _use_approx_meas_op (bool): Whether an approximate measurement operator
            is used in the iterative forward step.
        _im_min_itr (int): Minimum number of iterations before convergence is
            checked.
        _im_var_tol (float): Relative-variation tolerance for the image model.
        _im_peak_est (float | None): User-provided image-peak estimate, if any.
        _adapt_net_select (bool): Whether AIRI denoiser selection is adapted
            during optimisation.
        _peak_tol_min (float): Minimum allowed relative peak-variation
            threshold.
        _peak_tol (float): Current relative peak-variation threshold.
        _peak_tol_step (float): Multiplicative decay factor applied to the
            peak-variation threshold.
        _peak_val_range (list): Current valid peak-value range returned by the
            denoiser selector.
        _verbose (bool): Whether informational logs are printed.
        _iter_save (int): Interval between intermediate FITS snapshots.
        _prev_peak_val (torch.Tensor | float | None): Previous channel-wise
            image peak estimate.
        _im_rel_var (float): Current relative image variation.

    Example:
        >>> optimiser = FBAIRI(
        ...     meas=meas,
        ...     ch_freq=ch_freq,
        ...     meas_op=meas_op,
        ...     regulariser=regulariser,
        ... )
        >>> optimiser.initialisation()
        >>> optimiser.run()
        >>> optimiser.finalisation()

    References:
        Terris, M., Dabbech, A., Tang, C., & Wiaux, Y. (2023). `Image
        reconstruction algorithms in radio interferometry: From handcrafted to
        learned regularization denoisers <https://doi.org/10.1093/mnras/stac2672>`_.
        *MNRAS*, 518(1), 604-622.

        Terris, M., Tang, C., Jackson, A., & Wiaux, Y. (2025). `The AIRI
        plug-and-play algorithm for image reconstruction in radio-
        interferometry: variations and robustness
        <https://doi.org/10.1093/mnras/staf022>`_. *MNRAS*, 537(2), 1608-1619.
    """

    def __init__(
        self,
        meas: list[torch.Tensor],
        ch_freq: torch.Tensor,
        meas_op: MeasOpHyperspectral,
        regulariser: DenAIRI,
        rank_ch_start: int = 0,
        meas_op_approx: MeasOpHyperspectral | None = None,
        im_min_itr: int = 100,
        im_max_itr: int = 2000,
        im_var_tol: float = 1e-4,
        im_peak_est: float | None = None,
        heu_noise_scale: float = 1.0,
        adapt_net_select: bool = True,
        peak_tol_min: float = 1e-3,
        peak_tol_max: float = 0.1,
        peak_tol_step: float = 0.1,
        verbose: bool = True,
        save_pth: str = "results",
        file_prefix: str = "",
        iter_save: int = 1000,
    ) -> None:
        """Initialise the AIRI optimiser.

        Args:
            meas (List[torch.Tensor]): Local measurement tensors, typically one
                tensor per local channel.
            ch_freq (torch.Tensor): Frequencies for the local spectral
                channels.
            meas_op (MeasOpHyperspectral): Precise hyperspectral measurement
                operator used for diagnostics and final products.
            regulariser (DenAIRI): AIRI denoising regulariser.
            rank_ch_start (int, optional): Global starting channel index owned
                by the current rank. Default: 0.
            meas_op_approx (MeasOpHyperspectral | None, optional): Optional
                approximate measurement operator used during the iterative
                forward step. Default: None.
            im_min_itr (int, optional): Minimum number of iterations before
                convergence is checked. Default: 100.
            im_max_itr (int, optional): Maximum number of iterations.
                Default: 2000.
            im_var_tol (float, optional): Relative-variation tolerance.
                Default: 1e-4.
            im_peak_est (float | None, optional): User-provided image peak
                estimate. Default: None.
            heu_noise_scale (float, optional): Multiplicative scaling applied
                to heuristic noise levels. Default: 1.0.
            adapt_net_select (bool, optional): Adapt AIRI denoiser selection
                during optimisation. Default: True.
            peak_tol_min (float, optional): Minimum allowed relative
                peak-variation threshold. Default: 1e-3.
            peak_tol_max (float, optional): Initial relative peak-variation
                threshold. Default: 0.1.
            peak_tol_step (float, optional): Multiplicative decay factor for
                the peak-variation threshold. Default: 0.1.
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
            reference operator for dirty-image, PSF, peak estimation, and
            heuristic calculations.
        """
        super().__init__(
            meas,
            ch_freq,
            meas_op if meas_op_approx is None else meas_op_approx,
            regulariser,
            meas_op_precise=meas_op,
            heu_noise_scale=heu_noise_scale,
            im_max_itr=im_max_itr,
            save_pth=save_pth,
            file_prefix=file_prefix,
            rank_ch_start=rank_ch_start,
        )

        self._use_approx_meas_op = meas_op_approx is not None
        self._im_min_itr = im_min_itr
        self._im_var_tol = im_var_tol
        self._im_peak_est = im_peak_est
        self._adapt_net_select = adapt_net_select
        self._peak_tol_min = peak_tol_min
        self._peak_tol = peak_tol_max
        self._peak_tol_step = peak_tol_step
        self._peak_val_range = []
        self._verbose = verbose
        self._iter_save = iter_save

        self._prev_peak_val = None
        self._im_rel_var = 1.0

    def initialisation(self) -> None:
        """Initialise AIRI-specific optimisation parameters.

        This method sets the forward-backward step size, estimates channel-wise
        image peaks, computes heuristic noise levels, initialises the AIRI
        denoiser selector, and prints configuration information when verbose
        logging is enabled.
        """
        if self._is_distributed:
            dist.barrier()
        if self._rank == 0 and self._verbose:
            print("\nINFO: initialise AIRI optimiser...", flush=True)

        # Set the forward-backward step size from the precise operator norm.
        self._gd_step_size = 1.98 / self._meas_op_precise.get_op_norm()

        # Estimate channel-wise image peaks from the current dirty image.
        self._estimate_channel_wise_peak_val()

        # Estimate heuristic regularisation levels for local channels.
        self._calculate_channel_wise_heuristic()

        # Cache the threshold used by spectral-index estimation in finalisation.
        self._spec_idx_prec = self._heuristic.min()
        if self._is_distributed:
            dist.all_reduce(self._spec_idx_prec, op=dist.ReduceOp.MIN)
        self._spec_idx_prec = self._spec_idx_prec.item()

        # Initialise the AIRI denoiser and cache its supported peak range.
        self._peak_val_range = self._regulariser.update(
            self._heuristic, self._prev_peak_val
        )

        if self._is_distributed:
            dist.barrier()
        if self._verbose and self._rank == 0:
            print(
                f"INFO: forward-backward step size: {self._gd_step_size:.4e}",
                flush=True,
            )
            if self._use_approx_meas_op:
                print(
                    "INFO: use approximate measurement operator for reconstruction",
                    flush=True,
                )
            print("\n*************************************************", flush=True)
            print("********* STARTING ALGORITHM:    AIRI   *********", flush=True)
            print("*************************************************", flush=True)
        if self._is_distributed:
            dist.barrier()

    def _estimate_channel_wise_peak_val(self) -> None:
        """Estimate or set channel-wise image peak values.

        If ``self._im_peak_est`` is not provided, this method estimates peak
        values from the dirty image normalized by the PSF peak. The result is
        stored in ``self._prev_peak_val`` with shape ``(1, C_rank, 1, 1)``.
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

    def _each_iter_begin(self) -> None:
        """Run bookkeeping at the beginning of each iteration."""

    @torch.no_grad()
    def _stop_criteria(self) -> bool:
        """Evaluate the AIRI stopping criterion.

        This method computes the relative image variation across all ranks,
        reports iteration statistics when verbose logging is enabled, and
        terminates once the minimum iteration count has been reached and the
        relative variation falls below the tolerance.

        Returns:
            bool: True if the stopping criterion is met.
        """
        # Compute the relative image variation across all ranks.
        model_norm = torch.sum(self._model**2)
        model_diff_norm = torch.sum((self._model - self._model_prev) ** 2)
        if self._is_distributed:
            dist.all_reduce(model_norm, op=dist.ReduceOp.SUM)
            dist.all_reduce(model_diff_norm, op=dist.ReduceOp.SUM)
        self._im_rel_var = torch.sqrt(model_diff_norm / model_norm).item()

        if self._verbose and self._rank == 0:
            print(
                f"\nIter {self._iter+1}: "
                f"relative variation {self._im_rel_var:.3e}"
                f"\ntimings: gradient step {self._t_forward:#.3g} sec, "
                f"denoising step {self._t_backward:#.3g} sec, "
                f"iteration {self._t_iter:#.3g} sec.",
                flush=True,
            )

        # Stop once the minimum iteration count is reached and convergence
        # falls below the prescribed tolerance.
        if self._iter + 1 >= self._im_min_itr and self._im_rel_var < self._im_var_tol:
            return True

        return False

    @torch.no_grad()
    def _each_iter_end(self) -> None:
        """Save intermediate results and update AIRI denoiser selection."""
        # Save model and residual snapshots at the requested interval.
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

        # Update the AIRI denoiser if the peak estimate has moved enough and
        # leaves the current supported range.
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

    @torch.no_grad()
    def finalisation(self) -> None:
        """Finalise AIRI outputs and save the reconstructed products."""
        if self._verbose and self._rank == 0:
            print("\n**************************************", flush=True)
            print("********** END OF ALGORITHM **********", flush=True)
            print("**************************************\n", flush=True)
            print(
                f"Imaging finished in {self._t_total:#.4g} sec,",
                f"total number of iterations {self._iter+1}",
                flush=True,
            )

        # Save the final model and residual dirty image.
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

            # Generate a spectral-index map only for multi-channel outputs.
            if len(model_full_np.shape) > 2:
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
