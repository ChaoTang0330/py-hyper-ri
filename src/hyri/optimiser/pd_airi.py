"""Define the constrained AIRI optimiser.

This module provides the cAIRI optimiser built on the shared primal-dual
backbone with an AIRI denoiser in the primal update and a constrained
data-fidelity prox in the dual update.
"""

import os

import torch
import torch.distributed as dist
from astropy.io import fits

from ..interface import get_spec_idx
from ..measurement_operator import MeasOpHyperspectral
from ..regulariser import DenAIRI, ProxEllipsoid
from .primal_dual import PrimalDualPrecond


class PDAIRI(PrimalDualPrecond):
    """Represent the constrained AIRI primal-dual optimiser.

    This class combines the shared primal-dual optimisation loop with an AIRI
    denoiser in the primal image update and an ellipsoidal data-fidelity prox
    in the dual visibility update. It can also adapt AIRI denoiser selection
    from channel-wise peak estimates as the reconstruction evolves.

    Shape:
        Image tensor: ``(1, C_rank, H, W)``.
        Visibility tensor: tuple of ``C_rank`` tensors, each with shape
            ``(1, 1, N_c)``.
        Frequency tensor: ``(1, C_total, 1, 1)`` after reshaping in the base
            optimiser.

    Attributes:
        _im_min_itr (int): Minimum number of iterations before convergence is
            checked.
        _im_var_tol (float): Relative-variation tolerance for the image model.
        _im_peak_est (float | None): User-provided image peak estimate, if
            any.
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
        >>> optimiser = PDAIRI(
        ...     meas=meas,
        ...     ch_freq=ch_freq,
        ...     meas_op=meas_op,
        ...     regulariser_prime=regulariser_prime,
        ...     regulariser_dual_data=regulariser_dual_data,
        ... )
        >>> optimiser.initialisation()
        >>> optimiser.run()
        >>> optimiser.finalisation()

    References:
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
        regulariser_prime: DenAIRI,
        regulariser_dual_data: ProxEllipsoid,
        rank_ch_start: int = 0,
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
        """Initialise the constrained AIRI optimiser.

        Args:
            meas (List[torch.Tensor]): Local measurement tensors, typically one
                tensor per local channel.
            ch_freq (torch.Tensor): Frequencies for all spectral channels.
            meas_op (MeasOpHyperspectral): Hyperspectral measurement operator.
            regulariser_prime (DenAIRI): AIRI denoiser used in the primal
                image update.
            regulariser_dual_data (ProxEllipsoid): Proximal operator applied
                to the dual data variable.
            rank_ch_start (int, optional): Global starting channel index owned
                by the current rank. Default: 0.
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
        """
        super().__init__(
            meas,
            ch_freq,
            meas_op,
            regulariser_prime,
            regulariser_dual_data,
            heu_noise_scale=heu_noise_scale,
            rank_ch_start=rank_ch_start,
            im_max_itr=im_max_itr,
            save_pth=save_pth,
            file_prefix=file_prefix,
        )

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
        """Initialise cAIRI-specific optimisation parameters.

        This method computes the primal step size from the preconditioned
        operator norm, estimates image peaks, computes heuristic noise levels,
        and initialises the AIRI denoiser selector.
        """
        if self._is_distributed:
            dist.barrier()
        if self._rank == 0 and self._verbose:
            print("\nINFO: initialise cAIRI optimiser...", flush=True)

        # Compute the primal step size with the square-root preconditioner.
        self._meas_op.set_real_flag(False)
        self._precond_weight = self._regulariser_dual_data.get_precond_weight()
        precond_weight_sqrt = tuple(torch.sqrt(w) for w in self._precond_weight)
        self._meas_op.set_precond_weight(precond_weight_sqrt)
        self._pr_step_size = 0.5 / self._meas_op.get_op_norm()
        if self._verbose and self._rank == 0:
            print(
                f"INFO: measurement operator norm for primal-dual "
                f"{self._meas_op.get_op_norm():.4e}",
                flush=True,
            )
        # Restore the operator to its default unpreconditioned configuration.
        self._meas_op.set_precond_weight(
            tuple(
                torch.ones(1, 1, device=self._meas_op.get_device())
                for _ in range(self._rank_ch_num)
            )
        )
        self._meas_op.set_real_flag(True)

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
        self._peak_val_range = self._regulariser_prime.update(
            self._heuristic, self._prev_peak_val
        )

        if self._is_distributed:
            dist.barrier()
        if self._verbose and self._rank == 0:
            print(f"INFO: primal step size: {self._pr_step_size:.4e}", flush=True)
            print("\n*************************************************", flush=True)
            print("********* STARTING ALGORITHM:   cAIRI   *********", flush=True)
            print("*************************************************", flush=True)
        if self._is_distributed:
            dist.barrier()

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

    def _each_iter_begin(self) -> None:
        """Run logic at the beginning of each iteration.

        No per-iteration pre-processing is required for cAIRI.
        """

    @torch.no_grad()
    def _stop_criteria(self) -> bool:
        """Evaluate the cAIRI stopping criterion.

        The stopping criterion checks both image stabilisation and whether the
        current model satisfies the ellipsoidal data-fidelity constraint.

        Returns:
            bool: ``True`` when the optimisation should stop, otherwise
            ``False``.
        """
        # Compute the relative variation of the image model.
        model_norm = torch.sum(self._model**2)
        model_diff_norm = torch.sum((self._model - self._model_prev) ** 2)
        # Gather the global norms across all ranks.
        if self._is_distributed:
            dist.all_reduce(model_norm, op=dist.ReduceOp.SUM)
            dist.all_reduce(model_diff_norm, op=dist.ReduceOp.SUM)
        self._im_rel_var = torch.sqrt(model_diff_norm / model_norm).item()

        if self._iter + 1 >= self._im_min_itr and self._im_rel_var < self._im_var_tol:
            # Evaluate the current data-fidelity residual against the constraint.
            curr_meas_model = self._meas_op.forward_op(self._model)
            curr_l2_error_sq = torch.stack(
                [
                    torch.sum(torch.abs(self._meas[i] - curr_meas_model[i]) ** 2)
                    for i in range(self._rank_ch_num)
                ],
            )
            curr_l2_error_sum = curr_l2_error_sq.sum()
            curr_l2_error_valid = (
                curr_l2_error_sq.sqrt() < self._regulariser_dual_data.get_radius()
            )
            curr_l2_error_valid_all = curr_l2_error_valid.all().int()
            if self._is_distributed:
                dist.all_reduce(curr_l2_error_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(curr_l2_error_valid_all, op=dist.ReduceOp.MIN)
            curr_l2_error_sum = curr_l2_error_sum.sqrt().item()
            curr_l2_error_valid_all = curr_l2_error_valid_all.item()

            if self._verbose and self._rank == 0:
                print(
                    f"\nIter {self._iter+1}: "
                    f"relative variation {self._im_rel_var:.3e}, "
                    f"data fidelity {curr_l2_error_sum:.3e}\n"
                    f"timings: primal step {self._t_primal:#.3g} sec, "
                    f"dual step {self._t_dual:#.3g} sec, "
                    f"iteration {self._t_iter:#.3g} sec.",
                    flush=True,
                )
            if curr_l2_error_valid_all:
                return True
        elif self._verbose and self._rank == 0:
            print(
                f"\nIter {self._iter+1}: "
                f"relative variation {self._im_rel_var:.3e} "
                f"\ntimings: primal step {self._t_primal:#.3g} sec, "
                f"dual step {self._t_dual:#.3g} sec, "
                f"iteration {self._t_iter:#.3g} sec.",
                flush=True,
            )

        return False

    @torch.no_grad()
    def _each_iter_end(self) -> None:
        """Run post-processing at the end of each iteration.

        This method saves intermediate products and updates the AIRI denoiser
        selection when the channel-wise peak estimates have stabilised enough.
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

        # Update AIRI denoiser selection from the current channel-wise peaks.
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
                self._peak_val_range = self._regulariser_prime.update(
                    ch_heuristic=self._heuristic,
                    ch_peak_est=self._prev_peak_val,
                )

    @torch.no_grad()
    def finalisation(self) -> None:
        """Finalise cAIRI and save the reconstructed products.

        This method prints final log information, saves the reconstructed model
        and residual dirty images, and exports a spectral-index map when the
        final model has a spectral dimension.
        """
        if self._verbose and self._rank == 0:
            print("\n**************************************", flush=True)
            print("********** END OF ALGORITHM **********", flush=True)
            print("**************************************\n", flush=True)
            print(
                f"Imaging finished in {self._t_total:#.4g} sec, ",
                f"total number of iterations {self._iter+1}",
                flush=True,
            )

        # Save the final model, residual, and optional spectral-index products.
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

            # Generate a spectral-index map only for hyperspectral outputs.
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
