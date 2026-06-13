"""Define the uSARA optimiser built on the forward-backward backbone.

This module provides the reweighted sparse-analysis optimiser used for
hyperspectral RI imaging with positivity and SARA regularisation.
"""

import os
from timeit import default_timer as timer

import numpy as np
import torch
import torch.distributed as dist
from astropy.io import fits

from ..interface.spectral_index import get_spec_idx
from ..measurement_operator import MeasOpHyperspectral
from ..regulariser import ProxSARAPos
from .forward_backward import ForwardBackward


class FBSARA(ForwardBackward):
    """Represent the uSARA forward-backward optimiser.

    This class combines the shared forward-backward optimisation loop with the
    SARA reweighting strategy and a positivity-constrained proximal operator.
    It alternates inner forward-backward iterations with outer reweighting
    cycles until both inner-loop and outer-loop stopping criteria are met.

    Shape:
        Image tensor: ``(1, C_rank, H, W)``.
        Visibility tensor: tuple of ``C_rank`` tensors, each with shape
            ``(1, 1, N_c)``.
        Frequency tensor: ``(1, C_rank, 1, 1)`` after reshaping in the base
            optimiser.

    Attributes:
        _use_approx_meas_op (bool): Whether an approximate measurement operator
            is used in the iterative forward step.
        _im_max_itr_inner (int): Maximum number of inner forward-backward
            iterations per major cycle.
        _im_min_itr (int): Minimum number of inner iterations before checking
            convergence.
        _im_var_tol (float): Relative-variation tolerance for inner iterations.
        _im_max_itr_outer (int): Maximum number of reweighting cycles.
        _im_var_tol_outer (float): Relative-variation tolerance for
            reweighting cycles.
        _reweight_save (bool): Whether intermediate reweighting products are
            saved to disk.
        _verbose (bool): Whether informational logs are printed.
        _model_prev_re (torch.Tensor): Model image cached at the previous
            reweighting cycle.
        _iter_inner (int): Current inner-loop iteration index.
        _iter_outer (int): Current outer-loop iteration index.
        _reweighting_flag (bool): Whether a reweighting update should be
            applied at the end of the current iteration.
        _t_iter_outer (float): Elapsed time of the current reweighting cycle.
        _im_rel_var (float): Current inner-loop relative image variation.
        _im_rel_var_outer (float): Current outer-loop relative image
            variation.

    Example:
        >>> optimiser = FBSARA(
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
    """

    def __init__(
        self,
        meas: list[torch.Tensor],
        ch_freq: torch.Tensor,
        meas_op: MeasOpHyperspectral,
        regulariser: ProxSARAPos,
        rank_ch_start: int = 0,
        meas_op_approx: MeasOpHyperspectral | None = None,
        im_min_itr: int = 30,
        im_max_itr: int = 2000,
        im_var_tol: float = 1e-5,
        heu_reg_scale: float = 1.0,
        im_max_itr_outer: int = 20,
        im_var_tol_outer: float = 1e-4,
        save_pth: str = "results",
        file_prefix: str = "",
        reweight_save: bool = True,
        verbose: bool = True,
    ) -> None:
        """Initialise the uSARA optimiser.

        Args:
            meas (List[torch.Tensor]): Local measurement tensors, typically one
                tensor per local channel.
            ch_freq (torch.Tensor): Frequencies for the local spectral
                channels.
            meas_op (MeasOpHyperspectral): Precise hyperspectral measurement
                operator used for diagnostics and final products.
            regulariser (ProxSARAPos): Positivity-constrained SARA proximal
                operator.
            rank_ch_start (int, optional): Global starting channel index owned
                by the current rank. Default: 0.
            meas_op_approx (MeasOpHyperspectral | None, optional): Optional
                approximate measurement operator used during the iterative
                forward step. Default: None.
            im_min_itr (int, optional): Minimum number of inner-loop
                iterations. Default: 30.
            im_max_itr (int, optional): Maximum number of inner-loop
                iterations per reweighting cycle. Default: 2000.
            im_var_tol (float, optional): Inner-loop relative-variation
                tolerance. Default: 1e-5.
            heu_reg_scale (float, optional): Multiplicative scaling applied to
                heuristic regularisation levels. Default: 1.0.
            im_max_itr_outer (int, optional): Maximum number of reweighting
                cycles. Default: 20.
            im_var_tol_outer (float, optional): Outer-loop relative-variation
                tolerance. Default: 1e-4.
            save_pth (str, optional): Directory where optimiser products are
                saved. Default: ``"results"``.
            file_prefix (str, optional): Prefix prepended to saved file names.
                Default: ``""``.
            reweight_save (bool, optional): Save intermediate products after
                each reweighting cycle. Default: True.
            verbose (bool, optional): Enable informational logging.
                Default: True.

        Note:
            When ``meas_op_approx`` is provided, the approximate operator is
            used in the iterative forward step while ``meas_op`` remains the
            reference operator for dirty-image, PSF, and heuristic
            calculations.
        """
        super().__init__(
            meas,
            ch_freq,
            meas_op if meas_op_approx is None else meas_op_approx,
            regulariser,
            meas_op_precise=meas_op,
            heu_noise_scale=heu_reg_scale,
            im_max_itr=im_max_itr * im_max_itr_outer,
            save_pth=save_pth,
            file_prefix=file_prefix,
            rank_ch_start=rank_ch_start,
        )

        self._use_approx_meas_op = meas_op_approx is not None
        self._im_max_itr_inner = im_max_itr
        self._im_min_itr = im_min_itr
        self._im_var_tol = im_var_tol
        self._im_max_itr_outer = im_max_itr_outer
        self._im_var_tol_outer = im_var_tol_outer
        self._reweight_save = reweight_save
        self._verbose = verbose

        self._heuristic = 1.0
        self._model_prev_re = self._model
        self._iter_inner = 0
        self._iter_outer = 0
        self._reweighting_flag = False
        self._t_iter_outer = 0.0
        self._im_rel_var = 1.0
        self._im_rel_var_outer = 1.0

    def initialisation(self) -> None:
        """Initialise uSARA-specific optimisation parameters.

        This method sets the forward-backward step size, computes heuristic
        regularisation levels, configures the wavelet noise floor and soft
        thresholding value, and resets reweighting-loop state.
        """
        if self._is_distributed:
            dist.barrier()
        if self._rank == 0 and self._verbose:
            print("\nINFO: initialise uSARA optimiser...", flush=True)

        # Set the forward-backward step size from the precise operator norm.
        self._gd_step_size = 1.98 / self._meas_op_precise.get_op_norm()

        # Estimate the heuristic regularisation levels for local channels.
        self._calculate_channel_wise_heuristic()

        # The factor 1/3 maps the image-domain heuristic to one wavelet basis
        # out of the 9 bases used by the SARA dictionary.
        wl_noise_floor = self._heuristic / 3.0 / self._heu_noise_scale
        self._regulariser.set_noise_floor_level(wl_noise_floor)
        # Use the scaled heuristic directly for the soft-threshold level.
        sfth_level = self._heuristic / 3.0
        self._regulariser.set_soft_thresholding_value(sfth_level)

        # Cache the weakest channel heuristic for spectral-index masking.
        self._spec_idx_prec = self._heuristic.min()
        if self._is_distributed:
            dist.all_reduce(self._spec_idx_prec, op=dist.ReduceOp.MIN)
        self._spec_idx_prec = self._spec_idx_prec.item()

        # Initialise the reweighted l1 coefficients from the starting model.
        self._regulariser.update(self._model)

        # Reset inner-loop and reweighting-loop state.
        self._model_prev_re = self._model
        self._iter_inner = 0
        self._iter_outer = 0
        self._reweighting_flag = False

        if self._is_distributed:
            dist.barrier()
        if self._verbose:
            if self._rank == 0:
                print(
                    f"INFO: forward-backward step size (gamma): "
                    f"{self._gd_step_size:.4e}",
                    flush=True,
                )
                if self._use_approx_meas_op:
                    print(
                        "INFO: use approximate measurement operator "
                        "for reconstruction",
                        flush=True,
                    )
            if self._is_distributed:
                dist.barrier()
            reg_param = self._heuristic / 3.0 / self._gd_step_size
            reg_param_list = reg_param.squeeze().cpu().tolist()
            if not isinstance(reg_param_list, list):
                reg_param_list = [reg_param_list]
            print(
                f"INFO: channel {self._rank_ch_start + 1} - "
                f"{self._rank_ch_start + self._rank_ch_num}, "
                f"regularization parameter (lambda): "
                f"[{', '.join(f'{val:.4e}' for val in reg_param_list)}]\n",
                flush=True,
                end="",
            )
            if self._is_distributed:
                dist.barrier()
            sfth_level_list = sfth_level.squeeze().cpu().tolist()
            if not isinstance(sfth_level_list, list):
                sfth_level_list = [sfth_level_list]
            print(
                f"INFO: channel {self._rank_ch_start + 1} - "
                f"{self._rank_ch_start + self._rank_ch_num}, "
                f"heuristic soft thresholding level (gamma x lambda): "
                f"[{', '.join(f'{val:.4e}' for val in sfth_level_list)}]\n",
                flush=True,
                end="",
            )
            if self._is_distributed:
                dist.barrier()
            wl_noise_floor_list = wl_noise_floor.squeeze().cpu().tolist()
            if not isinstance(wl_noise_floor_list, list):
                wl_noise_floor_list = [wl_noise_floor_list]
            print(
                f"INFO: channel {self._rank_ch_start + 1} - "
                f"{self._rank_ch_start + self._rank_ch_num}, "
                f"estimated noise floor level in wavelet coefficients: "
                f"[{', '.join(f'{val:.4e}' for val in wl_noise_floor_list)}]\n",
                flush=True,
                end="",
            )
            if self._is_distributed:
                dist.barrier()
            if self._rank == 0:
                print("\n*************************************************", flush=True)
                print("********* STARTING ALGORITHM:   uSARA   *********", flush=True)
                print("*************************************************", flush=True)
        if self._is_distributed:
            dist.barrier()

    def _each_iter_begin(self) -> None:
        """Run bookkeeping at the beginning of each iteration."""
        if self._iter_inner == 0 and self._rank == 0 and self._verbose:
            print(
                "\n********************* Reweighting,",
                f"start major cycle {self._iter_outer+1}",
                "*********************\n",
                flush=True,
            )
            self._t_iter_outer = timer()

    @torch.no_grad()
    def _stop_criteria(self) -> bool:
        """Evaluate the inner-loop and reweighting stopping criteria.

        The imaging loop terminates when the current reweighting cycle has
        converged or reached its iteration budget and the outer reweighting
        loop has also met its own stopping criterion.

        Returns:
            bool: True if the global stopping criterion is met.
        """
        # Compute the inner-loop relative variation across all ranks.
        model_norm = torch.sum(self._model**2)
        model_diff_norm = torch.sum((self._model - self._model_prev) ** 2)
        if self._is_distributed:
            dist.all_reduce(model_norm, op=dist.ReduceOp.SUM)
            dist.all_reduce(model_diff_norm, op=dist.ReduceOp.SUM)
        self._im_rel_var = torch.sqrt(model_diff_norm / model_norm).item()

        if self._verbose and self._rank == 0:
            print(
                f"Cumulative itr: {self._iter+1},  "
                f"re-weighting itr: {self._iter_outer+1}, "
                f"forward-backward itr: {self._iter_inner+1}, "
                f"relative variation {self._im_rel_var:.3e}",
                f"\ntimings: gradient step {self._t_forward:#.3g} sec, "
                f"denoising step {self._t_backward:#.3g} sec, "
                f"iteration {self._t_iter:#.3g} sec.\n",
                flush=True,
            )

        # Promote the current iteration to a reweighting update once the inner
        # stopping criterion or the inner iteration budget is reached.
        if (
            self._iter_inner + 1 >= self._im_min_itr
            and self._im_rel_var < self._im_var_tol
        ) or self._iter_inner + 1 >= self._im_max_itr_inner:
            # Inner loop has converged or reached its maximum iteration count
            self._t_iter_outer = timer() - self._t_iter_outer
            self._reweighting_flag = True
            model_diff_norm = torch.sum((self._model - self._model_prev_re) ** 2)
            if self._is_distributed:
                dist.all_reduce(model_diff_norm, op=dist.ReduceOp.SUM)
            self._im_rel_var_outer = torch.sqrt(
                model_diff_norm / (model_norm + 1e-12)
            ).item()
            self._model_prev_re = self._model
            if self._verbose and self._rank == 0:
                print(
                    "************************** Major cycle",
                    f"{self._iter_outer+1} finished **************************",
                    flush=True,
                )
                print(
                    f"INFO: Re-weighting iteration {self._iter_outer+1} "
                    f"completed in {self._t_iter_outer:#.4g} sec.",
                    flush=True,
                )
                print(
                    f"INFO: Image relative variation of the major cycle "
                    f"{self._im_rel_var_outer:#.3g}",
                    flush=True,
                )

            if (
                self._im_rel_var_outer < self._im_var_tol_outer
                or self._iter_outer + 1 >= self._im_max_itr_outer
            ):
                # Outer loop has converged or reached its maximum iteration count
                return True

        return False

    @torch.no_grad()
    def _each_iter_end(self) -> None:
        """Update reweighting state and save intermediate products."""
        if self._reweighting_flag:
            # Refresh the reweighted l1 coefficients from the current model.
            self._regulariser.update(self._model)

            if self._verbose:
                residual_full_np = self.get_residual_image(dtype=torch.float32)
                if self._rank == 0:
                    print(
                        f"INFO: The std of the residual dirty image "
                        f"{np.std(residual_full_np).item():.3e}",
                        flush=True,
                    )

            # Save model and residual snapshots after each major cycle.
            if self._reweight_save:
                model_full_np = self.get_model_image(dtype=torch.float32)
                residual_full_np = self.get_residual_image(dtype=torch.float32)
                if self._rank == 0:
                    fits.writeto(
                        os.path.join(
                            self._save_pth,
                            self._file_prefix
                            + "tmp_model_major_itr_"
                            + str(self._iter_outer + 1)
                            + ".fits",
                        ),
                        model_full_np,
                        overwrite=True,
                    )
                    fits.writeto(
                        os.path.join(
                            self._save_pth,
                            self._file_prefix
                            + "tmp_residual_major_itr_"
                            + str(self._iter_outer + 1)
                            + ".fits",
                        ),
                        residual_full_np,
                        overwrite=True,
                    )

            self._iter_inner = 0
            self._iter_outer += 1
            self._reweighting_flag = False
        else:
            self._iter_inner += 1

    @torch.no_grad()
    def finalisation(self) -> None:
        """Finalise uSARA outputs and save the reconstructed products."""
        if self._verbose and self._rank == 0:
            print("\n**************************************", flush=True)
            print("********** END OF ALGORITHM **********", flush=True)
            print("**************************************\n", flush=True)
            print(
                f"Imaging finished in {self._t_total:#.4g} sec,",
                f"total number of iterations {self._iter + 1}",
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
