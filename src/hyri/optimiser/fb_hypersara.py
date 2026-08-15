"""Define the Hyper-uSARA optimiser.

This module provides the Hyper-uSARA optimiser built on the shared
forward-backward backbone with positivity and reweighted sparse analysis
regularisation.
"""

import os
from timeit import default_timer as timer

import numpy as np
import torch
import torch.distributed as dist
from astropy.io import fits

from ..interface.spectral_index import get_spec_idx
from ..measurement_operator import MeasOpHyperspectral
from ..regulariser import ProxHyperSARAPos
from .forward_backward import ForwardBackward


class FBHyperSARA(ForwardBackward):
    """Represent the Hyper-uSARA forward-backward optimiser.

    This class combines the shared forward-backward optimisation loop with a
    hyperspectral reweighted SARA prior under positivity. The optimisation is
    organised into inner forward-backward iterations and outer major cycles
    that refresh the wavelet-analysis weights.

    Shape:
        Image tensor: ``(1, C_rank, H, W)``.
        Visibility tensor: tuple of ``C_rank`` tensors, each with shape
            ``(1, 1, N_c)``.
        Frequency tensor: ``(1, C_total, 1, 1)`` after reshaping in the base
            optimiser.

    Attributes:
        _use_approx_meas_op (bool): Whether an approximate measurement operator
            is used in the iterative forward step.
        _im_max_itr_inner (int): Maximum number of iterations per major cycle.
        _im_min_itr (int): Minimum number of inner iterations before checking
            convergence.
        _im_var_tol (float): Relative-variation tolerance for the inner loop.
        _im_max_itr_outer (int): Maximum number of reweighting major cycles.
        _im_var_tol_outer (float): Relative-variation tolerance for the outer
            reweighting loop.
        _reweight_save (bool): Whether intermediate products are saved at the
            end of each major cycle.
        _verbose (bool): Whether informational logs are printed.
        _heuristic_l2 (torch.Tensor): Channel-collapsed heuristic level used
            by the hyperspectral regulariser.
        _model_prev_re (torch.Tensor): Model image saved at the start of the
            current major cycle.
        _iter_inner (int): Inner forward-backward iteration index within the
            current major cycle.
        _iter_outer (int): Major-cycle index for reweighting.
        _reweighting_flag (bool): Whether the current iteration ended a major
            cycle and triggered reweighting.
        _t_iter_outer (float): Elapsed time of the current major cycle.
        _im_rel_var (float): Current relative image variation in the inner
            loop.
        _im_rel_var_outer (float): Relative image variation between two
            successive major cycles.

    Example:
        >>> optimiser = FBHyperSARA(
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
        regulariser: ProxHyperSARAPos,
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
        """Initialise the Hyper-uSARA optimiser.

        Args:
            meas (list[torch.Tensor]): Local measurement tensors, typically one
                tensor per local channel.
            ch_freq (torch.Tensor): Frequencies for all spectral channels.
            meas_op (MeasOpHyperspectral): Precise hyperspectral measurement
                operator used for diagnostics and final products.
            regulariser (ProxHyperSARAPos): Hyper-uSARA proximal regulariser.
            rank_ch_start (int, optional): Global starting channel index owned
                by the current rank. Default: 0.
            meas_op_approx (MeasOpHyperspectral | None, optional): Optional
                approximate measurement operator used during the iterative
                forward step. Default: None.
            im_min_itr (int, optional): Minimum number of inner-loop
                iterations before checking convergence. Default: 30.
            im_max_itr (int, optional): Maximum number of inner-loop
                iterations per major cycle. Default: 2000.
            im_var_tol (float, optional): Relative-variation tolerance for the
                inner loop. Default: 1e-5.
            heu_reg_scale (float, optional): Multiplicative scaling applied to
                the heuristic regularisation levels. Default: 1.0.
            im_max_itr_outer (int, optional): Maximum number of reweighting
                major cycles. Default: 20.
            im_var_tol_outer (float, optional): Relative-variation tolerance
                for the outer reweighting loop. Default: 1e-4.
            save_pth (str, optional): Directory where optimiser products are
                saved. Default: ``"results"``.
            file_prefix (str, optional): Prefix prepended to saved file names.
                Default: ``""``.
            reweight_save (bool, optional): Save intermediate products at the
                end of each major cycle. Default: True.
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
        """Initialise Hyper-uSARA-specific optimisation parameters.

        This method sets the forward-backward step size, computes the
        channel-wise and collapsed heuristic levels, configures the wavelet
        noise floor and soft-thresholding levels, and initialises the first
        set of reweighting coefficients.
        """
        if self._is_distributed:
            dist.barrier()
        if self._rank == 0 and self._verbose:
            print("\nINFO: initialise Hyper-uSARA optimiser...", flush=True)

        # Set the forward-backward step size from the precise operator norm.
        self._gd_step_size = 1.98 / self._meas_op_precise.get_op_norm()

        # Estimate heuristic regularisation levels for local channels.
        self._calculate_channel_wise_heuristic()
        # Collapse the channel-wise heuristic into the l2 image heuristic.
        self._heuristic_l2 = torch.sum(self._heuristic**2, dim=1, keepdim=True)
        if self._is_distributed:
            self._heuristic_l2 = self._heuristic_l2.to(self._dist_device)
            dist.all_reduce(self._heuristic_l2, op=dist.ReduceOp.SUM)
            self._heuristic_l2 = self._heuristic_l2.to(self._regulariser.get_device())
        self._heuristic_l2 = torch.sqrt(self._heuristic_l2)

        # Set the wavelet-domain noise floor independently of heuristic scaling.
        # The factor 3.0 corresponds to the nine wavelet bases used internally.
        wl_noise_floor = self._heuristic_l2 / 3.0 / self._heu_noise_scale
        self._regulariser.set_noise_floor_level(wl_noise_floor)
        # Set the soft-thresholding level used by the proximal operator.
        sfth_level = self._heuristic_l2 / 3.0
        self._regulariser.set_soft_thresholding_value(sfth_level)

        # Cache the threshold used by spectral-index estimation in finalisation.
        self._spec_idx_prec = self._heuristic.min()
        if self._is_distributed:
            dist.all_reduce(self._spec_idx_prec, op=dist.ReduceOp.MIN)
        self._spec_idx_prec = self._spec_idx_prec.item()

        # Initialise the reweighting coefficients from the current model.
        self._regulariser.update(self._model)

        # Reset the nested-loop counters and cached model state.
        self._model_prev_re = self._model
        self._iter_inner = 0
        self._iter_outer = 0
        self._reweighting_flag = False

        if self._is_distributed:
            dist.barrier()
        if self._verbose and self._rank == 0:
            print(
                f"INFO: l2 norm of the heuristic noise level: "
                f"{self._heuristic_l2.squeeze().cpu().item():.4e}",
                flush=True,
            )
            print(
                f"INFO: forward-backward step size (gamma): "
                f"{self._gd_step_size:.4e}",
                flush=True,
            )
            if self._use_approx_meas_op:
                print(
                    "INFO: use approximate measurement operator for reconstruction",
                    flush=True,
                )
            reg_param = self._heuristic_l2 / 3.0 / self._gd_step_size
            print(
                f"INFO: regularization parameter (lambda): "
                f"{reg_param.squeeze().cpu().item():.4e}",
                flush=True,
            )
            print(
                f"INFO: heuristic soft thresholding level (gamma x lambda): "
                f"{sfth_level.squeeze().cpu().item():.4e}",
                flush=True,
            )
            print(
                f"INFO: estimated noise floor level in l2 wavelet coefficients: "
                f"{wl_noise_floor.squeeze().cpu().item():.4e}",
                flush=True,
            )
            print("\n*************************************************", flush=True)
            print("****** STARTING ALGORITHM:   Hyper-uSARA   ******", flush=True)
            print("*************************************************", flush=True)

        if self._is_distributed:
            dist.barrier()

    def _each_iter_begin(self) -> None:
        """Run logic at the beginning of each iteration.

        This method resets the outer-loop timer when a new major cycle starts.
        """
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
        """Evaluate the Hyper-uSARA stopping criteria.

        The algorithm stops only after both the inner forward-backward loop and
        the outer reweighting loop satisfy their respective stopping
        conditions.

        Returns:
            bool: ``True`` when the overall optimisation should stop,
            otherwise ``False``.
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
                f"Cumulative itr: {self._iter+1}, "
                f"re-weighting itr: {self._iter_outer+1}, "
                f"forward-backward itr: {self._iter_inner+1}, "
                f"relative variation {self._im_rel_var:.3e}"
                f"\ntimings: gradient step {self._t_forward:#.3g} sec, "
                f"denoising step {self._t_backward:#.3g} sec, "
                f"iteration {self._t_iter:#.3g} sec.\n",
                flush=True,
            )

        if (
            self._iter_inner + 1 >= self._im_min_itr
            and self._im_rel_var < self._im_var_tol
        ) or self._iter_inner + 1 >= self._im_max_itr_inner:
            # The inner loop has converged or reached the maximum number of iterations
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
                    f"************************** Major cycle "
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
                # The outer loop has converged or reached the maximum
                # number of major cycles
                return True

        return False

    @torch.no_grad()
    def _each_iter_end(self) -> None:
        """Run post-processing at the end of each iteration.

        When an inner loop finishes, this method refreshes the reweighting
        coefficients, optionally saves intermediate products, and advances the
        major-cycle counters.
        """
        if self._reweighting_flag:
            # Refresh the reweighting coefficients for the next major cycle.
            self._regulariser.update(self._model)

            if self._verbose:
                residual_full_np = self.get_residual_image(dtype=torch.float32)
                if self._rank == 0:
                    print(
                        "INFO: The std of the residual dirty image",
                        f"{np.std(residual_full_np).item():.3e}",
                        flush=True,
                    )

            # Save intermediate products at the end of each major cycle.
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
        """Finalise Hyper-uSARA and save the reconstructed products.

        This method prints final log information, saves the reconstructed model
        and residual dirty images, and exports a spectral-index map when the
        final model has a spectral dimension.
        """
        if self._verbose and self._rank == 0:
            print("\n**************************************", flush=True)
            print("********** END OF ALGORITHM **********", flush=True)
            print("**************************************\n", flush=True)
            print(
                f"Imaging finished in {self._t_total:.1f} sec,",
                f"total number of iterations {self._iter + 1}",
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
