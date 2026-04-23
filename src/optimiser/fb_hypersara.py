"""
uSARA algorithm, built on forward-backward algorithm
"""

import os
from typing import Optional
from timeit import default_timer as timer

import torch
import torch.distributed as dist
import numpy as np
from astropy.io import fits

from .forward_backward import ForwardBackward
from ..regulariser import ProxHyperSARAPos
from ..measurement_operator import MeasOpHyperspectral
from ..interface.spectral_index import get_spec_idx


class FBHyperSARA(ForwardBackward):
    """
    FBHyperSARA implements Hyper-uSARA algorithm based on forward-backward algorithm.

    It handles data fidelity in the forward step and uses SARA prior and
    positivity constrained in the backward step for regularization.
    """

    def __init__(
        self,
        meas: torch.Tensor,
        ch_freq: torch.Tensor,
        meas_op: MeasOpHyperspectral,
        regulariser: ProxHyperSARAPos,
        rank_ch_start: int = 0,
        meas_op_approx: Optional[MeasOpHyperspectral] = None,
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
        """
        Initializes the Hyper-uSARA optimiser, which is built on forward-backward algorithm.

        Args:
            meas (torch.Tensor): The measurements to be used in the algorithm.
            meas_op (MeasOpHyperspectral): The measurement operator.
            regulariser (ProxHyperSARAPos): The proximal operator.
            im_min_itr (int, optional): The minimum number of inner loop iterations.
                Defaults to 30.
            im_max_itr (int, optional): The maximum number of inner loop iterations.
                Defaults to 2000.
            im_var_tol (float, optional): The tolerance for inner loop image variation.
                Defaults to 1e-5.
            heu_reg_scale (float, optional): The heuristic regularization scale. Defaults to 1.0.
            im_max_itr_outer (int, optional): The maximum number of outer loop iterations.
                Defaults to 20.
            im_var_tol_outer (float, optional): The tolerance for outer loop image variation.
                Defaults to 1e-4.
            save_pth (str, optional): The path to save the results. Defaults to "results".
            file_prefix (str, optional): The prefix of the saving files. Defaults to "".
            reweight_save (bool, optional): Whether to save the reweighting results.
            Defaults to True.
            verbose (bool, optional): Whether to print verbose logs. Defaults to True.
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
        """
        Initialises specific parameters of Hyper-uSARA.

        This method initialises the step size and the Hyper-uSARA proximity operator,
        sets the heuristic regularization parameter and the noise floor level in
        the wavelet domain. Some information will be printed if `verbose` is True.
        """
        if self._is_distributed:
            dist.barrier()
        if self._rank == 0 and self._verbose:
            print("\nINFO: initialise Hyper-uSARA optimiser...", flush=True)

        # step size
        self._gd_step_size = 1.98 / self._meas_op_precise.get_op_norm()

        # heuristic noise level
        self._calculate_channel_wise_heuristic()
        # heuristic noise level for the l2 image
        self._heuristic_l2 = torch.sum(self._heuristic**2, dim=1, keepdim=True)
        if self._is_distributed:
            self._heuristic_l2 = self._heuristic_l2.to(self._dist_device)
            dist.all_reduce(self._heuristic_l2, op=dist.ReduceOp.SUM)
            self._heuristic_l2 = self._heuristic_l2.to(self._regulariser.get_device())
        self._heuristic_l2 = torch.sqrt(self._heuristic_l2)
            

        # set noise floor level in wavelet domain
        # The noise floor level is not affected by the heuristic scaling factor
        # divided by 3.0 -> 9 wavelet bases
        wl_noise_floor = self._heuristic_l2 / 3.0 / self._heu_noise_scale
        self._regulariser.set_noise_floor_level(wl_noise_floor)
        # set soft thresholding value
        sfth_level = self._heuristic_l2 / 3.0
        self._regulariser.set_soft_thresholding_value(sfth_level)

        # threshold used by spectral-index estimation in finalisation
        self._spec_idx_prec = self._heuristic.min()
        if self._is_distributed:
            dist.all_reduce(self._spec_idx_prec, op=dist.ReduceOp.MIN)
        self._spec_idx_prec = self._spec_idx_prec.item()

        # initialise weighting for the l1 norm
        self._regulariser.update(self._model)

        # other parameters
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
                f"INFO: forward-backward step size (gamma): {self._gd_step_size:.4e}",
                flush=True,
            )
            if self._use_approx_meas_op:
                print(
                    "INFO: use approximate measurement operator for reconstruction",
                    flush=True,
                )
            reg_param = self._heuristic_l2 / 3.0 / self._gd_step_size
            print(
                f"INFO: regularization parameter (lambda): {reg_param.squeeze().cpu().item():.4e}",
                flush=True,
            )
            print(
                f"INFO: heuristic soft threolding level (gamma x lambda): "
                f"{sfth_level.squeeze().cpu().item():.4e}",
                flush=True,
            )
            print(
                f"INFO: estimated noise floor level in l2 wavelet coeeficients: "
                f"{wl_noise_floor.squeeze().cpu().item():.4e}",
                flush=True,
            )
            print("\n*************************************************", flush=True)
            print("****** STARTING ALGORITHM:   Hyper-uSARA   ******", flush=True)
            print("*************************************************", flush=True)

        if self._is_distributed:
            dist.barrier()

    def _each_iter_begin(self) -> None:
        """
        Method to be executed at the beginning of each iteration.

        This method resets the timer at the start of each reweighting loop.
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
        """
        Checks the stop criteria for the algorithm.

        The imaging loop will be terminated if the stop criteria of both inner loop
        and reweighting loop are met. The stop criteria are: 1) meets the minimum number
        of iterations and 2) the relative variation is smaller than tolerance or the
        maximum number of iterations is reached. The `_reweighting_flag` will be set if
        the inner loop stop criteria is met.

        Returns:
            bool: Whether the stop criteria is met.
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
                f"Cumulative itr: {self._iter+1},  re-weighting itr: {self._iter_outer+1},",
                f"forward-backward itr: {self._iter_inner+1},",
                f"relative variation {self._im_rel_var:.3e}",
                f"\ntimings: gradient step {self._t_forward:#.3g} sec,",
                f"denoising step {self._t_backward:#.3g} sec,",
                f"iteration {self._t_iter:#.3g} sec.\n",
                flush=True,
            )

        # stop criteria
        if (
            self._iter_inner + 1 >= self._im_min_itr
            and self._im_rel_var < self._im_var_tol
        ) or self._iter_inner + 1 >= self._im_max_itr_inner:
            self._t_iter_outer = timer() - self._t_iter_outer
            self._reweighting_flag = True
            model_diff_norm = torch.sum((self._model - self._model_prev_re) ** 2)
            if self._is_distributed:
                dist.all_reduce(model_diff_norm, op=dist.ReduceOp.SUM)
            self._im_rel_var_outer = torch.sqrt(
                model_diff_norm / (model_norm + 1e-12)
            ).item()
            self._model_prev_re = self._model
            # log
            if self._verbose and self._rank == 0:
                print(
                    "************************** Major cycle",
                    f"{self._iter_outer+1} finished **************************",
                    flush=True,
                )
                print(
                    f"INFO: Re-weighting iteration {self._iter_outer+1}",
                    f"completed in {self._t_iter_outer:#.4g} sec.",
                    flush=True,
                )
                print(
                    f"INFO: Image relative variation of the major cycle {self._im_rel_var_outer:#.3g}",
                    flush=True,
                )

            if (
                self._im_rel_var_outer < self._im_var_tol_outer
                or self._iter_outer + 1 >= self._im_max_itr_outer
            ):
                return True

        return False

    @torch.no_grad()
    def _each_iter_end(self) -> None:
        """
        Method to be executed at the end of each iteration.

        This method saves intermediate results, and updates the HyperSARA wavelet coeeficients
        weights if the reweighting flag is set. The inner and outer iteration counters will
        also be updated accordingly.
        """
        if self._reweighting_flag:
            # update l1 weighting
            self._regulariser.update(self._model)

            if self._verbose:
                residual_full_np = self.get_residual_image(dtype=torch.float32)
                if self._rank == 0:
                    print(
                        "INFO: The std of the residual dirty image",
                        f"{np.std(residual_full_np).item():.3e}",
                        flush=True,
                    )

            # save intermediate results
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
        """
        Finalises the imaging process and saves the final images.

        This method calculates the residual image, prints information about
        the process if verbose mode is on, and saves the final model and
        residual images in FITS format.
        """
        if self._verbose and self._rank == 0:
            print("\n**************************************", flush=True)
            print("********** END OF ALGORITHM **********", flush=True)
            print("**************************************\n", flush=True)
            print(
                f"Imaging finished in {self._t_total:#.4g} sec,",
                f"total number of iterations {self._iter + 1}",
                flush=True,
            )

        # save final images
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

            # generate spectral index map if applicable
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
