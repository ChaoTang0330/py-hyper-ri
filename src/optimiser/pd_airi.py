"""
Constrained AIRI built on primal-dual algorithm
"""

import os
from typing import Optional

import torch
import torch.distributed as dist
from astropy.io import fits

from .primal_dual import PrimalDualPrecond
from ..regulariser import DenAIRI, ProxEllipsoid
from ..ri_measurement_operator import MeasOpHyperspectral
from ..interface import get_spec_idx


class PDAIRI(PrimalDualPrecond):
    """
    This class implements constrained AIRI built on the primal-dual algorithm.

    It handles data fidelity in the dual step and uses AIRI denoisers in the
    primal step for regularization.
    """

    def __init__(
        self,
        meas: torch.Tensor,
        ch_freq: torch.Tensor,
        meas_op: MeasOpHyperspectral,
        regulariser_prime: DenAIRI,
        regulariser_dual_data: ProxEllipsoid,
        rank_ch_start: int = 0,
        im_min_itr: int = 100,
        im_max_itr: int = 2000,
        im_var_tol: float = 1e-4,
        im_peak_est: Optional[float] = None,
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
        """
        Initialises the PDAIRI class.

        Args:
            meas (torch.Tensor): The measurement tensor.
            meas_op (MeasOpHyperspectral): The measurement operator.
            regulariser_prime (DenAIRI): The denoising operator for the prime variable.
            regulariser_dual_data (ProxEllipsoid): The proximal operator for the dual data.
            im_min_itr (int, optional): The minimum number of iterations. Defaults to 100.
            im_max_itr (int, optional): The maximum number of iterations. Defaults to 2000.
            im_var_tol (float, optional): The image variation tolerance. Defaults to 1e-4.
            im_peak_est (Optional[float], optional): The estimated peak of the image.
                Defaults to None.
            heu_noise_scale (float, optional): The heuristic noise scale. Defaults to 1.0.
            adapt_net_select (bool, optional): Whether to adaptively select the network.
                Defaults to True.
            peak_tol_min (float, optional): The minimum peak tolerance. Defaults to 1e-3.
            peak_tol_max (float, optional): The maximum peak tolerance. Defaults to 0.1.
            peak_tol_step (float, optional): The step size for peak tolerance. Defaults to 0.1.
            verbose (bool, optional): Whether to print verbose output. Defaults to True.
            save_pth (str, optional): The path to save the results. Defaults to "results".
            file_prefix (str, optional): The prefix of the saving files. Defaults to "".
            iter_save (int, optional): The number of iterations after which to save the results.
                Defaults to 1000.
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
        """
        Initialises specific parameters of the algorithm including step size,
        the maximum intensity of the target image, and heuristic noise level.
        The AIRI proximal operator is also initialised.
        """
        if self._is_distributed:
            dist.barrier()
        if self._rank == 0 and self._verbose:
            print("\nINFO: Initialisation cAIRI optimiser...", flush=True)

        # calculate step size
        self._meas_op.set_real_flag(False)
        self._precond_weight = self._regulariser_dual_data.get_precond_weight()
        precond_weight_sqrt = tuple(torch.sqrt(w) for w in self._precond_weight)
        self._meas_op.set_precond_weight(precond_weight_sqrt)
        self._pr_step_size = 0.5 / self._meas_op.get_op_norm()
        if self._verbose and self._rank == 0:
            print(
                f"INFO: measurement operator norm for primal-dual "
                f"{self._meas_op.get_op_norm()}",
                flush=True,
            )
        self._meas_op.set_precond_weight(
            tuple(
                    torch.ones(1, 1, device=self._meas_op.get_device())
                    for _ in range(self._rank_ch_num)
            )
        )
        self._meas_op.set_real_flag(True)

        # estimate peak value
        self._estimate_channel_wise_peak_val()

        # heuristic noise level
        self._calculate_channel_wise_heuristic()

        # threshold used by spectral-index estimation in finalisation
        self._spec_idx_prec = self._heuristic.min()
        if self._is_distributed:
            dist.all_reduce(self._spec_idx_prec, op=dist.ReduceOp.MIN)
        self._spec_idx_prec = self._spec_idx_prec.item()

        # initialise AIRI prox
        self._peak_val_range = self._regulariser_prime.update(
            self._heuristic, self._prev_peak_val
        )

        if self._is_distributed:
            dist.barrier()
        if self._verbose and self._rank == 0:
            print(f"INFO: primal step size: {self._pr_step_size}")
            print("\n*************************************************", flush=True)
            print("********* STARTING ALGORITHM:   cAIRI   *********", flush=True)
            print("*************************************************", flush=True)
        if self._is_distributed:
            dist.barrier()

    def _estimate_channel_wise_peak_val(self) -> None:
        """
        Estimates the peak value for each channel from the dirty image if the user does not
        specify it.

        The estimated peak value is saved in the attribute `_prev_peak_val`, which is a tensor
        of shape (1, C, 1, 1).
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
                    f"INFO: channel {self._rank_ch_start + 1} - "
                    f"{self._rank_ch_start + self._rank_ch_num}, "
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

    def _each_iter_begin(self):
        """
        Constrained AIRI does not require any specific operation at
        the beginning of each iteration.
        """

    @torch.no_grad()
    def _stop_criteria(self):
        """
        Determines the stop criteria for the algorithm based on image relative variation
        and data fidelity.

        Returns:
            bool: Whether the stop criteria has been met.
        """
        # img relative variation
        model_norm = torch.sum(self._model**2)
        model_diff_norm = torch.sum((self._model - self._model_prev) ** 2)
        # gather norms from all ranks
        if self._is_distributed:
            dist.all_reduce(model_norm, op=dist.ReduceOp.SUM)
            dist.all_reduce(model_diff_norm, op=dist.ReduceOp.SUM)
        self._im_rel_var = torch.sqrt(model_diff_norm / model_norm).item()

        # stop criteria
        if self._iter + 1 >= self._im_min_itr and self._im_rel_var < self._im_var_tol:
            curr_meas_bp = self._meas_op.forward_op(self._model)
            curr_l2_error_sq = torch.stack(
                [
                    torch.sum(torch.abs(self._meas[i] - curr_meas_bp[i]) ** 2)
                    for i in range(self._rank_ch_num)
                ],
                device=self._meas_op.get_device(),
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

            # log
            if self._verbose and self._rank == 0:
                print(
                    f"\nIter {self._iter+1}: relative variation {self._im_rel_var},",
                    f"data fidelity {curr_l2_error_sum}\ntimings: primal step {self._t_primal} sec,",
                    f"dual step {self._t_dual} sec, iteration {self._t_iter} sec.",
                    flush=True,
                )
            if curr_l2_error_valid_all:
                return True
        elif self._verbose and self._rank == 0:
            print(
                f"\nIter {self._iter+1}: relative variation {self._im_rel_var}",
                f"\ntimings: primal step {self._t_primal} sec, dual step {self._t_dual} sec,",
                f"iteration {self._t_iter} sec.",
                flush=True,
            )

        return False

    @torch.no_grad()
    def _each_iter_end(self):
        """
        Saves intermediate results for every `iter_save` iteration and updates the
        AIRI proximal operator with the latest image peak value if the relative
        variation of the peak value is smaller than certain tolerence.
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

        # AIRI denoiser selection
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

    @torch.no_grad()
    def finalisation(self):
        """
        Finalises the algorithm by printing the total time and number of iterations.
        The final model and residual images will be save to `save_pth`.
        """
        if self._verbose and self._rank == 0:
            print("\n**************************************", flush=True)
            print("********** END OF ALGORITHM **********", flush=True)
            print("**************************************\n", flush=True)
            print(
                f"Imaging finished in {self._t_total} sec, ",
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
