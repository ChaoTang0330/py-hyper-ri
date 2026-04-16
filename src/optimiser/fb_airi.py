"""
Unconstrained AIRI, built on forward-backward algorithm
"""

import os
from typing import Optional

import torch
import torch.distributed as dist
from astropy.io import fits

from .forward_backward import ForwardBackward
from ..regulariser import DenAIRI
from ..ri_measurement_operator import MeasOpHyperspectral
from ..interface import get_spec_idx


class FBAIRI(ForwardBackward):
    """
    FBAIRI implements the AIRI algorithm based on Forward-Backward.

    It handles data fidelity in the forward step and uses AIRI denoisers
    in the backward step for regularization.
    """

    def __init__(
        self,
        meas: torch.Tensor,
        ch_freq: torch.Tensor,
        meas_op: MeasOpHyperspectral,
        regulariser: DenAIRI,
        rank_ch_start: int = 0,
        meas_op_approx: Optional[MeasOpHyperspectral] = None,
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
        Initializes the AIRI optimiser, which is built on forward-backward algorithm.

        Args:
            meas (torch.Tensor): The measurement tensor.
            meas_op (MeasOpHyperspectral): The measurement operator.
            regulariser (DenAIRI): The denoising operator.
            meas_op_approx (Union[MeasOpHyperspectral, None], optional): The approximate measurement
                operator build on oversampled PSF. The AIRI algorithm will use this operator
                for reconstruction if it is not None. Defaults to None.
            im_min_itr (int, optional): The minimum number of iterations. Defaults to 100.
            im_max_itr (int, optional): The maximum number of iterations. Defaults to 2000.
            im_var_tol (float, optional): The tolerance for image variation. Defaults to 1e-4.
            im_peak_est (float, optional): The estimated peak of the image. Defaults to None.
            heu_noise_scale (float, optional): The heuristic noise scale. Defaults to 1.0.
            adapt_net_select (bool, optional): Whether to adapt network selection. Defaults to True.
            peak_tol_min (float, optional): The minimum tolerance for peak. Defaults to 1e-3.
            peak_tol_max (float, optional): The maximum tolerance for peak. Defaults to 0.1.
            peak_tol_step (float, optional): The step size for peak tolerance. Defaults to 0.1.
            verbose (bool, optional): Whether to print verbose logs. Defaults to True.
            save_pth (str, optional): The path where results will be saved. Defaults to "results".
            file_prefix (str, optional): The prefix of the saving files. Defaults to "".
            iter_save (int, optional): The number of iterations after which to save results.
                Defaults to 1000.
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
        """
        Initialises specific parameters of AIRI.

        This method initialises the step size, estimates the peak value, sets the
        heuristic noise level, initialises the AIRI proximal operator, and prints
        some information if verbose is True.
        """
        if self._is_distributed:
            dist.barrier()
        if self._rank == 0 and self._verbose:
            print("\nINFO: Initialisation AIRI optimiser...", flush=True)

        # step size
        self._gd_step_size = 1.98 / self._meas_op_precise.get_op_norm()

        # estimate peak value
        self._estimate_channel_wise_peak_val()

        # heuristic noise level
        self._calculate_channel_wise_heuristic()

        # initialise AIRI prox
        self._peak_val_range = self._regulariser.update(
            self._heuristic, self._prev_peak_val
        )

        if self._is_distributed:
            dist.barrier()
        if self._verbose and self._rank == 0:
            print(f"INFO: forward-backward step size: {self._gd_step_size}", flush=True)
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
                peak_val_list = self._prev_peak_val.squeeze().cpu().tolist()
                if not isinstance(peak_val_list, list):
                    peak_val_list = [peak_val_list]
                print(
                    f"INFO: channel {self._rank_ch_start + 1} - "
                    f"{self._rank_ch_start + self._rank_ch_num}, "
                    f"estimated image peak value [{', '.join(f'{val:#.4g}' for val in peak_val_list)}]",
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
                peak_val_list = self._prev_peak_val.squeeze().cpu().tolist()
                if not isinstance(peak_val_list, list):
                    peak_val_list = [peak_val_list]
                print(
                    f"INFO: channel {self._rank_ch_start + 1} - {self._rank_ch_start + self._rank_ch_num}, "
                    f"specified image peak value [{', '.join(f'{val:#.4g}' for val in peak_val_list)}]",
                    flush=True,
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
                f"\nIter {self._iter+1}: relative variation {self._im_rel_var:.3e}",
                f"\ntimings: gradient step {self._t_forward:#.3g} sec,",
                f"denoising step {self._t_backward:#.3g} sec,"
                f"iteration {self._t_iter:#.3g} sec.",
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

        This method saves intermediate results, and updates the AIRI denoiser selection
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
                        f"  Model image peak value: ch first {curr_peak_val_ch_1:#.3g}, "
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
        """
        Finalises the FBAIRI optimiser.

        This method prints final log information if `verbose` is True, and saves
        the result model and residual images.
        """
        if self._verbose and self._rank == 0:
            print("\n**************************************", flush=True)
            print("********** END OF ALGORITHM **********", flush=True)
            print("**************************************\n", flush=True)
            print(
                f"Imaging finished in {self._t_total:#.4g} sec,",
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
