"""
Primal-dual algorithm
"""

import os
from timeit import default_timer as timer

import torch
from torch.distributed import dist
import numpy as np
from astropy.io import fits

from .optimiser import Optimiser
from ..regulariser import Regulariser, ProxEllipsoid
from ..ri_measurement_operator import MeasOpHyperspectral


class PrimalDualPrecond(Optimiser):
    """
    Primal-dual algorithm with preconditioned dual data update.

    This implementation support solving objective function with two
    non-differentiable terms. The data-fidelity term with pre-conditioning is
    handeled in the dual update step. The regularization term is handeled in the
    primal update step.
    """

    def __init__(
        self,
        meas: torch.Tensor,
        meas_op: MeasOpHyperspectral,
        regulariser_prime: Regulariser,
        regulariser_dual_data: ProxEllipsoid,
        heu_noise_scale: float = 1.0,
        rank_ch_start: int = 0,
        im_max_itr: int = 2000,
        save_pth: str = "results",
        file_prefix: str = "",
    ) -> None:
        """
        Constructs all the necessary attributes for the PrimalDualPrecond object.

        Args:
            meas (torch.Tensor): The measurement tensor.
            meas_op (MeasOp): The measurement operator.
            regulariser_prime (ProxOp): The proximal operator for the prime variable.
            regulariser_dual_data (ProxOpElipse): The proximal operator for the dual data.
            rank_ch_start (int, optional): The starting channel for the rank. Defaults to 0.
            im_max_itr (int, optional): The maximum number of iterations. Defaults to 2000.
            save_pth (str, optional): The path to save the results. Defaults to "results".
            file_prefix (str, optional): The prefix of the saving files. Defaults to "".
        """
        super().__init__(
            meas,
            meas_op,
            save_pth=save_pth,
            file_prefix=file_prefix,
            rank_ch_start=rank_ch_start,
        )
        self._regulariser_prime = regulariser_prime
        self._regulariser_dual_data = regulariser_dual_data
        self._heu_noise_scale = heu_noise_scale
        self._start_iter = 0
        self._im_max_itr = im_max_itr
        self._pr_step_size = 1.0
        self._precond_weight = [
            torch.ones(1, 1, device=meas_op.get_device())
            for _ in range(self._rank_ch_num)
        ]
        self._dual = [torch.zeros_like(m) for m in self._meas]

        # timing
        self._iter = 1
        self._t_total = 0.0
        self._t_iter = 0.0
        self._t_primal = 0.0
        self._t_dual = 0.0

        # cuda event
        self._cuda_timing = False
        if self._meas_op.get_device() == torch.device("cuda"):
            self._cuda_timing = True

        # save dirty image and psf
        self._meas_bp = self._meas_op_precise.adjoint_op(self._meas).to(
            self._meas_op_precise.get_device()
        )
        self._psf = self._meas_op_precise.get_psf()
        self._psf_peak = self._psf.amax(dim=(2, 3), keepdim=True)

        dirty_image_np = self.get_dirty_image()
        psf_np = self.get_psf()
        if self._rank == 0:
            fits.writeto(
                os.path.join(self._save_pth, "dirty.fits"),
                dirty_image_np,
                overwrite=True,
            )
            fits.writeto(
                os.path.join(self._save_pth, "psf.fits"),
                psf_np,
                overwrite=True,
            )

        # heuristic noise level
        self._heuristic = None

    @torch.no_grad()
    def run(self) -> None:
        """
        Runs the main loop of primal-dual algorithm.

        This method iteratively updates the prime and dual variables until the
        stop criteria is met or the maximum number of iterations is reached.
        """
        if self._cuda_timing:
            primal_start_event = torch.cuda.Event(enable_timing=True)
            primal_end_event = torch.cuda.Event(enable_timing=True)
            dual_start_event = torch.cuda.Event(enable_timing=True)
            dual_end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()

        self._t_total = timer()
        for self._iter in range(self._start_iter, self._im_max_itr):
            self._t_iter = timer()
            self._each_iter_begin()

            # primal step
            if self._cuda_timing:
                primal_start_event.record()
            else:
                self._t_primal = timer()
            self._model = self._model - self._pr_step_size * self._meas_op.adjoint_op(
                [
                    self._precond_weight[i] * self._dual[i]
                    for i in range(self._rank_num_ch)
                ]
            )
            self._model = self._model.to(
                device=self._regulariser_prime.get_device(),
                dtype=self._regulariser_prime.get_data_type(),
            )
            self._model = self._regulariser_prime(self._model)
            if self._cuda_timing:
                primal_end_event.record()
                torch.cuda.synchronize()
                self._t_primal = primal_start_event.elapsed_time(primal_end_event) / 1e3
            else:
                self._t_primal = timer() - self._t_primal

            # dual step
            if self._cuda_timing:
                dual_start_event.record()
            else:
                self._t_dual = timer()
            self._model = self._model.to(device=self._meas_op.get_device()).to(
                dtype=self._meas_op.get_data_type()
            )
            dual_update = self._meas_op.forward_op(2 * self._model - self._model_prev)
            self._dual = [
                self._dual[i] + dual_update[i] for i in range(self._rank_ch_num)
            ]
            dual_proj = self._regulariser_dual_data(self._dual)
            self._dual = [
                self._dual[i] - dual_proj[i] for i in range(self._rank_ch_num)
            ]
            if self._cuda_timing:
                dual_end_event.record()
                torch.cuda.synchronize()
                self._t_dual = dual_start_event.elapsed_time(dual_end_event) / 1e3
            else:
                self._t_dual = timer() - self._t_dual
            self._t_iter = timer() - self._t_iter

            if self._stop_criteria():
                break

            self._each_iter_end()

            self._model_prev = self._model.clone()

        self._t_total = timer() - self._t_total

    @torch.no_grad()
    def _calculate_channel_wise_heuristic(self) -> None:
        """
        Calculates the heuristic noise level for each channel.

        This method computes the heuristic noise level based on the measurement operator
        and the measurements. It also applies any correction or scaling factors to the
        heuristic noise level if imaging weight and/or scaling factor are specified.

        The result heuristic noise level is saved in the attribute `_heuristic`, which
        is a tensor of shape (1, C, 1, 1).
        """

        self._heuristic = 1.0 / torch.sqrt(2 * self._meas_op_precise.get_op_norm_ch())
        if self._verbose:
            if self._is_distributed:
                dist.barrier()
            print(
                f"INFO: channel {self._ch_start + 1} - {self._ch_start + self._rank_ch_num}, "
                f"op norm {self._meas_op_precise.get_op_norm_ch().squeeze().cpu().tolist()}, "
                f"heuristic noise level {self._heuristic.squeeze().cpu().tolist()}",
                flush=True,
            )

        heu_corr_factor = torch.sqrt(
            self._meas_op_precise.get_op_norm_prime_ch()
            / self._meas_op_precise.get_op_norm_ch()
        )
        if not torch.allclose(
            heu_corr_factor,
            torch.tensor(
                1.0,
                device=heu_corr_factor.device,
                dtype=heu_corr_factor.dtype,
            ),
        ):
            self._heuristic *= heu_corr_factor
            if self._verbose:
                if self._is_distributed:
                    dist.barrier()
                print(
                    f"INFO: channel {self._ch_start + 1} - {self._ch_start + self._rank_ch_num}, "
                    f"heuristic noise level after correction: {self._heuristic.squeeze().cpu().tolist()}, "
                    f"correction factor {heu_corr_factor.squeeze().cpu().tolist()}",
                    flush=True,
                )
        if not np.isclose(self._heu_noise_scale, 1.0):
            self._heuristic *= self._heu_noise_scale
            if self._verbose:
                if self._is_distributed:
                    dist.barrier()
                if self._rank == 0:
                    print(
                        f"INFO: apply user specified heuristic noise scale {self._heu_noise_scale}",
                        flush=True,
                    )
                if self._is_distributed:
                    dist.barrier()
                print(
                    f"INFO: channel {self._ch_start + 1} - {self._ch_start + self._rank_ch_num}, "
                    f"heuristic noise level after scaling: {self._heuristic.squeeze().cpu().tolist()}",
                    flush=True,
                )
