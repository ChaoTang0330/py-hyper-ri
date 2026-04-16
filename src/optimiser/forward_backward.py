"""
Forward-backward algorithm
"""

import os
from typing import Optional
from timeit import default_timer as timer
import torch
import torch.distributed as dist
import numpy as np
from astropy.io import fits

from .optimiser import Optimiser
from ..regulariser import Regulariser
from ..ri_measurement_operator import MeasOp


class ForwardBackward(Optimiser):
    """
    This class implements the forward-backward algorithm.

    We here assume the data fidelity term is differentiable and the regularisation term is
    non-differentiable.
    """

    def __init__(
        self,
        meas: torch.Tensor,
        ch_freq: torch.Tensor,
        meas_op: MeasOp,
        regulariser: Regulariser,
        meas_op_precise: Optional[MeasOp] = None,
        heu_noise_scale: float = 1.0,
        im_max_itr: int = 2000,
        save_pth: str = "results",
        file_prefix: str = "",
        rank_ch_start: int = 0,
    ) -> None:
        """
        Initializes the ForwardBackward optimiser.

        Args:
            meas (torch.Tensor): The measurement tensor.
            ch_freq (torch.Tensor): The frequency of each channel.
            meas_op (MeasOp): The measurement operator.
            meas_op_precise (Optional[MeasOp]): The precise measurement operator.
            regulariser (Regulariser): The regulariser.
            im_max_itr (int, optional): The maximum number of iterations. Defaults to 2000.
            save_pth (str, optional): The path where results will be saved. Defaults to "results".
            file_prefix (str, optional): The prefix of the saving files. Defaults to None.
            rank_ch_start (int, optional): The starting channel index for the current rank. Defaults to 0.
        """
        super().__init__(
            meas,
            ch_freq,
            meas_op,
            save_pth=save_pth,
            file_prefix=file_prefix,
            rank_ch_start=rank_ch_start,
        )
        self._meas_op_precise = meas_op_precise
        if self._meas_op_precise is None:
            self._meas_op_precise = meas_op
        self._heu_noise_scale = heu_noise_scale
        self._regulariser = regulariser
        self._start_iter = 0
        self._im_max_itr = im_max_itr
        self._gd_step_size = 1.0

        # timing
        self._t_total = 0.0
        self._t_iter = 0.0
        self._t_forward = 0.0
        self._t_backward = 0.0

        self._iter = 0

        # cuda event
        self._forward_cuda_timing = False
        self._backward_cuda_timing = False
        if self._meas_op.get_device() == torch.device("cuda"):
            self._forward_cuda_timing = True
        if self._regulariser.get_device() == torch.device("cuda"):
            self._backward_cuda_timing = True

        # save dirty image and psf
        self._meas_bp = self._meas_op_precise.adjoint_op(self._meas).to(
            device=self._meas_op.get_device(), dtype=self._meas_op.get_data_type()
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

    @torch.inference_mode()
    def run(self) -> None:
        """
        Runs the main loop of the forward-backward algorithm.

        This method performs the forward and backward steps of the algorithm
        in a loop until the stop criteria is met.
        """
        # timing with cuda events
        if self._forward_cuda_timing and self._rank == 0:
            forward_start_event = torch.cuda.Event(enable_timing=True)
            forward_end_event = torch.cuda.Event(enable_timing=True)
        if self._backward_cuda_timing and self._rank == 0:
            backward_start_event = torch.cuda.Event(enable_timing=True)
            backward_end_event = torch.cuda.Event(enable_timing=True)

        self._t_total = timer()
        for self._iter in range(self._start_iter, self._im_max_itr):
            if self._is_distributed:
                dist.barrier()

            self._t_iter = timer()
            self._each_iter_begin()

            # forward step
            if self._forward_cuda_timing:
                if self._rank == 0:
                    forward_start_event.record()
            else:
                self._t_forward = timer()
            x_hat = self._model - self._gd_step_size * (
                self._meas_op.adjoint_op(self._meas_op.forward_op(self._model))
                - self._meas_bp
            )
            x_hat = x_hat.to(
                device=self._regulariser.get_device(),
                dtype=self._regulariser.get_data_type(),
            )

            if self._is_distributed:
                dist.barrier()

            if self._forward_cuda_timing:
                if self._rank == 0:
                    forward_end_event.record()
                    torch.cuda.synchronize()
                    self._t_forward = (
                        forward_start_event.elapsed_time(forward_end_event) / 1e3
                    )
            else:
                self._t_forward = timer() - self._t_forward

            # backward step
            if self._backward_cuda_timing:
                if self._rank == 0:
                    backward_start_event.record()
            else:
                self._t_backward = timer()
            self._model = self._regulariser(x_hat)
            self._model = self._model.to(device=self._meas_op.get_device()).to(
                dtype=self._meas_op.get_data_type()
            )

            if self._is_distributed:
                dist.barrier()

            if self._backward_cuda_timing:
                if self._rank == 0:
                    backward_end_event.record()
                    torch.cuda.synchronize()
                    self._t_backward = (
                        backward_start_event.elapsed_time(backward_end_event) / 1e3
                    )
            else:
                self._t_backward = timer() - self._t_backward
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
            op_norm_ch_list = (
                self._meas_op_precise.get_op_norm_ch().squeeze().cpu().tolist()
            )
            if not isinstance(op_norm_ch_list, list):
                op_norm_ch_list = [op_norm_ch_list]
            heuristic_list = self._heuristic.squeeze().cpu().tolist()
            if not isinstance(heuristic_list, list):
                heuristic_list = [heuristic_list]
            print(
                f"INFO: channel {self._rank_ch_start + 1} - {self._rank_ch_start + self._rank_ch_num}, "
                f"op norm [{', '.join(f'{val:.4e}' for val in op_norm_ch_list)}], "
                f"heuristic noise level [{', '.join(f'{val:.4e}' for val in heuristic_list)}]",
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
                heuristic_list = self._heuristic.squeeze().cpu().tolist()
                if not isinstance(heuristic_list, list):
                    heuristic_list = [heuristic_list]
                heu_corr_factor_list = heu_corr_factor.squeeze().cpu().tolist()
                if not isinstance(heu_corr_factor_list, list):
                    heu_corr_factor_list = [heu_corr_factor_list]
                print(
                    f"INFO: channel {self._rank_ch_start + 1} - {self._rank_ch_start + self._rank_ch_num}, "
                    f"heuristic noise level after correction: [{', '.join(f'{val:.4e}' for val in heuristic_list)}], "
                    f"correction factor [{', '.join(f'{val:.4e}' for val in heu_corr_factor_list)}]",
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
                heuristic_list = self._heuristic.squeeze().cpu().tolist()
                if not isinstance(heuristic_list, list):
                    heuristic_list = [heuristic_list]
                print(
                    f"INFO: channel {self._rank_ch_start + 1} - {self._rank_ch_start + self._rank_ch_num}, "
                    f"heuristic noise level after scaling: [{', '.join(f'{val:.4e}' for val in heuristic_list)}]",
                    flush=True,
                )
