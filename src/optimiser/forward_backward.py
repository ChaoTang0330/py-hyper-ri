"""
Forward-backward algorithm
"""

import os
from typing import Union
from timeit import default_timer as timer
import torch
import torch.distributed as dist
import numpy as np
from astropy.io import fits

from .optimiser import Optimiser
from ..prox_operator import ProxOp
from ..ri_measurement_operator.pysrc.measOperator import MeasOp


class ForwardBackward(Optimiser):
    """
    This class implements the forward-backward algorithm.

    We here assume the data fidelity term is differentiable and the regularisation term is
    non-differentiable.
    """

    def __init__(
        self,
        meas: torch.Tensor,
        meas_op: MeasOp,
        meas_op_precise: Union[MeasOp, None],
        prox_op: ProxOp,
        im_max_itr: int = 2000,
        save_pth: str = "results",
        file_prefix: str = "",
        ch_start: int = 0,
    ) -> None:
        """
        Initializes the ForwardBackward optimiser.

        Args:
            meas (torch.Tensor): The measurement tensor.
            meas_op (MeasOp): The measurement operator.
            meas_op_precise (Union[MeasOp, None]): The precise measurement operator.
            prox_op (ProxOp): The proximal operator.
            im_max_itr (int, optional): The maximum number of iterations. Defaults to 2000.
            save_pth (str, optional): The path where results will be saved. Defaults to "results".
            file_prefix (str, optional): The prefix of the saving files. Defaults to None.
            ch_start (int, optional): The starting channel index. Defaults to 0.
        """
        super().__init__(
            meas, meas_op, save_pth=save_pth, file_prefix=file_prefix, ch_start=ch_start
        )
        self._meas_op_precise = meas_op_precise
        self._prox_op = prox_op
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
        if self._prox_op.get_device() == torch.device("cuda"):
            self._backward_cuda_timing = True

        # save dirty image and psf
        self._meas_bp = self._meas_op_precise.adjoint_op(self._meas).to(
            self._meas_op_precise.get_device()
        )
        self._psf = self._meas_op_precise.get_psf()
        self._psf_peak = self._psf.amax(dim=(2, 3), keepdim=True)

        dirty_image_np = self.get_dirty_image(normalise=self._psf_peak, dest=0)
        psf_np = self.get_psf(dest=0)
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

    # @torch.no_grad()
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
            # x_hat = self._model - self._gd_step_size * (
            #     self._meas_op.adjoint_op(self._meas_op.forward_op(self._model))
            #     - self._meas_bp
            # )
            x_hat = self._model - self._gd_step_size * (
                self._meas_op.forward_adjoint_op(self._model) - self._meas_bp
            )
            x_hat = x_hat.to(
                device=self._prox_op.get_device(), dtype=self._prox_op.get_data_type()
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
            self._model = self._prox_op(x_hat)
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
