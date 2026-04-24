"""Define the shared forward-backward optimiser implementation.

This module provides the forward-backward backbone used by the AIRI and SARA
optimisers. It handles the common forward step, proximal backward step,
timing, and heuristic noise-level estimation shared by those variants.
"""

import os
from timeit import default_timer as timer
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
from astropy.io import fits

from ..measurement_operator import MeasOp
from ..regulariser import Regulariser
from .optimiser import Optimiser


class ForwardBackward(Optimiser):
    """Represent a shared forward-backward optimisation backbone.

    This class implements a forward-backward scheme in which the data-fidelity
    term is differentiable and the regularisation term is applied through a
    backward proximal step. Concrete optimiser subclasses define the
    initialisation, stopping criterion, per-iteration bookkeeping, and
    finalisation policy on top of this common loop.

    Attributes:
        _meas_op_precise (MeasOp): Measurement operator used to compute the
            dirty image, PSF, and norm-based heuristics.
        _heu_noise_scale (float): User-defined multiplicative scaling applied
            to heuristic noise levels.
        _regulariser (Regulariser): Regulariser or proximal operator used in
            the backward step.
        _start_iter (int): Starting iteration index.
        _im_max_itr (int): Maximum number of forward-backward iterations.
        _gd_step_size (float): Forward-step gradient-descent step size.
        _t_total (float): Total optimisation time in seconds.
        _t_iter (float): Last iteration time in seconds.
        _t_forward (float): Last forward-step time in seconds.
        _t_backward (float): Last backward-step time in seconds.
        _iter (int): Current iteration index.
        _forward_cuda_timing (bool): Whether CUDA events time the forward step.
        _backward_cuda_timing (bool): Whether CUDA events time the backward
            step.
        _psf_peak (torch.Tensor): Per-channel PSF peak values.
        _heuristic (torch.Tensor | None): Cached heuristic noise levels.

    References:
        Combettes, P. L., & Pesquet, J.-C. (2011). `Proximal splitting methods
        in signal processing <https://doi.org/10.1007/978-1-4419-9569-8_10>`_.
        In H. H. Bauschke, R. S. Burachik, P. L. Combettes, V. Elser,
        D. R. Luke, & H. Wolkowicz (Eds.), *Fixed-point algorithms for inverse
        problems in science and engineering* (pp. 185-212). Springer.
    """

    def __init__(
        self,
        meas: List[torch.Tensor],
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
        """Initialise the shared forward-backward optimiser state.

        Args:
            meas (List[torch.Tensor]): Local measurement tensors, typically one
                tensor per local channel.
            ch_freq (torch.Tensor): Frequencies for the local spectral
                channels.
            meas_op (MeasOp): Measurement operator used inside the iterative
                forward step.
            regulariser (Regulariser): Regulariser or proximal operator applied
                in the backward step.
            meas_op_precise (MeasOp | None, optional): Measurement operator
                used for dirty-image, PSF, and heuristic calculations. If
                None, reuses ``meas_op``. Default: None.
            heu_noise_scale (float, optional): User-specified multiplicative
                factor applied to heuristic noise estimates. Default: 1.0.
            im_max_itr (int, optional): Maximum number of iterations.
                Default: 2000.
            save_pth (str, optional): Directory where optimiser products are
                saved. Default: ``"results"``.
            file_prefix (str, optional): Prefix prepended to saved file names.
                Default: ``""``.
            rank_ch_start (int, optional): Global starting channel index owned
                by the current rank. Default: 0.

        Notes:
            This constructor computes and stores the dirty image and PSF before
            the main optimisation loop starts, then writes them to FITS files
            on rank 0.
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

        # Cache wall-clock timing statistics for logging and diagnostics.
        self._t_total = 0.0
        self._t_iter = 0.0
        self._t_forward = 0.0
        self._t_backward = 0.0

        self._iter = 0

        # Enable CUDA-event timing only when the relevant step runs on CUDA.
        self._forward_cuda_timing = False
        self._backward_cuda_timing = False
        if self._meas_op.get_device() == torch.device("cuda"):
            self._forward_cuda_timing = True
        if self._regulariser.get_device() == torch.device("cuda"):
            self._backward_cuda_timing = True

        # Precompute dirty image and PSF products used by concrete subclasses.
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

        # Allocate the heuristic-noise cache lazily.
        self._heuristic = None

    @torch.inference_mode()
    def run(self) -> None:
        """Run the forward-backward optimisation loop.

        This method alternates a gradient-descent step on the differentiable
        data-fidelity term with a proximal regularisation step until the
        subclass-defined stopping criterion is met or the maximum iteration
        count is reached.
        """
        # Create CUDA events only on rank 0 when CUDA timing is enabled.
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

            # Forward step: gradient descent on the data-fidelity term.
            if self._forward_cuda_timing:
                if self._rank == 0:
                    forward_start_event.record()
            else:
                self._t_forward = timer()
            x_hat = self._model - self._gd_step_size * (
                self._meas_op.adjoint_op(self._meas_op.forward_op(self._model))
                - self._meas_bp
            )
            # Move the gradient step to the regulariser device for the
            # proximal backward step.
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

            # Backward step: apply the proximal regulariser on its own device.
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
        r"""Compute heuristic noise levels for local channels.

        The heuristic is derived from channel-wise operator norms and can be
        corrected for twice-applied image weighting and scaled by the
        user-provided ``heu_noise_scale`` factor.

        For each local channel, the base heuristic is computed as
        :math:`h_c = 1 / \sqrt{2 L_c}`, where
        :math:`L_c = \lVert \Phi_c^\dagger \Phi_c \rVert_\mathrm{S}` is the
        channel-wise operator norm returned by
        ``self._meas_op_precise.get_op_norm_ch()``.

        If the image weighting is integrated, the heuristic noise level is corrected
        by the factor
        :math:`\sqrt{L^\prime_c / L_c}`, where
        :math:`L^\prime_c = \lVert \Phi_c^\dagger W^\dagger W \Phi_c \rVert_\mathrm{S}`.

        Returns:
            None: The heuristic is stored in ``self._heuristic``.

        References:
            Terris, M., Dabbech, A., Tang, C., & Wiaux, Y. (2023). `Image
            reconstruction algorithms in radio interferometry: From handcrafted to
            learned regularization denoisers <https://doi.org/10.1093/mnras/stac2672>`_.
            *MNRAS, 518*(1), 604-622.

            Wilber, A. G., Dabbech, A., Jackson, A., & Wiaux, Y. (2023). `Scalable
            precision wide-field imaging in radio interferometry: I. uSARA
            validated on ASKAP data <https://doi.org/10.1093/mnras/stad1351>`_.
            *MNRAS, 522*(4), 5558-5575.
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
                f"INFO: channel {self._rank_ch_start + 1} - "
                f"{self._rank_ch_start + self._rank_ch_num}, "
                f"op norm [{', '.join(f'{val:.4e}' for val in op_norm_ch_list)}], "
                f"heuristic noise level "
                f"[{', '.join(f'{val:.4e}' for val in heuristic_list)}]",
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
                    f"INFO: channel {self._rank_ch_start + 1} - "
                    f"{self._rank_ch_start + self._rank_ch_num}, "
                    f"heuristic noise level after correction: "
                    f"[{', '.join(f'{val:.4e}' for val in heuristic_list)}], "
                    f"correction factor "
                    f"[{', '.join(f'{val:.4e}' for val in heu_corr_factor_list)}]",
                    flush=True,
                )
        if not np.isclose(self._heu_noise_scale, 1.0):
            self._heuristic *= self._heu_noise_scale
            if self._verbose:
                if self._is_distributed:
                    dist.barrier()
                if self._rank == 0:
                    print(
                        f"INFO: apply user specified heuristic "
                        f"noise scale {self._heu_noise_scale}",
                        flush=True,
                    )
                if self._is_distributed:
                    dist.barrier()
                heuristic_list = self._heuristic.squeeze().cpu().tolist()
                if not isinstance(heuristic_list, list):
                    heuristic_list = [heuristic_list]
                print(
                    f"INFO: channel {self._rank_ch_start + 1} - "
                    f"{self._rank_ch_start + self._rank_ch_num}, "
                    f"heuristic noise level after scaling: "
                    f"[{', '.join(f'{val:.4e}' for val in heuristic_list)}]",
                    flush=True,
                )
