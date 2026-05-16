"""Define the shared primal-dual optimiser implementation.

This module provides a primal-dual backbone with a preconditioned dual update
for data fidelity and a proximal primal update for regularisation.
"""

import os
from timeit import default_timer as timer

import numpy as np
import torch
import torch.distributed as dist
from astropy.io import fits

from ..measurement_operator import MeasOpHyperspectral
from ..regulariser import ProxEllipsoid, Regulariser
from .optimiser import Optimiser


class PrimalDualPrecond(Optimiser):
    """Represent a preconditioned primal-dual optimisation backbone.

    This class implements a primal-dual splitting scheme with two
    non-differentiable terms. The preconditioned data-fidelity term is handled
    through a dual update, while the regularisation term is handled through a
    proximal update of the primal image variable.

    Shape:
        Image tensor: ``(1, C_rank, H, W)``.
        Visibility tensor: tuple of ``C_rank`` tensors, each with shape
            ``(1, 1, N_c)``.
        Frequency tensor: ``(1, C_total, 1, 1)`` after reshaping in the base
            optimiser.

    Attributes:
        _regulariser_prime (Regulariser): Proximal regulariser applied to the
            primal image variable.
        _regulariser_dual_data (ProxEllipsoid): Proximal operator applied to
            the dual data variable.
        _heu_noise_scale (float): User-defined multiplicative scaling applied
            to heuristic noise levels.
        _start_iter (int): Starting iteration index.
        _im_max_itr (int): Maximum number of primal-dual iterations.
        _pr_step_size (float): Primal step size.
        _precond_weight (list[torch.Tensor]): Channel-wise preconditioning
            weights applied in the adjoint dual update.
        _dual (list[torch.Tensor]): Dual variables associated with the
            visibility-domain data-fidelity term.
        _iter (int): Current iteration index.
        _t_total (float): Total optimisation time in seconds.
        _t_iter (float): Last iteration time in seconds.
        _t_primal (float): Last primal-step time in seconds.
        _t_dual (float): Last dual-step time in seconds.
        _cuda_timing (bool): Whether CUDA events are used for timing.
        _psf_peak (torch.Tensor): Per-channel PSF peak values.
        _heuristic (torch.Tensor | None): Cached heuristic noise levels.

    References:
        Condat, L. (2013). `A Primal-Dual Splitting Method for Convex
        Optimization Involving Lipschitzian, Proximable and Linear Composite
        Terms <https://link.springer.com/article/10.1007/s10957-012-0245-9>`_.
        *J. Optim. Theory Appl., 158*, 460-479.
    """

    def __init__(
        self,
        meas: list[torch.Tensor],
        ch_freq: torch.Tensor,
        meas_op: MeasOpHyperspectral,
        regulariser_prime: Regulariser,
        regulariser_dual_data: ProxEllipsoid,
        heu_noise_scale: float = 1.0,
        rank_ch_start: int = 0,
        im_max_itr: int = 2000,
        save_pth: str = "results",
        file_prefix: str = "",
    ) -> None:
        """Initialise the shared primal-dual optimiser state.

        Args:
            meas (List[torch.Tensor]): Local measurement tensors, typically one
                tensor per local channel.
            ch_freq (torch.Tensor): Frequencies for all spectral channels.
            meas_op (MeasOpHyperspectral): Hyperspectral measurement operator.
            regulariser_prime (Regulariser): Proximal regulariser applied to
                the primal image variable.
            regulariser_dual_data (ProxEllipsoid): Proximal operator applied
                to the dual data variable.
            heu_noise_scale (float, optional): User-specified multiplicative
                factor applied to heuristic noise estimates. Default: 1.0.
            rank_ch_start (int, optional): Global starting channel index owned
                by the current rank. Default: 0.
            im_max_itr (int, optional): Maximum number of iterations.
                Default: 2000.
            save_pth (str, optional): Directory where optimiser products are
                saved. Default: ``"results"``.
            file_prefix (str, optional): Prefix prepended to saved file names.
                Default: ``""``.

        Note:
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
        self._regulariser_prime = regulariser_prime
        self._regulariser_dual_data = regulariser_dual_data
        self._heu_noise_scale = heu_noise_scale
        self._start_iter = 0
        self._im_max_itr = im_max_itr
        self._pr_step_size = 1.0
        self._iter = 0

        # Store one preconditioning weight per local visibility channel.
        self._precond_weight = [
            torch.ones(1, 1, device=meas_op.get_device())
            for _ in range(self._rank_ch_num)
        ]
        # Allocate the dual variables with the same layout as the measurements.
        self._dual = [torch.zeros_like(m) for m in self._meas]

        # Cache wall-clock timing statistics for logging and diagnostics.
        self._t_total = 0.0
        self._t_iter = 0.0
        self._t_primal = 0.0
        self._t_dual = 0.0

        # Enable CUDA-event timing only when the measurement operator runs on CUDA.
        self._cuda_timing = self._meas_op.get_device() == torch.device("cuda")

        # Precompute dirty image and PSF products used by concrete subclasses.
        self._meas_bp = self._meas_op.adjoint_op(self._meas)
        self._psf = self._meas_op.get_psf()
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

    @torch.no_grad()
    def run(self) -> None:
        """Run the primal-dual optimisation loop.

        This method alternates a primal proximal update in the image domain
        with a dual proximal update in the visibility domain until the
        subclass-defined stopping criterion is met or the maximum iteration
        count is reached.
        """
        if self._cuda_timing:
            primal_start_event = torch.cuda.Event(enable_timing=True)
            primal_end_event = torch.cuda.Event(enable_timing=True)
            dual_start_event = torch.cuda.Event(enable_timing=True)
            dual_end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()

        self._t_total = timer()
        for iter_idx in range(self._start_iter, self._im_max_itr):
            self._iter = iter_idx
            self._t_iter = timer()
            self._each_iter_begin()

            # Primal step: apply the adjoint dual correction then the primal prox.
            if self._cuda_timing:
                primal_start_event.record()
            else:
                self._t_primal = timer()
            self._model = self._model - self._pr_step_size * self._meas_op.adjoint_op(
                [
                    self._precond_weight[i] * self._dual[i]
                    for i in range(self._rank_ch_num)
                ]
            )
            # Move the primal variable to the regulariser device for the prox.
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

            # Dual step: extrapolate the primal variable and project the dual state.
            if self._cuda_timing:
                dual_start_event.record()
            else:
                self._t_dual = timer()
            self._model = self._model.to(device=self._meas_op.get_device()).to(
                dtype=self._meas_op.get_data_type()
            )
            # Use the standard extrapolated primal point in the forward model.
            dual_update = self._meas_op.forward_op(2 * self._model - self._model_prev)
            self._dual = [
                self._dual[i] + dual_update[i] for i in range(self._rank_ch_num)
            ]
            # Apply the dual prox by subtracting the projection result.
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
        r"""Calculate channel-wise heuristic noise levels.

        The base heuristic is defined as

        .. math::
            h_c = \frac{1}{\sqrt{2 L_c}},

        where :math:`L_c = \|\Phi_c^\dagger \Phi_c\|_S` is the channel-wise
        operator norm returned by ``self._meas_op.get_op_norm_ch()``. The
        heuristic is then optionally corrected with
        ``self._meas_op.get_op_norm_prime_ch()`` and scaled by the
        user-provided factor ``self._heu_noise_scale``.

        The result is stored in ``self._heuristic`` with shape
        ``(1, C_rank, 1, 1)``.

        References:
            Terris, M., Dabbech, A., Tang, C., & Wiaux, Y. (2023). `Image
            reconstruction algorithms in radio interferometry: From handcrafted to
            learned regularization denoisers <https://doi.org/10.1093/mnras/stac2672>`_.
            *MNRAS*, 518(1), 604-622.

            Wilber, A. G., Dabbech, A., Jackson, A., & Wiaux, Y. (2023). `Scalable
            precision wide-field imaging in radio interferometry: I. uSARA
            validated on ASKAP data <https://doi.org/10.1093/mnras/stad1351>`_.
            *MNRAS*, 522(4), 5558-5575.
        """
        self._heuristic = 1.0 / torch.sqrt(2 * self._meas_op.get_op_norm_ch())
        if self._verbose:
            if self._is_distributed:
                dist.barrier()
            op_norm_ch_list = self._meas_op.get_op_norm_ch().squeeze().cpu().tolist()
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
            self._meas_op.get_op_norm_prime_ch() / self._meas_op.get_op_norm_ch()
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
