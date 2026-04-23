"""
Hyperspectral measurement operator
"""

from typing import Tuple, Type

import torch
import torch.distributed as dist

from .meas_op import MeasOp
from .meas_op_nufft_finufft import MeasOpFinufft


class MeasOpHyperspectral(MeasOp):
    """
    Measurement operator for hyperspectral imaging
    """

    def __init__(
        self,
        u: Tuple[torch.Tensor],
        v: Tuple[torch.Tensor],
        img_size: Tuple[int, int],
        natural_weight: Tuple[torch.Tensor] = (torch.ones(1, 1, 1),),
        image_weight: Tuple[torch.Tensor] = (torch.ones(1, 1, 1),),
        precond_weight: Tuple[torch.Tensor] = (torch.ones(1, 1, 1),),
        meas_op_class: Type[MeasOp] = MeasOpFinufft,
        real_flag: bool = True,
        grid_size: Tuple[int, int] = None,
        kernel_dim: int = 7,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float,
        verbose: bool = True,
        **kwargs,
    ):

        super().__init__(
            img_size=img_size,
            real_flag=real_flag,
            device=device,
            dtype=dtype,
        )

        self._num_ch_rank = len(u)
        self._num_ch = (self._num_ch_rank,)
        self._meas_op_ch = []

        self._op_norm_ch = None
        self._op_norm_prime_ch = None

        # Initialize the measurement operator for each channel
        assert (
            len(u) == len(v) == self._num_ch_rank
        ), "length of u and v must match num_ch_rank"
        assert len(natural_weight) in (
            1,
            self._num_ch_rank,
        ), "natural_weight must have length 1 or num_ch_rank"
        assert len(image_weight) in (
            1,
            self._num_ch_rank,
        ), "image_weight must have length 1 or num_ch_rank"
        assert len(precond_weight) in (
            1,
            self._num_ch_rank,
        ), "precond_weight must have length 1 or num_ch_rank"
        # Override default weights if they are all ones and length 1
        if len(natural_weight) == 1 and torch.equal(
            natural_weight[0], torch.ones(1, 1, 1)
        ):
            natural_weight = tuple(
                torch.ones(1, 1, 1, device=self._device, dtype=self._dtype_meas)
                for _ in range(self._num_ch_rank)
            )
        if len(image_weight) == 1 and torch.equal(image_weight[0], torch.ones(1, 1, 1)):
            image_weight = tuple(
                image_weight[0].to(device=self._device, dtype=self._dtype_meas)
                for _ in range(self._num_ch_rank)
            )
        if len(precond_weight) == 1 and torch.equal(
            precond_weight[0], torch.ones(1, 1, 1)
        ):
            precond_weight = tuple(
                torch.ones(1, 1, 1, device=self._device, dtype=self._dtype_meas)
                for _ in range(self._num_ch_rank)
            )

        for ch in range(self._num_ch_rank):
            self._meas_op_ch.append(
                meas_op_class(
                    u=u[ch],
                    v=v[ch],
                    img_size=img_size,
                    natural_weight=natural_weight[ch],
                    image_weight=image_weight[ch],
                    precond_weight=precond_weight[ch],
                    real_flag=real_flag,
                    grid_size=grid_size,
                    kernel_dim=kernel_dim,
                    device=device,
                    dtype=dtype,
                    verbose=verbose,
                    **kwargs,
                )
            )

        # distributed
        self._distributed = dist.is_available() and dist.is_initialized()
        self._rank = 0
        self._world_size = 1
        if self._distributed:
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()

            # gather num_ch from all ranks
            local_n = torch.tensor(
                [self._num_ch_rank], device=self._device, dtype=torch.int
            )
            gathered = torch.empty(
                self._world_size, device=self._device, dtype=torch.int
            )
            dist.all_gather_into_tensor(gathered, local_n)
            self._num_ch = tuple(gathered.cpu().tolist())

    def forward_op(self, x):
        """
        Hyperspectral forward measurement operator
        """
        y_ch = [None] * self._num_ch_rank
        for i in range(self._num_ch_rank):
            y_ch[i] = self._meas_op_ch[i].forward_op(x[:, i : i + 1, :, :])

        return tuple(y_ch)

    def adjoint_op(self, y):
        """
        Hyperspectral adjoint measurement operator
        """
        x_ch = torch.empty(
            1,
            self._num_ch_rank,
            *self._img_size,
            device=self._device,
            dtype=self._dtype if self._real_flag else self._dtype_meas,
        )
        for i in range(self._num_ch_rank):
            x_ch[:, i : i + 1, :, :].copy_(self._meas_op_ch[i].adjoint_op(y[i]))

        return x_ch

    @torch.no_grad()
    def get_op_norm(
        self, compute_flag=False, rel_tol=1e-5, max_iter=500, verbose=False
    ):
        """
        Get the spectral norm of the measurement operator
        Overriden to introduce distributed compatibility
        """
        if self._op_norm is None or compute_flag:
            x = torch.randn(
                1,
                self._num_ch_rank,
                *self._img_size,
                device=self._device,
                dtype=self._dtype,
            )
            init_val = torch.linalg.vector_norm(x)
            if self._distributed:
                tmp = init_val**2
                dist.all_reduce(tmp, op=dist.ReduceOp.SUM)
                init_val = tmp.sqrt()
            x = x / init_val.item()
            init_val = torch.ones_like(init_val)

            for k in range(max_iter):
                x = self.adjoint_op(self.forward_op(x))
                val = torch.linalg.vector_norm(x)
                if self._distributed:
                    tmp = val**2
                    dist.all_reduce(tmp, op=dist.ReduceOp.SUM)
                    val = tmp.sqrt()
                rel_var = (val - init_val).abs() / init_val
                if rel_var.item() < rel_tol:
                    break
                init_val = val
                x = x / val
                if verbose and self._rank == 0:
                    print(f"Iter = {k}, norm = {val.item()}", flush=True)
            self._op_norm = val.item()

        return self._op_norm

    @torch.no_grad()
    def get_op_norm_ch(
        self, compute_flag=False, rel_tol=1e-5, max_iter=500, verbose=False
    ):
        """
        Get the spectral norm of the measurement operator
        """
        if self._op_norm_ch is None or compute_flag:
            self._op_norm_ch = tuple(
                self._meas_op_ch[i].get_op_norm(
                    compute_flag=compute_flag,
                    rel_tol=rel_tol,
                    max_iter=max_iter,
                    verbose=verbose,
                )
                for i in range(self._num_ch_rank)
            )
            self._op_norm_ch = torch.tensor(
                self._op_norm_ch, device=self._device, dtype=self._dtype
            ).view(1, -1, 1, 1)

        return self._op_norm_ch

    @torch.no_grad()
    def get_op_norm_prime_ch(self, rel_tol=1e-5, max_iter=500, verbose=False):
        """
        Get the spectral norm of measurement operator with data weight applied twice.
        It will be used to calculated the correction factor of the heuristic noise level.

        :param rel_tol: The relative tolerance. Defaults to 1e-5.
        :type rel_tol: float, optional
        :param max_iter: The maximum number of iterations. Defaults to 500.
        :type max_iter: int, optional
        :param verbose: If True, print progress messages. Defaults to False.
        :type verbose: bool, optional

        :return: The spectral norm of the measurement operator with data weight applied twice.
        :rtype: float
        """
        if self._op_norm_prime_ch is None:
            self._op_norm_prime_ch = tuple(
                self._meas_op_ch[i].get_op_norm_prime(
                    rel_tol=rel_tol,
                    max_iter=max_iter,
                    verbose=verbose,
                )
                for i in range(self._num_ch_rank)
            )
            self._op_norm_prime_ch = torch.tensor(
                self._op_norm_prime_ch, device=self._device, dtype=self._dtype
            ).view(1, -1, 1, 1)
        return self._op_norm_prime_ch

    @torch.no_grad()
    def get_psf(self):
        """
        Get the point spread function of the measurement operator.

        If gather_to_rank_0 and distributed:
        - returns full PSF (concatenated across channels) on rank 0
        - returns None on other ranks
        Otherwise:
        - returns local PSF for this rank
        """
        dirac = torch.zeros(
            1,
            self._num_ch_rank,
            *self._img_size,
            dtype=self._dtype,
            device=self._device,
        )
        dirac[0, :, self._img_size[0] // 2, self._img_size[1] // 2] = 1.0
        psf = self.adjoint_op(self.forward_op(dirac))  # (1, C_rank, H, W)

        return psf

    def set_real_flag(self, real_flag):
        """
        Set real flag for adjoint measurement oprator
        overriden to handle hyperspectral case
        """
        if self._real_flag != real_flag:
            self._real_flag = real_flag
            for ch in range(self._num_ch_rank):
                self._meas_op_ch[ch].set_real_flag(real_flag)

            self._op_norm = None
            self._op_norm_ch = None
            self._op_norm_prime_ch = None

    def set_precond_weight(self, precond_weight: Tuple[torch.Tensor, ...]) -> None:
        """
        Set preconditioning weight for measurement operator.

        :param precond_weight: The preconditioning weight tensor for each channel.
        :type precond_weight: Tuple[torch.Tensor, ...]
        """
        for ch in range(self._num_ch_rank):
            self._meas_op_ch[ch].set_precond_weight(precond_weight[ch])
        self._op_norm = None
        self._op_norm_ch = None
