"""Define the hyperspectral measurement operator wrapper.

This file provides a channel-wise wrapper that stacks monochromatic
measurement operators and exposes a unified hyperspectral interface.
"""

from typing import Any, Tuple, Type

import torch
import torch.distributed as dist

from .meas_op import MeasOp
from .meas_op_nufft_finufft import MeasOpFinufft


class MeasOpHyperspectral(MeasOp):
    """Represent a hyperspectral measurement operator assembled per channel.

    This class wraps a monochromatic measurement-operator implementation and
    instantiates one local operator per spectral channel handled by the current
    process. Forward and adjoint applications are evaluated channel-wise, while
    norm estimation can aggregate local contributions across distributed ranks.

    Shape:
        Image tensors: ``(1, C_rank, H, W)``.
        Visibility-domain tensors: tuple of ``C_rank`` tensors, each with shape
            ``(1, 1, N_c)``.
        Sampling-pattern coordinates :math:`u` and :math:`v`: tuples of ``C_rank``
            tensors, each with shape ``(1, 1, N_c)``.

    Attributes:
        _num_ch_rank (int): Number of local spectral channels owned by this
            rank.
        _num_ch (tuple[int, ...]): Per-rank channel counts across the active
            distributed job.
        _meas_op_ch (list[MeasOp]): Monochromatic operators for local channels.
        _distributed (bool): Whether ``torch.distributed`` is initialised.
        _rank (int): Distributed rank index.
        _world_size (int): Number of distributed ranks.
        _op_norm_ch (torch.Tensor | None): Cached local per-channel norm
            estimates.
        _op_norm_prime_ch (torch.Tensor | None): Cached local per-channel
            twice-weighted norm estimates.

    Example:
        >>> import torch
        >>> u = ((2 * torch.pi) * torch.rand(1, 1, 8) - torch.pi,)
        >>> v = ((2 * torch.pi) * torch.rand(1, 1, 8) - torch.pi,)
        >>> op = MeasOpHyperspectral(u=u, v=v, img_size=(16, 16))
        >>> image = torch.randn(1, 1, 16, 16)
        >>> vis = op.forward_op(image)
        >>> len(vis)
        1
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
        **kwargs: Any,
    ) -> None:
        """Initialise channel-wise measurement operators.

        Args:
            u (Tuple[torch.Tensor]): Per-channel ``u`` sampling coordinates.
            v (Tuple[torch.Tensor]): Per-channel ``v`` sampling coordinates.
            img_size (Tuple[int, int]): Target image size ``(H, W)``.
            natural_weight (Tuple[torch.Tensor], optional): Per-channel natural
                weighting tensors. Length must be 1 or the local channel count.
            image_weight (Tuple[torch.Tensor], optional): Per-channel imaging
                weighting tensors. Length must be 1 or the local channel count.
            precond_weight (Tuple[torch.Tensor], optional): Per-channel
                preconditioning tensors. Length must be 1 or the local channel
                count.
            meas_op_class (Type[MeasOp], optional): Monochromatic operator
                class used for each channel. Default: MeasOpFinufft.
            real_flag (bool, optional): If True, keep adjoint outputs real when
                applicable. Default: True.
            grid_size (Tuple[int, int], optional): Backend-specific oversampled
                Fourier-grid size. Default: None.
            kernel_dim (int, optional): Backend-specific kernel support.
                Default: 7.
            device (torch.device, optional): Execution device. Default:
                ``torch.device("cpu")``.
            dtype (torch.dtype, optional): Image-domain dtype. Default:
                ``torch.float``.
            verbose (bool, optional): Enable backend/operator logs.
                Default: True.
            **kwargs: Additional backend-specific arguments forwarded to
                ``meas_op_class``.

        Notes:
            Each process constructs operators only for its local channel subset.
            When ``torch.distributed`` is active, global channel counts are
            gathered so distributed norm estimates can combine contributions
            across ranks.
        """

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

        # Validate per-channel inputs and optional shared-weight shortcuts.
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
        # Expand singleton defaults to per-channel tuples.
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

        # Distributed metadata and local-channel counts per rank.
        self._distributed = dist.is_available() and dist.is_initialized()
        self._rank = 0
        self._world_size = 1
        if self._distributed:
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()

            # Gather local channel counts from all ranks.
            local_n = torch.tensor(
                [self._num_ch_rank], device=self._device, dtype=torch.int
            )
            gathered = torch.empty(
                self._world_size, device=self._device, dtype=torch.int
            )
            dist.all_gather_into_tensor(gathered, local_n)
            self._num_ch = tuple(gathered.cpu().tolist())

    def forward_op(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Apply the hyperspectral forward operator channel-wise.

        Args:
            x (torch.Tensor): Input tensor with shape ``(1, C_rank, H, W)``.

        Returns:
            Tuple[torch.Tensor, ...]: Tuple of measurement-domain tensors, one
            per local channel.

        Notes:
            The channel dimension of ``x`` is split across the locally owned
            monochromatic operators in index order.
        """
        y_ch = [None] * self._num_ch_rank
        for i in range(self._num_ch_rank):
            y_ch[i] = self._meas_op_ch[i].forward_op(x[:, i : i + 1, :, :])

        return tuple(y_ch)

    def adjoint_op(self, y: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Apply the hyperspectral adjoint operator channel-wise.

        Args:
            y (Tuple[torch.Tensor, ...]): Tuple of measurement-domain tensors,
                one per local channel.

        Returns:
            torch.Tensor: Reconstructed local image cube with shape
            ``(1, C_rank, H, W)``.

        Notes:
            The input tuple must contain one measurement tensor per local
            channel in the same order used during construction.
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
    ) -> float:
        """Estimate the global spectral norm of the hyperspectral operator.

        This overrides the base implementation to aggregate vector norms across
        distributed ranks.

        Args:
            compute_flag (bool, optional): Force recomputation if True.
                Default: False.
            rel_tol (float, optional): Relative stopping tolerance.
                Default: 1e-5.
            max_iter (int, optional): Maximum power iterations. Default: 500.
            verbose (bool, optional): Print iteration progress on rank 0.
                Default: False.

        Returns:
            float: Estimated spectral norm.

        Notes:
            In distributed runs, the power iteration uses an all-reduce on the
            squared vector norm so the returned value reflects the global
            hyperspectral operator rather than only the local rank.
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
    ) -> torch.Tensor:
        """Return per-channel spectral-norm estimates.

        Args:
            compute_flag (bool, optional): Force recomputation if True.
                Default: False.
            rel_tol (float, optional): Relative stopping tolerance forwarded to
                channel operators. Default: 1e-5.
            max_iter (int, optional): Maximum power iterations forwarded to
                channel operators. Default: 500.
            verbose (bool, optional): Print channel progress when enabled.
                Default: False.

        Returns:
            torch.Tensor: Tensor of shape ``(1, C_rank, 1, 1)`` containing
            local per-channel norm estimates.
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
    def get_op_norm_prime_ch(self, rel_tol=1e-5, max_iter=500, verbose=False) -> torch.Tensor:
        """Return per-channel twice-weighted norm estimates.

        These values are typically used to correct heuristic noise levels for
        weighted imaging.

        Args:
            rel_tol (float, optional): Relative stopping tolerance forwarded to
                channel operators. Default: 1e-5.
            max_iter (int, optional): Maximum power iterations forwarded to
                channel operators. Default: 500.
            verbose (bool, optional): Print channel progress when enabled.
                Default: False.

        Returns:
            torch.Tensor: Tensor of shape ``(1, C_rank, 1, 1)`` containing
            local per-channel twice-weighted norm estimates.
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
    def get_psf(self) -> torch.Tensor:
        """Compute local-channel PSFs.

        Returns:
            torch.Tensor: Local PSF tensor with shape ``(1, C_rank, H, W)``.

        Notes:
            This method returns only the PSFs owned by the current rank. It
            does not gather channel blocks across distributed processes.
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

    def set_real_flag(self, real_flag: bool) -> None:
        """Set the real-output flag for all local channel operators.

        Args:
            real_flag (bool): New real-output flag.
        """
        if self._real_flag != real_flag:
            self._real_flag = real_flag
            for ch in range(self._num_ch_rank):
                self._meas_op_ch[ch].set_real_flag(real_flag)

            self._op_norm = None
            self._op_norm_ch = None
            self._op_norm_prime_ch = None

    def set_precond_weight(self, precond_weight: Tuple[torch.Tensor, ...]) -> None:
        """Set per-channel preconditioning weights.

        Args:
            precond_weight (Tuple[torch.Tensor, ...]): Preconditioning tensors,
                one per local channel.

        Notes:
            Updating these weights invalidates cached global and per-channel
            norm estimates.
        """
        for ch in range(self._num_ch_rank):
            self._meas_op_ch[ch].set_precond_weight(precond_weight[ch])
        self._op_norm = None
        self._op_norm_ch = None
