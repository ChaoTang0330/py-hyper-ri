"""Define the TKBNUFFT-backed measurement operator.

This implementation uses ``torchkbnufft`` for forward and adjoint NUFFT
operations.
"""

from typing import Any, Tuple

import numpy as np
import torch

from .meas_op_nufft import MeasOpNUFFT


class MeasOpTkbn(MeasOpNUFFT):
    """Represent a NUFFT measurement operator backed by TorchKbNUFFT.

    This class evaluates nonuniform Fourier sampling with ``torchkbnufft`` for
    single-channel measurement operators. It reuses the weighting and
    preconditioning logic defined in :class:`MeasOpNUFFT` while exposing the
    backend-specific interpolation modes supported by TorchKbNUFFT.

    Shape:
        Image tensors: ``(1, 1, H, W)``.
        Visibility-domain tensors: ``(1, 1, N)``.
        Sampling-pattern coordinates :math:`u` and :math:`v`: ``(1, 1, N)``.

    Attributes:
        _traj (torch.Tensor): Sampling trajectory stored in TorchKbNUFFT's
            expected layout.
        _data_weight (torch.Tensor): Combined natural and imaging weights
            applied in measurement space.
        _precond_weight (torch.Tensor): Preconditioning weights applied around
            the backend NUFFT calls.
        _nufft_obj (torch.nn.Module): Forward TorchKbNUFFT operator.
        _adj_nufft_obj (torch.nn.Module): Adjoint TorchKbNUFFT operator.
        _interp_mats (tuple[torch.Tensor, ...] | None): Sparse interpolation
            matrices for ``"matrix"`` mode, or ``None`` for ``"table"`` mode.

    Example:
        >>> import torch
        >>> u = (2 * torch.pi) * torch.rand(1, 1, 8) - torch.pi
        >>> v = (2 * torch.pi) * torch.rand(1, 1, 8) - torch.pi
        >>> op = MeasOpTkbn(
        ...     u=u,
        ...     v=v,
        ...     img_size=(16, 16),
        ...     mode="table",
        ... )
        >>> image = torch.randn(1, 1, 16, 16)
        >>> vis = op.forward_op(image)
        >>> vis.shape
        torch.Size([1, 1, 8])
    """

    def __init__(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        img_size: Tuple[int, int],
        natural_weight: torch.Tensor = torch.ones(1, 1, 1),
        image_weight: torch.Tensor = torch.ones(1, 1, 1),
        precond_weight: torch.Tensor = torch.ones(1, 1, 1),
        real_flag: bool = True,
        grid_size: Tuple[int, int] = None,
        kernel_dim: int = 7,
        mode: str = "table",
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float,
        **kwargs: Any
    ) -> None:
        """Initialise the torchkbnufft measurement operator.

        Args:
            u (torch.Tensor): ``u`` sampling coordinates in radians.
            v (torch.Tensor): ``v`` sampling coordinates in radians.
            img_size (Tuple[int, int]): Target image size ``(H, W)``.
            natural_weight (torch.Tensor, optional): Natural weighting tensor.
                Default: ``torch.ones(1, 1, 1)``.
            image_weight (torch.Tensor, optional): Imaging weighting tensor.
                Default: ``torch.ones(1, 1, 1)``.
            precond_weight (torch.Tensor, optional): Data-space
                preconditioning tensor. Default: ``torch.ones(1, 1, 1)``.
            real_flag (bool, optional): If True, keep adjoint outputs real when
                applicable. Default: True.
            grid_size (Tuple[int, int], optional): Oversampled Fourier-grid
                size. If None, uses ``2 * img_size``. Default: None.
            kernel_dim (int, optional): Kaiser-Bessel kernel support.
                Default: 7.
            mode (str, optional): Interpolation mode, either ``"table"`` or
                ``"matrix"``. Default: ``"table"``.
            device (torch.device, optional): Execution device. Default:
                ``torch.device("cpu")``.
            dtype (torch.dtype, optional): Image-domain dtype. Default:
                ``torch.float``.
            **kwargs: Reserved for API compatibility with other backends.

        Raises:
            ImportError: If ``torchkbnufft`` is unavailable.
            NotImplementedError: If ``mode`` is unsupported.
        """
        super().__init__(
            u=u,
            v=v,
            img_size=img_size,
            natural_weight=natural_weight,
            image_weight=image_weight,
            precond_weight=precond_weight,
            real_flag=real_flag,
            device=device,
            dtype=dtype,
        )

        # Import lazily to avoid hard dependency when this backend is unused.
        try:
            import torchkbnufft as tkbn
        except ImportError as e:
            raise ImportError(
                "torchkbnufft is required for MeasOpTkbn but is not installed.\n"
                "Please install it via `pip install torchkbnufft` or refer to "
                "the documentation for more details."
            ) from e

        # Match the historical backend default of 2x oversampling when the
        # caller does not provide an explicit Fourier-grid size.
        if grid_size is None:
            grid_size = tuple(np.array(self._img_size) * 2)

        # Forward NUFFT operator.
        self._nufft_obj = tkbn.KbNufft(
            im_size=self._img_size,
            grid_size=grid_size,
            numpoints=kernel_dim,
            dtype=self._dtype,
            device=self._device,
        )
        # Adjoint NUFFT operator.
        self._adj_nufft_obj = tkbn.KbNufftAdjoint(
            im_size=self._img_size,
            grid_size=grid_size,
            numpoints=kernel_dim,
            dtype=self._dtype,
            device=self._device,
        )

        if mode == "matrix":
            # Precompute sparse interpolation matrices for repeated NUFFT calls.
            self._interp_mats = tkbn.calc_tensor_spmatrix(
                self._traj.squeeze(),
                self._img_size,
                grid_size=grid_size,
                numpoints=kernel_dim,
            )
            self._interp_mats = tuple(
                [
                    t.to(device=self._device, dtype=self._dtype)
                    for t in self._interp_mats
                ]
            )
        elif mode == "table":
            # Use TorchKbNUFFT's built-in table-based interpolation path.
            self._interp_mats = None
        else:
            raise NotImplementedError("Unrecognised mode: " + mode)

    def _GA(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward TKBNUFFT and data weighting.

        Args:
            x (torch.Tensor): Image-domain input tensor.

        Returns:
            torch.Tensor: Weighted measurement-domain tensor.
        """
        return (
            self._nufft_obj(
                x.to(self._dtype_meas), self._traj, interp_mats=self._interp_mats
            )
            * self._data_weight
        )

    def _AtGt(self, y: torch.Tensor) -> torch.Tensor:
        """Apply adjoint TKBNUFFT and data weighting.

        Args:
            y (torch.Tensor): Measurement-domain input tensor.

        Returns:
            torch.Tensor: Image-domain adjoint output tensor.
        """
        return self._adj_nufft_obj(
            y * self._data_weight, self._traj, interp_mats=self._interp_mats
        )
