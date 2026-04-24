"""Define the FINUFFT-backed NUFFT measurement operator.

This implementation uses ``pytorch_finufft`` to evaluate nonuniform Fourier
transforms for forward and adjoint measurement operations.
"""

from typing import Any, Tuple

import torch

from .meas_op_nufft import MeasOpNUFFT


class MeasOpFinufft(MeasOpNUFFT):
    """Represent a NUFFT measurement operator backed by FINUFFT.

    This class evaluates nonuniform Fourier sampling with
    ``pytorch_finufft`` for single-channel measurement operators. It reuses
    the weighting and preconditioning logic defined in :class:`MeasOpNUFFT`
    while adapting the shared trajectory representation to FINUFFT's calling
    convention. The forward operator maps image-domain tensors to sampled
    visibilities, and the adjoint operator maps weighted visibilities back to
    the image domain.

    Shape:
        Image tensors: ``(1, 1, H, W)``.
        Visibility-domain tensors: ``(1, 1, N)``.
        Sampling-pattern coordinates :math:`u` and :math:`v`: ``(1, 1, N)``.

    Attributes:
        _traj (torch.Tensor): Sampling trajectory stored in FINUFFT-compatible
            ``(2, N)`` layout.
        _data_weight (torch.Tensor): Combined natural and imaging weights
            applied in measurement space.
        _precond_weight (torch.Tensor): Preconditioning weights applied around
            the backend NUFFT calls.
        _finufft_type1 (Callable): Imported FINUFFT adjoint transform.
        _finufft_type2 (Callable): Imported FINUFFT forward transform.

    Example:
        >>> import torch
        >>> u = (2 * torch.pi) * torch.rand(1, 1, 8) - torch.pi
        >>> v = (2 * torch.pi) * torch.rand(1, 1, 8) - torch.pi
        >>> op = MeasOpFinufft(
        ...     u=u,
        ...     v=v,
        ...     img_size=(16, 16),
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
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float,
        **kwargs: Any
    ) -> None:
        """Initialise the FINUFFT measurement operator.

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
            device (torch.device, optional): Execution device. Default:
                ``torch.device("cpu")``.
            dtype (torch.dtype, optional): Image-domain dtype. Default:
                ``torch.float``.
            **kwargs: Reserved for API compatibility with other backends.

        Raises:
            ImportError: If ``pytorch_finufft`` is unavailable.
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
            from pytorch_finufft.functional import finufft_type1, finufft_type2
            self._finufft_type1 = finufft_type1
            self._finufft_type2 = finufft_type2
        except ImportError as e:
            raise ImportError(
                "pytorch_finufft is required for MeasOpFinufft but is not installed.\n"
                "Please install it via `pip install pytorch-finufft` or refer to the "
                "documentation for more details."
            ) from e

        # FINUFFT consumes coordinates as a 2 x M trajectory tensor.
        self._traj = self._traj.squeeze(0)

    def _GA(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward FINUFFT and data weighting.

        Args:
            x (torch.Tensor): Image-domain input tensor.

        Returns:
            torch.Tensor: Weighted measurement-domain tensor.
        """
        return (
            self._finufft_type2(
                self._traj,
                x.to(self._dtype_meas),
                upsampfac=2.0,
                modeord=0,
                isign=-1,
            )
            * self._data_weight
        )

    def _AtGt(self, y: torch.Tensor) -> torch.Tensor:
        """Apply adjoint FINUFFT and data weighting.

        Args:
            y (torch.Tensor): Measurement-domain input tensor.

        Returns:
            torch.Tensor: Image-domain adjoint output tensor.
        """
        return self._finufft_type1(
            self._traj,
            y * self._data_weight,
            self._img_size,
            upsampfac=2.0,
            modeord=0,
            isign=1,
        )
