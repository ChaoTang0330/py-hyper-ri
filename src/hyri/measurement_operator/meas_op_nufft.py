"""Define NUFFT-based measurement-operator abstractions.

This file provides the shared NUFFT measurement-operator class used by concrete
backends (for example FINUFFT and TKBNUFFT implementations).
"""

from abc import abstractmethod
from typing import Any, Tuple

import torch

from .meas_op import MeasOp


class MeasOpNUFFT(MeasOp):
    """Base class for NUFFT-based measurement operators.

    This class stores shared sampling pattern (trajectory) and weighting tensors,
    and defines the common forward/adjoint interfaces used by backend-specific
    NUFFT operators. It also provides :meth:`get_op_norm_prime` for estimating
    the spectral norm of the effective operator when the image-weight term is
    applied twice, which is used for heuristic noise correction factor in the
    imaging pipeline.

    Shape:
        Image tensor: ``(1, 1, H, W)``.
        Visibility tensor: ``(1, 1, N)``.
        Sampling-pattern coordinates :math:`u` and :math:`v`: ``(1, 1, N)``.

    Attributes:
        _traj (torch.Tensor): Stacked sampling trajectory in ``(v, u)`` order.
        _natural_weight (torch.Tensor): Natural weighting tensor.
        _image_weight (torch.Tensor): Imaging weighting tensor.
        _data_weight (torch.Tensor): Product of natural and imaging weights.
        _precond_weight (torch.Tensor): Data-space preconditioning tensor.
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
        **kwargs: Any,
    ) -> None:
        """Initialise NUFFT trajectory and weighting state.

        Args:
            u (torch.Tensor): ``u`` sampling coordinates in radians.
            v (torch.Tensor): ``v`` sampling coordinates in radians.
            img_size (Tuple[int, int]): Target image size ``(H, W)``.
            natural_weight (torch.Tensor, optional): Natural weighting tensor.
                Default: ``torch.ones(1, 1, 1)``.
            image_weight (torch.Tensor, optional): Imaging weighting tensor.
                Default: ``torch.ones(1, 1, 1)``.
            precond_weight (torch.Tensor, optional): Preconditioning tensor
                applied in data space. Default: ``torch.ones(1, 1, 1)``.
            real_flag (bool, optional): If True, keep adjoint outputs real
                when applicable. Default: True.
            device (torch.device, optional): Execution device. Default:
                ``torch.device("cpu")``.
            dtype (torch.dtype, optional): Image-domain dtype. Default:
                ``torch.float``.
            **kwargs: Backend-specific optional arguments accepted by
                subclasses.
        """
        super().__init__(
            img_size=img_size,
            real_flag=real_flag,
            device=device,
            dtype=dtype,
        )
        # Use (v, u) ordering to match downstream NUFFT backend conventions.
        self._traj = torch.cat((v, u), dim=1).to(dtype=self._dtype, device=self._device)

        self._natural_weight = natural_weight.to(
            dtype=self._dtype_meas, device=self._device
        )
        self._image_weight = image_weight.to(
            dtype=self._dtype_meas, device=self._device
        )

        self._data_weight = (self._natural_weight * self._image_weight).to(
            dtype=self._dtype_meas, device=self._device
        )
        self._precond_weight = precond_weight.to(
            dtype=self._dtype_meas, device=self._device
        )
        self._grid_size = None
        self._kernel_dim = None
        self._mode = None

    @abstractmethod
    def _GA(self, x: torch.Tensor) -> torch.Tensor:
        """Apply backend-specific forward NUFFT with imaging weighting.

        Args:
            x (torch.Tensor): Image-domain input.

        Returns:
            torch.Tensor: Visibility-domain output.
        """
        return NotImplemented

    @abstractmethod
    def _AtGt(self, y: torch.Tensor) -> torch.Tensor:
        """Apply backend-specific adjoint NUFFT with imaging weighting.

        Args:
            y (torch.Tensor): Visibility-domain input.

        Returns:
            torch.Tensor: Image-domain adjoint output.
        """
        return NotImplemented

    def forward_op(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the forward measurement operator.

        Args:
            x (torch.Tensor): Image-domain input tensor.

        Returns:
            torch.Tensor: Visibility-domain output tensor.
        """

        return self._GA(x) * self._precond_weight

    def adjoint_op(self, y: torch.Tensor) -> torch.Tensor:
        """Apply the adjoint measurement operator.

        Args:
            y (torch.Tensor): Visibility-domain input tensor.

        Returns:
            torch.Tensor: Image-domain output tensor.
        """

        if self._real_flag:
            return torch.real(self._AtGt(y * self._precond_weight))

        return self._AtGt(y * self._precond_weight)

    def set_precond_weight(self, precond_weight: torch.Tensor) -> None:
        """Set the data-space preconditioning weight.

        Args:
            precond_weight (torch.Tensor): Preconditioning tensor.

        Notes:
            Updating this tensor invalidates the cached operator norm.
        """
        self._precond_weight = precond_weight.to(
            dtype=self._dtype_meas, device=self._device
        )
        self._op_norm = None

    @torch.no_grad()
    def get_op_norm_prime(
        self, rel_tol: float = 1e-5, max_iter: int = 500, verbose: bool = False
    ) -> float:
        r"""Estimate the norm used for heuristic-noise correction.

        This method temporarily sets the preconditioner to ``_image_weight`` and
        computes :math:`\|\Phi^\dagger \Phi\|_2` using :meth:`get_op_norm`.
        Under this substitution, the image-weight contribution is applied twice
        in the effective normal operator.

        Args:
            rel_tol (float, optional): Relative tolerance for power-iteration
                stopping. Default: 1e-5.
            max_iter (int, optional): Maximum number of power iterations.
                Default: 500.
            verbose (bool, optional): Print iteration progress. Default: False.

        Returns:
            float: Spectral norm estimate for the twice-weighted operator.
        """
        if not torch.allclose(self._image_weight, torch.ones_like(self._image_weight)):
            precond_weight = self._precond_weight
            op_norm = self._op_norm
            self._op_norm = None

            # Temporarily reuse the preconditioning slot to probe the
            # twice-weighted operator norm.
            self._precond_weight = self._image_weight
            op_norm_prime = self.get_op_norm(
                rel_tol=rel_tol, max_iter=max_iter, verbose=verbose
            )
            self._precond_weight = precond_weight
            self._op_norm = op_norm
        else:
            op_norm_prime = self.get_op_norm()

        return op_norm_prime
