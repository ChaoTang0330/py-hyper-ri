"""Define the abstract base class for measurement operators.

This file provides the shared interface and utility methods used by concrete
measurement operators (forward operator, adjoint operator, spectral-norm estimation,
and PSF evaluation).
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch


class MeasOp(ABC):
    """Abstract base class for measurement operators.

    Subclasses implement :meth:`forward_op` and :meth:`adjoint_op` for a
    specific measurement model. The base class also provides utility methods
    for operator-norm estimation and PSF evaluation.

    Shape:
        Image tensor: ``(B, C, H, W)``.
        Visibility tensor: ``(B, C, N)``.

    Attributes:
        _img_size (Tuple[int, int]): Target image size ``(H, W)``.
        _real_flag (bool): Whether adjoint outputs are returned as real-valued
            when applicable.
        _dtype (torch.dtype): Image-domain dtype.
        _dtype_meas (torch.dtype): Measurement-domain dtype.
        _device (torch.device | None): Execution device.
        _op_norm (Optional[float]): Cached spectral-norm estimate.
    """

    def __init__(
        self,
        img_size: Tuple[int, int],
        real_flag: bool = True,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float,
    ) -> None:
        """Initialise shared measurement-operator state.

        Args:
            img_size (tuple[int, int]): Spatial image size ``(H, W)``.
            real_flag (bool, optional): Whether the adjoint output is treated
                as real-valued when applicable. Default: True.
            device (torch.device | None, optional): Execution device.
                Default: None.
            dtype (torch.dtype, optional): Image-domain tensor dtype.
                Default: torch.float.
        """
        self._img_size = img_size
        self._real_flag = real_flag
        self._dtype = dtype
        # Choose measurement-domain complex dtype from image-domain precision.
        if self._dtype in [torch.float, torch.float32]:
            self._dtype_meas = torch.complex64
        else:
            self._dtype_meas = torch.complex128

        self._device = device
        self._op_norm = None

    @abstractmethod
    def forward_op(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the forward measurement operator.

        Args:
            x (torch.Tensor): Image-domain input tensor.

        Returns:
            torch.Tensor: Measurement-domain output tensor.
        """
        return NotImplemented

    @abstractmethod
    def adjoint_op(self, y: torch.Tensor) -> torch.Tensor:
        """Apply the adjoint measurement operator.

        Args:
            y (torch.Tensor): Measurement-domain input tensor.

        Returns:
            torch.Tensor: Image-domain output tensor.
        """
        return NotImplemented

    def set_real_flag(self, real_flag: bool) -> None:
        """Set whether adjoint outputs are returned as real values.

        Args:
            real_flag (bool): Real-output flag.

        Notes:
            Updating this flag invalidates any cached operator norm.
        """
        self._real_flag = real_flag
        self._op_norm = None

    def get_device(self) -> Optional[torch.device]:
        """Return the device used by the measurement operator.

        Returns:
            torch.device | None: Current execution device.
        """
        return self._device

    def get_data_type(self) -> torch.dtype:
        """Return the image-domain tensor dtype.

        Returns:
            torch.dtype: Image-domain dtype.
        """
        return self._dtype

    def get_data_type_meas(self) -> torch.dtype:
        """Return the measurement-domain tensor dtype.

        Returns:
            torch.dtype: Measurement-domain dtype.
        """
        return self._dtype_meas

    def get_img_size(self) -> Tuple[int, int]:
        """Return the configured image size.

        Returns:
            tuple[int, int]: Image size ``(H, W)``.
        """
        return self._img_size

    @torch.no_grad()
    def get_op_norm(
        self,
        compute_flag: bool = False,
        rel_tol: float = 1e-5,
        max_iter: int = 500,
        verbose: bool = False,
    ) -> float:
        """Estimate or return the spectral norm of the normal operator.

        Args:
            compute_flag (bool, optional): Force recomputation even if a cached
                value exists. Default: False.
            rel_tol (float, optional): Relative tolerance for the power-method
                stopping criterion. Default: 1e-5.
            max_iter (int, optional): Maximum number of power iterations.
                Default: 500.
            verbose (bool, optional): Print iteration progress. Default: False.

        Returns:
            float: Estimated spectral norm.
        """
        if self._op_norm is None or compute_flag:
            x = torch.randn(1, 1, *self._img_size, device=self._device, dtype=self._dtype)
            x = x / torch.linalg.vector_norm(x).item()
            init_val = 1.0

            for k in range(max_iter):
                # Power iteration on \Phi^\dagger \Phi.
                x = self.adjoint_op(self.forward_op(x))
                val = torch.linalg.vector_norm(x)
                rel_var = abs(val - init_val) / init_val
                if rel_var < rel_tol:
                    break
                init_val = val
                x = x / val
                if verbose:
                    print(f"Iter = {k}, norm = {val.item()}", flush=True)
            self._op_norm = val.item()

        return self._op_norm

    @torch.no_grad()
    def get_psf(self) -> torch.Tensor:
        r"""Compute the point-spread function (PSF).

        Returns:
            torch.Tensor: PSF obtained by applying :math:`\Phi^\dagger \Phi`
            to a centered Dirac impulse.
        """
        dirac = torch.zeros(1, 1, *self._img_size, dtype=self._dtype, device=self._device)
        dirac[0, 0, self._img_size[0] // 2, self._img_size[1] // 2] = 1.0
        psf = self.adjoint_op(self.forward_op(dirac))

        return psf
