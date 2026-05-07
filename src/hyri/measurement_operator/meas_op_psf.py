"""Define a PSF-based measurement operator.

This operator uses a precomputed oversampled PSF to perform forward and
adjoint compound operations that approximate the normal operator.
"""

from typing import Any

import torch

from .meas_op import MeasOp
from .meas_op_nufft_finufft import MeasOpFinufft

DEFAULT_DEVICE = torch.device("cpu")
DEFAULT_WEIGHT = torch.ones(1, 1, 1)


class MeasOpPSF(MeasOp):
    """Represent a PSF-based measurement operator for dirty-image updates.

    This class approximates the normal operator with convolution by a cached
    point-spread function (PSF). A 2x field-of-view PSF is constructed once
    with :class:`MeasOpFinufft`, transformed to Fourier space, and then reused
    to evaluate adjoint applications efficiently by FFT-based convolution.

    Shape:
        Image tensor: ``(1, 1, H, W)``.
        Visibility tensor: Not used explicitly; this operator applies a
            PSF-based approximation in the image domain.
        Sampling-pattern coordinates :math:`u` and :math:`v`: ``(1, 1, N)``.

    Attributes:
        _u (torch.Tensor): Sampling ``u`` coordinates.
        _v (torch.Tensor): Sampling ``v`` coordinates.
        _psf_fft (torch.Tensor): Fourier transform of the cached 2x-FoV PSF.

    Example:
        >>> import torch
        >>> u = (2 * torch.pi) * torch.rand(1, 1, 8) - torch.pi
        >>> v = (2 * torch.pi) * torch.rand(1, 1, 8) - torch.pi
        >>> op = MeasOpPSF(u=u, v=v, img_size=(16, 16))
        >>> image = torch.randn(1, 1, 16, 16)
        >>> dirty = op.adjoint_op(image)
        >>> dirty.shape
        torch.Size([1, 1, 16, 16])
    """

    def __init__(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        img_size: tuple[int, int],
        natural_weight: torch.Tensor = DEFAULT_WEIGHT,
        image_weight: torch.Tensor = DEFAULT_WEIGHT,
        precond_weight: torch.Tensor = DEFAULT_WEIGHT,
        real_flag: bool = True,
        device: torch.device = DEFAULT_DEVICE,
        dtype: torch.dtype = torch.float,
        **kwargs: Any,
    ) -> None:
        """Initialise the PSF-convolution measurement operator.

        Args:
            u (torch.Tensor): ``u`` sampling coordinates in radians.
            v (torch.Tensor): ``v`` sampling coordinates in radians.
            img_size (Tuple[int, int]): Target image size ``(H, W)``.
            natural_weight (torch.Tensor, optional): Natural weighting tensor.
                Default: ``torch.ones(1, 1)``.
            image_weight (torch.Tensor, optional): Imaging weighting tensor.
                Default: ``torch.ones(1, 1)``.
            precond_weight (torch.Tensor, optional): Data-space
                preconditioning tensor. Default: ``torch.ones(1, 1)``.
            real_flag (bool, optional): If True, return real adjoint outputs.
                Default: True.
            device (torch.device, optional): Execution device. Default:
                ``torch.device("cpu")``.
            dtype (torch.dtype, optional): Image-domain dtype. Default:
                ``torch.float``.
            **kwargs: Reserved for API compatibility.
        """
        super().__init__(
            img_size=img_size,
            real_flag=real_flag,
            device=device,
            dtype=dtype,
        )

        self._u = u.to(dtype=self._dtype, device=torch.device("cpu"))
        self._v = v.to(dtype=self._dtype, device=torch.device("cpu"))

        # Build the oversampled PSF on CPU and cache its FFT on the execution
        # device for repeated convolution-based adjoint applications.
        meas_op2 = MeasOpFinufft(
            u=u.to(dtype=self._dtype, device=torch.device("cpu")),
            v=v.to(dtype=self._dtype, device=torch.device("cpu")),
            img_size=tuple(i * 2 for i in img_size),
            natural_weight=natural_weight.to(
                dtype=self._dtype_meas, device=torch.device("cpu")
            ),
            image_weight=image_weight.to(
                dtype=self._dtype_meas, device=torch.device("cpu")
            ),
            precond_weight=precond_weight.to(
                dtype=self._dtype_meas, device=torch.device("cpu")
            ),
            real_flag=real_flag,
            device=torch.device("cpu"),
            dtype=dtype,
        )

        _psf = meas_op2.get_psf().view(1, 1, *tuple(i * 2 for i in img_size))
        self._psf_fft = torch.fft.fft2(_psf)
        self._psf_fft = self._psf_fft.to(dtype=self._dtype_meas, device=self._device)

    def forward_op(self, x: torch.Tensor) -> torch.Tensor:
        """Return the input image unchanged.

        Args:
            x (torch.Tensor): Image-domain input tensor.

        Returns:
            torch.Tensor: Unmodified input tensor.

        Notes:
            This class approximate the compound forward-adjoint operation with
            convolution with PSF. The forward operation is set to identity
            to ensure a consistent measuerement operator interface.
        """

        return x

    def adjoint_op(self, y: torch.Tensor) -> torch.Tensor:
        """Apply FFT-domain convolution with the cached 2x-FoV PSF.

        Args:
            y (torch.Tensor): Image-domain input tensor.

        Returns:
            torch.Tensor: Convolved image cropped back to the target FoV.

        Notes:
            The input is zero-padded implicitly through the FFT size argument,
            multiplied by the cached PSF spectrum, and cropped to the central
            field of view after inverse FFT.
        """

        # Perform convolution in Fourier space, then crop to the central FoV.
        y_fft = torch.fft.fft2(y, s=self._psf_fft.shape[-2:], dim=(-2, -1))
        res = torch.fft.ifft2(y_fft * self._psf_fft, dim=(-2, -1))[
            ...,
            self._img_size[0] : self._img_size[0] * 2,
            self._img_size[1] : self._img_size[1] * 2,
        ]
        if self._real_flag:
            return torch.real(res)

        return res
