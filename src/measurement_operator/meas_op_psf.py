"""
Measurement operators built on NUFFT
"""

from typing import Tuple

import torch

from .meas_op import MeasOp
from .meas_op_nufft_finufft import MeasOpFinufft


class MeasOpPSF(MeasOp):
    """
    Measurement operators sampling in continuous spatial Fourier domain.
    Use PSF with 2x FoV to compute residual dirty image. Such PSF is computed
    using the pytorch_finufft package.
    """

    def __init__(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        img_size: Tuple[int, int],
        natural_weight: torch.Tensor = torch.ones(1, 1),
        image_weight: torch.Tensor = torch.ones(1, 1),
        precond_weight: torch.Tensor = torch.ones(1, 1),
        real_flag: bool = True,
        # normalise_psf: bool = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float,
        **kwargs
    ) -> None:
        """
        Initialize the measurement operator.

        :param u: The u coordinates of the sampling points in units of radians.
        :type u: torch.Tensor
        :param v: The v coordinates of the sampling points in units of radians.
        :type v: torch.Tensor
        :param img_size: The target image size.
        :type img_size: Tuple[int, int]
        :param natural_weight: The natural weight for the measurement operator, defaults to torch.ones(1, 1)
        :type natural_weight: torch.Tensor, optional
        :param image_weight: The image weight for the measurement operator, defaults to torch.ones(1, 1)
        :type image_weight: torch.Tensor, optional
        :param precond_weight:  The preconditioning weight for the measurement operator, defaults to torch.ones(1, 1)
        :type precond_weight: torch.Tensor, optional
        :param real_flag: Flag to indicate the image is real of complex, defaults to True
        :type real_flag: bool, optional
        :param normalise_psf: Enable to normalise the PSF with 2x FoV by its max, defaults to False
        :type normalise_psf: bool, optional
        :param device: The device where the measurement operator will run on, defaults to torch.device("cpu")
        :type device: torch.device, optional
        :param dtype: Data precision, defaults to torch.float
        :type dtype: torch.dtype, optional
        """
        super().__init__(
            img_size=img_size,
            real_flag=real_flag,
            device=device,
            dtype=dtype,
        )

        self._u = u.to(dtype=self._dtype, device=torch.device("cpu"))
        self._v = v.to(dtype=self._dtype, device=torch.device("cpu"))

        # create the PSF with 2x FoV
        meas_op2 = MeasOpFinufft(
            u=u.to(dtype=self._dtype, device=torch.device("cpu")),
            v=v.to(dtype=self._dtype, device=torch.device("cpu")),
            img_size=tuple(i * 2 for i in img_size),
            natural_weight=natural_weight.to(dtype=self._dtype_meas, device=torch.device("cpu")),
            image_weight=image_weight.to(dtype=self._dtype_meas, device=torch.device("cpu")),
            precond_weight=precond_weight.to(dtype=self._dtype_meas, device=torch.device("cpu")),
            real_flag=real_flag,
            device=torch.device("cpu"),
            dtype=dtype,
        )

        _psf = meas_op2.get_psf().view(1, 1, *tuple(i * 2 for i in img_size))
        # if normalise_psf:
        #     _psf /= _psf.max()
        self._psf_fft = torch.fft.fft2(_psf)
        self._psf_fft = self._psf_fft.to(dtype=self._dtype_meas, device=self._device)

        # del meas_op2
        # gc.collect()

    def forward_op(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward measurement operator.

        :param x: The input tensor in image domain.
        :type x: torch.Tensor

        :return: The result of the forward measurement operator with natural, image and preconditioning weights.
        :rtype: torch.Tensor
        """

        return x

    def adjoint_op(self, y: torch.Tensor) -> torch.Tensor:
        """
        Adjoint measurement operator.

        :param y: The input tensor in Fourier domain.
        :type y: torch.Tensor

        :return: The result of the adjoint measurement operator with natural, image and preconditioning weights.
        :rtype: torch.Tensor
        """
        
        y_fft = torch.fft.fft2(y, s=self._psf_fft.shape[-2:], dim=(-2, -1))
        res = torch.fft.ifft2(y_fft * self._psf_fft, dim=(-2, -1))[
            ..., self._img_size[0] : self._img_size[0] * 2, self._img_size[0] : self._img_size[0] * 2
        ]
        if self._real_flag:
            return torch.real(res)

        return res
