"""
Measurement operators built on NUFFT using pytorch_finufft library
"""

from typing import Tuple

import torch

from .meas_op_nufft import MeasOpNUFFT


class MeasOpFinufft(MeasOpNUFFT):
    """
    Measurement operators sampling in continuous spatial Fourier domain.
    Use pytorch_finufft as the NUFFT package, which is a PyTorch wrapper for FINUFFT.
    Uses `exponential of semi-circle' kernel instead of Kaiser-Bessel kernel.
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
        :param device: The device where the measurement operator will run on, defaults to torch.device("cpu")
        :type device: torch.device, optional
        :param dtype: Data precision, defaults to torch.float
        :type dtype: torch.dtype, optional
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

        # import pytorch_finufft here to avoid hard dependency
        try:
            from pytorch_finufft.functional import finufft_type1, finufft_type2
            self._finufft_type1 = finufft_type1
            self._finufft_type2 = finufft_type2
        except ImportError as e:
            raise ImportError(
                "pytorch_finufft is required for MeasOpFinufft but is not installed. "
                "Please install it via `pip install pytorch-finufft` or refer to the documentation for more details."
            ) from e

        self._traj = self._traj.squeeze(0)

    def _GA(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward NUFFT with data weight

        :param x: The input tensor in the image domain.
        :type x: torch.Tensor
        :return: The result of the forward NUFFT operation.
        :rtype: torch.Tensor
        """
        x = x.view(*x.shape[-2:])
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
        """
        Adjoint NUFFT with data weight.

        :param y: The input tensor in the Fourier domain.
        :type y: torch.Tensor
        :return: The result of the adjoint NUFFT operation.
        :rtype: torch.Tensor
        """
        return self._finufft_type1(
            self._traj,
            y * self._data_weight,
            self._img_size,
            upsampfac=2.0,
            modeord=0,
            isign=1,
        )
