"""Implement a positivity projection regulariser.

This file provides a proximal operator that projects image-domain values
onto the non-negative orthant.
"""

import torch

from .regulariser import Regulariser

DEFAULT_DEVICE = torch.device("cpu")


class ProxOpPositivity(Regulariser):
    """Project tensors onto the non-negative orthant.

    This regulariser applies an element-wise positivity projection equivalent
    to ReLU and does not maintain adaptive state across iterations. The
    operator is :math:`\operatorname{prox}_{\iota_{\mathbb{R}_{+}}}(x)=\max(x,0)`.

    Shape:
        Input: Any tensor shape (for example ``(N, C, H, W)``).
        Output: Tensor with the same shape as input, with negative values
          clipped to zero.

    Example:
        >>> prox = ProxOpPositivity()
        >>> prox(torch.tensor([[-1.0, 2.0]]))
        tensor([[0., 2.]])
    """

    def __init__(
        self,
        rank_num_ch: int = 1,
        rank_ch_start: int = 0,
        device: torch.device = DEFAULT_DEVICE,
        dtype: torch.dtype = torch.float,
        verbose: bool = True,
    ) -> None:
        """Initialise the positivity regulariser.

        Args:
            rank_num_ch (int, optional): Number of channels handled by this
                rank. Default: 1.
            rank_ch_start (int, optional): Global start index of local
                channels. Default: 0.
            device (torch.device, optional): Execution device. Default:
                torch.device("cpu").
            dtype (torch.dtype, optional): Tensor dtype for computation.
                Default: torch.float.
            verbose (bool, optional): Enable progress messages. Default: True.
        """
        super().__init__(
            rank_num_ch=rank_num_ch,
            rank_ch_start=rank_ch_start,
            device=device,
            dtype=dtype,
        )
        self._verbose = verbose

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Project an input tensor onto the positivity set.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Element-wise non-negative tensor with the same shape
            and dtype as ``x``.
        """
        return torch.nn.functional.relu(x)

    def update(self, *args, **kwargs) -> None:
        """Keep the operator state unchanged.

        This regulariser is parameter-free during optimisation.
        """
        pass
