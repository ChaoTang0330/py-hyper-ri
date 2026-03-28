"""
Base class for proximity operator
"""

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.distributed as dist


class Regulariser(ABC):
    """
    Base class for regulariser.

    This class provides the base functionality for a regulariser.
    It defines common methods for various regularisers, such as

    applying the operator to an input and updating the operator.
    The actual implementation of these methods are left to the subclasses.
    """

    def __init__(
        self,
        rank_num_ch: int = 1,
        rank_ch_start: int = 0,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float,
    ) -> None:
        """
        Initialize the Regulariser class.

        Args:
            device (torch.device, optional): The device for the tensors.
                Defaults to torch.device("cpu").
            dtype (torch.dtype, optional): The data type for the tensors.
                Defaults to torch.float.
        """
        self._rank_num_ch = rank_num_ch
        self._rank_ch_start = rank_ch_start
        self._dtype = dtype
        self._device = device

        # distributed parameters
        self._rank = 0
        self._world_size = 1
        self._is_dist = dist.is_available() and dist.is_initialized()
        if self._is_dist:
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> Any:
        """
        Apply regulariser to input.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            NotImplemented: This method should be implemented in a subclass.
        """
        return NotImplemented

    @abstractmethod
    def update(self, *args, **kwargs) -> Any:
        """
        Update regulariser.

        This method should be implemented in a subclass.
        The input arguments can be arbitrary and should be specified in the subclass.
        """
        return NotImplemented

    def get_device(self) -> torch.device:
        """
        Return the device that the regulariser is running on.

        Returns:
            torch.device: The device that the regulariser is running on.
        """
        return self._device

    def get_data_type(self) -> torch.dtype:
        """
        Return the data type of the data that the regulariser will accept and return.

        Returns:
            torch.dtype: The data type of the data that the regulariser will accept
                and return.
        """
        return self._dtype
