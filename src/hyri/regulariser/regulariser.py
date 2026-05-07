"""Define the abstract base API for regularisers.

This file provides the shared constructor and minimal interface used by all
regulariser implementations in this repository.
"""

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.distributed as dist


class Regulariser(ABC):
    """Define the base interface for regulariser operators.

    Subclasses implement :meth:`__call__` to apply a regularisation step and
    :meth:`update` to refresh internal state between optimisation iterations.

    Attributes:
        _rank_num_ch (int): Number of channels assigned to this rank.
        _rank_ch_start (int): Global index of the first local channel.
        _device (torch.device): Device used for operator state and outputs.
        _dtype (torch.dtype): Tensor dtype used by the operator.
        _is_dist (bool): Whether distributed execution is enabled.
        _rank (int): Distributed rank identifier (0 in non-distributed mode).
        _world_size (int): Number of distributed ranks (1 in non-distributed mode).
    """

    def __init__(
        self,
        rank_num_ch: int = 1,
        rank_ch_start: int = 0,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float,
    ) -> None:
        """Initialise shared regulariser state.

        Args:
            rank_num_ch (int, optional): Number of channels handled by this
                rank. Default: 1.
            rank_ch_start (int, optional): Global start index of local
                channels. Default: 0.
            device (torch.device, optional): Execution device for regulariser
                state and outputs. Default: torch.device("cpu").
            dtype (torch.dtype, optional): Tensor dtype used by the
                regulariser. Default: torch.float.

        Notes:
            When torch.distributed is initialised, this constructor caches
            rank metadata and chooses a point-to-point communication device
            based on the active backend.
        """
        self._rank_num_ch = rank_num_ch
        self._rank_ch_start = rank_ch_start
        self._dtype = dtype
        self._device = device

        # Distributed execution metadata.
        self._rank = 0
        self._world_size = 1
        self._is_dist = dist.is_available() and dist.is_initialized()
        if self._is_dist:
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()
            # Select the device used for point-to-point communication buffers.
            dist_backend = dist.get_backend()
            if dist_backend == "nccl":
                self._dist_device = torch.device("cuda")
            elif dist_backend == "xccl":
                self._dist_device = torch.device("xpu")
            else:
                # Backends such as gloo and mpi use host tensors.
                self._dist_device = torch.device("cpu")

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> Any:
        """Apply the regulariser to an input tensor.

        Args:
            x (torch.Tensor): Input tensor in image domain.

        Returns:
            Any: Regularised output produced by a concrete implementation.
        """
        return NotImplemented

    @abstractmethod
    def update(self, *args, **kwargs) -> Any:
        """Update internal regulariser state.

        Args:
            *args (Any): Positional arguments required by the subclass.
            **kwargs (Any): Keyword arguments required by the subclass.

        Returns:
            Any: Optional update artefacts returned by the subclass.
        """
        return NotImplemented

    def get_device(self) -> torch.device:
        """Return the regulariser execution device.

        Returns:
            torch.device: Device used for internal state and outputs.
        """
        return self._device

    def get_data_type(self) -> torch.dtype:
        """Return the regulariser tensor dtype.

        Returns:
            torch.dtype: Dtype expected for inputs and produced for outputs.
        """
        return self._dtype
