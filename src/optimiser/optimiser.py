"""Define the abstract base class for optimisers.

This module provides the shared optimiser interface used by all imaging
algorithms together with helper methods for exporting distributed image-domain
products such as the model image, dirty image, PSF, and residual image.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional

import numpy as np
import torch
import torch.distributed as dist

from ..measurement_operator import MeasOp


class Optimiser(ABC):
    """Represent the shared interface for imaging optimisers.

    This class stores measurement data, channel metadata, and common image
    buffers used by optimiser implementations. It also defines the lifecycle
    hooks implemented by concrete forward-backward and primal-dual algorithms.
    In distributed runs, it tracks per-rank channel ownership and provides
    helper methods for gathering image-domain tensors back to a destination
    rank.

    Attributes:
        _meas_op (MeasOp): Measurement operator used by the optimiser.
        _ch_freq (torch.Tensor): Local channel frequencies with shape
            ``(1, C_rank, 1, 1)``.
        _meas (torch.Tensor | tuple[torch.Tensor, ...]): Local measurement
            tensors moved to the measurement-operator device.
        _save_pth (str): Output directory for saved products.
        _file_prefix (str): Prefix appended to saved file names.
        _rank_ch_num (int): Number of channels owned by the current rank.
        _rank_ch_start (int): Global starting channel index for the current
            rank.
        _meas_bp (torch.Tensor): Backprojected dirty image cube.
        _model (torch.Tensor): Current model image cube.
        _model_prev (torch.Tensor): Previous model image cube.
        _psf (torch.Tensor): Local PSF cube.
        _is_distributed (bool): Whether ``torch.distributed`` is initialised.
        _rank (int): Distributed rank index.
        _world_size (int): Number of distributed ranks.
        _ch_num_each_rank (tuple[int, ...]): Local channel counts for every
            rank.

    Example:
        >>> # Concrete subclasses implement the abstract optimisation hooks.
        >>> optimiser = SomeOptimiser(meas, ch_freq, meas_op)
        >>> optimiser.initialisation()
        >>> optimiser.run()
    """

    def __init__(
        self,
        meas: List[torch.Tensor],
        ch_freq: torch.Tensor,
        meas_op: MeasOp,
        save_pth: str = "results",
        file_prefix: str = "",
        rank_ch_start: int = 0,
    ) -> None:
        """Initialise shared optimiser state.

        Args:
            meas (List[torch.Tensor]): Local measurement tensors, typically one
                tensor per local channel.
            ch_freq (torch.Tensor): Frequencies for the local spectral
                channels.
            meas_op (MeasOp): Measurement operator used for forward and adjoint
                applications.
            save_pth (str, optional): Directory where optimiser products are
                saved. Default: ``"results"``.
            file_prefix (str, optional): Prefix prepended to saved file names.
                Default: ``""``.
            rank_ch_start (int, optional): Global starting channel index owned
                by the current rank. Default: 0.
        """
        self._meas_op = meas_op
        self._ch_freq = ch_freq.to(
            device=self._meas_op.get_device(), dtype=self._meas_op.get_data_type()
        ).view(1, -1, 1, 1)
        if isinstance(meas, torch.Tensor):
            self._meas = meas.to(
                dtype=self._meas_op.get_data_type_meas(),
                device=self._meas_op.get_device(),
            )
        else:
            self._meas = tuple(
                m.to(
                    dtype=self._meas_op.get_data_type_meas(),
                    device=self._meas_op.get_device(),
                )
                for m in meas
            )
        self._save_pth = save_pth
        self._file_prefix = file_prefix
        self._rank_ch_num = len(meas)
        self._rank_ch_start = rank_ch_start

        # Allocate shared image-domain buffers for all optimiser variants.
        self._meas_bp = torch.zeros(
            1,
            self._rank_ch_num,
            *self._meas_op.get_img_size(),
            device=self._meas_op.get_device(),
            dtype=self._meas_op.get_data_type()
        )
        self._model = torch.zeros_like(self._meas_bp)
        self._model_prev = torch.zeros_like(self._meas_bp)
        self._psf = torch.zeros_like(self._meas_bp)

        # Record distributed metadata and per-rank channel counts.
        self._is_distributed = dist.is_available() and dist.is_initialized()
        self._rank = dist.get_rank() if self._is_distributed else 0
        self._world_size = dist.get_world_size() if self._is_distributed else 1
        self._ch_num_each_rank = (self._rank_ch_num,)
        if self._is_distributed:
            ch_num_each_rank_tensor = torch.zeros(
                self._world_size,
                dtype=torch.int64,
                device=self._meas_op.get_device(),
            )
            dist.all_gather_into_tensor(
                ch_num_each_rank_tensor,
                torch.tensor(
                    [self._rank_ch_num],
                    dtype=torch.int64,
                    device=self._meas_op.get_device(),
                ),
            )
            self._ch_num_each_rank = tuple(
                ch_num_each_rank_tensor.cpu().numpy().tolist()
            )

            # Select a control device for distributed tensor exchanges.
            dist_backend = dist.get_backend()
            if dist_backend == "nccl":
                self._dist_device = torch.device("cuda")
            elif dist_backend == "xccl":
                self._dist_device = torch.device("xpu")
            else:
                # for gloo and mpi
                self._dist_device = torch.device("cpu")

    @abstractmethod
    def initialisation(self) -> Any:
        """Initialise algorithm-specific state before the optimisation loop.

        Returns:
            Any: Subclass-defined initialisation result.
        """
        return NotImplemented

    @abstractmethod
    def run(self) -> Any:
        """Run the main optimisation loop.

        Returns:
            Any: Subclass-defined run result.
        """
        return NotImplemented

    @abstractmethod
    def _each_iter_begin(self) -> Any:
        """Run subclass logic at the beginning of each iteration.

        Returns:
            Any: Subclass-defined iteration state.
        """
        return NotImplemented

    @abstractmethod
    def _stop_criteria(self) -> Any:
        """Evaluate the stopping criterion for the optimisation loop.

        Returns:
            Any: Subclass-defined stopping indicator.
        """
        return NotImplemented

    @abstractmethod
    def _each_iter_end(self) -> Any:
        """Run subclass logic at the end of each iteration.

        Returns:
            Any: Subclass-defined iteration state.
        """
        return NotImplemented

    @abstractmethod
    def finalisation(self) -> Any:
        """Finalise optimisation after the main loop terminates.

        Returns:
            Any: Subclass-defined finalisation result.
        """
        return NotImplemented

    def get_model_image(
        self, dtype: torch.dtype = torch.double, dest: int = 0
    ) -> Optional[np.ndarray]:
        """Return the gathered model image cube.

        Args:
            dtype (torch.dtype, optional): Output dtype used before converting
                to NumPy. Default: ``torch.double``.
            dest (int, optional): Destination rank for distributed gathering.
                Default: 0.

        Returns:
            np.ndarray | None: Gathered model image on ``dest``, or ``None`` on
            non-destination ranks in distributed runs.
        """
        model_full = self._gather_ch_img_to_tensor(
            self._model,
            dest=dest,
        )
        if self._rank != dest and self._world_size > 1:
            return None
        return model_full.squeeze().cpu().to(dtype).numpy()

    def get_psf(
        self, dtype: torch.dtype = torch.double, dest: int = 0
    ) -> Optional[np.ndarray]:
        """Return the gathered PSF cube.

        Args:
            dtype (torch.dtype, optional): Output dtype used before converting
                to NumPy. Default: ``torch.double``.
            dest (int, optional): Destination rank for distributed gathering.
                Default: 0.

        Returns:
            np.ndarray | None: Gathered PSF on ``dest``, or ``None`` on
            non-destination ranks in distributed runs.
        """
        psf_full = self._gather_ch_img_to_tensor(
            self._psf,
            dest=dest,
        )
        if self._rank != dest and self._world_size > 1:
            return None
        return psf_full.squeeze().cpu().to(dtype).numpy()

    def get_dirty_image(
        self,
        normalise: Optional[float] = None,
        dtype: torch.dtype = torch.double,
        dest: int = 0,
    ) -> Optional[np.ndarray]:
        """Return the gathered dirty image cube.

        Args:
            normalise (float | None, optional): Optional scalar divisor applied
                to the dirty image before gathering. Default: None.
            dtype (torch.dtype, optional): Output dtype used before converting
                to NumPy. Default: ``torch.double``.
            dest (int, optional): Destination rank for distributed gathering.
                Default: 0.

        Returns:
            np.ndarray | None: Gathered dirty image on ``dest``, or ``None`` on
            non-destination ranks in distributed runs.
        """
        dirty_full = self._gather_ch_img_to_tensor(
            self._meas_bp / (normalise if normalise is not None else 1.0),
            dest=dest,
        )
        if self._rank != dest and self._world_size > 1:
            return None
        return dirty_full.squeeze().cpu().to(dtype).numpy()

    def get_residual_image(
        self, dtype: torch.dtype = torch.double, dest: int = 0
    ) -> Optional[np.ndarray]:
        """Return the gathered residual image cube.

        Args:
            dtype (torch.dtype, optional): Output dtype used before converting
                to NumPy. Default: ``torch.double``.
            dest (int, optional): Destination rank for distributed gathering.
                Default: 0.

        Returns:
            np.ndarray | None: Gathered residual image on ``dest``, or ``None``
            on non-destination ranks in distributed runs.
        """
        residual_rank = self._meas_bp - self._meas_op.adjoint_op(
            self._meas_op.forward_op(self._model)
        )
        residual_full = self._gather_ch_img_to_tensor(
            residual_rank,
            dest=dest,
        )
        if self._rank != dest and self._world_size > 1:
            return None
        return residual_full.squeeze().cpu().to(dtype).numpy()

    def _gather_ch_img_to_tensor(
        self,
        tensor_ch_rank: torch.Tensor,
        dest: int = 0,
    ) -> Optional[torch.Tensor]:
        """Gather channel-partitioned image tensors to a destination rank.

        Args:
            tensor_ch_rank (torch.Tensor): Local tensor with shape
                ``(1, C_rank, H, W)``.
            dest (int, optional): Destination rank for distributed gathering.
                Default: 0.

        Returns:
            torch.Tensor | None: Gathered tensor with shape
            ``(1, C_total, H, W)`` on ``dest``, or ``None`` on non-destination
            ranks.

        Notes:
            Ranks with fewer channels are zero-padded temporarily so
            ``torch.distributed.gather`` can operate on tensors with matching
            shapes before the padding is removed on ``dest``.
        """
        # If not distributed, return the input tensor directly.
        if self._world_size <= 1:
            return tensor_ch_rank

        # Pad local tensors to the same channel count.
        max_ch_num = max(self._ch_num_each_rank)
        if tensor_ch_rank.shape[1] < max_ch_num:
            tensor_pad = torch.cat(
                [
                    tensor_ch_rank,
                    tensor_ch_rank.new_zeros(
                        1,
                        max_ch_num - tensor_ch_rank.shape[1],
                        tensor_ch_rank.shape[2],
                        tensor_ch_rank.shape[3],
                    ),
                ],
                dim=1,
            )
        else:
            tensor_pad = tensor_ch_rank

        # Gather the padded tensors to the destination rank.
        gather_list = (
            [torch.empty_like(tensor_pad) for _ in range(self._world_size)]
            if self._rank == dest
            else None
        )
        dist.gather(tensor_pad, gather_list=gather_list, dst=dest)

        # Exit early on non-destination ranks.
        if self._rank != dest:
            return None
        # Concatenate the gathered tensors and remove padding.
        tensor_full = torch.cat(
            [
                gather_list[r][:, : self._ch_num_each_rank[r], :, :]
                for r in range(self._world_size)
            ],
            dim=1,
        )
        return tensor_full
