"""
Base class for optimisers
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional

import torch
import torch.distributed as dist
import numpy as np

from ..ri_measurement_operator import MeasOp


class Optimiser(ABC):
    """
    Base class for optimisers.

    This class provides a common interface for different optimisation algorithms.
    It is initialized with a measurement, a measurement operator, and an optional save path.
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
        """
        General initialization of the Optimiser.
        Prepare the measurement data, measurement operator.
        Generate and svae the dirty image and PSF.

        Args:
            meas (torch.Tensor): The measurement tensor.
            ch_freq (torch.Tensor): The frequency of each channel.
            meas_op (MeasOp): The measurement operator.
            save_pth (str, optional): The path where results will be saved. Defaults to "results".
            file_prefix (str, optional): The prefix of the saving files. Defaults to None.
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

        # common initialisation
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

        # init distributed
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

    @abstractmethod
    def initialisation(self) -> Any:
        """
        Initalise parameters for specific parameters.
        This method should be called before running the optimiser.

        Returns:
            NotImplemented: This method should be implemented in a subclass.
        """
        return NotImplemented

    @abstractmethod
    def run(self) -> Any:
        """
        Runs the main loop of the optimiser.

        Returns:
            NotImplemented: This method should be implemented in a subclass.
        """
        return NotImplemented

    @abstractmethod
    def _each_iter_begin(self) -> Any:
        """
        This method should be called at the beginning of each iteration.

        Returns:
            NotImplemented: This method should be implemented in a subclass.
        """
        return NotImplemented

    @abstractmethod
    def _stop_criteria(self) -> Any:
        """
        Checks the stop criteria.

        Returns:
            NotImplemented: This method should be implemented in a subclass.
        """
        return NotImplemented

    @abstractmethod
    def _each_iter_end(self) -> Any:
        """
        This method should be called at the end of each iteration.

        Returns:
            NotImplemented: This method should be implemented in a subclass.
        """
        return NotImplemented

    @abstractmethod
    def finalisation(self) -> Any:
        """
        This method should be called after the optimiser's main loop finishes.
        It can be used to save the final results, print final log etc.

        Returns:
            NotImplemented: This method should be implemented in a subclass.
        """
        return NotImplemented

    def get_model_image(self, dtype=torch.double, dest: int = 0) -> np.ndarray:
        """
        Get the final model image.

        args:
            dtype (torch.dtype): The data type of the output model image.
            dest (int): Destination rank to gather tensors.

        Returns:
            np.ndarray: The final model.
        """
        model_full = self._gather_ch_img_to_tensor(
            self._model,
            dest=dest,
        )
        if self._rank != dest and self._world_size > 1:
            return None
        return model_full.squeeze().cpu().to(dtype).numpy()

    def get_psf(self, dtype=torch.double, dest: int = 0) -> np.ndarray:
        """
        Get the PSF.

        args:
            dtype (torch.dtype): The data type of the output PSF.
            dest (int): Destination rank to gather tensors.
        Returns:
            np.ndarray: The PSF.
        """
        psf_full = self._gather_ch_img_to_tensor(
            self._psf,
            dest=dest,
        )
        if self._rank != dest and self._world_size > 1:
            return None
        return psf_full.squeeze().cpu().to(dtype).numpy()

    def get_dirty_image(
        self, normalise=None, dtype=torch.double, dest: int = 0
    ) -> Optional[np.ndarray]:
        """
        Get the dirty image.

        args:
            dtype (torch.dtype): The data type of the output dirty image.
            dest (int): Destination rank to gather tensors.

        Returns:
            np.ndarray: The dirty image.
        """
        dirty_full = self._gather_ch_img_to_tensor(
            self._meas_bp / (normalise if normalise is not None else 1.0),
            dest=dest,
        )
        if self._rank != dest and self._world_size > 1:
            return None
        return dirty_full.squeeze().cpu().to(dtype).numpy()

    def get_residual_image(
        self, dtype=torch.double, dest: int = 0
    ) -> Optional[np.ndarray]:
        """
        Get the residual image.

        args:
            dtype (torch.dtype): The data type of the output residual image.
            dest (int): Destination rank to gather tensors.

        Returns:
            np.ndarray: The residual image.
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
        """
        Gather tensors have different number of channels from all ranks to the tensor at destination rank.
        Args:
            tensor_ch_rank (torch.Tensor): Tensor on each rank with shape (1, rank_ch_num, H, W).
        Returns:
            torch.Tensor: Gathered tensor on the destination rank with shape (1, total_ch_num, H, W)
                at the destination rank, None on other ranks.
        """
        if self._world_size <= 1:
            return tensor_ch_rank

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

        gather_list = (
            [torch.empty_like(tensor_pad) for _ in range(self._world_size)]
            if self._rank == dest
            else None
        )
        dist.gather(tensor_pad, gather_list=gather_list, dst=dest)
        if self._rank != dest:
            return None
        tensor_full = torch.cat(
            [
                gather_list[r][:, : self._ch_num_each_rank[r], :, :]
                for r in range(self._world_size)
            ],
            dim=1,
        )
        return tensor_full
