"""
Proximity operator for projecting vectors to elipse
"""

import math
from typing import List, Tuple, Optional

import torch
import torch.distributed as dist

from .regulariser import Regulariser


class ProxEllipsoid(Regulariser):
    """
    Proximity operator for projecting vectors to ellipsoid (weighted l2 ball
    with preconditioning matrix).

    It uses the forward-backward algorithm to solve the projection.
    """

    def __init__(
        self,
        center: List[torch.Tensor],
        precond_weight: Optional[List[torch.Tensor]] = None,
        radius: Optional[torch.Tensor] = None,
        rank_num_ch: int = 1,
        rank_ch_start: int = 0,
        itr_min: int = 1,
        itr_max: int = 10,
        rel_var_tol: float = 1e-6,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.complex64,
        verbose: bool = False,
    ):
        """
        Initializes the Elipse proximity operator with the given parameters.

        Args:
            center (List[torch.Tensor]): The center of the elipse.
            precond_weight (Optional[List[torch.Tensor]], optional): The preconditioning weight.
                Defaults to None, which means no preconditioning.
            radius (Optional[torch.Tensor], optional): The radius of the elipse.
                Defaults to None, which means no projection.
            itr_min (int, optional): The minimum number of iterations for the forward-backward algorithm.
                Defaults to 1.
            itr_max (int, optional): The maximum number of iterations for the forward-backward algorithm.
                Defaults to 10.
            rel_var_tol (float, optional): The relative variation tolerance for convergence in the forward-backward algorithm.
                Defaults to 1e-6.
            device (torch.device, optional): The device for the tensors. Defaults to torch.device("cpu").
            dtype (torch.dtype, optional): The data type for the tensors. Defaults to torch.complex64.
            verbose (bool, optional): Whether to print verbose information during the forward-backward algorithm. Defaults to False.
        """
        super().__init__(
            rank_num_ch=rank_num_ch,
            rank_ch_start=rank_ch_start,
            device=device,
            dtype=dtype,
        )

        self._center = tuple(
            c.to(device=self._device, dtype=self._dtype) for c in center
        )
        self._itr_min = max(1, itr_min)
        self._itr_max = max(self._itr_min, itr_max)
        self._rel_var_tol = rel_var_tol
        self._verbose = verbose
        self._rank_num_ch = len(center)
        self._rank_ch_start = rank_ch_start

        if precond_weight is None:
            precond_weight = [
                torch.ones(1, device=self._device, dtype=self._dtype)
                for _ in range(self._rank_num_ch)
            ]
        else:
            precond_weight = [
                w.to(device=self._device, dtype=self._dtype) for w in precond_weight
            ]
        self._precond_weight = tuple(precond_weight)
        if radius is None:
            # The theoretical l2 error bounds are used as the default radius of the \ell_2 ball,
            # assume chi-square distribution and tau=1
            radius = torch.tensor(
                [
                    math.sqrt(torch.numel(c) + 2.0 * math.sqrt(torch.numel(c)))
                    for c in center
                ]
            )
        if self._dtype in (torch.float64, torch.complex128):
            radius_dtype = torch.float64
        else:
            radius_dtype = torch.float32
        self._radius = radius.to(device=self._device, dtype=radius_dtype)

        # initialisation
        self._prev = [torch.zeros_like(c) for c in self._center]
        self._gd_step_size = tuple(
            1.0 / torch.max(torch.abs(self._precond_weight[i])).item() ** 2
            for i in range(self._rank_num_ch)
        )

    @torch.no_grad()
    def __call__(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Project input onto elipse based on forward-backward algorithm.

        Args:
            x (List[torch.Tensor]): The input tensors.

        Returns:
            List[torch.Tensor]: The projected tensors.
        """
        for i in range(self._itr_max):
            result = self._proj_l2_ball(self._grad_descent(self._prev, x))
            x_norm = sum(torch.linalg.vector_norm(x_ch) ** 2 for x_ch in x)
            diff_norm = sum(
                torch.linalg.vector_norm(result[i] - self._prev[i]) ** 2
                for i in range(self._rank_num_ch)
            )
            if self._is_dist:
                dist.all_reduce(x_norm, op=dist.ReduceOp.SUM)
                dist.all_reduce(diff_norm, op=dist.ReduceOp.SUM)
            rel_var = torch.sqrt(diff_norm / x_norm)
            self._prev = result
            if i >= self._itr_min - 1 and rel_var.item() <= self._rel_var_tol:
                break
        return result

    @torch.no_grad()
    def _grad_descent(
        self, result: List[torch.Tensor], target: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Perform one step of gradient descent.

        Args:
            result (List[torch.Tensor]): The current result tensors.
            target (List[torch.Tensor]): The target tensors.

        Returns:
            List[torch.Tensor]: The updated tensors after one step of gradient descent.
        """
        return [
            result[i]
            - self._gd_step_size[i] * self._precond_weight[i] * (result[i] - target[i])
            for i in range(self._rank_num_ch)
        ]

    @torch.no_grad()
    def _proj_l2_ball(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Project input onto l2 ball.

        Args:
            x (List[torch.Tensor]): The list of input tensors.

        Returns:
            List[torch.Tensor]: The projected tensors.
        """
        for i in range(self._rank_num_ch):
            x[i] -= self._center[i]
            x_i_norm = torch.linalg.vector_norm(x[i])
            x[i] *= torch.minimum(
                self._radius[i] / x_i_norm,
                torch.ones_like(x_i_norm),
            )
            x[i] += self._center[i]
        return x

    def update(self, *args, **kwargs):
        """
        No updates need for this prox operator.
        """
        pass

    def get_radius(self) -> Tuple[float, ...]:
        """
        Return the radius of the l2 ball when preconditioning weight is not applied.

        Returns:
            float: The radius of the l2 ball.
        """
        return self._radius

    def get_precond_weight(self) -> Tuple[torch.Tensor, ...]:
        """
        Return the preconditioning weight used for the proximity operator.

        Returns:
            torch.Tensor: The preconditioning weight.
        """
        return self._precond_weight
