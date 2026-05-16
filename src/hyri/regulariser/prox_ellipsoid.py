"""Implement projection onto weighted ellipsoidal constraint sets.

This file provides the proximal operator used by constrained optimisation
variants that require projection onto per-channel weighted :math:`\\ell_2` balls.
"""

import math

import torch
import torch.distributed as dist

from .regulariser import Regulariser

DEFAULT_DEVICE = torch.device("cpu")


class ProxEllipsoid(Regulariser):
    r"""Project channel tensors with a weighted ellipsoidal proximity operator.

    This class implements the non-Euclidean proximity operator used for
    preconditioned data-fidelity updates. For each local channel :math:`i`,
    the data-fidelity term is the indicator function of a shifted weighted
    ellipsoid,

    .. math::
        h_i(\mathbf{z}) = \iota_{\mathcal{B}_i}(\mathbf{z}),

    and the weighted proximity operator is

    .. math::
        \operatorname{prox}^{\mathbf{U}_i}_{h_i}(\mathbf{z})
        =
        \arg\min_{\bar{\mathbf{z}}}
        \iota_{\mathcal{B}_i}(\bar{\mathbf{z}})
        +
        \frac{1}{2}
        (\mathbf{z}-\bar{\mathbf{z}})^\dagger
        \mathbf{U}_i
        (\mathbf{z}-\bar{\mathbf{z}}).

    For diagonal positive preconditioning :math:`\mathbf{U}_i`, this operator is
    equivalent to a skewed projection onto a generalised ellipsoid. In this
    class, :math:`\mathbf{U}_i` is represented through per-channel diagonal
    preconditioning weights, while ``center`` and ``radius`` define the
    corresponding constraint set used by the projection step.

    Shape:
        Input: List of ``C`` tensors, where ``C = len(center) = rank_num_ch``,
            and each tensor ``x[i]`` has the same shape as ``center[i]``.
        Output: List of ``C`` tensors with the same list length and
            per-channel tensor shapes as the input list.

    Attributes:
        _center (tuple[torch.Tensor, ...]): Per-channel ellipsoid centres.
        _precond_weight (tuple[torch.Tensor, ...]): Per-channel preconditioners.
        _radius (torch.Tensor): Per-channel projection radii.
        _gd_step_size (tuple[float, ...]): Per-channel gradient step sizes.

    Example:
        >>> centre = [torch.zeros(4, 4)]
        >>> prox = ProxEllipsoid(center=centre)
        >>> x = [torch.ones(4, 4)]
        >>> y = prox(x)
        >>> y[0].shape
        torch.Size([4, 4])

    References:
        Onose, A., Dabbech, A., & Wiaux, Y. (2017). `An accelerated splitting
        algorithm for radio-interferometric imaging: when natural and uniform
        weighting meet <https://doi.org/10.1093/mnras/stx755>`_. *MNRAS*, 469(1),
        938-949.
    """

    def __init__(
        self,
        center: list[torch.Tensor],
        precond_weight: list[torch.Tensor] | None = None,
        radius: torch.Tensor | None = None,
        rank_num_ch: int = 1,
        rank_ch_start: int = 0,
        itr_min: int = 1,
        itr_max: int = 10,
        rel_var_tol: float = 1e-6,
        device: torch.device = DEFAULT_DEVICE,
        dtype: torch.dtype = torch.complex64,
        verbose: bool = False,
    ):
        """Initialise the ellipsoidal projection operator.

        Args:
            center (List[torch.Tensor]): Per-channel ellipsoid centres.
            precond_weight (Optional[List[torch.Tensor]], optional): Per-channel
                diagonal preconditioning weights. If None, unit weights are
                used. Default: None.
            radius (Optional[torch.Tensor], optional): Per-channel ellipsoid
                radii. If None, radii are initialised from a chi-square-inspired
                bound. Default: None.
            rank_num_ch (int, optional): Number of channels handled by this
                rank. Default: 1.
            rank_ch_start (int, optional): Global start index of local
                channels. Default: 0.
            itr_min (int, optional): Minimum projection iterations. Default: 1.
            itr_max (int, optional): Maximum projection iterations. Default: 10.
            rel_var_tol (float, optional): Relative variation tolerance used for
                stopping. Default: 1e-6.
            device (torch.device, optional): Execution device. Default:
                torch.device("cpu").
            dtype (torch.dtype, optional): Tensor dtype used for internal state.
                Default: torch.complex64.
            verbose (bool, optional): Enable verbose logging. Default: False.
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
            # The theoretical l2 error bounds are used as the default radius of
            # the \ell_2 ball, assume chi-square distribution and tau=1
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
    def __call__(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        """Project channel tensors onto the configured ellipsoids.

        Args:
            x (List[torch.Tensor]): Input tensors, one tensor per local channel.

        Returns:
            List[torch.Tensor]: Projected tensors with the same structure as
            ``x``.
        """
        for i in range(self._itr_max):
            # Alternate a weighted gradient step with projection onto the local
            # ellipsoids until the relative variation stops decreasing.
            result = self._proj_l2_ball(self._grad_descent(self._prev, x))
            x_norm = sum(torch.linalg.vector_norm(x_ch) ** 2 for x_ch in x)
            diff_norm = sum(
                torch.linalg.vector_norm(result[i] - self._prev[i]) ** 2
                for i in range(self._rank_num_ch)
            )
            if self._is_dist:
                # Aggregate the stopping criterion across distributed ranks.
                dist.all_reduce(x_norm, op=dist.ReduceOp.SUM)
                dist.all_reduce(diff_norm, op=dist.ReduceOp.SUM)
            rel_var = torch.sqrt(diff_norm / x_norm)
            self._prev = result
            if i >= self._itr_min - 1 and rel_var.item() <= self._rel_var_tol:
                break
        return result

    @torch.no_grad()
    def _grad_descent(
        self, result: list[torch.Tensor], target: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Perform one weighted forward gradient step.

        Args:
            result (List[torch.Tensor]): Current iterate.
            target (List[torch.Tensor]): Target tensors to project.

        Returns:
            List[torch.Tensor]: Updated tensors before the projection step.
        """
        # The per-channel weights encode the diagonal preconditioner used by
        # the accelerated splitting scheme.
        return [
            result[i]
            - self._gd_step_size[i] * self._precond_weight[i] * (result[i] - target[i])
            for i in range(self._rank_num_ch)
        ]

    @torch.no_grad()
    def _proj_l2_ball(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        """Project tensors onto per-channel Euclidean balls.

        Args:
            x (List[torch.Tensor]): Tensors to be projected.

        Returns:
            List[torch.Tensor]: Projected tensors.
        """
        for i in range(self._rank_num_ch):
            # Shift to the ellipsoid centre, project radially, then shift back.
            x[i] -= self._center[i]
            x_i_norm = torch.linalg.vector_norm(x[i])
            x[i] *= torch.minimum(
                self._radius[i] / x_i_norm,
                torch.ones_like(x_i_norm),
            )
            x[i] += self._center[i]
        return x

    def update(self, *args, **kwargs):
        """Keep the operator state unchanged.

        This operator does not require per-iteration parameter updates.
        """
        pass

    def get_radius(self) -> tuple[float, ...]:
        """Return configured per-channel projection radii.

        Returns:
            torch.Tensor: Radius values for each local channel.
        """
        return self._radius

    def set_radius(self, radius: tuple[float, ...]) -> None:
        """Set the projection radii for each local channel.

        Args:
            radius (Tuple[float, ...]): Radius values for each local channel.
        """
        assert (
            len(radius) == self._rank_num_ch
        ), "Radius tuple length must match local channel count."
        self._radius = radius

    def get_precond_weight(self) -> tuple[torch.Tensor, ...]:
        """Return per-channel preconditioning weights.

        Returns:
            Tuple[torch.Tensor, ...]: Preconditioning tensors.
        """
        return self._precond_weight
