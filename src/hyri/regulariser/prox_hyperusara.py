"""Implement the Hyper-uSARA positivity-constrained proximal operator.

This file provides a dual forward-backward proximal step combining weighted
joint-channel :math:`\\ell_{2,1}` wavelet sparsity with positivity projection.
Wavelet transforms are evaluated with :ptwt-docs:`PyTorch Wavelet Toolbox <>`.
"""

import math

import ptwt
import pywt
import torch
import torch.distributed as dist

from .regulariser import Regulariser

DEFAULT_DEVICE = torch.device("cpu")
DEFAULT_NOISE_LEVEL = torch.tensor(1e-4)


class ProxHyperSARAPos(Regulariser):
    r"""Apply the Hyper-uSARA proximal operator with positivity constraint.

    This class implements the proximal operator of the positivity-constrained
    weighted joint-channel wavelet prior used by Hyper-uSARA. It extends
    :class:`~hyri.regulariser.regulariser.Regulariser`. For an input image cube
    :math:`\mathbf{x}`, it computes

    .. math::
        \operatorname{prox}_{g}(\mathbf{x}),

    where

    .. math::
        g(\mathbf{z})
        =
        \iota_{\mathbb{R}_{+}}(\mathbf{z})
        +
        \left\|
        \mathbf{\Psi}^{\dagger}\mathbf{z}
        \right\|_{2,1, \mathbf{w}},

    with :math:`\iota_{\mathbb{R}_{+}}` the indicator function of the
    non-negative orthant, :math:`\mathbf{\Psi}^{\dagger}` the concatenated
    wavelet analysis transform, and :math:`\mathbf{w}` the weighting
    coefficients. In this implementation, the weighted :math:`\ell_{2,1}`
    term is handled by a dual forward-backward iteration, while the primal
    update enforces positivity.

    The wavelet dictionary may include several orthogonal wavelet bases and an
    optional Dirac basis. The transforms are evaluated with
    :ptwt-docs:`PyTorch Wavelet Toolbox <>`. The :math:`\ell_{2,1}` norm is
    computed jointly across channels, and when distributed execution is enabled,
    the required channel norms are aggregated across ranks.

    Shape:
        Input: ``(1, C, H, W)`` where ``C`` equals ``rank_num_ch`` for the
          current rank.
        Output: ``(1, C, H, W)`` with the same shape as input.

    Attributes:
        _wl_dict (list[str]): Active non-Dirac wavelet bases.
        _dirac (bool): Whether the Dirac basis is included.
        _dual (torch.Tensor): Packed dual wavelet coefficients.
        _weights (torch.Tensor): Packed reweighting coefficients.

    Example:
        >>> prox = ProxHyperSARAPos(
        ...     img_size=(32, 32),
        ...     rank_num_ch=2,
        ...     sfth_val=torch.tensor(1e-3),
        ...     wl_noise_floor=torch.tensor(1e-4),
        ... )
        >>> x = torch.zeros(1, 2, 32, 32)
        >>> prox.update(x)
        >>> y = prox(x)
        >>> y.shape
        torch.Size([1, 2, 32, 32])

    References:
        Tang, C., Dabbech, A., Jackson, A., & Wiaux, Y. (2026).
        `HyperAIRI: a plug-and-play algorithm for precise hyperspectral image
        reconstruction in radio interferometry
        <https://doi.org/10.3847/1538-4365/ae3149>`_. *ApJS*, 283(1), 9.

        Combettes, P. L., & Pesquet, J.-C. (2011). `Proximal splitting methods
        in signal processing <https://doi.org/10.1007/978-1-4419-9569-8_10>`_.
        In H. H. Bauschke, R. S. Burachik, P. L. Combettes, V. Elser,
        D. R. Luke, & H. Wolkowicz (Eds.), *Fixed-point algorithms for inverse
        problems in science and engineering* (pp. 185-212). Springer.
    """

    def __init__(
        self,
        img_size: tuple[int, int],
        rank_num_ch: int = 1,
        rank_ch_start: int = 0,
        wl_dict: tuple[str, ...] = (
            "dirac",
            "db1",
            "db2",
            "db3",
            "db4",
            "db5",
            "db6",
            "db7",
            "db8",
        ),
        dec_lev: int = 4,
        mode: str = "zero",
        sfth_val: torch.Tensor = DEFAULT_NOISE_LEVEL,
        wl_noise_floor: torch.Tensor = DEFAULT_NOISE_LEVEL,
        max_iter: int = 20,
        obj_tol: float = 1e-4,
        device: torch.device = DEFAULT_DEVICE,
        dtype: torch.dtype = torch.float,
        verbose: bool | int = True,
    ) -> None:
        """Initialise Hyper-uSARA proximal-operator state.

        Args:
            img_size (Tuple[int, int]): Image spatial shape ``(H, W)``.
            rank_num_ch (int, optional): Number of channels handled by this
                rank. Default: 1.
            rank_ch_start (int, optional): Global start index of local
                channels. Default: 0.
            wl_dict (Tuple[str, ...], optional): Wavelet dictionary including
                optional ``"dirac"``. Default: ("dirac", "db1", "db2", "db3",
                "db4", "db5", "db6", "db7", "db8").
            dec_lev (int, optional): Wavelet decomposition depth. Default: 4.
            mode (str, optional): Boundary mode for wavelet transforms.
                Default: "zero".
            sfth_val (torch.Tensor, optional): Soft-thresholding velue in the
                :math:`\\ell_{2}`-norm wavelet domain. Default: torch.tensor(1e-4).
            wl_noise_floor (torch.Tensor, optional): Noise floor level in the
                :math:`\\ell_{2}`-norm wavelet domain. Default: torch.tensor(1e-4).
            max_iter (int, optional): Maximum dual forward-backward iterations.
                Default: 20.
            obj_tol (float, optional): Relative objective tolerance used for
                stopping. Default: 1e-4.
            device (torch.device, optional): Execution device. Default:
                torch.device("cpu").
            dtype (torch.dtype, optional): Tensor dtype for computation.
                Default: torch.float.
            verbose (Union[bool, int], optional): Verbosity level. Default:
                True.

        Raises:
            AssertionError: If the wavelet dictionary is empty or contains
                unsupported bases.
            AssertionError: If the soft-thresholding value is not a scalar.
            AssertionError: If the noise floor level is not a scalar.
        """
        super().__init__(
            rank_num_ch=rank_num_ch,
            rank_ch_start=rank_ch_start,
            device=device,
            dtype=dtype,
        )

        assert wl_dict, "Wavelet dictionary cannot be empty."
        for basis in wl_dict:
            if basis != "dirac":
                assert basis in pywt.wavelist(
                    kind="discrete"
                ), f"Unsupported wavelet basis: {basis}"

        self._img_size = img_size
        self._rank_num_ch = rank_num_ch
        if sfth_val is not None:
            assert (
                sfth_val.numel() == 1
            ), "Soft thresholding value must be a single scalar for Hyper-uSARA prox."
            sfth_val = sfth_val.reshape(1, 1, 1).to(
                device=self._device, dtype=self._dtype
            )
        self._sfth_val = sfth_val
        self._dec_lev = dec_lev
        if wl_noise_floor is not None:
            assert (
                wl_noise_floor.numel() == 1
            ), "Noise floor level must be a single scalar for Hyper-uSARA prox."
            wl_noise_floor = wl_noise_floor.reshape(1, 1, 1).to(
                device=self._device, dtype=self._dtype
            )
        self._wl_noise_floor = wl_noise_floor
        self._mode = mode
        self._scale_factor = math.sqrt(len(wl_dict))
        self._wl_dict = list(wl_dict)
        self._dirac = "dirac" in self._wl_dict
        self._verbose = verbose
        self._weights = torch.ones(1, device=self._device, dtype=self._dtype)
        self._norm_l21 = torch.zeros(1, device=self._device, dtype=self._dtype)
        self._max_iter = max_iter
        self._obj_tol = obj_tol
        if self._dirac:
            self._wl_dict.remove("dirac")

        # Initialise dual variable and coefficient indexing metadata.
        self._dual = None
        # List of coefficient shapes
        self._dual_shape = []
        # List of (start, end) index tuples for each coefficient
        self._dual_slices = []
        self._init_dual()

    def _init_dual(self) -> None:
        """Initialise packed dual variables and coefficient metadata."""
        x0 = torch.zeros(
            (1, self._rank_num_ch, *self._img_size),
            device=self._device,
            dtype=self._dtype,
        )

        idx_start = 0
        for basis in self._wl_dict:
            coeff = ptwt.wavedec2(x0, basis, level=self._dec_lev, mode=self._mode)

            basis_shapes = []
            basis_slices = []

            # Approximation coefficients
            approx_shape = coeff[0][0, 0].shape
            approx_numel = coeff[0][0, 0].numel()
            basis_shapes.append(approx_shape)
            basis_slices.append((idx_start, idx_start + approx_numel))
            idx_start += approx_numel

            # Detail coefficients per level
            for j in range(1, self._dec_lev + 1):
                level_shapes = []
                level_slices = []
                for k in range(3):
                    detail_shape = coeff[j][k][0, 0].shape
                    detail_numel = coeff[j][k][0, 0].numel()
                    level_shapes.append(detail_shape)
                    level_slices.append((idx_start, idx_start + detail_numel))
                    idx_start += detail_numel
                basis_shapes.append(level_shapes)
                basis_slices.append(level_slices)

            self._dual_shape.append(basis_shapes)
            self._dual_slices.append(basis_slices)

        if self._dirac:
            dirac_shape = x0[0, 0].shape
            dirac_numel = x0[0, 0].numel()
            self._dual_shape.append([dirac_shape])
            self._dual_slices.append([(idx_start, idx_start + dirac_numel)])

        self._dual = self._wavedec2_dict(x0)

    def _wavedec2_dict(self, x: torch.Tensor) -> torch.Tensor:
        """Pack wavelet coefficients from all bases into one tensor.

        Args:
            x (torch.Tensor): Input tensor with shape ``(1, C, H, W)``.

        Returns:
            torch.Tensor: Packed coefficient tensor with shape ``(1, C, N)``.
        """
        coeff_list: list[torch.Tensor] = []
        for basis in self._wl_dict:
            curr_coeff = ptwt.wavedec2(x, basis, level=self._dec_lev, mode=self._mode)
            coeff_list.append(curr_coeff[0].flatten(start_dim=2))
            for j in range(1, self._dec_lev + 1):
                for k in range(3):
                    coeff_list.append(curr_coeff[j][k].flatten(start_dim=2))
        if self._dirac:
            coeff_list.append(x.flatten(start_dim=2))

        return torch.cat(coeff_list, dim=2) / self._scale_factor

    def _waverec2_dict(self, y: torch.Tensor) -> torch.Tensor:
        """Reconstruct tensors from packed wavelet coefficients.

        Args:
            y (torch.Tensor): Packed coefficient tensor with shape ``(1, C, N)``.

        Returns:
            torch.Tensor: Reconstructed tensor with shape ``(1, C, H, W)``.
        """
        result = torch.zeros(
            1,
            self._rank_num_ch,
            *self._img_size,
            device=self._device,
            dtype=self._dtype,
        )

        for basis_idx, basis in enumerate(self._wl_dict):
            # Extract approximation coefficients
            approx_start, approx_end = self._dual_slices[basis_idx][0]
            approx_shape = self._dual_shape[basis_idx][0]
            approx = y[:, :, approx_start:approx_end].reshape(
                1, self._rank_num_ch, *approx_shape
            )

            # Reconstruct nested list: [approx, (cH1, cV1, cD1), (cH2, cV2, cD2), ...]
            reconstructed = [approx]
            for level_idx in range(1, self._dec_lev + 1):
                details_tuple = []
                for subband_idx in range(3):
                    detail_start, detail_end = self._dual_slices[basis_idx][level_idx][
                        subband_idx
                    ]
                    detail_shape = self._dual_shape[basis_idx][level_idx][subband_idx]
                    detail = y[:, :, detail_start:detail_end].reshape(
                        1, self._rank_num_ch, *detail_shape
                    )
                    details_tuple.append(detail)
                reconstructed.append(tuple(details_tuple))

            result += ptwt.waverec2(reconstructed, basis)

        if self._dirac:
            dirac_start, dirac_end = self._dual_slices[-1][0]
            dirac_shape = self._dual_shape[-1][0]
            dirac = y[:, :, dirac_start:dirac_end].reshape(
                1, self._rank_num_ch, *dirac_shape
            )
            result += dirac

        return result / self._scale_factor

    def _prox_l21_adj(self, z: torch.Tensor, sfth: torch.Tensor) -> torch.Tensor:
        """Apply the proximal map of the adjoint :math:`\\ell_{2,1}` norm.

        Args:
            z (torch.Tensor): Input packed coefficients.
            sfth (torch.Tensor): Soft-thresholding levels.

        Returns:
            torch.Tensor: Thresholded packed coefficients.
        """
        # Compute joint-channel l2 norms coefficient-wise.
        z_l2 = torch.sum(z**2, dim=1, keepdim=True)
        if self._is_dist:
            dist.all_reduce(z_l2, op=dist.ReduceOp.SUM)
        z_l2 = torch.sqrt(z_l2)
        # Apply l2 shrinkage induced by the adjoint l2,1 proximal map.
        z_l2_sfth = torch.maximum(
            z_l2 - sfth, torch.zeros(1, device=self._device, dtype=self._dtype)
        )
        # Rescale each coefficient vector along the channel dimension.
        result = z - z / (z_l2 + 1e-16) * z_l2_sfth
        return result

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Run the Hyper-uSARA proximal step.

        Args:
            x (torch.Tensor): Input tensor with shape ``(1, C, H, W)``.

        Returns:
            torch.Tensor: Positivity-constrained proximal output.

        Note:
            This method uses a dual forward-backward loop with joint-channel
            :math:`\\ell_{2,1}` shrinkage, and synchronises channel-norm terms
            across distributed ranks when needed.
        """
        # Dual forward-backward iterations: primal positivity projection and
        # dual weighted-thresholding updates.
        obj_val_prev = 0
        for i in range(self._max_iter):
            # Primal update (positivity projection).
            result = torch.maximum(
                x - self._waverec2_dict(self._dual),
                torch.tensor(
                    0.0,
                    device=self._device,
                    dtype=self._dtype,
                ),
            )
            norm_l2_2 = torch.sum((result - x) ** 2)
            # Dual update stage 1: compute wavelet coefficients of current iterate
            psit_img = self._wavedec2_dict(result)
            psit_img_l2 = torch.sum(psit_img**2, dim=1, keepdim=True)

            # Compute objective and stopping criterion.
            if self._is_dist:
                dist.all_reduce(norm_l2_2, op=dist.ReduceOp.SUM)
                dist.all_reduce(psit_img_l2, op=dist.ReduceOp.SUM)
            self._norm_l21 = (
                torch.sum(torch.sqrt(psit_img_l2) * self._weights) * self._sfth_val
            )
            obj_val = 0.5 * norm_l2_2.item() + self._norm_l21.item()
            obj_rel_var = abs(obj_val - obj_val_prev) / max(abs(obj_val), 1e-12)
            obj_val_prev = obj_val
            if self._verbose > 1 and self._rank == 0:
                print(
                    f"  Prox Iter {i+1}, prox_fval = {obj_val:.3e},",
                    f"rel_fval = {obj_rel_var:.3e}, "
                    f"l21norm_w = {self._norm_l21.item():.3e}",
                    flush=True,
                )
            if obj_rel_var < self._obj_tol:
                break

            # Dual update stage 2: ascent step followed by l2,1 shrinkage.
            self._dual = self._dual + self._wavedec2_dict(result)
            self._dual = self._prox_l21_adj(self._dual, self._sfth_val * self._weights)

        if self._verbose and self._rank == 0:
            print(
                f"  Prox converged: Iter {i+1}, rel_fval = {obj_rel_var:.3e},",
                f"l21norm_w = {self._norm_l21.item():.3e}",
                flush=True,
            )

        return result

    def update(self, x: torch.Tensor) -> None:
        """Update joint-channel reweighting factors.

        Args:
            x (torch.Tensor): Input tensor used to recompute Hyper-uSARA
                weights.

        Note:
            The :math:`\\ell_2` coefficient norms are aggregated across ranks in
            distributed execution before updating the shared weights.
        """

        x = x.to(self._dtype).to(self._device)
        psit_x = self._wavedec2_dict(x)
        psit_x_l2 = torch.sum(psit_x**2, dim=1, keepdim=True)
        if self._is_dist:
            dist.all_reduce(psit_x_l2, op=dist.ReduceOp.SUM)
        psit_x_l2 = torch.sqrt(psit_x_l2)
        self._weights = self._wl_noise_floor / (self._wl_noise_floor + psit_x_l2)

    def get_l21_norm(self) -> float:
        """Return the latest weighted :math:`\\ell_{2,1}` norm value.

        Returns:
            float: Last objective regularisation term.
        """
        return self._norm_l21.item()

    def set_noise_floor_level(self, wl_noise_floor: torch.Tensor) -> None:
        """Set the scalar wavelet reweighting noise floor.

        Args:
            wl_noise_floor (torch.Tensor): Scalar noise-floor tensor.

        Raises:
            AssertionError: If the noise floor level is not a scalar.
        """
        assert (
            wl_noise_floor.numel() == 1
        ), "Noise floor level should be a scalar for Hyper-uSARA prox."
        self._wl_noise_floor = wl_noise_floor.reshape(1, 1, 1).to(
            device=self._device, dtype=self._dtype
        )

    def set_soft_thresholding_value(self, sfth_val: torch.Tensor) -> None:
        """Set the scalar wavelet soft-thresholding level.

        Args:
            sfth_val (torch.Tensor): Scalar threshold tensor.

        Raises:
            AssertionError: If the soft-thresholding value is not a scalar.
        """
        assert (
            sfth_val.numel() == 1
        ), "Soft thresholding value should be a scalar for Hyper-uSARA prox."
        self._sfth_val = sfth_val.reshape(1, 1, 1).to(
            device=self._device, dtype=self._dtype
        )
