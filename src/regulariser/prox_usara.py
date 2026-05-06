"""Implement the uSARA positivity-constrained proximal operator.

This file provides a dual forward-backward proximal step combining weighted
wavelet :math:`\ell_1` sparsity with positivity projection.
"""

from typing import Tuple, List, Union, Optional
import math

import torch
import torch.distributed as dist
import ptwt
import pywt

from .regulariser import Regulariser

WaveletCoeff = List[
    Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
]
WaveletDictCoeff = List[Union[WaveletCoeff, torch.Tensor]]


class ProxSARAPos(Regulariser):
    r"""Apply the uSARA proximal operator with positivity constraint.

    This class implements the proximal operator of the positivity-constrained
    weighted wavelet prior used by uSARA. For an input image cube
    :math:`\mathbf{x}`, it computes :math:`\operatorname{prox}_{g}(\mathbf{x})`,
    where

    .. math::
        g(\mathbf{z})
        =
        \iota_{\mathbb{R}_{+}}(\mathbf{z})
        +
        \left\|
        \mathbf{\Psi}^{\dagger}\mathbf{z}
        \right\|_{1,\mathbf{w}},

    with :math:`\iota_{\mathbb{R}_{+}}` the indicator function of the
    non-negative orthant, :math:`\mathbf{\Psi}^{\dagger}` the concatenated
    wavelet transform operator, and :math:`\mathbf{w}` the weighting
    coefficients. In this implementation, the weighted :math:`\ell_{1}` term
    is handled by a dual forward-backward iteration, while the primal update
    enforces positivity.

    The wavelet dictionary may include several wavelet bases and an optional
    Dirac basis. The weighted :math:`\ell_{1}` norm is evaluated channel-wise,
    with one set of reweighting coefficients per channel and per wavelet
    coefficient block.

    Shape:
        - Input: ``(1, C, H, W)`` where ``C`` equals ``rank_num_ch`` for the
          current rank.
        - Output: ``(1, C, H, W)`` with the same shape as input.

    Attributes:
        _wl_dict (list[str]): Active non-Dirac wavelet bases.
        _dirac (bool): Whether the Dirac basis is included.
        _dual (WaveletDictCoeff): Current dual variables.
        _weights (list): Current wavelet reweighting coefficients.

    Example:
        >>> prox = ProxSARAPos(
        ...     img_size=(32, 32),
        ...     rank_num_ch=2,
        ...     sfth_val=torch.tensor([1e-3, 1e-3]),
        ...     wl_noise_floor=torch.tensor([1e-4, 1e-4]),
        ... )
        >>> x = torch.zeros(1, 2, 32, 32)
        >>> prox.update(x)
        >>> y = prox(x)
        >>> y.shape
        torch.Size([1, 2, 32, 32])

    References:
        Terris, M., Dabbech, A., Tang, C., & Wiaux, Y. (2023). `Image
        reconstruction algorithms in radio interferometry: From handcrafted to
        learned regularization denoisers <https://doi.org/10.1093/mnras/stac2672>`_.
        *MNRAS, 518*(1), 604-622.

        Combettes, P. L., & Pesquet, J.-C. (2011). `Proximal splitting methods
        in signal processing <https://doi.org/10.1007/978-1-4419-9569-8_10>`_.
        In H. H. Bauschke, R. S. Burachik, P. L. Combettes, V. Elser,
        D. R. Luke, & H. Wolkowicz (Eds.), *Fixed-point algorithms for inverse
        problems in science and engineering* (pp. 185-212). Springer.
    """

    def __init__(
        self,
        img_size: Tuple[int, int],
        rank_num_ch: int = 1,
        rank_ch_start: int = 0,
        wl_dict: Tuple[str, ...] = (
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
        sfth_val: torch.Tensor = torch.tensor(1e-4),
        wl_noise_floor: torch.Tensor = torch.tensor(1e-4),
        max_iter: int = 20,
        obj_tol: float = 1e-4,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float,
        verbose: Union[bool, int] = True,
    ) -> None:
        """Initialise the uSARA positivity proximal operator.

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
            sfth_val (torch.Tensor): Soft-thresholding value per channel.
                It should have a number of elements equals to the number of
                channels or equals to 1. Default: torch.tensor(1e-4).
            wl_noise_floor (torch.Tensor): Reweighting noise floor per channel.
                It should have a number of elements equals to the number of
                channels or equals to 1. Default: torch.tensor(1e-4).
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
            AssertionError: If the soft-thresholding value is not a scalar
                or a vector with one value per channel.
            AssertionError: If the noise floor level is not a scalar
                or a vector with one value per channel.
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
        assert (
            sfth_val.numel() == rank_num_ch or sfth_val.numel() == 1
        ), "Soft-thresholding value must be a scalar or a vector with one value per channel."
        self._sfth_val = sfth_val.reshape(1, -1, 1, 1).to(
            device=self._device, dtype=self._dtype
        )
        self._dec_lev = dec_lev
        assert (
            wl_noise_floor.numel() == rank_num_ch or wl_noise_floor.numel() == 1
        ), "Noise floor level must be a scalar or a vector with one value per channel."
        self._wl_noise_floor = wl_noise_floor.reshape(1, -1, 1, 1).to(
            device=self._device, dtype=self._dtype
        )
        self._mode = mode
        self._scale_factor = math.sqrt(len(wl_dict))
        self._wl_dict = list(wl_dict)
        self._dirac = "dirac" in self._wl_dict
        self._verbose = verbose
        self._weights = []
        self._norm_l1 = torch.zeros(1, device=self._device, dtype=self._dtype)
        self._max_iter = max_iter
        self._obj_tol = obj_tol
        if self._dirac:
            self._wl_dict.remove("dirac")

        # Initialise the dual variables with the coefficient structure induced
        # by the configured wavelet dictionary.
        self._dual = self._wavedec2_dict(
            torch.zeros(
                (1, self._rank_num_ch, *self._img_size),
                device=self._device,
                dtype=self._dtype,
            )
        )
        # Initialise the reweighting coefficients to ones with zero image cube.
        self.update(
            torch.zeros(
                (1, self._rank_num_ch, *self._img_size),
                device=self._device,
                dtype=self._dtype,
            )
        )

    def _wavedec2_dict(self, x: torch.Tensor) -> WaveletDictCoeff:
        """Decompose input tensors with the configured wavelet dictionary.

        Args:
            x (torch.Tensor): Input tensor with shape ``(1, C, H, W)``.

        Returns:
            WaveletDictCoeff: Structured wavelet coefficients for each basis.
        """
        coeff: WaveletDictCoeff = []
        for basis in self._wl_dict:
            curr_coeff = list(
                ptwt.wavedec2(
                    x / self._scale_factor, basis, level=self._dec_lev, mode=self._mode
                )
            )
            coeff.append(curr_coeff)
        if self._dirac:
            coeff.append(x / self._scale_factor)
        return coeff

    def _waverec2_dict(self, y: WaveletDictCoeff) -> torch.Tensor:
        """Reconstruct tensors from dictionary wavelet coefficients.

        Args:
            y (WaveletDictCoeff): Coefficients generated by
                :meth:`_wavedec2_dict`.

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
        for i, basis in enumerate(self._wl_dict):
            result += ptwt.waverec2(y[i], basis) / self._scale_factor
        if self._dirac:
            result += y[-1] / self._scale_factor
        return result

    def _prox_l1_adj(self, z: torch.Tensor, sfth: torch.Tensor) -> torch.Tensor:
        """Apply the proximal map of the adjoint :math:`\ell_1` norm.

        Args:
            z (torch.Tensor): Input coefficient tensor.
            sfth (torch.Tensor): Soft-thresholding levels.

        Returns:
            torch.Tensor: Thresholded coefficients.
        """
        return z - (
            torch.sign(z)
            * torch.maximum(
                torch.abs(z) - sfth,
                torch.tensor(0.0, device=self._device, dtype=self._dtype),
            )
        )

    def _sfth_dual(self, psit_img: WaveletDictCoeff) -> None:
        """Threshold coefficients and update dual variables.

        Args:
            psit_img (WaveletDictCoeff): Wavelet coefficients of the current
                primal iterate.

        Notes:
            This method updates ``self._dual`` in place and refreshes
            ``self._norm_l1`` for the stopping criterion.
        """
        norm_l1_ch = torch.zeros(
            1, self._rank_num_ch, 1, 1, device=self._device, dtype=self._dtype
        )
        for i in range(len(self._wl_dict)):
            self._dual[i][0] = self._prox_l1_adj(
                self._dual[i][0] + psit_img[i][0], self._sfth_val * self._weights[i][0]
            )
            norm_l1_ch += torch.sum(
                torch.abs(psit_img[i][0] * self._weights[i][0]),
                dim=(2, 3),
                keepdim=True,
            )
            for j in range(1, self._dec_lev + 1):
                self._dual[i][j] = tuple(
                    self._prox_l1_adj(
                        self._dual[i][j][k] + psit_img[i][j][k],
                        self._sfth_val * self._weights[i][j][k],
                    )
                    for k in range(3)
                )
                for k in range(3):
                    norm_l1_ch += torch.sum(
                        torch.abs(psit_img[i][j][k] * self._weights[i][j][k]),
                        dim=(2, 3),
                        keepdim=True,
                    )
        if self._dirac:
            self._dual[-1] = self._prox_l1_adj(
                self._dual[-1] + psit_img[-1], self._sfth_val * self._weights[-1]
            )
            norm_l1_ch += torch.sum(
                torch.abs(psit_img[-1] * self._weights[-1]), dim=(2, 3), keepdim=True
            )

        self._norm_l1 = torch.sum(norm_l1_ch * self._sfth_val)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Run the uSARA proximal step.

        Args:
            x (torch.Tensor): Input tensor with shape ``(1, C, H, W)``.

        Returns:
            torch.Tensor: Positivity-constrained proximal output.

        Notes:
            This method runs a dual forward-backward loop that alternates
            primal positivity projection and weighted wavelet soft thresholding.
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
            # Dual update (weighted wavelet soft-thresholding).
            self._sfth_dual(self._wavedec2_dict(result))
            # Objective-based stopping criterion.
            if self._is_dist:
                # Aggregate objective terms across distributed ranks.
                dist.all_reduce(norm_l2_2, op=dist.ReduceOp.SUM)
                dist.all_reduce(self._norm_l1, op=dist.ReduceOp.SUM)
            obj_val = 0.5 * norm_l2_2.item() + self._norm_l1.item()
            obj_rel_var = abs(obj_val - obj_val_prev) / max(abs(obj_val), 1e-12)
            obj_val_prev = obj_val
            if self._verbose > 1 and self._rank == 0:
                print(
                    f"  Prox Iter {i+1}, prox_fval = {obj_val:.3e},",
                    f"rel_fval = {obj_rel_var:.3e}, l1norm_w = {self._norm_l1.item():.3e}",
                    flush=True,
                )
            if obj_rel_var < self._obj_tol:
                break
        if self._verbose and self._rank == 0:
            print(
                f"  Prox converged: Iter {i+1}, rel_fval = {obj_rel_var:.3e},",
                f"l1norm_w = {self._norm_l1.item():.3e}",
                flush=True,
            )

        return result

    def update(self, x: torch.Tensor) -> None:
        """Update per-coefficient reweighting factors.

        Args:
            x (torch.Tensor): Input tensor used to recompute SARA weights.

        Notes:
            Reweighting is computed independently for each wavelet basis and,
            when enabled, for the Dirac component.
        """
        self._weights = []

        x = x.to(self._dtype).to(self._device)
        for basis in self._wl_dict:
            curr_coeff = ptwt.wavedec2(
                x / self._scale_factor, basis, level=self._dec_lev, mode=self._mode
            )
            weighting_i = [
                self._wl_noise_floor / (self._wl_noise_floor + torch.abs(curr_coeff[0]))
            ]
            for i in range(1, self._dec_lev + 1):
                weighting_i.append(
                    tuple(
                        self._wl_noise_floor
                        / (self._wl_noise_floor + torch.abs(curr_coeff[i][j]))
                        for j in range(3)
                    )
                )
            self._weights.append(weighting_i)
        if self._dirac:
            self._weights.append(
                self._wl_noise_floor
                / (self._wl_noise_floor + torch.abs(x / self._scale_factor))
            )

    def get_l1_norm(self) -> float:
        """Return the latest weighted :math:`\ell_1` norm value.

        Returns:
            float: Last objective regularisation term.
        """
        return self._norm_l1.item()

    def set_noise_floor_level(self, wl_noise_floor: torch.Tensor) -> None:
        """Set the wavelet reweighting noise floor.

        Args:
            wl_noise_floor (torch.Tensor): Per-channel noise-floor in the
                wavelet coefficients.

        Raises:
            AssertionError: If the noise floor level is not a scalar or a
                vector with one value per channel.

        """
        assert (
            wl_noise_floor.numel() == self._rank_num_ch or wl_noise_floor.numel() == 1
        ), "Noise floor level must be a scalar or a vector with one value per channel."
        self._wl_noise_floor = wl_noise_floor.reshape(1, -1, 1, 1).to(
            device=self._device, dtype=self._dtype
        )

    def set_soft_thresholding_value(self, sfth_val: torch.Tensor) -> None:
        """Set wavelet soft-thresholding levels.

        Args:
            sfth_val (torch.Tensor): Per-channel threshold tensor.

        Raises:
            AssertionError: If the soft-thresholding value is not a scalar or a
                vector with one value per channel.
        """
        assert (
            sfth_val.numel() == self._rank_num_ch or sfth_val.numel() == 1
        ), "Soft-thresholding value must be a scalar or a vector with one value per channel."
        self._sfth_val = sfth_val.reshape(1, -1, 1, 1).to(
            device=self._device, dtype=self._dtype
        )
