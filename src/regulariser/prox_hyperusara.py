"""
Proximity operator for hyper-uSARA with positivity constraint.
"""

from typing import Tuple, List, Union, Optional
import math

import torch
import torch.distributed as dist
import ptwt

from .regulariser import Regulariser

WaveletCoeff = List[
    Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
]
WaveletDictCoeff = List[Union[WaveletCoeff, torch.Tensor]]


class ProxHyperSARAPos(Regulariser):
    """
    Proximity operator for Hyper-uSARA with positivity constrained.

    It provides methods for wavelet decomposition and reconstruction with multiple bases,
    proximity operator for the adjoint of l1 norm, soft thresholding, and updating weights
    for l1 norm. We use pytorch wavelet toolbox (ptwt) for wavelet decomposition and
    reconstruction.
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
        sfth_val: Optional[torch.Tensor] = None,
        wl_noise_floor: Optional[torch.Tensor] = None,
        max_iter: int = 20,
        obj_tol: float = 1e-4,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float,
        verbose: Union[bool, int] = True,
    ) -> None:
        """
        Initializes the Hyper-uSARA positivity proximity operator with the given parameters.

        Args:
            img_size (Tuple[int, int]): The size of the image.
            sfth_val (torch.Tensor, optional): The soft thresholding value. Defaults to 1.0e-4.
            wl_dict (Tuple[str, ...], optional): The wavelet dictionary.
                Defaults to ("dirac", "db1", "db2", "db3", "db4", "db5", "db6", "db7", "db8").
            dec_lev (int, optional): The decomposition level. Defaults to 4.
            wl_noise_floor (torch.Tensor, optional): The noise floor level in wavelet coefficient.
                Defaults to 1.0e-4.
            mode (str, optional): The mode for wavelet decomposition and reconstruction.
                Defaults to "zero".
            max_iter (int, optional): The maximum number of iterations. Defaults to 20.
            obj_tol (float, optional): The tolerance for the objective function. Defaults to 1e-4.
            device (torch.device, optional): The device on which the computations are performed.
                Defaults to torch.device("cpu").
            dtype (torch.dtype, optional): The data type of the input. Defaults to torch.float.
            verbose (bool, optional): If True, print progress messages. Defaults to True.
        """
        super().__init__(
            rank_num_ch=rank_num_ch,
            rank_ch_start=rank_ch_start,
            device=device,
            dtype=dtype,
        )

        assert wl_dict, "Wavelet dictionary cannot be empty."
        for basis in wl_dict:
            assert basis in ptwt.wavelist(), f"Unsupported wavelet basis: {basis}"

        self._img_size = img_size
        self._rank_num_ch = rank_num_ch
        self._sfth_val = sfth_val
        self._dec_lev = dec_lev
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

        # initialise dual variable
        self._dual = []
        self._dual_numel_ch = 0
        self._dual_shape = []
        self._dual_slices = []
        self._init_dual()

    def _init_dual(self) -> None:
        """
        Initializes the dual variable to zeros.
        """
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
            self._dual_numel_ch += approx_numel
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
                    self._dual_numel_ch += detail_numel
                    idx_start += detail_numel
                basis_shapes.append(level_shapes)
                basis_slices.append(level_slices)

            self._dual_shape.append(basis_shapes)
            self._dual_slices.append(basis_slices)

        if self._dirac:
            dirac_shape = x0[0, 0].shape
            dirac_numel = x0[0, 0].numel()
            self._dual_shape.append([dirac_shape])
            self._dual_numel_ch += dirac_numel
            self._dual_slices.append([(idx_start, idx_start + dirac_numel)])

        self._dual = self._wavedec2_dict(x0)

    def _wavedec2_dict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs wavelet decomposition with multiple bases and packs coefficients
        into a single tensor.

        Args:
            x (torch.Tensor): The input tensor to be decomposed.

        Returns:
            torch.Tensor:
                Packed coefficient tensor with shape (1, C, N), where N is the total
                number of spatial coefficients per channel across all bases and levels.

        """
        coeff_list: List[torch.Tensor] = []
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
        """
        Performs wavelet reconstruction with multiple bases from a packed tensor.

        Args:
            y (torch.Tensor):
                Packed coefficient tensor with shape (1, C, N), where N is the total
                number of spatial coefficients per channel across all bases and levels.

        Returns:
            torch.Tensor: The reconstructed image with shape (1, C, H, W).
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
        """
        Proximity operator for the adjoint of l21 norm.

        Args:
            z (torch.Tensor): The input tensor.
            sfth (torch.Tensor): The soft thresholding level. The level can be
                different for each channel.
        Returns:
            torch.Tensor: The result of applying the proximity operator on `z`.
        """
        # calculate the l2 norm of z across channels
        z_l2 = torch.sum(z**2, dim=1, keepdim=True)
        if self._is_dist:
            dist.all_reduce(z_l2.to(self._dist_device), op=dist.ReduceOp.SUM)
            z_l2 = z_l2.to(self._device)
        z_l2 = torch.sqrt(z_l2)
        # soft-thresholding
        z_l2_sfth = torch.maximum(
            z_l2 - sfth, torch.zeros(1, device=self._device, dtype=self._dtype)
        )
        # scaling
        result = z - z / (z_l2 + 1e-16) * z_l2_sfth
        return result

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the proximity operator to the input tensor.
        The SARA prior and positivity constraint are splited with dual forward-backward algorithm.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the proximity operator.
        """
        # dual forward backward
        obj_val_prev = 0
        for i in range(self._max_iter):
            # primal update
            result = torch.maximum(
                x - self._waverec2_dict(self._dual),
                torch.tensor(
                    0.0,
                    device=self._device,
                    dtype=self._dtype,
                ),
            )
            norm_l2_2 = torch.sum((result - x) ** 2)
            # dual update 1, calculate Phi^T(result)
            psit_img = self._wavedec2_dict(result)
            psit_img_l2 = torch.sum(psit_img**2, dim=1, keepdim=True)

            # stop criterion
            if self._is_dist:
                norm_l2_2 = norm_l2_2.to(self._dist_device)
                psit_img_l2 = psit_img_l2.to(self._dist_device)
                dist.all_reduce(norm_l2_2, op=dist.ReduceOp.SUM)
                dist.all_reduce(psit_img_l2, op=dist.ReduceOp.SUM)
                norm_l2_2 = norm_l2_2.to(self._device)
                psit_img_l2 = psit_img_l2.to(self._device)
            self._norm_l21 = torch.sum(
                torch.sqrt(psit_img_l2) * self._sfth_val * self._weights
            )
            obj_val = 0.5 * norm_l2_2.item() + self._norm_l21.item()
            obj_rel_var = abs(obj_val - obj_val_prev) / max(abs(obj_val), 1e-12)
            obj_val_prev = obj_val
            if self._verbose > 1 and self._rank == 0:
                print(
                    f"  Prox Iter {i+1}, prox_fval = {obj_val:.3e},",
                    f"rel_fval = {obj_rel_var:.3e}, l21norm_w = {self._norm_l21.item():.3e}",
                    flush=True,
                )
            if obj_rel_var < self._obj_tol:
                break

            # dual update 2: soft thresholding
            self._dual = self._dual + self._wavedec2_dict(result)
            self._dual = self._prox_l21_adj(self._dual, self._sfth_val * self._weights)

        if self._verbose and self._rank == 0:
            print(
                f"  Prox converged: Iter {i+1}, rel_fval = {obj_rel_var:.3e},",
                f"l21norm_w = {self._norm_l21.item():.3e}",
                flush=True,
            )

        return result

    def update(self, x: torch.Tensor) -> None:  # initialisation: bool = False
        """
        Update the weight for l1 norm.

        Args:
            x (torch.Tensor): The input tensor.
            initialisation (bool, optional): If True, initialize the weights to 1.
                Defaults to False.
        """

        x = x.to(self._dtype).to(self._device)
        psit_x = self._wavedec2_dict(x)
        psit_x_l2 = torch.sum(psit_x**2, dim=1, keepdim=True)
        if self._is_dist:
            psit_x_l2 = dist.all_reduce(
                psit_x_l2.to(self._dist_device), op=dist.ReduceOp.SUM
            )
            psit_x_l2 = psit_x_l2.to(self._device)
        psit_x_l2 = torch.sqrt(psit_x_l2)
        self._weights = self._wl_noise_floor / (self._wl_noise_floor + psit_x_l2)

    def get_l21_norm(self) -> float:
        """
        Gets the latest l21 norm calculated in `_sfth_dual`.

        Returns:
            float: The latest l21 norm.
        """
        return self._norm_l21.item()

    def set_noise_floor_level(self, wl_noise_floor: torch.Tensor) -> None:
        """
        Sets the noise floor level in wavelet coefficient.

        Args:
            wl_noise_floor (float): The noise floor level to set.
        """
        self._wl_noise_floor = wl_noise_floor.reshape(1, self._rank_num_ch, 1, 1).to(
            device=self._device, dtype=self._dtype
        )

    def set_soft_thresholding_value(self, sfth_val: torch.Tensor) -> None:
        """
        Sets the soft thresholding level of the proximity operator.

        Args:
            sfth_val (float): The soft thresholding level for the wavelet coefficients.
        """
        self._sfth_val = sfth_val.reshape(1, self._rank_num_ch, 1, 1).to(
            device=self._device, dtype=self._dtype
        )
