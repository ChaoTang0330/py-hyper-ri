"""
AIRI proximity operator
"""

import csv
import os
from typing import Tuple

import torch
from onnx2torch import convert

from .regulariser import Regulariser


class DenAIRI(Regulariser):
    """
    AIRI denoiser

    This class implements the AIRI denoiser which uses AIRI denoisers for regularisations.
    It uses a shelf of pre-trained AIRI denoisers and selects the appropriate one based on
    the estimated maximum intensitie of target image and the heuristic noise level.
    """

    def __init__(
        self,
        shelf_path: str,
        rand_trans: bool = True,
        rank_num_ch: int = 1,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float,
        verbose: bool = True,
    ) -> None:
        """
        Initializes the AIRI denoiser with the given parameters.

        Args:
            shelf_path (str): Path to the CSV file containing the denoiser shelf.
            rand_trans (bool, optional): If True, apply random transformations to the input
                before applying denoisers. Defaults to True.
            device (torch.device, optional): The device on which the computations are
                performed. Defaults to torch.device("cpu").
            dtype (torch.dtype, optional): The data type of the input. Defaults to torch.float.
            verbose (bool, optional): If True, print progress messages. Defaults to True.
        """
        super().__init__(device=device, dtype=dtype)

        self._rand_trans = rand_trans
        self._net_scaling = 1.0
        self._rank_num_ch = rank_num_ch
        self._shelf = {}
        self._list_network = [None] * rank_num_ch
        self._list_net_scaling = [1.0] * rank_num_ch
        self._list_network_sigma = [0.0] * rank_num_ch
        self._verbose = verbose

        # load paths of networks
        if not os.path.isfile(shelf_path):
            raise FileNotFoundError("Shelf file not found: " + shelf_path)
        with open(shelf_path, newline="", encoding="utf-8") as shelf_file:
            shelf_reader = csv.reader(shelf_file, delimiter=",")
            for row in shelf_reader:
                self._shelf[float(row[0])] = row[1]
                if not os.path.isfile(row[1]):
                    raise FileNotFoundError("Denoiser file not found: " + row[1])
        if not self._shelf:
            raise RuntimeError("Shelf is empty: " + shelf_path)
        self._shelf = dict(sorted(self._shelf.items()))

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the AIRI prox operator to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The denoised tensor.
        """
        result = x
        for ch in range(self._rank_num_ch):
            result[:, ch : ch + 1, :, :] = self._den_ch_single(result, ch)

        return result

    @torch.no_grad()
    def _den_ch_single(self, net_input: torch.Tensor, ch: int) -> torch.Tensor:
        """
        Applies the AIRI denoiser to a single channel of the input tensor.

        Args:
            net_input (torch.Tensor): The input tensor for a single channel.
            ch (int): The channel index to be denoised.

        Returns:
            torch.Tensor: The denoised tensor for the specified channel.
        """
        x = net_input[:, ch : ch + 1, :, :]
        if self._rand_trans:
            flip_dim = ([], (2,), (3,), (2, 3))[torch.randint(4, (1,)).item()]
            rot_times = torch.randint(4, (1,)).item()
            x = torch.flip(x, dims=flip_dim)
            x = torch.rot90(x, k=rot_times, dims=[2, 3])
            x = x.contiguous()
        result = (
            self._list_network[ch](x / self._list_net_scaling[ch])
            * self._list_net_scaling[ch]
        )
        if self._rand_trans:
            result = torch.rot90(result, k=-rot_times, dims=[2, 3])
            result = torch.flip(result, dims=flip_dim)
            result = result.contiguous()

        return result

    @torch.no_grad()
    def update(
        self,
        ch_heuristic: torch.Tensor,
        ch_peak_est: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the denoiser selection and scaling factor based on the given
        heuristic noise level and maximum intensity.

        Args:
            heuristic (float): The heuristic noise level.
            peak_est (float): The estimated maximum intensity.
        """
        peak_min = torch.zeros(self._rank_num_ch, dtype=self._dtype)
        peak_max = torch.zeros(self._rank_num_ch, dtype=self._dtype)

        if ch_heuristic is not None and ch_peak_est is not None:
            ch_heu_peak = ch_heuristic / ch_peak_est
            ch_heu_peak = ch_heu_peak.cpu().squeeze().tolist()

            for i in range(self._rank_num_ch):
                sigma_s = max(
                    filter(
                        lambda sigma: sigma <= ch_heu_peak[i],
                        self._shelf.keys(),
                    ),
                    default=None,
                )
                if sigma_s is not None:
                    peak_max[i] = ch_heuristic.view(-1)[i] / sigma_s
                    sigma_s1 = min(
                        filter(lambda sigma: sigma > sigma_s, self._shelf.keys()),
                        default=None,
                    )
                    if sigma_s1 is not None:
                        peak_min[i] = ch_heuristic.view(-1)[i] / sigma_s1
                else:
                    sigma_s = min(self._shelf.keys())
                    peak_min[i] = ch_heuristic.view(-1)[i] / sigma_s
                    peak_max[i] = float("inf")

                if abs(self._list_network_sigma[i] - sigma_s) > 1e-10:
                    self._list_network[i] = (
                        convert(self._shelf[sigma_s]).to(self._device).eval()
                    )

                    self._list_net_scaling[i] = ch_heuristic.view(-1)[i] / sigma_s
                    self._list_network_sigma[i] = sigma_s

                    if self._verbose:
                        print(
                            f"SHELF *** Channel {self._rank_ch_start + i}: "
                            f"heu/peak {ch_heu_peak[i]:.3e}, "
                            f"using network: {os.path.basename(self._shelf[sigma_s])}, "
                            f"peak range: [{peak_min[i].item():.3e}, {peak_max[i].item():.3e}], "
                            f"scaling factor: {self._list_net_scaling[i]:.3e}",
                            flush=True,
                        )

            peak_min = peak_min.view(1, -1, 1, 1).to(device=self._device)
            peak_max = peak_max.view(1, -1, 1, 1).to(device=self._device)

            return (peak_min, peak_max)
