"""Implement the AIRI denoiser regulariser.

This file provides channel-wise AIRI denoising with per-channel network
selection driven by heuristic noise and peak-intensity estimates.
"""

import csv
import os
from importlib.resources import files

import torch
from onnx2torch import convert

from ..interface.io import resolve_existing_file_path
from .regulariser import Regulariser

DEFAULT_DEVICE = torch.device("cpu")


class DenAIRI(Regulariser):
    r"""Apply AIRI denoisers selected from a shelf of pretrained networks.

    This class implements the learned regularisation operator used by AIRI.
    For each local channel :math:`c`, a pretrained denoiser
    :math:`\mathcal{D}_{\sigma_c}` is selected from a denoiser shelf according
    to the ratio between the heuristic noise level and the current peak-intensity
    estimate. The denoised channel is then computed
    as

    .. math::
        \hat{\mathbf{x}}_c
        =
        \rho_c \, \mathcal{D}_{\sigma_c}(\mathbf{x}_c / \rho_c),

    where :math:`\rho_c` is the channel-dependent scaling factor and
    :math:`\sigma_c` is the selected denoiser noise level.

    Optional random flips and right-angle rotations are applied before
    denoising and inverted afterwards as a lightweight self-ensembling step.

    Shape:
        Input: ``(1, C, H, W)`` where ``C`` equals ``rank_num_ch`` for the
          current rank.
        Output: ``(1, C, H, W)`` with the same shape as the input tensor.

    Attributes:
        _shelf (dict[float, str]): Mapping from shelf noise level to ONNX model
            path.
        _list_network (list): Loaded denoiser used for each local channel.
        _list_net_scaling (list[float]): Per-channel scaling factor
            :math:`\rho_c`.
        _list_network_sigma (list[float]): Active shelf noise level for each
            local channel.

    Example:
        >>> den = DenAIRI(
        ...     shelf_path="airi_denoisers/shelf_airi_oaid.csv",
        ...     rank_num_ch=1,
        ... )
        >>> heu = torch.ones(1, 1, 1, 1)
        >>> peak = torch.ones(1, 1, 1, 1)
        >>> den.update(heu, peak)
        >>> x = torch.zeros(1, 1, 16, 16)
        >>> y = den(x)

    References:
        Terris, M., Dabbech, A., Tang, C., & Wiaux, Y. (2023). `Image
        reconstruction algorithms in radio interferometry: From handcrafted to
        learned regularization denoisers <https://doi.org/10.1093/mnras/stac2672>`_.
        *MNRAS*, 518(1), 604-622.
    """

    def __init__(
        self,
        shelf_path: str,
        rand_trans: bool = True,
        rank_num_ch: int = 1,
        rank_ch_start: int = 0,
        device: torch.device = DEFAULT_DEVICE,
        dtype: torch.dtype = torch.float,
        verbose: bool = True,
    ) -> None:
        """Initialise AIRI shelf metadata and runtime state.

        Args:
            shelf_path (str): Path to the denoiser shelf CSV.
            rand_trans (bool, optional): Apply random flips/rotations before and
                after denoising. Default: True.
            rank_num_ch (int, optional): Number of channels handled by this
                rank. Default: 1.
            rank_ch_start (int, optional): Global start index of local
                channels. Default: 0.
            device (torch.device, optional): Execution device. Default:
                torch.device("cpu").
            dtype (torch.dtype, optional): Tensor dtype for denoising tensors.
                Default: torch.float.
            verbose (bool, optional): Enable shelf-selection logs. Default: True.

        Raises:
            FileNotFoundError: If the denoiser shelf CSV or network files in the
                shelf cannot be found.
            RuntimeError: If the denoiser shelf CSV contains no usable entries.
        """
        super().__init__(
            rank_num_ch=rank_num_ch,
            rank_ch_start=rank_ch_start,
            device=device,
            dtype=dtype,
        )

        self._rand_trans = rand_trans
        self._net_scaling = 1.0
        self._shelf = {}
        self._list_network = [None] * rank_num_ch
        self._list_net_scaling = [1.0] * rank_num_ch
        self._list_network_sigma = [0.0] * rank_num_ch
        self._verbose = verbose

        # load paths of networks
        package_root = files("hyri").joinpath("airi_denoisers")
        resolved_shelf_path = resolve_existing_file_path(shelf_path, package_root)

        with open(resolved_shelf_path, newline="", encoding="utf-8") as shelf_file:
            shelf_reader = csv.reader(shelf_file, delimiter=",")
            for row in shelf_reader:
                sigma = float(row[0])
                resolved_net_path = resolve_existing_file_path(row[1], package_root)
                self._shelf[sigma] = resolved_net_path
        if not self._shelf:
            raise RuntimeError("Shelf is empty: " + shelf_path)
        self._shelf = dict(sorted(self._shelf.items()))

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Denoise all local channels of an image cube.

        Args:
            x (torch.Tensor): Input tensor with shape ``(1, C, H, W)``.

        Returns:
            torch.Tensor: Denoised tensor with the same shape as ``x``.

        Note:
            The method :meth:`update` must be called with valid heuristic noise level
            and peak-intensity estimates before this method is used, so that the
            appropriate denoisers are selected and loaded for each channel.
        """
        result = x
        for ch in range(self._rank_num_ch):
            result[:, ch : ch + 1, :, :] = self._den_ch_single(result, ch)

        return result

    @torch.no_grad()
    def _den_ch_single(self, net_input: torch.Tensor, ch: int) -> torch.Tensor:
        """Denoise one local channel using the selected shelf network.

        Args:
            net_input (torch.Tensor): Local image cube for this rank.
            ch (int): Local channel index to denoise.

        Returns:
            torch.Tensor: Denoised tensor for channel ``ch`` with shape
            ``(1, 1, H, W)``.
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
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Update per-channel network selection and scaling factors.

        Args:
            ch_heuristic (torch.Tensor): Per-channel heuristic noise levels
                with one value per local channel.
            ch_peak_est (torch.Tensor): Per-channel peak-intensity estimates
                with one value per local channel.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Lower and upper admissible peak
            ranges for each local channel, each with shape ``(1, C, 1, 1)``.
            Returns ``None`` when ``ch_heuristic`` or ``ch_peak_est`` is
            ``None``.

        Note:
            The selected network for each channel is refreshed only when the
            admissible shelf sigma changes.
        """
        peak_min = torch.zeros(self._rank_num_ch, dtype=self._dtype)
        peak_max = torch.zeros(self._rank_num_ch, dtype=self._dtype)

        if ch_heuristic is not None and ch_peak_est is not None:
            ch_heu_peak = ch_heuristic / ch_peak_est
            ch_heu_peak = ch_heu_peak.cpu().squeeze().tolist()
            if isinstance(ch_heu_peak, (float, int)):
                ch_heu_peak = [float(ch_heu_peak)]

            for i in range(self._rank_num_ch):
                # Pick the shelf entry whose sigma is closest from below to the
                # heuristic ratio for this channel.
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
                    # If the heuristic ratio is below the shelf range, use the
                    # smallest available denoiser sigma as a safe fallback.
                    sigma_s = min(self._shelf.keys())
                    peak_min[i] = ch_heuristic.view(-1)[i] / sigma_s
                    peak_max[i] = float("inf")

                if abs(self._list_network_sigma[i] - sigma_s) > 1e-10:
                    # Reload the ONNX model only when the selected shelf entry
                    # changes.
                    self._list_network[i] = (
                        convert(self._shelf[sigma_s]).to(self._device).eval()
                    )

                    self._list_net_scaling[i] = ch_heuristic.view(-1)[i] / sigma_s
                    self._list_network_sigma[i] = sigma_s

                    if self._verbose:
                        print(
                            f"INFO: SHELF *** Channel {self._rank_ch_start + i}: "
                            f"heu/peak {ch_heu_peak[i]:.3e}, "
                            f"using network: {os.path.basename(self._shelf[sigma_s])}, "
                            f"peak range: [{peak_min[i].item():#.3g}, "
                            f"{peak_max[i].item():#.3g}], "
                            f"scaling factor: {self._list_net_scaling[i]:#.3g}",
                            flush=True,
                        )

            peak_min = peak_min.view(1, -1, 1, 1).to(device=self._device)
            peak_max = peak_max.view(1, -1, 1, 1).to(device=self._device)

            return (peak_min, peak_max)

        return None
