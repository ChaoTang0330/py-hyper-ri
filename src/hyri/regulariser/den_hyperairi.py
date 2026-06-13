"""Implement the HyperAIRI denoiser regulariser.

This file provides multi-channel AIRI denoising with neighbour-channel context,
distributed boundary exchange, and optional spectral-correction updates.
"""

import csv
import os
from importlib.resources import files

import torch
import torch.distributed as dist
from onnx2torch import convert

from ..interface.io import resolve_existing_file_path
from .regulariser import Regulariser

DEFAULT_DEVICE = torch.device("cpu")


class DenHyperAIRI(Regulariser):
    r"""Apply the HyperAIRI denoiser to a local channel block.

    This class implements the learned regularisation operator used by
    HyperAIRI. For each target channel, the denoiser exploits a local
    three-channel spectral neighbourhood together with the current
    spectral-index map under a power-law spectral model. In distributed
    execution, boundary channels are exchanged with adjacent ranks so
    that the same local spectral context is available at rank boundaries.

    For channel :math:`i`, let

    .. math::
        \mathbf{z}_i =
        \begin{bmatrix}
        \mathbf{x}_{i-1},\ \mathbf{x}_{i},\ \mathbf{x}_{i+1}
        \end{bmatrix}

    denote the three-channel input centred at the target channel, and let
    :math:`\boldsymbol{\alpha}` denote the current spectral-index map. With
    local frequency ratios
    :math:`\mathbf{r}_i = (\nu_i/\nu_{i-1},\, 1,\, \nu_i/\nu_{i+1})`, the
    HyperAIRI denoiser is written abstractly as

    .. math::
        \mathbf{x}_i^{\mathrm{den}}
        =
        \rho_i \,
        \mathcal{D}_{\sigma_i}
        \bigl(
        \mathbf{z}_i / \rho_i,\,
        \boldsymbol{\alpha},\,
        \mathbf{r}_i
        \bigr),

    where :math:`\rho_i` is the channel-dependent scaling factor and
    :math:`\sigma_i` is the denoiser noise level selected from the shelf.

    In this implementation, the denoiser of each channel is selected
    dynamically according to the heuristic noise level and peak-intensity
    estimate.

    Shape:
        Input: ``(1, C, H, W)`` where ``C`` equals ``rank_num_ch`` for the
          current rank.
        Output: ``(1, C, H, W)`` with the same shape as input.

    Attributes:
        _shelf (dict[float, str]): Mapping from shelf noise level to ONNX model
            path.
        _list_network (list): Loaded denoiser used for each local channel.
        _list_network_sigma (list[float]): Shelf noise level currently assigned
            to each local channel.
        _list_net_scaling (list[float]): Per-channel scaling factor
            :math:`\rho_i`.
        _list_spec_corr_weight (list[torch.Tensor]): Internal frequency-dependent
            coupling weights derived from the spectral-index map and local
            frequency ratios.
        _ch_freq (torch.Tensor): Global channel frequencies.
        _rand_trans (bool): Whether random flips and right-angle rotations are
            applied as denoiser self-ensembling.
        _verbose (bool): Whether shelf-update and spectral-update messages are
            printed.

    Example:
        >>> den = DenHyperAIRI(
        ...     img_size=(64, 64),
        ...     ch_freq=torch.linspace(1.1e9, 1.4e9, 4),
        ...     shelf_path="airi_denoisers/shelf_hyperairi_mix.csv",
        ...     rank_num_ch=4,
        ... )
        >>> x = torch.zeros(1, 4, 64, 64)
        >>> den.update(spec_idx=torch.zeros(1, 1, 64, 64))
        >>> y = den(x)
        >>> y.shape
        torch.Size([1, 4, 64, 64])

    References:
        Tang, C., Dabbech, A., Jackson, A., & Wiaux, Y. (2026).
        `HyperAIRI: a plug-and-play algorithm for precise hyperspectral image
        reconstruction in radio interferometry
        <https://doi.org/10.3847/1538-4365/ae3149>`_. *ApJS*, 283(1), 9.
    """

    def __init__(
        self,
        img_size: tuple,
        ch_freq: torch.Tensor,
        shelf_path: str,
        rank_num_ch: int = 1,
        rank_ch_start: int = 0,
        rand_trans: bool = True,
        device: torch.device = DEFAULT_DEVICE,
        dtype: torch.dtype = torch.float,
        verbose: bool = True,
    ) -> None:
        """Initialise HyperAIRI shelf metadata and runtime state.

        Args:
            img_size (tuple): Image spatial shape ``(H, W)``.
            ch_freq (torch.Tensor): Global channel frequencies.
            shelf_path (str): Path to the denoiser shelf CSV.
            rank_num_ch (int, optional): Number of channels handled by this
                rank. Default: 1.
            rank_ch_start (int, optional): Global start index of local
                channels. Default: 0.
            rand_trans (bool, optional): Apply random flips/rotations before and
                after denoising. Default: True.
            device (torch.device, optional): Execution device. Default:
                torch.device("cpu").
            dtype (torch.dtype, optional): Tensor dtype for denoising tensors.
                Default: torch.float.
            verbose (bool, optional): Enable shelf-selection and spectral-update
                logs. Default: True.

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

        self._img_size = img_size
        self._rand_trans = rand_trans
        self._flip_table = (tuple(), (2,), (3,), (2, 3))
        self._ch_freq = ch_freq.flatten()
        self._list_net_scaling = [1.0 for _ in range(rank_num_ch)]
        self._shelf = {}
        self._list_network = [None for _ in range(rank_num_ch)]
        self._list_network_sigma = [0.0 for _ in range(rank_num_ch)]
        self._list_spec_corr_weight = [
            torch.ones(1, 3, 1, 1, device=self._device, dtype=self._dtype) / 3.0
            for _ in range(rank_num_ch)
        ]
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
        """Denoise local channels using neighbour-channel context.

        Args:
            x (torch.Tensor): Input tensor with shape ``(1, C, H, W)``.

        Returns:
            torch.Tensor: Denoised and positivity-projected tensor.

        Note:
            The method :meth:`update` must be called with valid heuristic noise level
            and peak-intensity estimates before this method is used, so that the
            appropriate denoisers are selected and loaded for each channel.
        """
        # Gather the one-channel halos needed by the first and last local slices.
        x_to_left = x[:, :1, :, :]
        x_from_left = torch.empty_like(x_to_left)
        x_to_right = x[:, -1:, :, :]
        x_from_right = torch.empty_like(x_to_right)

        left_rank_req = None
        right_rank_req = None

        if self._rank == 0:
            x_from_left = x_to_left
        elif self._world_size > 1:
            x_to_left = x_to_left.to(self._dist_device)
            x_from_left = x_from_left.to(self._dist_device)
            # Exchange the left halo with the previous rank.
            list_p2p_op = [
                dist.P2POp(dist.isend, x_to_left, self._rank - 1),
                dist.P2POp(dist.irecv, x_from_left, self._rank - 1),
            ]
            left_rank_req = dist.batch_isend_irecv(list_p2p_op)

        if self._rank == self._world_size - 1:
            x_from_right = x_to_right
        elif self._world_size > 1:
            x_to_right = x_to_right.to(self._dist_device)
            x_from_right = x_from_right.to(self._dist_device)
            # Exchange the right halo with the next rank.
            list_p2p_op = [
                dist.P2POp(dist.isend, x_to_right, self._rank + 1),
                dist.P2POp(dist.irecv, x_from_right, self._rank + 1),
            ]
            right_rank_req = dist.batch_isend_irecv(list_p2p_op)

        result = torch.empty_like(x)

        for i in range(self._rank_num_ch):
            # Assemble the three-slice context expected by the HyperAIRI denoiser.
            if i == 0:
                if left_rank_req is not None:
                    for req in left_rank_req:
                        req.wait()
                net_input = torch.cat(
                    [x_from_left.to(self._device), x[:, i : i + 2, :, :]], dim=1
                )
            elif i == self._rank_num_ch - 1:
                if right_rank_req is not None:
                    for req in right_rank_req:
                        req.wait()
                net_input = torch.cat(
                    [x[:, i - 1 : i + 1, :, :], x_from_right.to(self._device)], dim=1
                )
            else:
                net_input = x[:, i - 1 : i + 2, :, :]
            result_ch = self._den_ch_single(net_input, i)
            result[:, i : i + 1, :, :].copy_(result_ch)

        result = result.relu_()

        return result

    def _den_ch_single(self, net_input, i) -> torch.Tensor:
        """Denoise one channel with its two immediate neighbouring channels
            as contextual input.

        Args:
            net_input (torch.Tensor): Tensor containing the previous, current,
                and next channel slices.
            i (int): Local channel index.

        Returns:
            torch.Tensor: Denoised tensor for channel ``i`` with shape
            ``(1, 1, H, W)``.
        """
        if self._rand_trans:
            flip_dim = self._flip_table[torch.randint(4, (1,), device="cpu").item()]
            rot_times = torch.randint(4, (1,), device="cpu").item()
            x_trans = torch.flip(net_input, dims=flip_dim) if flip_dim else net_input
            x_trans = torch.rot90(x_trans, k=rot_times, dims=[2, 3])
            x_trans = x_trans.contiguous()
        else:
            x_trans = net_input.contiguous()
        # Match the denoiser's expected dynamic range, then restore the scale.
        result_ch = (
            self._list_network[i](x_trans / self._list_net_scaling[i])
            * self._list_net_scaling[i]
        )
        if self._rand_trans:
            result_ch = torch.rot90(result_ch, k=-rot_times, dims=[2, 3])
            result_ch = torch.flip(result_ch, dims=flip_dim)
            result_ch = result_ch.contiguous()

        # Add the spectral-correction residual predicted from the local context.
        result_ch = result_ch + torch.sum(
            net_input * self._list_spec_corr_weight[i], dim=1, keepdim=True
        )

        return result_ch

    @torch.no_grad()
    def update(
        self,
        ch_heuristic: torch.Tensor = None,
        ch_peak_est: torch.Tensor = None,
        spec_idx: torch.Tensor = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Update denoiser shelf selection and spectral correction weights.

        Args:
            ch_heuristic (torch.Tensor, optional): Per-channel heuristic noise
                levels. Default: None.
            ch_peak_est (torch.Tensor, optional): Per-channel peak-intensity
                estimates. Default: None.
            spec_idx (torch.Tensor, optional): Spectral-index map used to
                update channel-coupling weights. Default: None.

        Returns:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]: Lower and
            upper admissible peak ranges for local channels, or ``(None,
            None)`` when no shelf update is requested.

        Note:
            Shelf selection and spectral-correction updates are independent;
            either can be refreshed on its own.
        """
        peak_min = None
        peak_max = None

        # Adaptive shelf selection based on the heuristic noise-to-peak ratio.
        if ch_heuristic is not None and ch_peak_est is not None:
            peak_min = torch.zeros(
                1, self._rank_num_ch, 1, 1, device=self._device, dtype=self._dtype
            ).flatten()
            peak_max = torch.zeros(
                1, self._rank_num_ch, 1, 1, device=self._device, dtype=self._dtype
            ).flatten()
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
                            f"INFO: SHELF *** Channel {self._rank_ch_start + i}: "
                            f"heu/peak {ch_heu_peak[i]:.3e}, "
                            f"using network: {os.path.basename(self._shelf[sigma_s])}, "
                            f"peak range: [{peak_min[i].item():#.3g}, "
                            f"{peak_max[i].item():#.3g}], "
                            f"scaling factor: {self._list_net_scaling[i]:#.3g}",
                            flush=True,
                        )

            peak_min = peak_min.view(1, -1, 1, 1)
            peak_max = peak_max.view(1, -1, 1, 1)

        # Update the spectral-correction weights based on the input
        # spectral-index map and local frequency ratios.
        if spec_idx is not None:
            for ch in range(
                self._rank_ch_start, self._rank_num_ch + self._rank_ch_start
            ):
                # Build the three-channel correction kernel from local
                # frequency ratios and the current spectral-index map.
                freq_ratio = torch.ones(3, device=self._device, dtype=self._dtype)
                if ch > 0:
                    freq_ratio[0] = self._ch_freq[ch] / self._ch_freq[ch - 1]
                if ch < len(self._ch_freq) - 1:
                    freq_ratio[2] = self._ch_freq[ch] / self._ch_freq[ch + 1]

                # Use the spectral-index map to modulate the neighbour-channel
                # correction for this local channel.
                self._list_spec_corr_weight[ch - self._rank_ch_start] = (
                    freq_ratio.view(1, 3, 1, 1) ** spec_idx / 3.0
                )

            if self._is_dist:
                dist.barrier()
            if self._verbose and self._rank == 0:
                print("INFO: Update spectral correction weights.", flush=True)

        return (peak_min, peak_max)
