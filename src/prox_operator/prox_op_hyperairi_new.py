"""
HyperAIRI proximity operator
Try to denoise in batch.
Seems to have the same speed as the serial version.
This version still needs further testing and debugging.
"""

import csv
import os

from typing import Tuple
import torch
import torch.distributed as dist
from onnx2torch import convert

from .prox_op import ProxOp

# TODO: Add faceting functionality


class ProxOpHyperAIRI(ProxOp):
    """
    HyperAIRI proximity operator

    This class implements the HyperAIRI proximity operator which uses AIRI denoisers for regularisations.
    It uses a shelf of pre-trained AIRI denoisers and selects the appropriate one based on
    the estimated maximum intensities of the target image and the heuristic noise level.
    """

    def __init__(
        self,
        img_size: tuple,
        rank_num_ch: int,
        rank_ch_start: int,
        ch_freq: torch.Tensor,
        shelf_path: str,
        rand_trans: bool = True,
        multi_stream: bool = False,
        compile_net: bool = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float,
        verbose: bool = True,
    ) -> None:
        """
        Initializes the HyperAIRI proximity operator with the given parameters.

        Args:
            rank_num_ch (int): Number of channels assigned to the current rank.
            rank_ch_start (int): Starting channel index for the current rank.
            ch_freq (torch.Tensor): Frequencies of the channels.
            shelf_path (str): Path to the CSV file containing the denoiser shelf.
            rand_trans (bool, optional): If True, apply random transformations to the input
                before applying denoisers. Defaults to True.
            multi_stream (bool, optional): If True, use multiple CUDA streams for parallelism. Defaults to True.
            compile_net (bool, optional): If True, compile the network for optimization. Defaults to True.
            device (torch.device, optional): The device on which computations will be
                performed. Defaults to torch.device("cpu").
            dtype (torch.dtype, optional): The data type of the input. Defaults to torch.float.
            verbose (bool, optional): If True, print progress messages. Defaults to True.
        """
        super().__init__(device=device, dtype=dtype)
        self._rank_num_ch = rank_num_ch
        self._rank_ch_start = rank_ch_start
        self._rand_trans = rand_trans
        self._ch_freq = ch_freq.flatten()
        self._compile_net = compile_net
        self._net_scaling = torch.ones(rank_num_ch, 1, 1, 1, device=device, dtype=dtype)
        self._shelf = {}
        self._dict_network = {}
        self._dict_network_ch = {}
        self._spec_corr_weight = (
            torch.ones(
                self._rank_num_ch, 3, 1, 1, device=self._device, dtype=self._dtype
            )
            / 3.0
        )
        self._verbose = verbose

        self._rank = 0
        self._world_size = 1
        self._is_dist = dist.is_available() and dist.is_initialized()
        if self._is_dist:
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()

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

        # multi strem (no obvious speed up, relying on batch processing instead)
        # self._multi_stream = multi_stream and (device.type == "cuda")
        # if self._multi_stream:
        #     self._streams_ch = [
        #         torch.cuda.Stream(device=self._device) for _ in range(rank_num_ch)
        #     ]
        #     self._stream_comm_left = torch.cuda.Stream(device=self._device)
        #     self._stream_comm_right = torch.cuda.Stream(device=self._device)
        #     if self._verbose and self._rank == 0:
        #         print(
        #             "INFO: Using multi CUDA streams in HyperAIRI denoising.",
        #             flush=True,
        #         )

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the HyperAIRI prox operator to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The denoised tensor.
        """
        # # exchange border channels with neighbouring ranks
        # x_to_left = x[0, :1, :, :].clone()
        # x_from_left = torch.empty_like(x_to_left)
        # x_to_right = x[0, -1:, :, :].clone()
        # x_from_right = torch.empty_like(x_to_right)

        # # communicate with left rank
        # left_rank_req = None
        # right_rank_req = None
        # if self._world_size > 1:
        #     if self._rank == 0:
        #         x_from_left = x_to_left
        #     else:
        #         list_p2p_op = [
        #             dist.P2POp(dist.isend, x_to_left, self._rank - 1),
        #             dist.P2POp(dist.irecv, x_from_left, self._rank - 1),
        #         ]
        #         left_rank_req = dist.batch_isend_irecv(list_p2p_op)

        #     # communicate with right rank
        #     if self._rank == self._world_size - 1:
        #         x_from_right = x_to_right
        #     else:
        #         list_p2p_op = [
        #             dist.P2POp(dist.isend, x_to_right, self._rank + 1),
        #             dist.P2POp(dist.irecv, x_from_right, self._rank + 1),
        #         ]
        #         right_rank_req = dist.batch_isend_irecv(list_p2p_op)
        # exchange border channels with neighbouring ranks
        x_to_left = x[0, :1, :, :]
        x_from_left = torch.empty_like(x_to_left)
        x_to_right = x[0, -1:, :, :]
        x_from_right = torch.empty_like(x_to_right)

        left_rank_req = None
        right_rank_req = None
        if self._rank == 0:
            x_from_left = x_to_left
        elif self._world_size > 1:
            x_to_left = x_to_left.clone()
            # communicate with left rank
            list_p2p_op = [
                dist.P2POp(dist.isend, x_to_left, self._rank - 1),
                dist.P2POp(dist.irecv, x_from_left, self._rank - 1),
            ]
            left_rank_req = dist.batch_isend_irecv(list_p2p_op)

        if self._rank == self._world_size - 1:
            x_from_right = x_to_right
        elif self._world_size > 1:
            x_to_right = x_to_right.clone()
            # communicate with right rank
            list_p2p_op = [
                dist.P2POp(dist.isend, x_to_right, self._rank + 1),
                dist.P2POp(dist.irecv, x_from_right, self._rank + 1),
            ]
            right_rank_req = dist.batch_isend_irecv(list_p2p_op)
        # wait for communication to finish
        if left_rank_req is not None:
            for req in left_rank_req:
                req.wait()
        if right_rank_req is not None:
            for req in right_rank_req:
                req.wait()

        result = torch.empty_like(x)
        for sigma_s, list_ch in self._dict_network_ch.items():
            # prepare input for the denoiser
            net_input = torch.empty(
                len(list_ch),
                3,
                x.shape[2],
                x.shape[3],
                device=self._device,
                dtype=self._dtype,
            )
            for b_idx, ch_idx in enumerate(list_ch):
                if ch_idx == 0:
                    net_input[b_idx, 0:1, :, :].copy_(x_from_left)
                    net_input[b_idx, 1:3, :, :].copy_(x[0, 0:2, :, :])
                elif ch_idx == self._rank_num_ch - 1:
                    net_input[b_idx, 0:2, :, :].copy_(
                        x[0, ch_idx - 1 : ch_idx + 1, :, :]
                    )
                    net_input[b_idx, 2:3, :, :].copy_(x_from_right)
                else:
                    net_input[b_idx, :, :, :].copy_(
                        x[0, ch_idx - 1 : ch_idx + 2, :, :]
                    )

            spec_corr = torch.sum(
                net_input * self._spec_corr_weight[list_ch, ...], dim=1, keepdim=True
            )
            result_batch = self._den_ch_batch(
                self._dict_network[sigma_s], net_input, list_ch
            )
            result_batch += spec_corr
            result[0, list_ch, :, :].copy_(result_batch[:, 0, :, :])

        result = result.relu_()

        # for i in range(self._rank_num_ch):
        #     # prepare input for the denoiser
        #     if i == 0:
        #         if left_rank_req is not None:
        #             for req in left_rank_req:
        #                 req.wait()
        #         net_input = torch.cat([x_from_left, x[:, i : i + 2, :, :]], dim=1)
        #     elif i == self._rank_num_ch - 1:
        #         if right_rank_req is not None:
        #             for req in right_rank_req:
        #                 req.wait()
        #         net_input = torch.cat([x[0, i - 1 : i + 1, :, :], x_from_right], dim=1)
        #     else:
        #         net_input = x[:, i - 1 : i + 2, :, :]
        #     result_ch = self._den_ch_single(net_input, i)
        #     result[:, i : i + 1, :, :].copy_(torch.clamp(result_ch, min=0.0))

        return result

    def _den_ch_batch(self, net, net_input, list_ch_idx) -> torch.Tensor:
        """
        Denoises a single channel input using the corresponding denoiser.

        Args:
            net_input (torch.Tensor): The input tensor for a single channel.
            i (int): The channel index.

        Returns:
            torch.Tensor: The denoised tensor for the single channel.
        """
        # apply denoiser
        if self._rand_trans:
            flip_dim = (tuple(), (2,), (3,), (2, 3))[torch.randint(4, (1,)).item()]
            rot_times = torch.randint(4, (1,)).item()
            if flip_dim:
                x_trans = torch.flip(net_input, dims=flip_dim)
            else:
                x_trans = net_input
            x_trans = torch.rot90(x_trans, k=rot_times, dims=[2, 3])
            x_trans = x_trans.contiguous()
        else:
            x_trans = net_input.contiguous()

        scale = self._net_scaling[list_ch_idx, ...]
        result_ch = net(x_trans / scale) * scale

        if self._rand_trans:
            result_ch = torch.rot90(result_ch, k=-rot_times, dims=[2, 3])
            if flip_dim:
                result_ch = torch.flip(result_ch, dims=flip_dim)
            result_ch = result_ch.contiguous()

        # apply spectral correction & positivity
        # result_ch = result_ch + torch.sum(
        #     net_input * self._list_spec_corr_weight[i], dim=1, keepdim=True
        # )

        return result_ch

    def update(
        self,
        ch_heuristic: torch.Tensor = None,
        ch_peak_est: torch.Tensor = None,
        spec_idx: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the denoiser selection and scaling factor based on the given
        heuristic noise level and maximum intensity.
        Update the spectral correction weights based on the given spectral index map.

        Args:
            heuristic (float): The heuristic noise level.
            peak_est (float): The estimated maximum intensity.
        """
        peak_min = torch.empty(
            1, self._rank_num_ch, 1, 1, device=self._device, dtype=self._dtype
        ).flatten()
        peak_max = torch.empty(
            1, self._rank_num_ch, 1, 1, device=self._device, dtype=self._dtype
        ).flatten()

        if ch_heuristic is not None and ch_peak_est is not None:
            ch_heu_peak = ch_heuristic / ch_peak_est
            ch_heu_peak = ch_heu_peak.cpu().flatten().tolist()

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
                    peak_max[i] = torch.tensor(
                        float("inf"), device=self._device, dtype=self._dtype
                    )

                net_updated = False
                if sigma_s not in self._dict_network:
                    self._dict_network[sigma_s] = (
                        convert(self._shelf[sigma_s]).to(self._device).eval()
                    )
                    if self._compile_net:
                        self._dict_network[sigma_s] = torch.compile(
                            self._dict_network[sigma_s], dynamic=False
                        )
                    self._dict_network_ch[sigma_s] = [
                        i,
                    ]
                    self._net_scaling[i] = ch_heuristic.view(-1)[i] / sigma_s
                    net_updated = True
                else:
                    # check if the channel index is already assigned
                    prev_sigma = None
                    for net_sigma, ch_set in self._dict_network_ch.items():
                        if i in ch_set:
                            prev_sigma = net_sigma
                            break

                    if prev_sigma is None or abs(prev_sigma - sigma_s) > 1e-10:
                        if prev_sigma is not None:
                            self._dict_network_ch[prev_sigma].remove(i)
                            # remove network with no assigned channels
                            if not self._dict_network_ch[prev_sigma]:
                                self._dict_network.pop(prev_sigma, None)
                                self._dict_network_ch.pop(prev_sigma, None)

                        self._dict_network_ch[sigma_s].append(i)
                        self._dict_network_ch[sigma_s].sort()
                        self._net_scaling[i] = ch_heuristic.view(-1)[i] / sigma_s
                        net_updated = True

                if net_updated and self._verbose:
                    print(
                        f"SHELF *** Channel {self._rank_ch_start + i}: heu/peak {ch_heu_peak[i]:.3e}, "
                        f"using network: {os.path.basename(self._shelf[sigma_s])}, "
                        f"peak range: [{peak_min[i].item():.3e}, {peak_max[i].item():.3e}], "
                        f"scaling factor: {self._net_scaling[i].cpu().item():.3e}",
                        flush=True,
                    )

                # if abs(self._list_network_sigma[i] - sigma_s) > 1e-10:
                #     self._list_network_sigma[i] = sigma_s
                #     self._list_network[i] = (
                #         convert(self._shelf[sigma_s]).to(self._device).eval()
                #     )
                #     if self._compile_net:
                #         self._list_network[i] = torch.compile(self._list_network[i], dynamic=False)
                #     self._list_net_scaling[i] = ch_heuristic.view(-1)[i] / sigma_s

                #     if self._verbose:
                #         print(
                #             f"SHELF *** Channel {self._rank_ch_start + i}: heu/peak {ch_heu_peak.view(-1)[i].item():.3e}, "
                #             f"using network: {os.path.basename(self._shelf[sigma_s])}, "
                #             f"peak range: [{peak_min[i].item():.3e}, {peak_max[i].item():.3e}], "
                #             f"scaling factor: {self._list_net_scaling[i]:.3e}",
                #             flush=True,
                #         )

        if spec_idx is not None:
            if self._spec_corr_weight.shape[-2:] != spec_idx.shape[-2:]:
                self._spec_corr_weight = torch.empty(
                    self._rank_num_ch,
                    3,
                    spec_idx.shape[-2],
                    spec_idx.shape[-1],
                    device=self._device,
                    dtype=self._dtype,
                )
            for ch in range(
                self._rank_ch_start, self._rank_num_ch + self._rank_ch_start
            ):
                freq_ratio = torch.ones(3, device=self._device, dtype=self._dtype)
                if ch > 0:
                    freq_ratio[0] = self._ch_freq[ch] / self._ch_freq[ch - 1]
                if ch < len(self._ch_freq) - 1:
                    freq_ratio[2] = self._ch_freq[ch] / self._ch_freq[ch + 1]

                self._spec_corr_weight[
                    ch - self._rank_ch_start : ch - self._rank_ch_start + 1
                ] = (freq_ratio.view(1, 3, 1, 1) ** spec_idx / 3.0)

            if self._verbose and self._rank == 0:
                print("INFO: Update spectral correction weights.", flush=True)

        return (peak_min.view(1, -1, 1, 1), peak_max.view(1, -1, 1, 1))
