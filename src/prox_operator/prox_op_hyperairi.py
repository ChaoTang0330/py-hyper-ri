"""
AIRI proximity operator
"""

import csv
import os
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
        rank_num_ch: int,
        rank_ch_start: int,
        ch_freq: torch.Tensor,
        shelf_path: str,
        rand_trans: bool = True,
        multi_stream: bool = True,
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
                performed. Defaults to torch.device("cpu").
            dtype (torch.dtype, optional): The data type of the input. Defaults to torch.float.
            verbose (bool, optional): If True, print progress messages. Defaults to True.
        """
        super().__init__(device=device, dtype=dtype)
        self._rank_num_ch = rank_num_ch
        self._rank_ch_start = rank_ch_start
        self._rand_trans = rand_trans
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

        # multi strem
        self._multi_stream = multi_stream and (device.type == "cuda")
        if self._multi_stream:
            self._streams_ch = [
                torch.cuda.Stream(device=self._device) for _ in range(rank_num_ch)
            ]
            self._stream_comm_left = torch.cuda.Stream(device=self._device)
            self._stream_comm_right = torch.cuda.Stream(device=self._device)
            if self._verbose and self._rank == 0:
                print(
                    "INFO: Using multi CUDA streams in HyperAIRI denoising.",
                    flush=True,
                )


    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the HyperAIRI prox operator to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The denoised tensor.
        """
        # exchange border channels with neighbouring ranks
        x_to_left = x[:, :1, :, :].clone()
        x_from_left = torch.empty_like(x[:, :1, :, :])
        x_to_right = x[:, -1:, :, :].clone()
        x_from_right = torch.empty_like(x[:, -1:, :, :])

        if self._multi_stream:
            curr_stream = torch.cuda.current_stream(device=self._device)

        # communicate with left rank
        left_rank_req = None
        right_rank_req = None
        if self._world_size > 1:
            if self._rank == 0:
                x_from_left = x_to_left
            else:
                list_p2p_op = [
                    dist.P2POp(dist.isend, x_to_left, self._rank - 1),
                    dist.P2POp(dist.irecv, x_from_left, self._rank - 1),
                ]
                if self._multi_stream:
                    self._stream_comm_left.wait_stream(curr_stream)
                    with torch.cuda.stream(self._stream_comm_left):
                        left_rank_req = dist.batch_isend_irecv(list_p2p_op)
                else:
                    left_rank_req = dist.batch_isend_irecv(list_p2p_op)

            # communicate with right rank
            if self._rank == self._world_size - 1:
                x_from_right = x_to_right
            else:
                list_p2p_op = [
                    dist.P2POp(dist.isend, x_to_right, self._rank + 1),
                    dist.P2POp(dist.irecv, x_from_right, self._rank + 1),
                ]
                if self._multi_stream:
                    self._stream_comm_right.wait_stream(curr_stream)
                    with torch.cuda.stream(self._stream_comm_right):
                        right_rank_req = dist.batch_isend_irecv(list_p2p_op)
                else:
                    right_rank_req = dist.batch_isend_irecv(list_p2p_op)

        result = torch.empty_like(x)

        if not self._multi_stream:
            for i in range(self._rank_num_ch):
                # prepare input for the denoiser
                if i == 0:
                    if left_rank_req is not None:
                        for req in left_rank_req:
                            req.wait()
                    net_input = torch.cat([x_from_left, x[:, i : i + 2, :, :]], dim=1)
                elif i == self._rank_num_ch - 1:
                    if right_rank_req is not None:
                        for req in right_rank_req:
                            req.wait()
                    net_input = torch.cat([x[:, i - 1 : i + 1, :, :], x_from_right], dim=1)
                else:
                    net_input = x[:, i - 1 : i + 2, :, :]
                result_ch = self._den_ch_single(net_input, i)
                result[:, i : i + 1, :, :].copy_(torch.clamp(result_ch, min=0.0))
        else:
            # denoise all channels in parallel streams
            for i in range(self._rank_num_ch):
                self._streams_ch[i].wait_stream(curr_stream)
                if i == 0 and self._rank > 0:
                    self._streams_ch[i].wait_stream(self._stream_comm_left)
                elif i == self._rank_num_ch - 1 and self._rank < self._world_size - 1:
                    self._streams_ch[i].wait_stream(self._stream_comm_right)
                with torch.cuda.stream(self._streams_ch[i]):
                    # prepare input for the denoiser
                    if i == 0:
                        net_input = torch.cat([x_from_left, x[:, i : i + 2, :, :]], dim=1)
                    elif i == self._rank_num_ch - 1:
                        net_input = torch.cat([x[:, i - 1 : i + 1, :, :], x_from_right], dim=1)
                    else:
                        net_input = x[:, i - 1 : i + 2, :, :]
                    result_ch = self._den_ch_single(net_input, i)
                    result[:, i : i + 1, :, :].copy_(torch.clamp(result_ch, min=0.0))
            # synchronize all streams
            for s in self._streams_ch:
                curr_stream.wait_stream(s)
            
            # if left_rank_req is not None:
            #     for req in left_rank_req:
            #         req.wait()
            # if right_rank_req is not None:
            #     for req in right_rank_req:
            #         req.wait()

        return result
    
    def _den_ch_single(self, net_input, i) -> torch.Tensor:
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
            flip_dim = ([], (2,), (3,), (2, 3))[torch.randint(4, (1,)).item()]
            rot_times = torch.randint(4, (1,)).item()
            x_trans = torch.flip(net_input, dims=flip_dim)
            x_trans = torch.rot90(x_trans, k=rot_times, dims=[2, 3])
            x_trans = x_trans.contiguous()
        else:
            x_trans = net_input.contiguous()
        result_ch = (
            self._list_network[i](x_trans / self._list_net_scaling[i])
            * self._list_net_scaling[i]
        )
        if self._rand_trans:
            result_ch = torch.rot90(result_ch, k=-rot_times, dims=[2, 3])
            result_ch = torch.flip(result_ch, dims=flip_dim)
            result_ch = result_ch.contiguous()

        # apply spectral correction & positivity
        result_ch = result_ch + torch.sum(
            net_input * self._list_spec_corr_weight[i], dim=1, keepdim=True
        )

        return result_ch

    def update(
        self,
        ch_heuristic: torch.Tensor = None,
        ch_peak_est: torch.Tensor = None,
        spec_idx: torch.Tensor = None,
    ) -> None:
        """
        Updates the denoiser selection and scaling factor based on the given
        heuristic noise level and maximum intensity.
        Update the spectral correction weights based on the given spectral index map.

        Args:
            heuristic (float): The heuristic noise level.
            peak_est (float): The estimated maximum intensity.
        """
        peak_min = torch.empty(
            1, self._rank_num_ch, 1 , 1, device=self._device, dtype=self._dtype
        ).flatten()
        peak_max = torch.empty(
            1, self._rank_num_ch, 1 , 1, device=self._device, dtype=self._dtype
        ).flatten()

        if ch_heuristic is not None and ch_peak_est is not None:
            ch_heu_peak = ch_heuristic / ch_peak_est

            for i in range(self._rank_num_ch):
                sigma_s = max(
                    filter(
                        lambda sigma: sigma <= ch_heu_peak.view(-1)[i].item(),
                        self._shelf.keys(),
                    ),
                    default=None,
                )
                if sigma_s:
                    peak_max[i] = ch_heuristic.view(-1)[i] / sigma_s
                    sigma_s1 = min(
                        filter(lambda sigma: sigma > sigma_s, self._shelf.keys()),
                        default=None,
                    )
                    if sigma_s1:
                        peak_min[i] = ch_heuristic.view(-1)[i] / sigma_s1
                else:
                    sigma_s = min(self._shelf.keys())
                    peak_min[i] = ch_heuristic.view(-1)[i] / sigma_s
                    peak_max[i] = float("inf")

                if abs(self._list_network_sigma[i] - sigma_s) > 1e-10:
                    self._list_network_sigma[i] = sigma_s
                    self._list_network[i] = (
                        convert(self._shelf[sigma_s]).to(self._device).eval()
                    )
                    self._list_net_scaling[i] = ch_heuristic.view(-1)[i] / sigma_s

                    if self._verbose:
                        print(
                            f"SHELF *** Channel {self._rank_ch_start + i}: heu/peak {ch_heu_peak.view(-1)[i].item():.3e}, "
                            f"using network: {os.path.basename(self._shelf[sigma_s])}, "
                            f"peak range: [{peak_min[i].item():.3e}, {peak_max[i].item():.3e}], "
                            f"scaling factor: {self._list_net_scaling[i]:.3e}",
                            flush=True,
                        )

        if spec_idx is not None:
            for ch in range(
                self._rank_ch_start, self._rank_num_ch + self._rank_ch_start
            ):
                freq_ratio = torch.ones(3, device=self._device, dtype=self._dtype)
                if ch > 0:
                    freq_ratio[0] = self._ch_freq[ch] / self._ch_freq[ch - 1]
                if ch < len(self._ch_freq) - 1:
                    freq_ratio[2] = self._ch_freq[ch] / self._ch_freq[ch + 1]

                self._list_spec_corr_weight[ch - self._rank_ch_start] = (
                    freq_ratio.view(1, 3, 1, 1) ** spec_idx / 3.0
                )

            if self._verbose and self._rank == 0:
                print("INFO: Update spectral correction weights.", flush=True)

        return (peak_min.view(1, -1, 1, 1), peak_max.view(1, -1, 1, 1))
