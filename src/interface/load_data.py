"""
This module contains the function to load data from files and convert them to torch tensors.
"""

import os
import glob
import numpy as np
import torch
import torch.distributed as dist
from scipy.constants import speed_of_light
from typing import List, Optional, Tuple

from .io import verbose_print, load_mat_to_np
from .gen_imaging_weights import gen_imaging_weights


def load_data_to_tensor(
    uv_file_path: str,
    super_resolution: Optional[float] = 1.5,
    image_pixel_size: Optional[float] = None,
    data_weighting: bool = True,
    load_weight: bool = False,
    img_size: Tuple[int, int] = (512, 512),
    weight_type: str = "briggs",
    weight_gridsize: float = 2.0,
    weight_robustness: float = 0.0,
    precond_weighting: bool = False,
    precond_weight_gridsize: float = 2.0,
    precond_weight_type: str = "uniform",
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
    data: Optional[dict] = None,
    uvw_flag: Optional[tuple] = None,
    invert_uvw_flag: bool = False,
    ch_start: int = 1,
    ch_last: Optional[int] = None,
    ch_effctive: Optional[int] = None,
    ch_partition: Optional[List[int]] = None,
    verbose: bool = True,
):
    """Read u, v and imweight from specified path.
    In this path, there should be a file name `msSpecs.mat` containing the following fields:
        - freqs: a 1D array contains the frequencies of each channel.
        - uvw: a 2D array contains the uvw coordinates of the sampling points.
    Then for each channel, there should be a file named `data_ch_{channel}.mat` containing the following fields:
        - data_I: a 1D array of compext values contains the visibilities of the channel.
        - flag[optional]: a 1D array of boolean values marks the flagged uvw points.
        - weightsNat: the natural weighting of the channel.
        - weightsImg: the imaging weighting of the channel.

    Parameters
    ----------
    uv_file_path : str
        Path to the file containing sampling pattern,
        natural weights and (optional) imaging weights.
    super_resolution : float
        Super resolution factor.
    image_pixel_size : float, optional
        Image pixel size in arcsec, by default None
    data_weighting : bool, optional
        Flag to apply imaging weights, by default True
    load_weight : bool, optional
        Flag to load imaging weights from the file, by default False.
        If set to False and data_weighting is True, the imaging weights will be generated.
    dtype : torch.dtype, optional
        Data type to be used, by default torch.float64
    device : torch.device, optional
        Device to be used, by default torch.device('cpu')
    data : dict, optional
        Dictionary to store the data, by default None
    uvw_flag : tuple, optional
        Tuple of flags for each channel, by default None.
        If None, the flags will be loaded from the data file.
    invert_uvw_flag : bool, optional
        Flag to invert the uvw flags, by default False and flagged points will be removed.
        If True, the uvw flag will be inverted and flagged points will be keeped.
    ch_start : int, optional
        Starting channel number (1-based index), by default 1
    ch_last : int, optional
        Last channel number (1-based index, included), by default None.
        If None, all channels will be loaded.
    ch_effctive : int, optional
        Number of effective channels, by default None. If None, all channels will be loaded.
    verbose : bool, optional
        Flag to print information, by default True

    Returns
    -------
    data: dict
        Dictionary containing u, v, w, (optional) y, natural_weight, (optional) imaging_weight and other information.
    """

    # check if running in distributed mode
    is_distributed = dist.is_available() and dist.is_initialized()
    rank = 0
    world_size = 1
    if is_distributed:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    verbose_general = verbose and rank == 0
    verbose_print(
        f"INFO: loading sampling pattern from {uv_file_path}", verbose_general
    )

    # initialisation
    if data is None:
        data = {}

    data["u"] = []
    data["v"] = []
    data["w"] = []
    data["natural_weight"] = []
    data["imaging_weight"] = []
    data["precond_weight"] = []
    data["y"] = []
    data["max_proj_baseline"] = 0
    max_proj_baseline = 0

    freqs_physical = []
    u_physical = []
    v_physical = []
    w_physical = []
    natural_weight_physical = []
    imaging_weight_physical = []
    y_physical = []

    # load msSpecs.mat
    ms_specs = load_mat_to_np(os.path.join(uv_file_path, "msSpecs.mat"))
    ms_specs["uvw"] = ms_specs["uvw"].squeeze()
    if ms_specs["uvw"].shape[0] == 3:
        ms_specs["uvw"] = ms_specs["uvw"].T
    freqs_physical = ms_specs["freqs"].flatten()
    num_freqs = freqs_physical.size

    # range of channel index
    ch_start = max(0, ch_start - 1)
    if ch_last is None:
        ch_last = num_freqs
    else:
        ch_last = min(ch_last, num_freqs)
    if ch_effctive is None or ch_effctive < 0:
        ch_effctive = num_freqs
    else:
        ch_effctive = min(ch_effctive, ch_last - ch_start)
    if ch_partition:
        ch_par_total = sum(ch_partition)
        if ch_par_total != ch_last - ch_start:
            raise ValueError(
                f"The sum of ch_partition {ch_par_total} should be equal to "
                f"the number of channels to be loaded {ch_last - ch_start}."
            )
    else:
        ch_cat_num = (ch_last - ch_start) // ch_effctive
        ch_partition = [ch_cat_num] * ch_effctive
        ch_last = ch_start + ch_cat_num * ch_effctive
    verbose_print(
        f"INFO: loading data from channel {ch_start + 1} to {ch_last} "
        f"with {ch_effctive} effective channels. ",
        verbose_general,
    )
    data["freqs"] = np.zeros(ch_effctive, dtype=np.float64)
    for ch, ch_cat_num in enumerate(ch_partition):
        data["freqs"][ch] = np.mean(
            freqs_physical[
                ch_start + ch * ch_cat_num : ch_start + (ch + 1) * ch_cat_num
            ]
        )

    # update channel index if running in distributed mode
    rank_ch_start = 0
    rank_ch_last = sum(ch_partition)
    rank_eff_ch_start = 0
    rank_eff_ch_last = ch_effctive
    if is_distributed:
        num_eff_ch_per_proc = ch_effctive // world_size
        num_eff_ch_proc_rest = ch_effctive % world_size
        num_eff_ch_proc_cut = world_size - num_eff_ch_proc_rest
        rank_eff_ch_start = rank * num_eff_ch_per_proc
        rank_eff_ch_last = rank_eff_ch_start + num_eff_ch_per_proc
        if rank >= num_eff_ch_proc_cut:
            rank_eff_ch_start += rank - num_eff_ch_proc_cut
            rank_eff_ch_last += rank - num_eff_ch_proc_cut + 1

        rank_ch_start = sum(ch_partition[:rank_eff_ch_start]) + ch_start
        rank_ch_last = sum(ch_partition[:rank_eff_ch_last]) + ch_start

    data["rank_eff_ch_start"] = rank_eff_ch_start

    if is_distributed:
        dist.barrier()
    verbose_print(
        f"INFO: rank {rank} loading channels from {rank_ch_start + 1} to {rank_ch_last},"
        f"corresponding to effective channels from {rank_eff_ch_start + 1} to {rank_eff_ch_last}.",
        verbose,
    )
    if is_distributed:
        dist.barrier()

    # load data of each physical channel
    for ch, freq in enumerate(freqs_physical.tolist()[rank_ch_start:rank_ch_last]):
        u = ms_specs["uvw"][:, 0].copy().flatten()
        v = ms_specs["uvw"][:, 1].copy().flatten()
        w = ms_specs["uvw"][:, 2].copy().flatten()

        # load data_ch_{ch}.mat
        data_ch_file = glob.glob(
            os.path.join(uv_file_path, f"*data_ch_{ch+rank_ch_start+1}.mat")
        )[0]
        data_ch = {}
        if os.path.exists(data_ch_file):
            data_ch = load_mat_to_np(data_ch_file)

        if uvw_flag:
            flag = uvw_flag[ch].flatten()
            verbose_print(
                "INFO: uvw flag provided as user input.", verbose_general and ch == 0
            )
        elif data_ch and "flag" in data_ch:
            flag = data_ch["flag"].flatten()
            verbose_print(
                "INFO: uvw flag found in the data file.", verbose_general and ch == 0
            )
        else:
            flag = np.zeros(u.shape, dtype=bool)
            verbose_print(
                "INFO: no uvw flag found, all points are considered valid.",
                verbose_general and ch == 0,
            )
        if invert_uvw_flag:
            flag = np.logical_not(flag)

        # convert flagged uvw in units of the wavelength
        wavelength = speed_of_light / freq
        u = u[np.logical_not(flag)] / wavelength
        v = v[np.logical_not(flag)] / wavelength
        w = w[np.logical_not(flag)] / wavelength

        # max projection baseline in the unit of wavelength
        max_proj_baseline = max(max_proj_baseline, np.max(np.sqrt(u**2 + v**2)))

        # append to data
        u_physical.append(u)
        v_physical.append(v)
        w_physical.append(w)

        # load data of each channel if available
        if data_ch and "data_I" in data_ch:
            y_physical.append(data_ch["data_I"].flatten())

        if data_ch and "weightsNat" in data_ch:
            natural_weight_physical.append(data_ch["weightsNat"])
            if natural_weight_physical[-1].size == 1:
                natural_weight_physical[-1] = natural_weight_physical[
                    -1
                ].item() * np.ones_like(u_physical[-1])
        else:
            natural_weight_physical.append(np.ones_like(u_physical[-1]))

        if load_weight and data_ch and "weightsImg" in data_ch:
            imaging_weight_physical.append(data_ch["weightsImg"])
        else:
            imaging_weight_physical.append(np.ones_like(u_physical[-1]))
            load_weight = False

    # communicate max_proj_baseline if in distributed mode
    if is_distributed:
        max_proj_baseline = torch.tensor(
            max_proj_baseline, dtype=torch.float64, device=device
        )
        dist.all_reduce(max_proj_baseline, op=dist.ReduceOp.MAX)
        max_proj_baseline = max_proj_baseline.cpu().item()

    data["max_proj_baseline"] = max_proj_baseline

    # calculate pixel size
    if (
        "super_resolution" in data
        and ms_specs["super_resolution"].item() != super_resolution
    ):
        verbose_print(
            f"INFO: super resolution factor in the data file is "
            f'{ms_specs["super_resolution"].item():.4f}, '
            f"while the user specified {super_resolution}.",
            verbose_general,
        )
        verbose_print(
            f"INFO: super resolution factor of "
            f'{ms_specs["super_resolution"].item():.4f} will be used.',
            verbose_general,
        )
        super_resolution = ms_specs["super_resolution"].item()
    elif "super_resolution" not in data:
        data["super_resolution"] = super_resolution

    spatial_bandwidth = 2 * data["max_proj_baseline"]
    if image_pixel_size:
        verbose_print(
            f"INFO: user specified pixelsize: {image_pixel_size:.4e} arcsec.",
            verbose_general,
        )
        super_resolution = (
            (180.0 / np.pi) * 3600.0 / (image_pixel_size * spatial_bandwidth)
        )
        verbose_print(
            f"INFO: corresponding to the super resolution factor: {super_resolution:.4f}.",
            verbose_general,
        )
    else:
        image_pixel_size = (
            (180.0 / np.pi) * 3600.0 / (super_resolution * spatial_bandwidth)
        )
        verbose_print(
            f"INFO: default pixelsize: {image_pixel_size:.4e} arcsec, "
            f"that is {super_resolution:.4f} x nominal resolution.",
            verbose_general,
        )

    data["image_pixel_size"] = image_pixel_size
    halfSpatialBandwidth = (180.0 / np.pi) * 3600.0 / (image_pixel_size) / 2.0

    for ch in range(rank_eff_ch_last - rank_eff_ch_start):
        # initialize empty tensors
        data["u"].append(torch.empty(1, 1, 0, dtype=dtype, device=device))
        data["v"].append(torch.empty(1, 1, 0, dtype=dtype, device=device))
        data["w"].append(torch.empty(1, 1, 0, dtype=dtype, device=device))
        data["natural_weight"].append(torch.empty(1, 1, 0, dtype=dtype, device=device))
        data["imaging_weight"].append(torch.empty(1, 1, 0, dtype=dtype, device=device))
        data["precond_weight"].append(torch.empty(1, 1, 0, dtype=dtype, device=device))
        data["y"].append(torch.empty(1, 1, 0, dtype=dtype, device=device))

        for ch_physical in range(sum(ch_partition[:ch]), sum(ch_partition[: ch + 1])):
            # convert u,v,w to radians and save in torch tensor
            data["u"][ch] = torch.cat(
                (
                    data["u"][ch],
                    torch.tensor(
                        u_physical[ch_physical] * np.pi / halfSpatialBandwidth,
                        dtype=dtype,
                        device=device,
                    ).view(1, 1, -1),
                ),
                dim=-1,
            )
            data["v"][ch] = torch.cat(
                (
                    data["v"][ch],
                    -torch.tensor(
                        v_physical[ch_physical] * np.pi / halfSpatialBandwidth,
                        dtype=dtype,
                        device=device,
                    ).view(1, 1, -1),
                ),
                dim=-1,
            )
            data["w"][ch] = torch.cat(
                (
                    data["w"][ch],
                    -torch.tensor(
                        w_physical[ch_physical], dtype=dtype, device=device
                    ).view(1, 1, -1),
                ),
                dim=-1,
            )

            # convert natural_weight to torch tensor
            data["natural_weight"][ch] = torch.cat(
                (
                    data["natural_weight"][ch],
                    torch.tensor(
                        natural_weight_physical[ch_physical], dtype=dtype, device=device
                    ).view(1, 1, -1),
                ),
                dim=-1,
            )

            # concatenate image weights
            data["imaging_weight"][ch] = torch.cat(
                (
                    data["imaging_weight"][ch],
                    torch.tensor(
                        imaging_weight_physical[ch_physical],
                        dtype=dtype,
                        device=device,
                    ).view(1, 1, -1),
                ),
                dim=-1,
            )

            # convert visibilities to torch tensor
            if y_physical:
                data["y"][ch] = torch.cat(
                    (
                        data["y"][ch],
                        torch.tensor(
                            y_physical[ch_physical],
                            dtype=(
                                torch.complex64
                                if dtype == torch.float32
                                else torch.complex128
                            ),
                            device=device,
                        ).view(1, 1, -1),
                    ),
                    dim=-1,
                )

        # calculate image weight if needed
        if data_weighting:
            if not load_weight:
                verbose_print(
                    "INFO: computing imaging weights...", verbose_general and ch == 0
                )
                if "weight_robustness" in data:
                    weight_robustness = data["weight_robustness"].item()
                    verbose_print(
                        f"INFO: load weight_robustness from data file {weight_robustness}",
                        verbose_general and ch == 0,
                    )
                else:
                    verbose_print(
                        f"INFO: weight_robustness {weight_robustness}",
                        verbose_general and ch == 0,
                    )
                    data["imaging_weight"][ch] = gen_imaging_weights(
                        data["u"][ch].clone(),
                        data["v"][ch].clone(),
                        data["natural_weight"][ch].clone(),
                        img_size,
                        weight_type=weight_type,
                        weight_gridsize=weight_gridsize,
                        weight_robustness=weight_robustness,
                    ).to(device=device, dtype=dtype)
        else:
            verbose_print(
                "INFO: imaging weights will not be applied.",
                verbose_general and ch == 0,
            )

        # calculate preconditioning weights if needed
        if precond_weighting:
            verbose_print(
                "INFO: computing preconditioning weights...",
                verbose_general and ch == 0,
            )
            data["precond_weight"][ch] = (
                gen_imaging_weights(
                    data["u"][ch].clone(),
                    data["v"][ch].clone(),
                    data["natural_weight"][ch].clone(),
                    img_size,
                    weight_type=precond_weight_type,
                    weight_gridsize=precond_weight_gridsize,
                ).to(device=device, dtype=dtype)
                ** 2
            )
        else:
            verbose_print(
                "INFO: preconditioning weights will not be generated.",
                verbose_general and ch == 0,
            )
            data["precond_weight"][ch] = torch.ones(1, 1, 1, dtype=dtype, device=device)

        # apply natural weights and imaging weights to visibilities
        if y_physical:
            with torch.no_grad():
                data["y"][ch] *= data["natural_weight"][ch] * data["imaging_weight"][ch]

            # check dimensions
            assert data["y"][ch].shape == data["u"][ch].shape, (
                f'dimensions do not match: y -> {data["y"][ch].shape}, '
                f'u -> {data["u"][ch].shape}'
            )

        # check dimensions
        assert (
            data["u"][ch].shape == data["natural_weight"][ch].shape
            and data["u"][ch].shape == data["imaging_weight"][ch].shape
        ), (
            f'dimensions do not match: u -> {data["u"][ch].shape}, '
            f'natural_weight -> {data["natural_weight"][ch].shape}, '
            f'imaging_weight -> {data["imaging_weight"][ch].shape}'
        )

    if is_distributed:
        dist.barrier()
    for ch in range(rank_eff_ch_start, rank_eff_ch_last):
        verbose_print(
            f"INFO: effective channel {ch + 1} loaded "
            f"{data['u'][ch - rank_eff_ch_start].shape[-1]} uv points, "
            f"frequency {data['freqs'][ch]}.",
            verbose,
        )

    # convert list to tuple to avoid modification
    data["u"] = tuple(data["u"])
    data["v"] = tuple(data["v"])
    data["w"] = tuple(data["w"])
    data["natural_weight"] = tuple(data["natural_weight"])
    data["imaging_weight"] = tuple(data["imaging_weight"])
    data["precond_weight"] = tuple(data["precond_weight"])
    if data["y"]:
        data["y"] = tuple(data["y"])

    data["freqs"] = torch.tensor(data["freqs"], dtype=dtype, device=device).view(
        1, -1, 1, 1
    )

    if is_distributed:
        dist.barrier()
    verbose_print("INFO: data loading completed.", verbose_general)

    return data
