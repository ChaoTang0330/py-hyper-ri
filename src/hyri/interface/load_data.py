"""Load RI measurement data and convert them to torch tensors.

This file reads channel-wise measurement files, applies channel partitioning
and optional weighting logic, and returns tensors ready for measurement
operator and optimiser initialisation.
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
) -> dict:
    """Load channel-wise RI measurements and convert them into torch tensors.

    The input directory is expected to contain one ``msSpecs.mat`` file with
    global metadata (including ``freqs`` and ``uvw``), plus per-channel files
    matching ``*data_ch_<index>.mat``.

    Args:
        uv_file_path (str): Path to the input data directory.
        super_resolution (float, optional): Super-resolution factor used when
            pixel size is not explicitly provided. Default: 1.5.
        image_pixel_size (float, optional): Image pixel size in arcsec.
            Default: None.
        data_weighting (bool, optional): Whether to apply imaging weights.
            Default: True.
        load_weight (bool, optional): Whether to load imaging weights from the
            channel files. If False and ``data_weighting`` is True, imaging
            weights are generated. Default: False.
        img_size (Tuple[int, int], optional): Image size used by weighting
            generation. Default: (512, 512).
        weight_type (str, optional): Imaging-weight type. Default: "briggs".
        weight_gridsize (float, optional): Weight-grid size factor.
            Default: 2.0.
        weight_robustness (float, optional): Briggs robustness parameter.
            Default: 0.0.
        precond_weighting (bool, optional): Whether to generate preconditioning
            weights. Default: False.
        precond_weight_gridsize (float, optional): Grid size for
            preconditioning-weight generation. Default: 2.0.
        precond_weight_type (str, optional): Preconditioning-weight type.
            Default: "uniform".
        dtype (torch.dtype, optional): Real dtype for stored tensors.
            Default: ``torch.float64``.
        device (torch.device, optional): Target tensor device.
            Default: ``torch.device("cpu")``.
        data (dict, optional): Existing data dictionary to populate. If None,
            a new dictionary is created. Default: None.
        uvw_flag (tuple, optional): Optional per-channel flag arrays. If None,
            flags are read from channel files when available. Default: None.
        invert_uvw_flag (bool, optional): Whether to invert the loaded/provided
            UVW flags before filtering samples. Default: False.
        ch_start (int, optional): First physical channel index (1-based).
            Default: 1.
        ch_last (int, optional): Last physical channel index (1-based,
            inclusive). If None, all available channels are considered.
            Default: None.
        ch_effctive (int, optional): Number of effective channels after
            grouping. If None, all selected channels are used. Default: None.
        ch_partition (List[int], optional): Channel-group sizes used to map
            physical channels into effective channels. Default: None.
        verbose (bool, optional): Whether to print progress information.
            Default: True.

    Returns:
        dict: Data dictionary used by downstream operators/optimisers. Key
        entries include ``u``, ``v``, ``w``, ``natural_weight``,
        ``imaging_weight``, ``precond_weight``, optional ``y``, ``freqs``,
        ``image_pixel_size``, ``super_resolution``, ``max_proj_baseline``, and
        ``rank_eff_ch_start``.

    Raises:
        ValueError: If the provided channel partition does not match the number
            of selected physical channels.

    Notes:
        Tensor containers ``u``, ``v``, ``w``, ``natural_weight``,
        ``imaging_weight``, ``precond_weight``, and optional ``y`` are stored
        per effective channel, with each tensor shaped as ``(1, 1, N)``.

        In distributed mode, each rank loads only its assigned physical/effective
        channel slice. The returned ``rank_eff_ch_start`` gives the global
        effective-channel offset for the current rank.

        If ``data`` is provided, this function populates and updates it in
        place. If channel files do not contain ``data_I``, visibility tensors
        remain empty.

        Expected file content:

        - ``msSpecs.mat`` should provide ``freqs`` and ``uvw``.
        - ``data_ch_*.mat`` may provide ``flag``, ``weightsNat``,
          ``weightsImg``, and ``data_I``.

    Examples:
        Load all channels with default weighting options:

        >>> data = load_data_to_tensor("./data/measurement_set")
        >>> data["freqs"].shape[1] > 0
        True

        Load a channel range and aggregate into four effective channels:

        >>> data = load_data_to_tensor(
        ...     "./data/measurement_set",
        ...     ch_start=1,
        ...     ch_last=16,
        ...     ch_effctive=4,
        ...     ch_partition=[4, 4, 4, 4],
        ...     load_weight=False,
        ... )
        >>> len(data["u"])
        4
    """

    # Detect distributed execution context.
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

    # Initialise output containers.
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

    # Load global metadata from msSpecs.mat.
    ms_specs = load_mat_to_np(os.path.join(uv_file_path, "msSpecs.mat"))
    ms_specs["uvw"] = ms_specs["uvw"].squeeze()
    if ms_specs["uvw"].shape[0] == 3:
        ms_specs["uvw"] = ms_specs["uvw"].T
    freqs_physical = ms_specs["freqs"].flatten()
    num_freqs = freqs_physical.size

    # Resolve the physical/effective channel ranges.
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

    # Derive per-rank channel ranges in distributed mode.
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

    # Load each physical channel assigned to this rank.
    for ch, freq in enumerate(freqs_physical.tolist()[rank_ch_start:rank_ch_last]):
        u = ms_specs["uvw"][:, 0].copy().flatten()
        v = ms_specs["uvw"][:, 1].copy().flatten()
        w = ms_specs["uvw"][:, 2].copy().flatten()

        # Load channel-specific file data_ch_<index>.mat.
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

        # Convert valid UVW samples into wavelength units.
        wavelength = speed_of_light / freq
        u = u[np.logical_not(flag)] / wavelength
        v = v[np.logical_not(flag)] / wavelength
        w = w[np.logical_not(flag)] / wavelength

        # Track maximum projected baseline in wavelength units.
        max_proj_baseline = max(max_proj_baseline, np.max(np.sqrt(u**2 + v**2)))

        # Accumulate physical-channel arrays.
        u_physical.append(u)
        v_physical.append(v)
        w_physical.append(w)

        # Load per-channel visibilities and weights when available.
        if data_ch and "data_I" in data_ch:
            y_physical.append(data_ch["data_I"].flatten())

        if data_ch and "weightsNat" in data_ch and data_ch["weightsNat"]:
            natural_weight_physical.append(data_ch["weightsNat"])
            if natural_weight_physical[-1].size == 1:
                natural_weight_physical[-1] = natural_weight_physical[
                    -1
                ].item() * np.ones_like(u_physical[-1])
        else:
            natural_weight_physical.append(np.ones_like(u_physical[-1]))

        if load_weight:
            if data_ch and "weightsImg" in data_ch and data_ch["weightsImg"]:
                imaging_weight_physical.append(data_ch["weightsImg"])
            else:
                imaging_weight_physical = []
                load_weight = False
                data_weighting = False
                verbose_print(
                    "INFO: no imaging weights are available in the data file.",
                    verbose_general and ch == 0,
                )

    # Reduce max projected baseline across ranks if distributed.
    if is_distributed:
        all_reduce_device = device
        dist_backend = dist.get_backend()
        if dist_backend == "nccl":
            all_reduce_device = torch.device("cuda")
        elif dist_backend == "xccl":
            all_reduce_device = torch.device("xpu")
        else:
            all_reduce_device = torch.device("cpu")
        max_proj_baseline = torch.tensor(
            max_proj_baseline, dtype=torch.float64, device=all_reduce_device
        )
        dist.all_reduce(max_proj_baseline, op=dist.ReduceOp.MAX)
        max_proj_baseline = max_proj_baseline.cpu().item()

    data["max_proj_baseline"] = max_proj_baseline

    # Resolve pixel size from user input or super-resolution.
    if (
        "super_resolution" in data
        and ms_specs["super_resolution"].item() != super_resolution
    ):
        verbose_print(
            f"INFO: super resolution factor in the data file is "
            f'{ms_specs["super_resolution"].item():#.4g}, '
            f"while the user specified {super_resolution}.",
            verbose_general,
        )
        verbose_print(
            f"INFO: super resolution factor of "
            f'{ms_specs["super_resolution"].item():#.4g} will be used.',
            verbose_general,
        )
        super_resolution = ms_specs["super_resolution"].item()
    elif "super_resolution" not in data:
        data["super_resolution"] = super_resolution

    spatial_bandwidth = 2 * data["max_proj_baseline"]
    if image_pixel_size:
        verbose_print(
            f"INFO: user specified pixelsize: {image_pixel_size:#.4g} arcsec.",
            verbose_general,
        )
        super_resolution = (
            (180.0 / np.pi) * 3600.0 / (image_pixel_size * spatial_bandwidth)
        )
        verbose_print(
            f"INFO: corresponding to the super resolution factor: {super_resolution:#.4g}.",
            verbose_general,
        )
    else:
        image_pixel_size = (
            (180.0 / np.pi) * 3600.0 / (super_resolution * spatial_bandwidth)
        )
        verbose_print(
            f"INFO: default pixelsize: {image_pixel_size:#.4g} arcsec, "
            f"that is {super_resolution:#.4g} x nominal resolution.",
            verbose_general,
        )

    data["image_pixel_size"] = image_pixel_size
    halfSpatialBandwidth = (180.0 / np.pi) * 3600.0 / (image_pixel_size) / 2.0

    for ch in range(rank_eff_ch_last - rank_eff_ch_start):
        # Initialise empty tensors for one effective channel.
        data["u"].append(torch.empty(1, 1, 0, dtype=dtype, device=device))
        data["v"].append(torch.empty(1, 1, 0, dtype=dtype, device=device))
        data["w"].append(torch.empty(1, 1, 0, dtype=dtype, device=device))
        data["natural_weight"].append(torch.empty(1, 1, 0, dtype=dtype, device=device))
        data["imaging_weight"].append(torch.empty(1, 1, 0, dtype=dtype, device=device))
        data["precond_weight"].append(torch.empty(1, 1, 0, dtype=dtype, device=device))
        data["y"].append(torch.empty(1, 1, 0, dtype=dtype, device=device))

        for ch_physical in range(sum(ch_partition[:ch]), sum(ch_partition[: ch + 1])):
            # Convert UVW coordinates and append them as torch tensors.
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

            # Append natural weights.
            data["natural_weight"][ch] = torch.cat(
                (
                    data["natural_weight"][ch],
                    torch.tensor(
                        natural_weight_physical[ch_physical], dtype=dtype, device=device
                    ).view(1, 1, -1),
                ),
                dim=-1,
            )

            # Append imaging weights when loaded from files.
            if load_weight:
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

            # Append visibility tensors.
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

        # Compute imaging weights when requested.
        if data_weighting:
            if not load_weight:
                verbose_print(
                    "INFO: computing imaging weights...", verbose_general and ch == 0
                )
                if "weight_robustness" in data:
                    weight_robustness = data["weight_robustness"].item()
                    verbose_print(
                        f"INFO: load weight_robustness from data file {weight_robustness:#.4g}.",
                        verbose_general and ch == 0,
                    )
                else:
                    verbose_print(
                        f"INFO: weight_robustness {weight_robustness:#.4g}",
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
            data["imaging_weight"][ch] = torch.ones(1, 1, 1, dtype=dtype, device=device)

        # Compute preconditioning weights when requested.
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

        # Apply natural and imaging weights to visibilities.
        if y_physical:
            with torch.no_grad():
                data["y"][ch] *= data["natural_weight"][ch] * data["imaging_weight"][ch]

            # Validate weighted-visibility dimensions.
            assert data["y"][ch].shape == data["u"][ch].shape, (
                f'dimensions do not match: y -> {data["y"][ch].shape}, '
                f'u -> {data["u"][ch].shape}'
            )

        # Validate UV and weight dimensions.
        assert (
            data["u"][ch].shape == data["natural_weight"][ch].shape
            or data["natural_weight"][ch].numel() == 1
        ) and (
            data["u"][ch].shape == data["imaging_weight"][ch].shape
            or data["imaging_weight"][ch].numel() == 1
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
            f"frequency {data['freqs'][ch]:#.4g} Hz.",
            verbose,
        )

    # Freeze list containers as tuples.
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
