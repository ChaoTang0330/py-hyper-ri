"""
Prepare proper measurement operator, prior and algorithm for imaging task
"""

from typing import Dict
import os
import torch
import torch.distributed as dist
import numpy as np
from astropy.io import fits

from .prox_operator import ProxOpHyperAIRI
from .optimiser import FBHyperAIRI

# from .utils import gen_imaging_weight
from .ri_measurement_operator.pysrc.utils.io_hs import load_data_to_tensor_hs
from .ri_measurement_operator.pysrc.measOperator.meas_op_hs import MeasOpHyperspectral
# from .utils.dist import gather_ch_to_tensor


def env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v is not None else default


def imager_hs(param_optimiser: Dict, param_measop: Dict, param_proxop: Dict) -> None:
    """
    Imager for small scale RI imaging task.

    This function prepares the measurement operator, prior, and algorithm for the imaging task.
    It supports different algorithms such as 'airi', 'usara', and 'cairi'. The function also
    handles the imaging process if the 'flag_imaging' is set in the 'param_optimiser'.

    Args:
        param_optimiser (dict): A dictionary containing the parameters for the optimiser.
            It includes parameters like 'algorithm', 'im_min_itr', 'im_max_itr', 'im_var_tol',
            'im_peak_est', 'heu_noise_scale', 'dnn_adaptive_peak', 'dnn_adaptive_peak_tol_min',
            'dnn_adaptive_peak_tol_max', 'dnn_adaptive_peak_tol_step', 'result_path', 'itr_save',
            'verbose', and 'flag_imaging'.
        param_measop (dict): A dictionary containing the parameters for the measurement operator.
            It includes parameters like 'superresolution', 'im_pixel_size', 'flag_data_weighting',
            'weight_load', 'img_size', 'weight_type', 'weight_gridsize', 'weight_robustness',
            'dtype', 'device', 'nufft_grid_size', 'nufft_kb_kernel_dim', and 'nufft_mode'.
        param_proxop (dict): A dictionary containing the parameters for the proximal operator.
            It includes parameters like 'dnn_shelf_path', 'dnn_apply_transform', 'device', 'dtype',
            and 'verbose'.
    """
    # torch.backends.fp32_precision = "ieee"
    # torch.backends.cuda.matmul.fp32_precision = "ieee"
    # torch.backends.cudnn.fp32_precision = "ieee"
    # torch.backends.cudnn.conv.fp32_precision = "ieee"
    # torch.backends.cudnn.rnn.fp32_precision = "ieee"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True

    # init torch.distributed
    # Get process info from environment variables
    param_comm = {
        "rank": env_int("RANK", 0),
        "world_size": env_int("WORLD_SIZE", 1),
        "local_rank": env_int("LOCAL_RANK", 0),
        "local_world_size": env_int("LOCAL_WORLD_SIZE", 1),
        "node_rank": env_int("NODE_RANK", 0),
    }

    # Bind process to one GPU
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=param_comm["rank"],
        world_size=param_comm["world_size"],
        device_id=device,
    )

    # Small header
    dist.barrier()
    if param_comm["rank"] == 0:
        print(
            f"INFO: torch.distributed initialized with backend={dist.get_backend()}",
            flush=True,
        )
        print(
            f"INFO: num of processes = {param_comm['world_size']}, "
            f"num of nodes = {param_comm['world_size'] // param_comm['local_world_size']}",
            flush=True,
        )
    dist.barrier()

    # load data
    data = load_data_to_tensor_hs(
        uv_file_path=param_optimiser["data_file"],
        super_resolution=param_measop["superresolution"],
        image_pixel_size=param_measop["im_pixel_size"],
        data_weighting=param_measop["flag_data_weighting"],
        load_weight=param_measop["weight_load"],
        img_size=param_optimiser["img_size"],
        weight_type=param_measop["weight_type"],
        weight_gridsize=param_measop["weight_gridsize"],
        weight_robustness=param_measop["weight_robustness"],
        dtype=param_measop["dtype"],
        device=device,
        verbose=param_optimiser["verbose"],
        ch_start=param_optimiser["ch_start"],
        ch_last=param_optimiser["ch_last"],
        ch_effctive=param_optimiser["ch_effective"],
        invert_uvw_flag=param_measop["invert_uvw_flag"],
    )

    # prepare measurement operator
    meas_op = MeasOpHyperspectral(
        u=data["u"],
        v=data["v"],
        img_size=param_optimiser["img_size"],
        natural_weight=data["nW"],
        image_weight=data["nWimag"],
        multi_stream=False,
        device=device,
        dtype=param_measop["dtype"],
    )

    # prepare prior
    prox_op_hyperairi = ProxOpHyperAIRI(
        img_size=param_optimiser["img_size"],
        rank_num_ch=len(data["y"]),
        rank_ch_start=data["rank_eff_ch_start"],
        ch_freq=data["freqs"],
        shelf_path=param_proxop["dnn_shelf_path"],
        rand_trans=param_proxop["dnn_apply_transform"],
        compile_net=False,
        multi_stream=False,
        device=device,
        dtype=param_proxop["dtype"],
        verbose=param_proxop["verbose"],
    )

    # prepare optimiser
    optimiser = FBHyperAIRI(
        meas=data["y"],
        ch_freq=data["freqs"],
        meas_op=meas_op,
        prox_op=prox_op_hyperairi,
        ch_start=data["rank_eff_ch_start"],
        meas_op_approx=None,
        im_min_itr=param_optimiser["im_min_itr"],
        im_max_itr=param_optimiser["im_max_itr"],
        im_var_tol=param_optimiser["im_var_tol"],
        im_peak_est=param_optimiser["im_peak_est"],
        heu_noise_scale=param_optimiser["heu_noise_scale"],
        adapt_net_select=param_optimiser["dnn_adaptive_peak"],
        peak_tol_min=param_optimiser["dnn_adaptive_peak_tol_min"],
        peak_tol_max=param_optimiser["dnn_adaptive_peak_tol_max"],
        peak_tol_step=param_optimiser["dnn_adaptive_peak_tol_step"],
        save_pth=param_optimiser["result_path"],
        file_prefix=param_optimiser["file_prefix"],
        iter_save=param_optimiser["itr_save"],
        verbose=param_optimiser["verbose"],
    )
    # left to defult values:
    # spec_update: bool = True,
    # spec_max_update_count: int = 20,
    # spec_max_update_interval: int = 200,
    # spec_update_tol_max: float = 1e-2,
    # spec_update_tol_min: float = 1e-5,
    # spec_update_tol_step: float = 0.5,

    # imaging
    if param_optimiser["flag_imaging"]:
        # initialisation
        optimiser.initialisation()
        # run imaging loop
        optimiser.run()
        # finalisation
        optimiser.finalisation()

        dist.barrier()
        print(
            f"INFO: rank {param_comm['rank']}: The maximum GPU memory allocated {torch.cuda.memory_allocated()}",
            flush=True,
        )

    # clsoing down torch.distributed
    torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()
