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
# from .optimiser import HyperAIRI
# from .utils import gen_imaging_weight
from .ri_measurement_operator.pysrc.utils.io_hs import load_data_to_tensor_hs
from .ri_measurement_operator.pysrc.measOperator.meas_op_hs import MeasOpHyperspectral
from .utils.dist import gather_ch_to_tensor

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
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

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
        print(f"INFO: torch.distributed initialized with backend={dist.get_backend()}", flush=True)
        print(f"INFO: num of processes = {param_comm['world_size']}, num of nodes = {param_comm['world_size'] // param_comm['local_world_size']}", flush=True)
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
        device=device,
        dtype=param_measop["dtype"],
    )

    # prepare prior
    prox_op = ProxOpHyperAIRI(
        rank_num_ch=len(data["y"]),
        rank_ch_start=data["rank_eff_ch_start"],
        ch_freq=data["freqs"],
        shelf_path=param_proxop["dnn_shelf_path"],
        rand_trans=False, #param_proxop["dnn_apply_transform"],
        device=device,
        dtype=param_proxop["dtype"],
        verbose=param_proxop["verbose"],
    )

    # test hyperairi denoiser
    img_gdth = fits.getdata("/home/dp330/dp330/dc-tang2/Small-scale-RI-imaging-hs/data/3c353_uvw_id_00_dt_9.46_6.21_freqratio_1.70_nfreq_36_rotation_330.40/groundtruth.fits").astype(np.float32)
    img_gdth_rank = torch.tensor(img_gdth[data["rank_eff_ch_start"]:data["rank_eff_ch_start"]+len(data["y"]), :, :], device=device).unsqueeze(0)
    img_noisy = fits.getdata("/home/dp330/dp330/dc-tang2/Small-scale-RI-imaging-hs/data/img_noisy.fits").astype(np.float32)
    img_noisy_rank = torch.tensor(img_noisy[data["rank_eff_ch_start"]:data["rank_eff_ch_start"]+len(data["y"]), :, :], device=device).unsqueeze(0)
    spec_idx_map = fits.getdata("/home/dp330/dp330/dc-tang2/Small-scale-RI-imaging-hs/data/3c353_uvw_id_00_dt_9.46_6.21_freqratio_1.70_nfreq_36_rotation_330.40/index_map.fits").astype(np.float32)
    spec_idx_map = torch.from_numpy(spec_idx_map).to(device).unsqueeze(0).unsqueeze(0)
    prox_op.update(
        list_heuristic = [1.6e-4 for _ in range(len(data["y"]))],
        list_peak_est = [1.0 for _ in range(len(data["y"]))],
        spec_idx = spec_idx_map,
    )
    img_denoised_rank = prox_op(img_noisy_rank)

    # calculate SNR of the noisy and denoised images
    gdth_norm_rank = torch.linalg.vector_norm(img_gdth_rank)**2
    diff_norm_rank_noisy = torch.linalg.vector_norm(img_noisy_rank - img_gdth_rank)**2
    diff_norm_rank_denoised = torch.linalg.vector_norm(img_denoised_rank - img_gdth_rank)**2
    # gather norms from all ranks
    dist.all_reduce(gdth_norm_rank, op=dist.ReduceOp.SUM)
    dist.all_reduce(diff_norm_rank_noisy, op=dist.ReduceOp.SUM)
    dist.all_reduce(diff_norm_rank_denoised, op=dist.ReduceOp.SUM)

    snr_noisy = 10.0 * torch.log10(gdth_norm_rank / diff_norm_rank_noisy).item()
    snr_denoised = 10.0 * torch.log10(gdth_norm_rank / diff_norm_rank_denoised).item()

    if param_comm["rank"] == 0:
        print(f"INFO: Noisy Image SNR: {snr_noisy:.2f} dB", flush=True)
        print(f"INFO: Denoised Image SNR: {snr_denoised:.2f} dB", flush=True)

    img_denoised_full = gather_ch_to_tensor(
        img_denoised_rank,
        dest=0,
    )
    if param_comm["rank"] == 0:
        img_denoised_full = img_denoised_full.squeeze(0).cpu().numpy()
        fits.writeto(
            os.path.join(param_optimiser["result_path"], "img_denoised.fits"),
            img_denoised_full,
            overwrite=True,
        )

    # prepare optimiser

    # imaging
    
    # clsoing down torch.distributed
    torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()
