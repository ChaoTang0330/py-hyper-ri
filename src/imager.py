"""
Prepare proper measurement operator, prior and algorithm for imaging task
"""

from typing import Dict

import torch
import torch.distributed as dist

from .interface.io import env_int
from .regulariser import ProxSARAPos
from .regulariser import ProxEllipsoid
from .regulariser import DenAIRI, DenHyperAIRI
from .optimiser import FBSARA
from .optimiser import FBAIRI, PDAIRI, FBHyperAIRI

from .interface.load_data import load_data_to_tensor
from .ri_measurement_operator import MeasOpFinufft, MeasOpTkbn, MeasOpPSF
from .ri_measurement_operator import MeasOpHyperspectral


def imager(param_optimiser: Dict, param_measop: Dict, param_proxop: Dict) -> None:
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

    # set torch flags for reproducibility and performance
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
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
    use_distributed = param_comm["world_size"] > 1

    # sanity check
    if param_comm["world_size"] > param_optimiser["ch_effective"]:
        raise ValueError(
            f"Number of processes ({param_comm['world_size']}) should not exceed"
            f"the number of effective channels ({param_optimiser['ch_effective']})."
        )

    if use_distributed:
        dist.init_process_group(
            backend=param_optimiser["dist_backend"],
            init_method=param_optimiser["dist_init_method"],
            rank=param_comm["rank"],
            world_size=param_comm["world_size"],
        )

    # Small header
    if use_distributed:
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
    data = load_data_to_tensor(
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
        device=torch.device("cpu"),
        verbose=param_optimiser["verbose"],
        ch_start=param_optimiser["ch_start"],
        ch_last=param_optimiser["ch_last"],
        ch_effctive=param_optimiser["ch_effective"],
        ch_partition=param_optimiser["ch_partition"],
        invert_uvw_flag=param_measop["invert_uvw_flag"],
    )

    # prepare measurement operator
    MeasOpClass = None
    if param_measop["nufft_package"] == "finufft":
        MeasOpClass = MeasOpFinufft
    elif param_measop["nufft_package"] == "tkbn":
        MeasOpClass = MeasOpTkbn
    else:
        raise ValueError(
            f"Unsupported measurement operator type: {param_measop['nufft_package']}"
        )

    meas_op = MeasOpHyperspectral(
        u=data["u"],
        v=data["v"],
        img_size=param_optimiser["img_size"],
        natural_weight=data["natural_weight"],
        image_weight=data["imaging_weight"],
        meas_op_class=MeasOpClass,
        device=param_measop["device"],
        dtype=param_measop["dtype"],
        grid_size=param_measop["nufft_grid_size"],
        kernel_dim=param_measop["nufft_kb_kernel_dim"],
        mode=param_measop["nufft_mode"],
    )

    meas_op_approx = None
    if param_optimiser["approx_meas_op"]:
        meas_op_approx = MeasOpHyperspectral(
            u=data["u"],
            v=data["v"],
            img_size=param_optimiser["img_size"],
            natural_weight=data["natural_weight"],
            image_weight=data["imaging_weight"],
            meas_op_class=MeasOpPSF,
            device=param_measop["device"],
            dtype=param_measop["dtype"],
        )

    # prepare regulariser and optimiser for specfic algorithm
    if param_optimiser["algorithm"] == "usara":
        # prepare proximal operator of SARA with positivity constraint
        prox_op_sara = ProxSARAPos(
            param_optimiser["img_size"],
            rank_num_ch=len(data["y"]),
            rank_ch_start=data["rank_eff_ch_start"],
            device=param_proxop["device"],
            dtype=param_proxop["dtype"],
            verbose=param_proxop["verbose"],
        )

        # prepare uSARA optimiser
        optimiser = FBSARA(
            meas=data["y"],
            ch_freq=data["freqs"],
            meas_op=meas_op,
            regulariser=prox_op_sara,
            rank_ch_start=data["rank_eff_ch_start"],
            meas_op_approx=meas_op_approx,
            im_min_itr=param_optimiser["im_min_itr"],
            im_max_itr=param_optimiser["im_max_itr"],
            im_var_tol=param_optimiser["im_var_tol"],
            heu_reg_scale=param_optimiser["heu_reg_param_scale"],
            im_max_itr_outer=param_optimiser["im_max_outer_itr"],
            im_var_tol_outer=param_optimiser["im_var_outer_tol"],
            save_pth=param_optimiser["result_path"],
            file_prefix=param_optimiser["file_prefix"],
            reweight_save=param_optimiser["reweighting_save"],
            verbose=param_optimiser["verbose"],
        )

    elif param_optimiser["algorithm"] == "hyper-usara":
        return NotImplemented

    elif param_optimiser["algorithm"] == "airi":
        # prepare AIRI denoiser
        denoiser_airi = DenAIRI(
            param_proxop["dnn_shelf_path"],
            rand_trans=param_proxop["dnn_apply_transform"],
            rank_num_ch=len(data["y"]),
            rank_ch_start=data["rank_eff_ch_start"],
            device=param_proxop["device"],
            dtype=param_proxop["dtype"],
            verbose=param_proxop["verbose"],
        )

        # prepare AIRI optimiser
        optimiser = FBAIRI(
            meas=data["y"],
            ch_freq=data["freqs"],
            meas_op=meas_op,
            regulariser=denoiser_airi,
            meas_op_approx=meas_op_approx,
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

    elif param_optimiser["algorithm"] == "cairi":
        # prepare AIRI denoiser
        denoiser_airi = DenAIRI(
            param_proxop["dnn_shelf_path"],
            rand_trans=param_proxop["dnn_apply_transform"],
            device=param_proxop["device"],
            dtype=param_proxop["dtype"],
            verbose=param_proxop["verbose"],
        )

        prox_op_dual_data = ProxEllipsoid(
            center=data["y"],
            precond_weight=data["precond_weight"],
            rank_num_ch=len(data["y"]),
            rank_ch_start=data["rank_eff_ch_start"],
            device=meas_op.get_device(),
            dtype=meas_op.get_data_type_meas(),
        )

        optimiser = PDAIRI(
            meas=data["y"],
            ch_freq=data["freqs"],
            meas_op=meas_op,
            regulariser_prime=denoiser_airi,
            regulariser_dual_data=prox_op_dual_data,
            rank_ch_start=data["rank_eff_ch_start"],
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

    elif param_optimiser["algorithm"] == "hyperairi":
        # prepare prior
        prox_op_hyperairi = DenHyperAIRI(
            img_size=param_optimiser["img_size"],
            rank_num_ch=len(data["y"]),
            rank_ch_start=data["rank_eff_ch_start"],
            ch_freq=data["freqs"],
            shelf_path=param_proxop["dnn_shelf_path"],
            rand_trans=param_proxop["dnn_apply_transform"],
            device=param_proxop["device"],
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

    else:
        raise NotImplementedError(
            f"Unsupported algorithm: {param_optimiser['algorithm']}"
        )

    # imaging
    if param_optimiser["flag_imaging"]:

        # initialisation
        optimiser.initialisation()

        # run imaging loop
        optimiser.run()

        # finalisation
        optimiser.finalisation()

        dist.barrier()
        if param_optimiser["verbose"] and torch.cuda.is_available():
            print(
                f"INFO: rank {param_comm['rank']}: "
                f"The maximum GPU memory allocated {torch.cuda.memory_allocated()}",
                flush=True,
            )

    # clsoing down torch.distributed
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if use_distributed:
        dist.barrier()
        dist.destroy_process_group()
