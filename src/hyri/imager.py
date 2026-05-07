"""Configure and run imaging workflows.

This module assembles the measurement operator, regulariser, and optimiser
from user-provided parameter dictionaries, then optionally runs the imaging
loop.
"""

import torch
import torch.distributed as dist

from .interface.io import env_int
from .interface.load_data import load_data_to_tensor
from .measurement_operator import (
    MeasOpFinufft,
    MeasOpHyperspectral,
    MeasOpPSF,
    MeasOpTkbn,
)
from .optimiser import FBAIRI, FBSARA, PDAIRI, FBHyperAIRI, FBHyperSARA
from .regulariser import (
    DenAIRI,
    DenHyperAIRI,
    ProxEllipsoid,
    ProxHyperSARAPos,
    ProxSARAPos,
)


def imager(param_optimiser: dict, param_measop: dict, param_proxop: dict) -> None:
    """Prepare the imaging components and optionally run reconstruction.

    This function loads the input data, instantiates the requested
    hyperspectral measurement operator, builds the corresponding regulariser
    and optimiser, and executes the full imaging loop when requested.

    Args:
        param_optimiser (Dict): Optimiser configuration dictionary, including
            the algorithm choice, iteration controls, output paths, and
            distributed-imaging options.
        param_measop (Dict): Measurement-operator configuration dictionary,
            including Fourier backend, weighting options, image geometry, and
            numerical settings.
        param_proxop (Dict): Regulariser configuration dictionary, including
            denoiser shelf paths, devices, dtypes, and verbosity settings.

    Raises:
        ValueError: If the number of distributed processes exceeds the number
            of effective channels or if the requested NUFFT backend is not
            supported.
        NotImplementedError: If the requested imaging algorithm is not
            implemented.
    """

    # Set PyTorch flags for deterministic behaviour and runtime performance.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = True

    # Collect distributed-runtime metadata from environment variables.
    param_comm = {
        "rank": env_int("RANK", 0),
        "world_size": env_int("WORLD_SIZE", 1),
        "local_rank": env_int("LOCAL_RANK", 0),
        "local_world_size": env_int("LOCAL_WORLD_SIZE", 1),
        "node_rank": env_int("NODE_RANK", 0),
        "rank_device_id": (
            env_int("RANK_DEVICE_ID", 0) if torch.cuda.is_available() else None
        ),
    }
    use_distributed = param_comm["world_size"] > 1

    # Ensure that channel partitioning is compatible with the process count.
    if param_comm["world_size"] > param_optimiser["ch_effective"]:
        raise ValueError(
            f"Number of processes ({param_comm['world_size']}) should not exceed"
            f"the number of effective channels ({param_optimiser['ch_effective']})."
        )

    # Initialise torch.distributed only when multi-process imaging is requested.
    if use_distributed:
        dist.init_process_group(
            backend=param_optimiser["dist_backend"],
            init_method=param_optimiser["dist_init_method"],
            rank=param_comm["rank"],
            world_size=param_comm["world_size"],
            device_id=param_comm["rank_device_id"],
        )

    # Print a short distributed-runtime summary from rank 0.
    if use_distributed:
        dist.barrier()
        if param_comm["rank"] == 0:
            print(
                f"INFO: torch.distributed initialized with backend "
                f"{dist.get_backend()}",
                flush=True,
            )
            print(
                f"INFO: num of processes = {param_comm['world_size']}, "
                f"num of nodes = "
                f"{param_comm['world_size'] // param_comm['local_world_size']}",
                flush=True,
            )
        dist.barrier()

    # Load visibilities, weights, frequencies, and channel partition metadata.
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
        precond_weighting=param_optimiser["precond_flag"],
        dtype=param_measop["dtype"],
        device=torch.device("cpu"),
        verbose=param_optimiser["verbose"],
        ch_start=param_optimiser["ch_start"],
        ch_last=param_optimiser["ch_last"],
        ch_effctive=param_optimiser["ch_effective"],
        ch_partition=param_optimiser["ch_partition"],
        invert_uvw_flag=param_measop["invert_uvw_flag"],
    )

    # Select the requested single-channel NUFFT backend.
    MeasOpClass = None
    if param_measop["nufft_package"] == "finufft":
        MeasOpClass = MeasOpFinufft
    elif param_measop["nufft_package"] == "tkbn":
        MeasOpClass = MeasOpTkbn
    else:
        raise ValueError(
            f"Unsupported measurement operator type: {param_measop['nufft_package']}"
        )

    # Build the precise hyperspectral measurement operator used for imaging.
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
        # Optionally build a PSF-based surrogate for faster iterative updates.
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

    # Instantiate the regulariser and optimiser requested by the algorithm name.
    if param_optimiser["algorithm"] == "usara":
        # Build the uSARA positivity-constrained proximal operator.
        prox_op_sara = ProxSARAPos(
            param_optimiser["img_size"],
            rank_num_ch=len(data["y"]),
            rank_ch_start=data["rank_eff_ch_start"],
            device=param_proxop["device"],
            dtype=param_proxop["dtype"],
            verbose=param_proxop["verbose"],
        )

        # Build the uSARA optimiser.
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
        # Build the Hyper-uSARA positivity-constrained proximal operator.
        prox_op_sara = ProxHyperSARAPos(
            param_optimiser["img_size"],
            rank_num_ch=len(data["y"]),
            rank_ch_start=data["rank_eff_ch_start"],
            device=param_proxop["device"],
            dtype=param_proxop["dtype"],
            verbose=param_proxop["verbose"],
        )

        # Build the Hyper-uSARA optimiser.
        optimiser = FBHyperSARA(
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

    elif param_optimiser["algorithm"] == "airi":
        # Build the AIRI denoiser.
        denoiser_airi = DenAIRI(
            param_proxop["dnn_shelf_path"],
            rand_trans=param_proxop["dnn_apply_transform"],
            rank_num_ch=len(data["y"]),
            rank_ch_start=data["rank_eff_ch_start"],
            device=param_proxop["device"],
            dtype=param_proxop["dtype"],
            verbose=param_proxop["verbose"],
        )

        # Build the AIRI optimiser.
        optimiser = FBAIRI(
            meas=data["y"],
            ch_freq=data["freqs"],
            meas_op=meas_op,
            regulariser=denoiser_airi,
            rank_ch_start=data["rank_eff_ch_start"],
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
        # Build the AIRI denoiser for the primal update.
        denoiser_airi = DenAIRI(
            param_proxop["dnn_shelf_path"],
            rand_trans=param_proxop["dnn_apply_transform"],
            rank_num_ch=len(data["y"]),
            rank_ch_start=data["rank_eff_ch_start"],
            device=param_proxop["device"],
            dtype=param_proxop["dtype"],
            verbose=param_proxop["verbose"],
        )

        # Build the ellipsoidal dual-data prox from the measured visibilities.
        prox_op_dual_data = ProxEllipsoid(
            center=data["y"],
            precond_weight=data["precond_weight"],
            rank_num_ch=len(data["y"]),
            rank_ch_start=data["rank_eff_ch_start"],
            device=meas_op.get_device(),
            dtype=meas_op.get_data_type_meas(),
        )

        # Build the constrained AIRI optimiser.
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
        # Build the HyperAIRI denoiser.
        den_hyperairi = DenHyperAIRI(
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

        # Build the HyperAIRI optimiser.
        optimiser = FBHyperAIRI(
            meas=data["y"],
            ch_freq=data["freqs"],
            meas_op=meas_op,
            regulariser=den_hyperairi,
            meas_op_approx=meas_op_approx,
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
            spec_update=param_optimiser["spec_update"],
            spec_idx_init=param_optimiser["spec_idx_init"],
            spec_max_update_count=param_optimiser["spec_max_update_count"],
            spec_max_update_interval=param_optimiser["spec_max_update_interval"],
            spec_update_tol_max=param_optimiser["spec_update_tol_max"],
            spec_update_tol_min=param_optimiser["spec_update_tol_min"],
            spec_update_tol_step=param_optimiser["spec_update_tol_step"],
            save_pth=param_optimiser["result_path"],
            file_prefix=param_optimiser["file_prefix"],
            iter_save=param_optimiser["itr_save"],
            verbose=param_optimiser["verbose"],
        )

    else:
        raise NotImplementedError(
            f"Unsupported algorithm: {param_optimiser['algorithm']}"
        )

    # Run the imaging workflow only when explicitly requested.
    if param_optimiser["flag_imaging"]:
        # Initialise the optimiser state and auxiliary quantities.
        optimiser.initialisation()

        # Execute the main optimisation loop.
        optimiser.run()

        # Save final products and print the closing summary.
        optimiser.finalisation()

        # Print the maximum GPU memory allocated during execution.
        if param_optimiser["verbose"] and torch.cuda.is_available():
            if use_distributed:
                dist.barrier()
            mem_allocated = torch.cuda.memory_allocated() / 1024**3
            print(
                f"INFO: rank {param_comm['rank']}: "
                f"The maximum GPU memory allocated {mem_allocated:.4f} GB.",
                flush=True,
            )

    # Synchronise and close the distributed runtime cleanly.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if use_distributed:
        dist.barrier()
        dist.destroy_process_group()
