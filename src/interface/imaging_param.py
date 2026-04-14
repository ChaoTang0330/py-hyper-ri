"""
Generate parameter dictionaries for measuemt operator, prox operator and optimiser
from input parameter dictionary.
"""

from typing import Any, Dict, Tuple
import os
import torch
import torch.distributed as dist


def set_imaging_params_ri(
    param_general: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Set parameters of measuemt operator, prox operator and optimiser for
    small scale RI imaging task.

    Args:
        param_general (dict): A dictionary containing general parameters.

    Returns:
        tuple: A tuple containing three dictionaries:
            - param_measop (dict): A dictionary containing parameters for the measurement operator.
            - param_proxop (dict): A dictionary containing parameters for the prox operator.
            - param_optimiser (dict): A dictionary containing parameters for the optimiser.
    """
    # initialisation
    param_measop = {}
    param_proxop = {}
    param_optimiser = {}

    # parameters shared by all algorithms
    param_optimiser["algorithm"] = param_general["algorithm"]
    param_optimiser["data_file"] = param_general["data_file"]
    # set default values
    param_measop["invert_uvw_flag"] = param_general.get("invert_uvw_flag", False)
    param_optimiser["flag_imaging"] = param_general.get("flag_imaging", True)
    param_proxop["verbose"] = param_general.get("verbose", True)
    param_optimiser["verbose"] = param_general.get("verbose", True)

    # image size
    if (
        param_general.get("im_dim_x", None)
        and param_general["im_dim_y"] >= 1
        and param_general.get("im_dim_y", None)
        and param_general["im_dim_y"] >= 1
    ):
        param_optimiser["img_size"] = (
            int(param_general["im_dim_x"]),
            int(param_general["im_dim_y"]),
        )
    else:
        param_optimiser["img_size"] = (512, 512)

    # channels
    param_optimiser["ch_start"] = int(param_general.get("ch_start", 1))
    if param_optimiser["ch_start"] < 1:
        param_optimiser["ch_start"] = 1
    param_optimiser["ch_last"] = int(param_general.get("ch_last", 0))
    if param_optimiser["ch_last"] < param_optimiser["ch_start"]:
        param_optimiser["ch_last"] = None
    param_optimiser["ch_effective"] = int(param_general.get("ch_effective", 1))
    if param_optimiser["ch_effective"] < 1:
        param_optimiser["ch_effective"] = 1
    param_optimiser["ch_partition"] = param_general.get("ch_partition", [])
    param_optimiser["ch_partition"] = tuple(
        int(x) for x in param_optimiser["ch_partition"]
    )
    ch_partition_valid = all(int(x) > 0 for x in param_optimiser["ch_partition"])
    if ch_partition_valid:
        if len(param_optimiser["ch_partition"]) > 0:
            # overwrite ch_effective
            param_optimiser["ch_effective"] = len(param_optimiser["ch_partition"])
    else:
        param_optimiser["ch_partition"] = []

    # image pixel size
    if (
        param_general.get("im_pixel_size", None)
        and param_general["im_pixel_size"] > 0.0
    ):
        param_measop["im_pixel_size"] = param_general["im_pixel_size"]
    else:
        param_measop["im_pixel_size"] = None
    # super-resolution factor
    if (
        param_general.get("superresolution", None)
        and param_general["superresolution"] >= 1.0
    ):
        param_measop["superresolution"] = float(param_general["superresolution"])
    else:
        param_measop["superresolution"] = 1.0

    # weighting
    param_measop["flag_data_weighting"] = param_general.get("flag_data_weighting", True)
    param_measop["weight_load"] = param_general.get("weight_load", True)
    if param_general.get("weight_type", None) and param_general["weight_type"] in [
        "briggs",
        "uniform",
    ]:
        param_measop["weight_type"] = param_general["weight_type"]
    else:
        param_measop["weight_type"] = "briggs"
    if param_general.get("weight_robustness", None):
        param_measop["weight_robustness"] = float(param_general["weight_robustness"])
    else:
        param_measop["weight_robustness"] = 0.0
    if (
        param_general.get("weight_gridsize", None)
        and param_general.get("weight_gridsize", None) > 0.0
    ):
        param_measop["weight_gridsize"] = float(param_general["weight_gridsize"])
    else:
        param_measop["weight_gridsize"] = 2

    # nufft settings
    # package
    if param_general.get("nufft_package", None) and param_general["nufft_package"] in [
        "finufft",
        "pynufft",
        "tkbnufft",
    ]:
        param_measop["nufft_package"] = param_general["nufft_package"]
    else:
        param_measop["nufft_package"] = "finufft"

    # oversampling factor
    if (
        param_general.get("nufft_oversampling_factor", None)
        and param_general["nufft_oversampling_factor"][0] >= 1.0
        and param_general["nufft_oversampling_factor"][1] >= 1.0
    ):
        param_measop["nufft_grid_size"] = (
            int(
                param_general["nufft_oversampling_factor"][0]
                * param_optimiser["img_size"][0]
            ),
            int(
                param_general["nufft_oversampling_factor"][1]
                * param_optimiser["img_size"][1]
            ),
        )
    else:
        param_measop["nufft_grid_size"] = (
            int(2.0 * param_optimiser["img_size"][0]),
            int(2.0 * param_optimiser["img_size"][1]),
        )
    # KB kernel dimension
    if (
        param_general.get("nufft_kb_kernel_dim", None)
        and param_general["nufft_kb_kernel_dim"] >= 1.0
    ):
        param_measop["nufft_kb_kernel_dim"] = int(param_general["nufft_kb_kernel_dim"])
    else:
        param_measop["nufft_kb_kernel_dim"] = 7
    # nufft mode
    if param_general.get("nufft_mode", None) and param_general["nufft_mode"] in [
        "table",
        "matrix",
    ]:
        param_measop["nufft_mode"] = param_general["nufft_mode"]
    else:
        param_measop["nufft_mode"] = "table"

    # computing resources
    # devices
    list_devices = []
    if torch.cuda.is_available():
        list_devices.append("cuda")
    if torch.backends.mps.is_available():
        list_devices.append("mps")
    if torch.xpu.is_available():
        list_devices.append("xpu")
    list_devices.append("cpu")
    measop_device_type = param_general.get("meas_device", None)
    regulariser_device_type = param_general.get("regulariser_device", None)
    if measop_device_type in list_devices:
        param_measop["device"] = torch.device(measop_device_type)
    else:
        param_measop["device"] = torch.device("cpu")
    if regulariser_device_type in list_devices:
        param_proxop["device"] = torch.device(regulariser_device_type)
    else:
        param_proxop["device"] = torch.device(list_devices[0])
    # distributed settings
    param_optimiser["dist_backend"] = param_general.get("dist_backend", "gloo")
    if not dist.is_backend_available(param_optimiser["dist_backend"]):
        param_optimiser["dist_backend"] = "gloo"
    param_optimiser["dist_init_method"] = param_general.get(
        "dist_init_method", "env://"
    )
    if param_optimiser["dist_init_method"].split("://")[0] not in [
        "env",
        "file",
        "tcp",
    ]:
        param_optimiser["dist_init_method"] = "env://"

    # max number of iterations
    if param_general.get("im_max_itr", None) and param_general["im_max_itr"] > 0:
        param_optimiser["im_max_itr"] = int(param_general["im_max_itr"])
    else:
        param_optimiser["im_max_itr"] = 2000
    # min number of iterations
    if param_general.get("im_min_itr", None) and param_general["im_min_itr"] > 0:
        param_optimiser["im_min_itr"] = int(param_general["im_min_itr"])
    else:
        param_optimiser["im_min_itr"] = 200
    # image variation tolerance
    if param_general.get("im_var_tol", None) and param_general["im_var_tol"] > 0:
        param_optimiser["im_var_tol"] = float(param_general["im_var_tol"])
    else:
        param_optimiser["im_var_tol"] = 1e-5

    # data type
    if param_general.get("meas_dtype", None) and param_general.get("meas_dtype") in [
        "float",
        "float32",
        "single",
    ]:
        param_measop["dtype"] = torch.float
    else:
        param_measop["dtype"] = torch.double
    if param_general.get("regulariser_dtype", None) and param_general.get("regulariser_dtype") in [
        "double",
        "float64",
    ]:
        param_proxop["dtype"] = torch.double
    else:
        param_proxop["dtype"] = torch.float

    # interval for saveing intermediate results
    if param_general.get("itr_save", None) and param_general["itr_save"] >= 1:
        param_optimiser["itr_save"] = int(param_general["itr_save"])
    else:
        param_optimiser["itr_save"] = param_optimiser["im_max_itr"] + 1

    # using approximate measurement operator
    param_optimiser["approx_meas_op"] = False

    # parameters shared by AIRI algorithms
    if param_optimiser["algorithm"] in ["airi", "cairi", "hyperairi"]:
        # heuristic noise scale
        if (
            param_general.get("heu_noise_scale", None)
            and param_general["heu_noise_scale"] > 0
        ):
            param_optimiser["heu_noise_scale"] = float(param_general["heu_noise_scale"])
        else:
            param_optimiser["heu_noise_scale"] = 1.0

        # AIRI shelf path
        if param_general.get("dnn_shelf_path", None) and isinstance(
            param_general["dnn_shelf_path"], str
        ):
            param_proxop["dnn_shelf_path"] = param_general["dnn_shelf_path"]

        # Adaptive network selection scheme
        # estimated image peak value
        if param_general.get("im_peak_est", None) and param_general["im_peak_est"] > 0:
            param_optimiser["im_peak_est"] = float(param_general["im_peak_est"])
        else:
            param_optimiser["im_peak_est"] = None
        param_optimiser["dnn_adaptive_peak"] = param_general.get(
            "dnn_adaptive_peak", True
        )
        if (
            param_general.get("dnn_adaptive_peak_tol_max", None)
            and param_general["dnn_adaptive_peak_tol_max"] > 0
        ):
            param_optimiser["dnn_adaptive_peak_tol_max"] = float(
                param_general["dnn_adaptive_peak_tol_max"]
            )
        else:
            param_optimiser["dnn_adaptive_peak_tol_max"] = 0.1
        if (
            param_general.get("dnn_adaptive_peak_tol_min", None)
            and param_general["dnn_adaptive_peak_tol_min"] > 0
        ):
            param_optimiser["dnn_adaptive_peak_tol_min"] = float(
                param_general["dnn_adaptive_peak_tol_min"]
            )
        else:
            param_optimiser["dnn_adaptive_peak_tol_min"] = 1e-3
        if (
            param_general.get("dnn_adaptive_peak_tol_step", None)
            and param_general["dnn_adaptive_peak_tol_step"] > 0
        ):
            param_optimiser["dnn_adaptive_peak_tol_step"] = float(
                param_general["dnn_adaptive_peak_tol_step"]
            )
        else:
            param_optimiser["dnn_adaptive_peak_tol_step"] = 0.1

        # random input image flip & 90-degree rotation
        param_proxop["dnn_apply_transform"] = param_general.get(
            "dnn_apply_transform", True
        )

        # specific parameters for AIRI
        if param_optimiser["algorithm"] == "airi":
            param_optimiser["approx_meas_op"] = param_general.get(
                "approx_meas_op", False
            )

    # parameters shared by SARA algorithms
    elif param_optimiser["algorithm"] in ["usara"]:
        # heuristic regularisation parameter scale
        if (
            param_general.get("heu_reg_param_scale", None)
            and param_general["heu_reg_param_scale"] > 0
        ):
            param_optimiser["heu_reg_param_scale"] = float(
                param_general["heu_reg_param_scale"]
            )
        else:
            param_optimiser["heu_reg_param_scale"] = 1.0

        param_optimiser["reweighting_save"] = param_general.get(
            "reweighting_save", False
        )
        if (
            param_general.get("im_max_outer_itr", None)
            and param_general["im_max_outer_itr"] >= 1
        ):
            param_optimiser["im_max_outer_itr"] = int(param_general["im_max_outer_itr"])
        else:
            param_optimiser["im_max_outer_itr"] = 20
        if (
            param_general.get("im_var_outer_tol", None)
            and param_general["im_var_outer_tol"] > 0
        ):
            param_optimiser["im_var_outer_tol"] = float(
                param_general["im_var_outer_tol"]
            )
        else:
            param_optimiser["im_var_outer_tol"] = 1e-3

        # specific parameters for uSARA
        if param_optimiser["algorithm"] == "usara":
            param_optimiser["approx_meas_op"] = param_general.get(
                "approx_meas_op", False
            )

    # parameters shared by constrained algorithms
    if param_optimiser["algorithm"] in ["cairi"]:
        param_optimiser["precond_flag"] = param_general.get("precond_flag", True)

    # set path for saving results
    if not param_general.get("src_name", None):
        param_general["src_name"] = os.path.basename(param_general["data_file"])
    if not param_general.get("result_path", None):
        param_general["result_path"] = os.path.join(".", "results")
    param_optimiser["result_path"] = os.path.join(
        param_general["result_path"], param_general["src_name"]
    )
    os.makedirs(param_optimiser["result_path"], exist_ok=True)

    file_prefix = ""
    if param_optimiser["algorithm"] == "airi":
        file_prefix = "AIRI_heuScale_" + str(param_optimiser["heu_noise_scale"]) + "_"
    elif param_optimiser["algorithm"] == "cairi":
        file_prefix = "cAIRI_heuScale_" + str(param_optimiser["heu_noise_scale"]) + "_"
    elif param_optimiser["algorithm"] == "usara":
        file_prefix = (
            "uSARA_heuRegScale_" + str(param_optimiser["heu_reg_param_scale"]) + "_"
        )
    if param_optimiser["algorithm"] == "hyperairi":
        file_prefix = (
            "HyperAIRI_heuScale_" + str(param_optimiser["heu_noise_scale"]) + "_"
        )
    if param_general.get("run_id", None):
        file_prefix += "runID_" + str(param_general["run_id"]) + "_"
    param_optimiser["file_prefix"] = file_prefix

    return (
        dict(sorted(param_measop.items())),
        dict(sorted(param_proxop.items())),
        dict(sorted(param_optimiser.items())),
    )
