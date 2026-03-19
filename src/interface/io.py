"""
Utility functions for input/output operations.
"""

import os
import argparse
import json
import h5py
from typing import Any, Dict, Optional

import numpy as np
from scipy.io import loadmat
from scipy.io.matlab import matfile_version

def env_int(name: str, default: int) -> int:
    """
    Retrieve an integer value from the environment variable.

    Args:
        name (str): The name of the environment variable.
        default (int): The default value to return if the environment variable is not set.
    Returns:
        int: The integer value of the environment variable or the default value.
    """
    v = os.environ.get(name)
    return int(v) if v is not None else default


def print_dict(curr_dict: Dict[str, Any], indentation: int = 4, flush: bool = True) -> None:
    """
    Print items in a dictionary.

    Args:
        curr_dict (Dict[str, Any]): Dictionary to print.
        indentation (int, optional): The number of spaces to indent each line. Defaults to 4.
        flush (bool, optional): Whether to flush the output buffer. Defaults to True.
    """
    for k, v in curr_dict.items():
        print(' ' * indentation + f"{k}: {v}", flush=flush)

def verbose_print(*args: str, verbose: bool = True):
    """
    Print messages if verbose is True.

    Args:
        args (str): Variable length argument list of strings to print.
        verbose (bool, optional): Flag to indicate whether to print the messages. Defaults to True.

    """
    if verbose:
        print(*args, flush=True)


def load_mat_to_np(file_name: str) -> Dict[str, Any]:
    """
    Check mat file version and load data. If the mat file has the latest version (v7.3),
    it will be loaded using h5py. Otherwise, it will be loaded using scipy.io.loadmat.

    Args:
        file_name (str): Path of the .mat file to load.
    Returns:
        dict: A dictionary containing the data loaded from the .mat file.
    """
    data = {}
    mat_version, _ = matfile_version(file_name)
    if mat_version == 2:
        data = dict()
        with h5py.File(file_name, "r") as h5_file:
            for key, h5obj in h5_file.items():
                if isinstance(h5obj, h5py.Dataset):
                    data[key] = np.array(h5obj)
                    if data[key].dtype.names and "imag" in data[key].dtype.names:
                        data[key] = data[key]["real"] + 1j * data[key]["imag"]
    else:
        data = loadmat(file_name)

    return data

def parsing_arguments() -> argparse.Namespace:
    """
    Parses input arguments from the command line.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="imaging main parameters")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./config/default.json",
        help="path of the configuration file",
    )
    parser.add_argument(
        "--src_name", type=str, default=None, help="Initial reg parameter"
    )
    parser.add_argument(
        "--data_file", type=str, default=None, help="path of the datafile"
    )
    parser.add_argument(
        "--result_path", type=str, default=None, help="path of the result folder"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        choices=["airi", "cairi", "usara"],
        help="imaging algorithm",
    )
    parser.add_argument(
        "--im_dim_x", type=int, default=None, help="result horizental image size"
    )
    parser.add_argument(
        "--im_dim_y", type=int, default=None, help="result vertical image size"
    )
    parser.add_argument(
        "--dnn_shelf_path",
        type=str,
        default=None,
        help="path of the denoiser shelf configuration file",
    )
    parser.add_argument(
        "--im_pixel_size",
        type=float,
        default=None,
        help="pixel size of the estimated image in the unit of arcsec",
    )
    parser.add_argument(
        "--superresolution",
        type=float,
        default=None,
        help="imaging super-resolution factor",
    )
    parser.add_argument(
        "--ch_start",
        type=int,
        default=None,
        help="starting physical channel index of the data",
    )
    parser.add_argument(
        "--ch_last",
        type=int,
        default=None,
        help="last physical channel index of the data",
    )
    parser.add_argument(
        "--ch_effective",
        type=int,
        default=None,
        help="number of effective channels of the result image cube",
    )
    parser.add_argument(
        "--ch_partition",
        type=int,
        nargs="+",
        default=None,
        help="list of channel indices for partitioning the data",
    )
    parser.add_argument(
        "--run_id",
        type=int,
        default=None,
        help="identification number of the current task",
    )
    return parser.parse_args()


def parsing_parameters(
    config_file: str, input_param: Optional[argparse.Namespace] = None
) -> Dict[str, Any]:
    """
    Mixes input parameters from the command line and config file.

    Args:
        config_file (str): Path of the configuration file.
        input_param (argparse.Namespace, optional): Parsed command line arguments.

    Returns:
        Dict[str, Any]: Dictionary of parameters.
    """
    with open(config_file, "r", encoding="utf-8") as file:
        json_data = json.load(file)

    json_dict = {}
    for json_data_sub in json_data:
        for k, v in json_data_sub.items():
            if k != "__comment":
                json_dict[k] = v

    # load main parameters
    param = json_dict["main"]
    if input_param is not None:
        dict_input_param = vars(input_param)
        for k, v in dict_input_param.items():
            if k in param and v is not None:
                param[k] = v

    # load general parameters
    list_keys = ["flag", "weighting", "computing", "nufft"]
    if param["algorithm"] == "airi":
        list_keys.extend(["airi", "airi_default"])
    elif param["algorithm"] == "cairi":
        list_keys.extend(["cairi", "cairi_default"])
    elif param["algorithm"] == "usara":
        list_keys.extend(["usara", "usara_default"])
    elif param["algorithm"] == "hyperairi":
        list_keys.extend(["hyperairi", "hyperairi_default"])
    elif param["algorithm"] == "hyper-usara":
        list_keys.extend(["hyper-usara", "hyper-usara_default"])
    else:
        raise NotImplementedError(f"Algorithm {param['algorithm']} not found\n")
    for key in list_keys:
        for k, v in json_dict[key].items():
            if "_comment_" not in k:
                param[k] = v
    if input_param is not None and input_param.dnn_shelf_path is not None:
        param["dnn_shelf_path"] = input_param.dnn_shelf_path

    return param