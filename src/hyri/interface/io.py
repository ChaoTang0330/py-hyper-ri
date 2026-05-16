"""Provide interface IO helpers for configuration and MAT-file loading.

This file implements environment/config parsing utilities used by the
imaging entry points, plus compatibility loading for MATLAB files stored in
either legacy formats or HDF5-based v7.3 format.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.io import loadmat
from scipy.io.matlab import matfile_version


def env_int(name: str, default: int) -> int:
    """Retrieve the value of an environment variable as an integer.

    Args:
        name (str): Environment variable name.
        default (int): Fallback value used when the variable is unset.

    Returns:
        int: Parsed integer value from the environment variable, or ``default`` when
        the variable does not exist.

    Raises:
        ValueError: If the environment variable exists but cannot be parsed as
            an integer.

    Examples:
        >>> import os
        >>> os.environ["RI_NUM_WORKERS"] = "4"
        >>> env_int("RI_NUM_WORKERS", 1)
        4
        >>> env_int("RI_UNSET_KEY", 1)
        1
    """
    v = os.environ.get(name)
    return int(v) if v is not None else default


def resolve_existing_file_path(
    path: str,
    library_root: str,
    relative_anchor: str | None = None,
) -> str:
    """Resolve a file path against common project roots and validate existence.

    The input path may be absolute, relative, or include environment variable
    and user-home expansions. Relative paths are resolved in priority order
    against the current working directory, an optional anchor, and the
    provided library root.

    Args:
        path (str): Raw file path.
        library_root (str): Package/library root used as a fallback search
            location for relative paths.
        relative_anchor (str, optional): Additional root used before
            ``library_root`` for relative resolution. Default: ``None``.

    Returns:
        str: Absolute resolved path to an existing file.

    Raises:
        FileNotFoundError: If no candidate path resolves to an existing file.

    Note:
        Relative paths are resolved in this order: current working directory,
        ``relative_anchor`` when provided, then ``library_root``.

    Examples:
        >>> resolve_existing_file_path("config/default.json", ".")
        '/abs/path/to/config/default.json'
    """
    # Normalise raw path text from config/CSV input.
    raw = os.path.expandvars(path.strip().strip('"').strip("'"))
    p = Path(raw).expanduser()

    attempts = []

    # Handle absolute paths directly.
    if p.is_absolute():
        rp = p.resolve(strict=False)
        attempts.append(rp)
        if rp.is_file():
            return str(rp)
    else:
        # Resolve relative paths against priority roots.
        candidate_roots = [Path.cwd()]
        if relative_anchor is not None:
            candidate_roots.append(Path(relative_anchor))
        candidate_roots.append(Path(library_root))

        seen = set()
        for root in candidate_roots:
            cand = (root / p).resolve(strict=False)
            key = str(cand)
            if key in seen:
                continue
            seen.add(key)
            attempts.append(cand)
            if cand.is_file():
                return key

    raise FileNotFoundError(
        "File not found: "
        + path
        + "; attempted: "
        + ", ".join(str(a) for a in attempts)
    )


def print_dict(
    curr_dict: dict[str, Any], indentation: int = 4, flush: bool = True
) -> None:
    """Print key-value pairs from a dictionary.

    Args:
        curr_dict (Dict[str, Any]): Dictionary whose items are printed.
        indentation (int, optional): Number of leading spaces per printed
            line. Default: 4.
        flush (bool, optional): Whether to flush stdout after each print.
            Default: True.
    """
    for k, v in curr_dict.items():
        print(" " * indentation + f"{k}: {v}", flush=flush)


def verbose_print(args: str, verbose: bool = True) -> None:
    """Print a message only when verbose mode is enabled.

    Args:
        args (str): Message text to print.
        verbose (bool, optional): If ``True``, print ``args`` to stdout.
            Default: True.

    Returns:
        None: This function does not return a value.
    """
    if verbose:
        print(args, flush=True)


def load_mat_to_np(file_name: str) -> dict[str, Any]:
    """Load MATLAB data into a Python dictionary.

    Files with MATLAB v7.3 format are read with :mod:`h5py`, while older
    formats are read with :func:`scipy.io.loadmat`. Complex-valued v7.3 HDF5
    datasets stored with ``real`` and ``imag`` fields are converted to complex
    NumPy arrays.

    Args:
        file_name (str): Path to the ``.mat`` file.

    Returns:
        Dict[str, Any]: Mapping from MATLAB variable names to loaded arrays or
        values.

    Raises:
        FileNotFoundError: If ``file_name`` does not exist.
        OSError: If the file cannot be opened or is not a valid MAT file.

    Note:
        MATLAB v7.3 files are HDF5 containers and are read dataset-by-dataset.
        For structured datasets with ``real`` and ``imag`` fields, this
        function reconstructs complex arrays as ``real + 1j * imag``.
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
    """Parse command-line arguments for the imaging entry point.

    Returns:
        argparse.Namespace: Parsed CLI arguments including configuration path,
        dataset selection, imaging dimensions, algorithm choice, and
        channel-selection options.

    Raises:
        SystemExit: If command-line parsing fails or ``--help`` is requested.

    Note:
        This function parses values from :data:`sys.argv` using
        :class:`argparse.ArgumentParser`.

    Examples:
        Run with a custom config file::

            python run_imager.py --config ./config/example.json

        Override algorithm and image size from the command line::

            python run_imager.py --algorithm hyperairi --im_dim_x 512 --im_dim_y 512
    """
    parser = argparse.ArgumentParser(description="Imaging main parameters.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./config/default.json",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--src_name", type=str, default=None, help="Source name identifier."
    )
    parser.add_argument(
        "--data_file", type=str, default=None, help="Path to the input data file."
    )
    parser.add_argument(
        "--result_path", type=str, default=None, help="Path to the result directory."
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        choices=["airi", "cairi", "usara", "hyperairi", "hyper-usara"],
        help="Imaging algorithm.",
    )
    parser.add_argument(
        "--im_dim_x",
        type=int,
        default=None,
        help="Reconstructed image width in pixels.",
    )
    parser.add_argument(
        "--im_dim_y",
        type=int,
        default=None,
        help="Reconstructed image height in pixels.",
    )
    parser.add_argument(
        "--dnn_shelf_path",
        type=str,
        default=None,
        help="Path to the denoiser shelf configuration file.",
    )
    parser.add_argument(
        "--im_pixel_size",
        type=float,
        default=None,
        help="Estimated image pixel size in arcsec.",
    )
    parser.add_argument(
        "--superresolution",
        type=float,
        default=None,
        help="Imaging super-resolution factor.",
    )
    parser.add_argument(
        "--ch_start",
        type=int,
        default=None,
        help="Starting physical channel index in the input data.",
    )
    parser.add_argument(
        "--ch_last",
        type=int,
        default=None,
        help="Last physical channel index in the input data.",
    )
    parser.add_argument(
        "--ch_effective",
        type=int,
        default=None,
        help="Number of effective channels in the reconstructed image cube.",
    )
    parser.add_argument(
        "--ch_partition",
        type=int,
        nargs="+",
        default=None,
        help="Channel-index list used for data partitioning.",
    )
    parser.add_argument(
        "--run_id",
        type=int,
        default=None,
        help="Identifier for the current run.",
    )
    return parser.parse_args()


def parsing_parameters(
    config_file: str, input_param: argparse.Namespace | None = None
) -> dict[str, Any]:
    """Merge configuration-file parameters with CLI overrides.

    The returned dictionary starts from the ``main`` config section, applies
    non-null CLI overrides for matching keys, then extends algorithm-specific
    and shared parameter groups.

    Args:
        config_file (str): Path to the JSON configuration file.
        input_param (argparse.Namespace, optional): Parsed CLI arguments used
            as overrides when values are not ``None``. Default: ``None``.

    Returns:
        Dict[str, Any]: Flattened runtime parameter dictionary used by the
        imaging pipeline.

    Raises:
        NotImplementedError: If the selected algorithm is not supported by the
        parameter-routing logic.

    Note:
        Command-line inputs have the highest priority. Algorithm-specific
        parameters are expected not to overlap with keys from the ``main``
        section or command-line inputs.
    """
    with open(config_file, encoding="utf-8") as file:
        json_data = json.load(file)

    json_dict = {}
    for json_data_sub in json_data:
        for k, v in json_data_sub.items():
            if k != "__comment":
                json_dict[k] = v

    # Start from the main config section.
    param = json_dict["main"]
    if input_param is not None:
        # Apply non-null CLI overrides for matching keys.
        dict_input_param = vars(input_param)
        for k, v in dict_input_param.items():
            if k in param and v is not None:
                param[k] = v

    # Merge shared and algorithm-specific parameter sections.
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

    # Override algorithm-specific parameters with CLI inputs when provided.
    if input_param is not None:
        dict_input_param = vars(input_param)
        for k, v in dict_input_param.items():
            if k in param and v is not None:
                param[k] = v

    return param
