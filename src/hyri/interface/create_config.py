"""Create HYRI imaging configuration files.

The ``hyri-config`` command starts an interactive wizard by default.  Users can
also pass ``--default`` to copy the default JSON template without prompts.
"""

import argparse
import copy
import json
from collections.abc import Callable
from importlib import resources
from pathlib import Path
from typing import Any

EMPTY_VALUE: list[Any] = []
ConfigUpdate = tuple[str, str, Any]
ALGORITHMS = ["usara", "hyper-usara", "airi", "cairi", "hyperairi"]
SPEC_INDEX_INIT_MODES = ["zero", "dirty"]
DEVICES = ["cpu", "cuda", "mps", "xpu"]
DTYPES = ["single", "double", "float", "float32", "float64"]
DIST_BACKENDS = ["gloo", "nccl", "mpi", "xccl"]
NUFFT_PACKAGES = ["finufft", "tkbnufft"]
NUFFT_MODES = ["table", "matrix"]
WEIGHT_TYPES = ["briggs", "uniform"]
SARA_ALGORITHMS = {"usara", "hyper-usara"}
AIRI_ALGORITHMS = {"airi", "cairi", "hyperairi"}
CONSTRAINED_ALGORITHMS = {"cairi"}
UNCONSTRAINED_ALGORITHMS = {"usara", "hyper-usara", "airi", "hyperairi"}


def create_config(argv: list[str] | None = None) -> None:
    """Create a HYRI imaging configuration file.

    ``--default`` copies the default template.  Without ``--default``, this
    function starts an interactive wizard and reprompts until values are valid.

    Args:
        argv (list[str] | None, optional): Command-line arguments to parse.
            Default: ``None``. When ``None``, :mod:`argparse` reads from
            ``sys.argv``.

    Raises:
        RuntimeError: If interactive input ends before the wizard is complete.
        SystemExit: If the output file already exists and overwrite is disabled.

    Notes:
        Invalid interactive inputs should be handled by asking the same
        question again.  The wizard should not collect all answers and then
        fail with a validation error.

        Output overwrite safety is checked before the default copy or long
        interactive question flow starts.

    Examples:
        >>> create_config(["--default", "--output", "hyri_config.json"])
    """
    args = parsing_config_arguments(argv)
    output_path = args.output or ("default.json" if args.default else None)

    if output_path is None:
        output_path = ask_until_valid(
            "Output configuration file path",
            "hyri_config.json",
            parse_non_empty_str,
            "JSON file to create.",
        )
    prepare_output_path(output_path, overwrite=args.overwrite)

    config = load_default_config()

    if args.default:
        write_config(config, output_path)
        print_config_created(output_path, config)
        return

    answers = ask_config_questions(config)
    config = apply_config_answers(config, answers)
    write_config(config, output_path)
    print_config_created(output_path, config)


def parsing_config_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse arguments for the ``hyri-config`` command.

    Args:
        argv (list[str] | None, optional): Command-line arguments to parse.
            Default: ``None``. When ``None``, :mod:`argparse` reads from
            ``sys.argv``.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Create a HYRI imaging JSON configuration file."
    )
    parser.add_argument(
        "--default",
        action="store_true",
        help="Copy the default configuration without entering interactive mode.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output JSON config path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args(argv)


def load_default_config() -> list[dict[str, Any]]:
    """Load the JSON template used by the config creator.

    The template is bundled as package data at ``hyri/config/default.json``.

    Returns:
        list[dict[str, Any]]: Parsed default configuration template.
    """
    template = resources.files("hyri").joinpath("config/default.json")
    with template.open("r", encoding="utf-8") as file:
        return json.load(file)


def prepare_output_path(output_path: str, *, overwrite: bool = False) -> None:
    """Create the output parent directory and protect existing files.

    Args:
        output_path (str): JSON configuration path to prepare.

    Keyword args:
        overwrite (bool, optional): Whether an existing file may be replaced.
            Default: ``False``.

    Raises:
        SystemExit: If ``output_path`` exists and ``overwrite`` is ``False``.
    """
    output = Path(output_path).expanduser()
    if output.exists() and not overwrite:
        raise SystemExit(f"{output} already exists. Use --overwrite to replace it.")
    output.parent.mkdir(parents=True, exist_ok=True)


def write_config(
    config: list[dict[str, Any]],
    output_path: str,
) -> None:
    """Write a HYRI JSON configuration file.

    Args:
        config (list[dict[str, Any]]): Configuration data to serialise.
        output_path (str): JSON file path to write.
    """
    output = Path(output_path).expanduser()

    with output.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=4)
        file.write("\n")


def ask_config_questions(
    config: list[dict[str, Any]],
) -> dict[str, Any]:
    """Ask the interactive wizard questions and return configuration updates.

    Args:
        config (list[dict[str, Any]]): Default configuration template used for
            current values and section names.

    Returns:
        dict[str, Any]: Wizard answers with an ``"updates"`` list containing
            ``(section, key, value)`` tuples.

    Raises:
        RuntimeError: If interactive input ends before the wizard is complete.
    """
    answers: dict[str, Any] = {"updates": []}
    updates: list[ConfigUpdate] = answers["updates"]
    sections = config_sections(config)

    if ask_until_valid(
        "Configure data options",
        True,
        parse_bool,
        "Show data, image grid, and channel settings.",
    ):
        updates.extend(ask_data_questions(config))
    if ask_until_valid(
        "Configure imaging weighting options",
        True,
        parse_bool,
        "Show data-weighting and imaging-weight settings.",
    ):
        updates.extend(ask_weighting_questions(config))
    if ask_until_valid(
        "Configure general options",
        True,
        parse_bool,
        "Show output, run-metadata, and common flag settings.",
    ):
        updates.extend(ask_general_questions(config))
    if ask_until_valid(
        "Configure compute options",
        True,
        parse_bool,
        "Show device, data type, and distributed-computing settings.",
    ):
        updates.extend(ask_compute_questions(config))
    if ask_until_valid(
        "Configure measurement operator options",
        False,
        parse_bool,
        "Show NUFFT settings.",
    ):
        updates.extend(ask_measurement_operator_questions(config))

    algorithm = ask_choice(
        "Algorithm",
        ALGORITHMS,
        sections["main"]["algorithm"],
        "Reconstruction method to configure.",
    )
    updates.append(("main", "algorithm", algorithm))

    if ask_until_valid(
        "Configure algorithm-specific options",
        True,
        parse_bool,
        "Show settings for the selected reconstruction algorithm.",
    ):
        if algorithm in SARA_ALGORITHMS:
            updates.extend(ask_sara_family_questions(config, algorithm))
        if algorithm in AIRI_ALGORITHMS:
            updates.extend(ask_airi_family_questions(config, algorithm))
        if algorithm == "hyperairi":
            updates.extend(ask_hyperairi_questions(config))
        if algorithm in CONSTRAINED_ALGORITHMS:
            updates.extend(ask_constrained_questions(config, algorithm))
        if algorithm in UNCONSTRAINED_ALGORITHMS:
            updates.extend(ask_unconstrained_questions(config, algorithm))

    return answers


def ask_data_questions(config: list[dict[str, Any]]) -> list[ConfigUpdate]:
    """Ask data, image-grid, and channel questions.

    Args:
        config (list[dict[str, Any]]): Configuration template used for default
            answers.

    Returns:
        list[ConfigUpdate]: Updates for data, image, and channel settings.

    Raises:
        RuntimeError: If interactive input ends before these questions finish.
    """
    sections = config_sections(config)
    updates = [
        (
            "main",
            "data_file",
            ask_str(
                "Input measurement data path",
                sections["main"]["data_file"],
            ),
        ),
        (
            "main",
            "im_dim_x",
            ask_int(
                "Image width",
                sections["main"]["im_dim_x"],
                minimum=1,
            ),
        ),
        (
            "main",
            "im_dim_y",
            ask_int(
                "Image height",
                sections["main"]["im_dim_y"],
                minimum=1,
            ),
        ),
    ]

    im_pixel_size = ask_until_valid(
        "Pixel size in arcsec",
        sections["main"]["im_pixel_size"],
        lambda text: parse_optional_float(text, minimum=0.0),
        "Leave empty to infer it from superresolution.",
    )
    updates.append(("main", "im_pixel_size", im_pixel_size))
    if is_empty_value(im_pixel_size):
        updates.append(
            (
                "main",
                "superresolution",
                ask_float(
                    "Super resolution factor",
                    sections["main"]["superresolution"],
                    minimum=1.0,
                ),
            )
        )
    else:
        updates.append(("main", "superresolution", []))

    ch_start = ask_int(
        "First physical channel index",
        sections["main"]["ch_start"],
        minimum=1,
    )
    ch_last = ask_until_valid(
        "Last physical channel index",
        sections["main"]["ch_last"],
        lambda text: parse_optional_int(text, minimum=ch_start),
        "Leave this field empty to use all physical channels.",
    )
    updates.extend(
        [
            ("main", "ch_start", ch_start),
            ("main", "ch_last", ch_last),
            (
                "main",
                "ch_effective",
                ask_int(
                    "Number of effective channels",
                    sections["main"]["ch_effective"],
                    "Number of channels for the reconstructed image.",
                    minimum=1,
                ),
            ),
            (
                "main",
                "ch_partition",
                ask_optional_int_list(
                    "Physical channel partition",
                    sections["main"]["ch_partition"],
                    "Number of physical channels per effective channel; "
                    "leave empty to be evenly partitioned.",
                    minimum=1,
                ),
            ),
            (
                "flag",
                "invert_uvw_flag",
                ask_until_valid(
                    "Invert the flag for the uvw coordinates",
                    sections["flag"]["invert_uvw_flag"],
                    parse_bool,
                    "By default, the uvw coordinates are ignored if the associated "
                    "data flag is True. Enable this option to invert this logic",
                ),
            ),
        ]
    )
    return updates


def ask_weighting_questions(config: list[dict[str, Any]]) -> list[ConfigUpdate]:
    """Ask imaging-weighting questions.

    Args:
        config (list[dict[str, Any]]): Configuration template used for default
            answers.

    Returns:
        list[ConfigUpdate]: Updates for data-weighting flags and imaging-weight
            settings.

    Raises:
        RuntimeError: If interactive input ends before these questions finish.
    """
    sections = config_sections(config)
    updates: list[ConfigUpdate] = []
    flag_data_weighting = ask_until_valid(
        "Enable data weighting",
        sections["flag"]["flag_data_weighting"],
        parse_bool,
        "Apply imaging weighting for measurement operators.",
    )
    updates.append(("flag", "flag_data_weighting", flag_data_weighting))
    if not flag_data_weighting:
        return updates

    weight_load = ask_until_valid(
        "Load weights from data file",
        sections["weighting"]["weight_load"],
        parse_bool,
    )
    updates.append(("weighting", "weight_load", weight_load))
    if weight_load:
        return updates

    weight_type = ask_choice(
        "Imaging weight type",
        WEIGHT_TYPES,
        sections["weighting"]["weight_type"],
        "It can be 'briggs' for Briggs weighting or 'uniform' "
        "for uniform weighting.",
    )
    updates.append(("weighting", "weight_type", weight_type))
    if weight_type == "briggs":
        updates.append(
            (
                "weighting",
                "weight_robustness",
                ask_float(
                    "Briggs weight robustness factor",
                    sections["weighting"]["weight_robustness"],
                ),
            )
        )
    updates.append(
        (
            "weighting",
            "weight_gridsize",
            ask_float(
                "Imaging weight grid size",
                sections["weighting"]["weight_gridsize"],
                minimum=0.0,
            ),
        ),
    )
    return updates


def ask_general_questions(config: list[dict[str, Any]]) -> list[ConfigUpdate]:
    """Ask output, run-metadata, and common flag questions.

    Args:
        config (list[dict[str, Any]]): Configuration template used for default
            answers.

    Returns:
        list[ConfigUpdate]: Updates for result path, source metadata, run ID,
            and common flags.

    Raises:
        RuntimeError: If interactive input ends before these questions finish.
    """
    sections = config_sections(config)
    return [
        (
            "main",
            "result_path",
            ask_str(
                "Result path",
                sections["main"]["result_path"],
                "Directory where imaging results will be saved.",
            ),
        ),
        (
            "main",
            "src_name",
            ask_str(
                "Source name",
                sections["main"]["src_name"],
                "Source tag used in output folder names. The input data folder name "
                "is used if this field is empty.",
            ),
        ),
        (
            "main",
            "run_id",
            ask_until_valid(
                "Run ID",
                sections["main"]["run_id"],
                lambda text: parse_optional_int(text),
                "This numeric run identifier will be used in output folder names "
                "if specified.",
            ),
        ),
        (
            "flag",
            "flag_imaging",
            ask_until_valid(
                "Enable imaging",
                sections["flag"]["flag_imaging"],
                parse_bool,
                "Disable to generate dirty image and PSF only.",
            ),
        ),
        (
            "flag",
            "verbose",
            ask_until_valid(
                "Verbose output",
                sections["flag"]["verbose"],
                parse_bool,
                "Print effective parameters when running.",
            ),
        ),
    ]


def ask_measurement_operator_questions(
    config: list[dict[str, Any]],
) -> list[ConfigUpdate]:
    """Ask optional measurement-operator questions.

    Args:
        config (list[dict[str, Any]]): Configuration template used for default
            answers.

    Returns:
        list[ConfigUpdate]: Updates for the NUFFT package and, when using
            TorchKbNUFFT, the NUFFT mode.

    Raises:
        RuntimeError: If interactive input ends before these questions finish.
    """
    sections = config_sections(config)
    nufft = sections["nufft"]
    updates: list[ConfigUpdate] = []
    nufft_package = ask_choice(
        "NUFFT package",
        NUFFT_PACKAGES,
        nufft["nufft_package"],
        "NUFFT backend used for the measurement operators.",
    )
    updates.append(("nufft", "nufft_package", nufft_package))
    if nufft_package == "tkbnufft":
        updates.append(
            (
                "nufft",
                "nufft_mode",
                ask_choice(
                    "NUFFT mode",
                    NUFFT_MODES,
                    nufft["nufft_mode"],
                    "Interpolation mode for TorchKbNUFFT. It can be 'table' "
                    "or 'matrix'.",
                ),
            )
        )
    return updates


def ask_compute_questions(config: list[dict[str, Any]]) -> list[ConfigUpdate]:
    """Ask compute-resource questions.

    Args:
        config (list[dict[str, Any]]): Configuration template used for default
            answers.

    Returns:
        list[ConfigUpdate]: Updates for devices, dtypes, and distributed
            initialisation settings.

    Raises:
        RuntimeError: If interactive input ends before these questions finish.
    """
    sections = config_sections(config)
    return [
        (
            "computing",
            "meas_device",
            ask_choice(
                "Measurement device",
                DEVICES,
                sections["computing"]["meas_device"],
                "Device used by measurement operators.",
            ),
        ),
        (
            "computing",
            "regulariser_device",
            ask_choice(
                "Regulariser device",
                DEVICES,
                sections["computing"]["regulariser_device"],
                "Device used by regularisers.",
            ),
        ),
        (
            "computing",
            "meas_dtype",
            ask_choice(
                "Measurement dtype",
                DTYPES,
                sections["computing"]["meas_dtype"],
                "Numeric precision for measurement operators.",
            ),
        ),
        (
            "computing",
            "regulariser_dtype",
            ask_choice(
                "Regulariser dtype",
                DTYPES,
                sections["computing"]["regulariser_dtype"],
                "Numeric precision for regularisers.",
            ),
        ),
        (
            "computing",
            "dist_backend",
            ask_choice(
                "PyTorch distributed backend",
                DIST_BACKENDS,
                sections["computing"]["dist_backend"],
                "PyTorch supports 4 backends. Normally, gloo is paried with CPU, "
                "NCCL with CUDA and XCCL with XPU. "
                "See https://docs.pytorch.org/docs/stable/distributed.html"
                "#which-backend-to-use for details.",
            ),
        ),
        (
            "computing",
            "dist_init_method",
            ask_str(
                "PyTorch distributed initialisation method",
                sections["computing"]["dist_init_method"],
                "It is a URL string passed to torch.distributed indicating where "
                "and how to discover peers. "
                "See https://docs.pytorch.org/docs/stable/distributed.html"
                "#torch.distributed.init_process_group for details.",
            ),
        ),
    ]


def apply_config_answers(
    config: list[dict[str, Any]], answers: dict[str, Any]
) -> list[dict[str, Any]]:
    """Apply flat wizard answers to a copy of the JSON template.

    The implementation should update the existing nested config sections rather
    than constructing a new schema from scratch.  That keeps this command
    aligned with ``config/default.json`` and
    :func:`hyri.interface.io.parsing_parameters`.

    Since invalid values should be handled during prompting, this function
    should be a straightforward mapping step rather than another validation
    layer.

    Args:
        config (list[dict[str, Any]]): Configuration template to update.
        answers (dict[str, Any]): Wizard answers containing an ``"updates"``
            list of ``(section, key, value)`` tuples.

    Returns:
        list[dict[str, Any]]: Updated configuration with unselected algorithm
        sections removed when an algorithm answer is present.
    """
    updated_config = copy.deepcopy(config)
    sections = config_sections(updated_config)
    algorithm = None
    for section, key, value in answers["updates"]:
        sections[section][key] = value
        if section == "main" and key == "algorithm":
            algorithm = value
    if algorithm is not None:
        prune_unselected_algorithms(updated_config, algorithm)
    return updated_config


def print_config_created(output_path: str, config: list[dict[str, Any]]) -> None:
    """Print the generated config path and next imaging command.

    A detailed config summary is not necessary.  The user can inspect the JSON
    file directly if needed.  Keep the completion message focused:

    - created output path;
    - ``hyri-imaging --config <output_path>``.

    Args:
        output_path (str): Path of the generated JSON configuration file.
        config (list[dict[str, Any]]): Generated configuration data used to
            decide whether the run command needs ``--data_file``.
    """
    print(f"\nWrote {output_path}", flush=True)
    print("\nRun with:", flush=True)
    sections = config_sections(config)
    data_file = sections["main"].get("data_file")
    if is_empty_value(data_file):
        print(f"  hyri-imaging --config {output_path} --data_file <path>", flush=True)
        print(
            "\nThe configuration file has no data_file set; "
            "pass it when running the imager.",
            flush=True,
        )
    else:
        print(f"  hyri-imaging --config {output_path}", flush=True)


def ask_sara_family_questions(
    config: list[dict[str, Any]], algorithm: str
) -> list[ConfigUpdate]:
    """Ask SARA-family algorithm questions.

    Args:
        config (list[dict[str, Any]]): Configuration template used for default
            answers.
        algorithm (str): Selected SARA-family algorithm section to configure.

    Returns:
        list[ConfigUpdate]: Updates for SARA regularisation scale and
            reweighting options.

    Raises:
        RuntimeError: If interactive input ends before these questions finish.
    """
    section = config_sections(config)[algorithm]
    return [
        (
            algorithm,
            "heu_reg_param_scale",
            ask_float(
                "Scaling factor for the heuristic regularisation parameter",
                section["heu_reg_param_scale"],
                "The regularisation is stronger when this factor is larger, which will "
                "typically lead to smoother reconstructions, and vice versa.",
                minimum=0.0,
            ),
        ),
        (
            algorithm,
            "reweighting",
            ask_until_valid(
                "Use reweighting",
                section["reweighting"],
                parse_bool,
                "Enable the major reweighting loop for SARA algorithms.",
            ),
        ),
    ]


def ask_airi_family_questions(
    config: list[dict[str, Any]], algorithm: str
) -> list[ConfigUpdate]:
    """Ask AIRI-family algorithm questions.

    Args:
        config (list[dict[str, Any]]): Configuration template used for default
            answers.
        algorithm (str): Selected AIRI-family algorithm section to configure.

    Returns:
        list[ConfigUpdate]: Updates for denoiser shelf, noise scale, image peak,
            and adaptive denoising options.

    Raises:
        RuntimeError: If interactive input ends before these questions finish.
    """
    section = config_sections(config)[algorithm]
    return [
        (
            algorithm,
            "dnn_shelf_path",
            ask_str(
                "Denoiser shelf path",
                section["dnn_shelf_path"],
                "It is a CSV file mapping noise levels to ONNX denoisers. "
                "This library includes three shelves: 'shelf_airi_oaid.csv' and "
                "'shelf_airi_mrid.csv' for AIRI and cAIRI algorithms, and "
                "'shelf_hyperairi_mix.csv' for hyperAIRI.",
            ),
        ),
        (
            algorithm,
            "heu_noise_scale",
            ask_float(
                "Scaling factor for the heuristic noise level estimate",
                section["heu_noise_scale"],
                "The regularisation is stronger when this factor is larger, which will "
                "typically lead to smoother reconstructions, and vice versa.",
                minimum=0.0,
            ),
        ),
        (
            algorithm,
            "im_peak_est",
            ask_optional_float(
                "Estimated image peak intensity",
                section["im_peak_est"],
                "It will affect the initial target dynamic range estimation and the "
                "denoiser selection. Leave empty to estimate from the dirty image.",
                minimum=0.0,
            ),
        ),
        (
            algorithm,
            "dnn_adaptive_peak",
            ask_until_valid(
                "Adaptive denoiser selection during reconstruction",
                section["dnn_adaptive_peak"],
                parse_bool,
                "Adaptively select denoisers during reconstruction based on the latest "
                "maximum intensity of the reconstructed image.",
            ),
        ),
        (
            algorithm,
            "dnn_apply_transform",
            ask_until_valid(
                "Apply random denoising transforms",
                section["dnn_apply_transform"],
                parse_bool,
                "Use random rotations and flips in the denoising process. "
                "It will enhance the denoising performance but reduce the "
                "convergence speed.",
            ),
        ),
    ]


def ask_constrained_questions(
    config: list[dict[str, Any]], algorithm: str
) -> list[ConfigUpdate]:
    """Ask constrained algorithm questions.

    Args:
        config (list[dict[str, Any]]): Configuration template used for default
            answers.
        algorithm (str): Selected constrained algorithm section to configure.

    Returns:
        list[ConfigUpdate]: Updates for constrained preconditioning when the
            selected section supports it.

    Raises:
        RuntimeError: If interactive input ends before these questions finish.
    """
    section = config_sections(config)[algorithm]
    if "precond_flag" not in section:
        return []
    return [
        (
            algorithm,
            "precond_flag",
            ask_until_valid(
                "Use preconditioning",
                section["precond_flag"],
                parse_bool,
                "Enable preconditioned weighting scheme for constrained algorithms.",
            ),
        )
    ]


def ask_hyperairi_questions(config: list[dict[str, Any]]) -> list[ConfigUpdate]:
    """Ask HyperAIRI-specific questions.

    Args:
        config (list[dict[str, Any]]): Configuration template used for default
            answers.

    Returns:
        list[ConfigUpdate]: Updates for spectral-index map behaviour.

    Raises:
        RuntimeError: If interactive input ends before these questions finish.
    """
    section = config_sections(config)["hyperairi"]
    return [
        (
            "hyperairi",
            "spec_update",
            ask_until_valid(
                "Update spectral index",
                section["spec_update"],
                parse_bool,
                "Enable adaptive spectral-index map updates for HyperAIRI.",
            ),
        ),
        (
            "hyperairi",
            "spec_idx_init",
            ask_until_valid(
                "Spectral-index map initialisation",
                section["spec_idx_init"],
                parse_spec_idx_init,
                "It can be 'zero' for a flat spectral index map, "
                "'dirty' for inferring from the dirty image cube, "
                "or the path of a FITS file.",
            ),
        ),
    ]


def ask_unconstrained_questions(
    config: list[dict[str, Any]], algorithm: str
) -> list[ConfigUpdate]:
    """Ask unconstrained algorithm questions.

    Args:
        config (list[dict[str, Any]]): Configuration template used for default
            answers.
        algorithm (str): Selected unconstrained algorithm section to configure.

    Returns:
        list[ConfigUpdate]: Updates for the approximate measurement operator
            when the selected section supports it.

    Raises:
        RuntimeError: If interactive input ends before these questions finish.
    """
    section = config_sections(config)[algorithm]
    if "approx_meas_op" not in section:
        return []
    return [
        (
            algorithm,
            "approx_meas_op",
            ask_until_valid(
                "Use approximate measurement operator",
                section["approx_meas_op"],
                parse_bool,
                "Use a PSF-based measurement operator during reconstruction.",
            ),
        )
    ]


def ask_str(label: str, default: Any = None, description: str | None = None) -> Any:
    """Ask for a string value while preserving empty-list defaults.

    Args:
        label (str): Prompt label shown to the user.
        default (Any, optional): Value returned when the user submits an empty
            response. Default: ``None``.
        description (str | None, optional): Extra prompt detail. Default:
            ``None``.

    Returns:
        Any: Parsed string input, or a copy of ``default`` for an empty response.

    Raises:
        RuntimeError: If interactive input ends before a valid value is read.
    """
    return ask_until_valid(label, default, parse_str, description)


def ask_int(
    label: str,
    default: Any = None,
    description: str | None = None,
    *,
    minimum: int | None = None,
) -> int:
    """Ask for an integer value.

    Args:
        label (str): Prompt label shown to the user.
        default (Any, optional): Value returned when the user submits an empty
            response. Default: ``None``.
        description (str | None, optional): Extra prompt detail. Default:
            ``None``.

    Keyword args:
        minimum (int | None, optional): Inclusive lower bound for accepted
            values. Default: ``None``.

    Returns:
        int: Parsed integer input, or a copy of ``default`` for an empty response.

    Raises:
        RuntimeError: If interactive input ends before a valid value is read.
    """
    return ask_until_valid(
        label, default, lambda text: parse_int(text, minimum=minimum), description
    )


def ask_optional_int_list(
    label: str,
    default: Any = None,
    description: str | None = None,
    *,
    minimum: int | None = None,
) -> list[int] | list[Any]:
    """Ask for an optional space-separated integer list.

    Args:
        label (str): Prompt label shown to the user.
        default (Any, optional): Value returned when the user submits an empty
            response. Default: ``None``.
        description (str | None, optional): Extra prompt detail. Default:
            ``None``.

    Keyword args:
        minimum (int | None, optional): Inclusive lower bound for each accepted
            integer. Default: ``None``.

    Returns:
        list[int] | list[Any]: Parsed integer list, an empty-list sentinel, or a
            copy of ``default`` for an empty response.

    Raises:
        RuntimeError: If interactive input ends before a valid value is read.
    """
    return ask_until_valid(
        label,
        default,
        lambda text: parse_optional_int_list(text, minimum=minimum),
        description,
    )


def ask_float(
    label: str,
    default: Any = None,
    description: str | None = None,
    *,
    minimum: float | None = None,
) -> float:
    """Ask for a floating-point value.

    Args:
        label (str): Prompt label shown to the user.
        default (Any, optional): Value returned when the user submits an empty
            response. Default: ``None``.
        description (str | None, optional): Extra prompt detail. Default:
            ``None``.

    Keyword args:
        minimum (float | None, optional): Inclusive lower bound for accepted
            values. Default: ``None``.

    Returns:
        float: Parsed float input, or a copy of ``default`` for an empty response.

    Raises:
        RuntimeError: If interactive input ends before a valid value is read.
    """
    return ask_until_valid(
        label, default, lambda text: parse_float(text, minimum=minimum), description
    )


def ask_optional_float(
    label: str,
    default: Any = None,
    description: str | None = None,
    *,
    minimum: float | None = None,
) -> float | list[Any]:
    """Ask for an optional floating-point value.

    Args:
        label (str): Prompt label shown to the user.
        default (Any, optional): Value returned when the user submits an empty
            response. Default: ``None``.
        description (str | None, optional): Extra prompt detail. Default:
            ``None``.

    Keyword args:
        minimum (float | None, optional): Inclusive lower bound for accepted
            values. Default: ``None``.

    Returns:
        float | list[Any]: Parsed float, an empty-list sentinel, or a copy of
            ``default`` for an empty response.

    Raises:
        RuntimeError: If interactive input ends before a valid value is read.
    """
    return ask_until_valid(
        label,
        default,
        lambda text: parse_optional_float(text, minimum=minimum),
        description,
    )


def ask_choice(
    label: str,
    choices: list[str],
    default: Any = None,
    description: str | None = None,
) -> str:
    """Ask for a value from a fixed set of choices.

    Args:
        label (str): Prompt label shown to the user.
        choices (list[str]): Accepted string values.
        default (Any, optional): Preferred default value. Default: ``None``. If
            this is not in ``choices``, the first choice is used.
        description (str | None, optional): Extra prompt detail. Default:
            ``None``.

    Returns:
        str: Selected choice.

    Raises:
        RuntimeError: If interactive input ends before a valid value is read.
    """
    valid_default = default if default in choices else choices[0]
    choice_text = "/".join(choices)
    return ask_until_valid(
        f"{label} ({choice_text})",
        valid_default,
        lambda text: parse_choice(text, choices),
        description,
    )


def ask_until_valid(
    label: str,
    default: Any,
    parser: Callable[[str], Any],
    description: str | None = None,
) -> Any:
    """Prompt until ``parser`` accepts the user input.

    Args:
        label (str): Prompt label shown to the user.
        default (Any): Value returned when the user submits an empty response.
        parser (Callable[[str], Any]): Function that converts non-empty input
            text to a configuration value.
        description (str | None, optional): Extra prompt detail. Default:
            ``None``.

    Returns:
        Any: Parsed value, or a copy of ``default`` for an empty response.

    Raises:
        RuntimeError: If interactive input ends before a valid value is read.
    """
    while True:
        prompt = format_prompt(label, default, description)
        try:
            raw = input(prompt)
        except EOFError as exc:
            raise RuntimeError(
                "Interactive input ended before config was complete."
            ) from exc

        text = raw.strip()
        if text == "":
            return copy.deepcopy(default)

        try:
            return parser(text)
        except ValueError as exc:
            print(f"Invalid value: {exc}", flush=True)


def format_prompt(label: str, default: Any, description: str | None = None) -> str:
    """Format a prompt with a readable default value.

    Args:
        label (str): Prompt label shown to the user.
        default (Any): Default value displayed in square brackets.
        description (str | None, optional): Extra prompt detail. Default:
            ``None``.

    Returns:
        str: Complete prompt text ending with ``": "``.
    """
    detail = f" - {description}" if description else ""
    return f"{label}{detail} [{format_default(default)}]: "


def format_default(value: Any) -> str:
    """Format configuration defaults for display.

    Args:
        value (Any): Default value from the configuration template.

    Returns:
        str: Empty string for unset values, otherwise ``str(value)``.
    """
    if is_empty_value(value):
        return ""
    return str(value)


def is_empty_value(value: Any) -> bool:
    """Return whether a configuration value should be treated as unset.

    Args:
        value (Any): Configuration value to inspect.

    Returns:
        bool: ``True`` when ``value`` is the empty-list sentinel, ``None``, or
            an empty string.
    """
    return value == EMPTY_VALUE or value is None or value == ""


def parse_str(text: str) -> str:
    """Parse a string prompt.

    Args:
        text (str): Prompt text to parse.

    Returns:
        str: Unmodified prompt text.
    """
    return text


def parse_non_empty_str(text: str) -> str:
    """Parse a string that cannot be empty.

    Args:
        text (str): Prompt text to parse.

    Returns:
        str: Unmodified prompt text.

    Raises:
        ValueError: If ``text`` is empty.
    """
    if text == "":
        raise ValueError("value cannot be empty")
    return text


def parse_int(text: str, *, minimum: int | None = None) -> int:
    """Parse an integer with an optional lower bound.

    Args:
        text (str): Prompt text to parse.

    Keyword args:
        minimum (int | None, optional): Inclusive lower bound for the parsed
            value. Default: ``None``.

    Returns:
        int: Parsed integer value.

    Raises:
        ValueError: If ``text`` is not an integer or is less than ``minimum``.
    """
    try:
        value = int(text)
    except ValueError as exc:
        raise ValueError("enter an integer") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"enter an integer >= {minimum}")
    return value


def parse_optional_int(text: str, *, minimum: int | None = None) -> int | list[Any]:
    """Parse an optional integer.

    Users can type ``none`` or ``[]`` to store the empty-list sentinel used by
    the existing JSON files.

    Args:
        text (str): Prompt text to parse.

    Keyword args:
        minimum (int | None, optional): Inclusive lower bound for the parsed
            integer. Default: ``None``.

    Returns:
        int | list[Any]: Parsed integer value, or ``[]`` for empty sentinels.

    Raises:
        ValueError: If ``text`` is not an accepted sentinel or integer, or if
            the integer is less than ``minimum``.
    """
    if text.lower() in {"none", "null", "[]"}:
        return []
    return parse_int(text, minimum=minimum)


def parse_optional_int_list(
    text: str, *, minimum: int | None = None
) -> list[int] | list[Any]:
    """Parse an optional list of integers.

    Args:
        text (str): Prompt text to parse. Commas and spaces may separate list
            entries.

    Keyword args:
        minimum (int | None, optional): Inclusive lower bound for each parsed
            integer. Default: ``None``.

    Returns:
        list[int] | list[Any]: Parsed integer list, or ``[]`` for empty
            sentinels and empty input.

    Raises:
        ValueError: If any list entry is not an integer or is less than
            ``minimum``.
    """
    if text.lower() in {"none", "null", "[]"}:
        return []
    normalised = text.replace(",", " ")
    values = [parse_int(item, minimum=minimum) for item in normalised.split()]
    if len(values) == 0:
        return []
    return values


def parse_float(text: str, *, minimum: float | None = None) -> float:
    """Parse a float with an optional lower bound.

    Args:
        text (str): Prompt text to parse.

    Keyword args:
        minimum (float | None, optional): Inclusive lower bound for the parsed
            value. Default: ``None``.

    Returns:
        float: Parsed floating-point value.

    Raises:
        ValueError: If ``text`` is not numeric or is less than ``minimum``.
    """
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError("enter a number") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"enter a number >= {minimum}")
    return value


def parse_optional_float(
    text: str, *, minimum: float | None = None
) -> float | list[Any]:
    """Parse an optional float.

    Args:
        text (str): Prompt text to parse.

    Keyword args:
        minimum (float | None, optional): Inclusive lower bound for the parsed
            value. Default: ``None``.

    Returns:
        float | list[Any]: Parsed floating-point value, or ``[]`` for empty
        sentinels.

    Raises:
        ValueError: If ``text`` is not an accepted sentinel or numeric value,
            or if the value is less than ``minimum``.
    """
    if text.lower() in {"none", "null", "[]"}:
        return []
    return parse_float(text, minimum=minimum)


def parse_bool(text: str) -> bool:
    """Parse a yes/no or true/false answer.

    Args:
        text (str): Prompt text to parse.

    Returns:
        bool: Parsed boolean value.

    Raises:
        ValueError: If ``text`` is not a recognised boolean answer.
    """
    value = text.lower()
    if value in {"y", "yes", "true", "t", "1"}:
        return True
    if value in {"n", "no", "false", "f", "0"}:
        return False
    raise ValueError("enter yes/no or true/false")


def parse_choice(text: str, choices: list[str]) -> str:
    """Parse a fixed-choice answer.

    Args:
        text (str): Prompt text to parse.
        choices (list[str]): Accepted string values.

    Returns:
        str: Parsed choice.

    Raises:
        ValueError: If ``text`` is not in ``choices``.
    """
    if text in choices:
        return text
    raise ValueError("choose one of: " + ", ".join(choices))


def parse_spec_idx_init(text: str) -> str:
    """Parse a spectral-index initialisation mode or FITS path.

    Args:
        text (str): Prompt text to parse.

    Returns:
        str: Spectral-index initialisation mode or FITS file path.

    Raises:
        ValueError: If ``text`` is neither a supported mode nor a ``.fits``
            path.
    """
    if text in SPEC_INDEX_INIT_MODES:
        return text
    if text.lower().endswith(".fits"):
        return text
    raise ValueError("enter 'zero', 'dirty', or a FITS file path")


def config_sections(config: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return a flat mapping from section name to section dictionary.

    The template is stored as a list of dictionaries with optional
    ``"__comment"`` entries.  This helper should hide that layout from the rest
    of the implementation.

    Args:
        config (list[dict[str, Any]]): Configuration template or generated
            configuration data.

    Returns:
        dict[str, dict[str, Any]]: Mapping from section name to section
            dictionary.
    """
    sections = {}
    for block in config:
        for key, value in block.items():
            if key != "__comment":
                sections[key] = value
    return sections


def prune_unselected_algorithms(config: list[dict[str, Any]], algorithm: str) -> None:
    """Remove unused algorithm sections from the final configuration.

    Args:
        config (list[dict[str, Any]]): Configuration data to mutate in place.
        algorithm (str): Selected algorithm whose sections should be kept.
    """
    keep = {algorithm, f"{algorithm}_default"}
    algorithm_sections = set(ALGORITHMS)
    algorithm_sections.update(f"{name}_default" for name in ALGORITHMS)
    for block in config:
        for key in list(block):
            if key in algorithm_sections and key not in keep:
                del block[key]


if __name__ == "__main__":
    create_config()
