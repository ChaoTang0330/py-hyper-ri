"""Parse user inputs and launch the imaging workflow."""

import json

from .interface import (
    env_int,
    imager,
    parsing_arguments,
    parsing_parameters,
    print_dict,
    set_imaging_params_ri,
)


def run_imager() -> None:
    """Run the command-line imaging entry point.

    This function parses command-line arguments, loads and merges configuration
    parameters, converts them into imaging-specific parameter dictionaries, and
    then launches :func:`.interface.imager`.
    """

    # Parse command-line arguments.
    input_args = parsing_arguments()
    # Load the configuration file and override it with CLI arguments.
    param_general = parsing_parameters(input_args.config, input_param=input_args)
    # Split the general configuration into imaging sub-configurations.
    param_measop, param_proxop, param_optimiser = set_imaging_params_ri(param_general)

    # Print the effective configuration once from rank 0.
    rank = env_int("RANK", 0)
    if param_optimiser["verbose"] and rank == 0:
        print("Input parameters", flush=True)
        print(json.dumps(param_general, indent=4), flush=True)
        print("_" * 64 + "\n", flush=True)
        print("Imaging parameters:", flush=True)
        print("  param_optimiser:", flush=True)
        print_dict(param_optimiser, flush=True)
        print("  param_measop:", flush=True)
        print_dict(param_measop, flush=True)
        print("  param_proxop:", flush=True)
        print_dict(param_proxop, flush=True)
        print("_" * 64 + "\n", flush=True)

    # Launch the main imaging workflow.
    imager(param_optimiser, param_measop, param_proxop)


if __name__ == "__main__":
    run_imager()
