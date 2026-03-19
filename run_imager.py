"""
Prepare input variables for imager
"""

import json

from src.imager import imager
from src.interface import set_imaging_params_ri
from src.interface import env_int, print_dict
from src.interface.io import parsing_arguments, parsing_parameters


def run_imager() -> None:
    """
    Main function to launch the imager.

    This function parses the input arguments, sets the imaging parameters, and calls the imager function.
    It also handles the logging of input parameters and imaging parameters if the 'verbose' flag is set in the 'param_optimiser'.
    """

    # parsing command line arguments
    input_args = parsing_arguments()
    # parsing parameters from config file and override with command line arguments
    param_general = parsing_parameters(input_args.config, input_param=input_args)
    # set imaging parameters
    param_measop, param_proxop, param_optimiser = set_imaging_params_ri(param_general)

    # Initial logging
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

    # launch imager
    imager(param_optimiser, param_measop, param_proxop)


if __name__ == "__main__":
    run_imager()
