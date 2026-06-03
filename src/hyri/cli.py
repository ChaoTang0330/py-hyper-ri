"""Parse user inputs and launch package command-line workflows."""


def run_hyri() -> None:
    """Run the top-level pyHyperRI command-line entry point.

    This function is linked to the command-line entry point ``hyri``. It
    dispatches subcommands such as ``hyri imaging``, ``hyri config``, and
    ``hyri ms2mat`` to the same workflow functions used by the standalone
    command-line entry points.
    """
    import sys

    commands = {
        "imaging": ("hyri-imaging", run_imager),
        "config": ("hyri-config", run_config),
        "ms2mat": ("hyri-ms2mat", run_ms2mat),
    }
    command_descriptions = {
        "imaging": "run hyperspectral radio interferometric imaging",
        "config": "create or fetch an imaging configuration file",
        "ms2mat": "convert a Measurement Set into package-compatible .mat files",
    }

    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print("Usage: hyri <command> [options]\n", flush=True)
        print(
            "pyHyperRI is a Python package for astronomical radio "
            "interferometric imaging.",
            flush=True,
        )
        print("Run pyHyperRI command-line workflows.\n", flush=True)
        print("Commands:", flush=True)
        for command in commands:
            print(f"  {command:<8} {command_descriptions[command]}", flush=True)
        print("\nExamples:", flush=True)
        print("  hyri imaging --config config.json", flush=True)
        print("  hyri config", flush=True)
        print("  hyri ms2mat --help", flush=True)
        print("\nUse 'hyri <command> --help' for command-specific options.", flush=True)
        print("Documentation: <TODO: doc link>", flush=True)
        return

    command_name = sys.argv[1]
    if command_name not in commands:
        print(f"Unknown command: {command_name}", file=sys.stderr, flush=True)
        print(
            "Run 'hyri --help' to list available commands.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(2)

    script_name, command = commands[command_name]
    sys.argv = [script_name, *sys.argv[2:]]
    command()


def run_imager() -> None:
    """Run the command-line imaging entry point.

    This function is linked to the command-line entry point ``hyri-imaging``. It parses
    command-line arguments, loads and merges configuration parameters, converts
    them into imaging-specific parameter dictionaries, and then launches
    :func:`.interface.imager.imager`.
    """
    import json

    from .interface import (
        env_int,
        parsing_arguments,
        parsing_parameters,
        print_dict,
        set_imaging_params_ri,
    )
    from .interface.imager import imager

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


def run_config() -> None:
    """Run the command-line configuration entry point.

    This function is linked to the command-line entry point ``hyri-config``. It launches
    the interactive configuration creator
    :func:`.interface.create_config.create_config`. This allows users to fetch the
    default configuration template, or interactively create a new configuration file
    by answering a series of questions.
    """
    from .interface.create_config import create_config

    create_config()


def run_ms2mat() -> None:
    """Run the command-line Measurement Set conversion entry point.

    This function is linked to the command-line entry point ``hyri-ms2mat``. It
    launches :func:`.interface.ms2mat.ms2mat`, which parses command-line
    arguments and converts a Measurement Set into the ``.mat`` data layout used
    by this package.
    """
    from .interface.ms2mat import ms2mat

    ms2mat()


if __name__ == "__main__":
    run_imager()
