# Installation

## Requirements

pyHyperRI requires Python 3.11 or later and a PyTorch-supported execution
environment.

TODO:

- Confirm the supported Python versions.
- Confirm the supported operating systems.
- Confirm the supported CPU, GPU, and distributed execution environments.

## Create an Environment

Create and activate a Python environment for pyHyperRI.

Using conda:

```bash
conda create -n hyri python=3.11
conda activate hyri
```

Using `venv` and pip:

```bash
python -m venv .hyri-env
source .hyri-env/bin/activate
python -m pip install --upgrade pip
```

On Windows, activate a `venv` environment with:

```bat
.hyri-env\Scripts\activate
```

## Install PyTorch

Install PyTorch for the target hardware and accelerator stack.

Use the official PyTorch installation selector to choose the command matching
the operating system, package manager, Python version, and compute platform.
For CUDA-enabled devices, select the CUDA version supported by the local driver
and hardware.

```bash
# Select the appropriate command from:
# https://pytorch.org/get-started/locally/
```

## Install hyri

Install the package from PyPI.

```bash
pip install hyri
```

When using CUDA devices, install the CUDA optional dependency group.

```bash
pip install "hyri[cuda]"
```

## Install from Source

Install the package from a local source checkout.

```bash
git clone <REPOSITORY_URL>
cd <REPOSITORY_NAME>
pip install -e .
```

TODO:

- Add the final repository URL.
- Confirm whether editable installs should include development or optional extras.

## Optional Dependencies

For development based on pyHyperRI, install the development dependency group
from a source checkout.

```bash
pip install -e ".[dev]"
```

TODO:

- Confirm the complete development workflow.
- Document the recommended testing and formatting commands.

## Verify Installation

Check that the package and command-line tools are available.

```bash
python -c "import hyri"
hyri-config --help
hyri-imaging --help
```

TODO:

- Add a minimal command that validates a full installation.
- Add expected output snippets for the command-line checks.

## Common Issues

TODO.
