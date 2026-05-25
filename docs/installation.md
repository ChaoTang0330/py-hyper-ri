# Installation

## Requirements

pyHyperRI follows the platform support of the PyTorch build installed in the
environment. PyTorch documents requirements by operating system, Python version,
package manager, and compute platform, so choose a PyTorch build that matches
the target machine before installing pyHyperRI.

- **Python:** pyHyperRI requires Python 3.11 or later. The selected Python
  version must also satisfy the Python requirements for the chosen PyTorch build.
- **Operating system:** [Linux](https://pytorch.org/get-started/locally/#installing-on-linux),
  [macOS](https://pytorch.org/get-started/locally/#installing-on-macos), and
  [Windows](https://pytorch.org/get-started/locally/#installing-on-windows) are
  supported when a compatible PyTorch build is available for the selected Python
  and compute platform.
- **Compute platform:** CPU execution is supported through the CPU PyTorch
  build. GPU execution requires a PyTorch build for the target accelerator stack,
  such as [CUDA](https://docs.pytorch.org/docs/stable/notes/cuda.html),
  [ROCm](https://docs.pytorch.org/docs/stable/notes/hip.html),
  [XPU](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html), or
  [MPS](https://docs.pytorch.org/docs/stable/notes/mps.html).
- **Distributed execution:** distributed workflows use
  [`torch.distributed`](https://docs.pytorch.org/docs/stable/distributed.html#backends-that-come-with-pytorch).
  PyTorch distributed support differs by operating system and backend: Linux and
  macOS are described as stable, Windows as prototype. The backend Gloo is
  included in precompiled binaries for Linux and macOS, and NCCL is available
  with CUDA builds on Linux. Users may build PyTorch from source with MPI
  installed on the target machine to enable the MPI backend.

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

Use the official PyTorch installation selector at
[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) to
choose the command matching the operating system, package manager, Python
version, and compute platform. For CUDA-enabled devices, select the CUDA version
supported by the local driver and hardware.

## Install pyHyperRI

Install the package from PyPI.

```bash
pip install pyHyperRI
```

When using CUDA devices, install the CUDA optional dependency group.

```bash
pip install "pyHyperRI[cuda]"
```

## Install from Source

Install the package from a local source checkout.

```bash
git clone <TODO: REPO_URL>
cd py-hyper-ri
pip install -e .
```

For a source install with development tools, include the `dev` optional
dependency group.

```bash
pip install -e ".[dev]"
```

## Optional Dependencies

pyHyperRI provides optional dependency groups for common workflows:

- `notebook`: Jupyter notebook support.
- `cuda`: CUDA NUFFT support through `cufinufft`.
- `dev`: testing, formatting, packaging, and release tools.
- `docs`: documentation build tools.

Install one or more groups with pip:

```bash
pip install "pyHyperRI[docs,notebook]"
```

From a source checkout, combine optional groups with an editable install:

```bash
pip install -e ".[dev,docs,notebook]"
```

## Verify Installation

Check that the package and command-line tools are available.

```bash
hyri-config --help
hyri-imaging --help
```

<!--
## Common Issues

TBD.
-->
