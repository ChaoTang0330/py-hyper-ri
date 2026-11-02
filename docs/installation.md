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
git clone https://github.com/ChaoTang0330/py-hyper-ri.git
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
- `ms`: Measurement Set conversion support through `python-casacore`.
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

(measurement-set-support)=
### Measurement Set Support

Measurement Set conversion requires `python-casacore`, which depends on native
casacore libraries. Installation support depends on the operating system and
package manager. See the
[`python-casacore` documentation](https://casacore.github.io/python-casacore/),
the
[`python-casacore` conda-forge package](https://anaconda.org/conda-forge/python-casacore),
and the
[`casacore` installation guide](https://github.com/casacore/casacore#installation)
for upstream installation details.

On Linux, install the optional pip dependency group:

```bash
pip install "pyHyperRI[ms]"
```

From a source checkout on Linux, use:

```bash
pip install -e ".[ms]"
```

On macOS, using conda-forge to install `python-casacore` is simpler because it
also installs the native `casacore` dependency. Users can install the rest of
pyHyperRI with pip and install `python-casacore` with conda:

```bash
conda install conda-forge::python-casacore
```

Then verify the installation with:

```bash
python -c "import casacore"
```

If the import fails with a GSL library error, reinstall GSL from conda-forge:

```bash
conda install conda-forge::gsl --force-reinstall
```

Windows users need a
working `casacore` installation before installing `python-casacore`; refer to
the [`casacore` installation guide](https://github.com/casacore/casacore#installation)
for upstream instructions.

(docker-images)=
## Docker Images

We provide two Docker images through
[GitHub Container Registry](https://github.com/ChaoTang0330/py-hyper-ri/pkgs/container/py-hyper-ri),
both based on official PyTorch images with CUDA 12.8 for `linux/amd64`:

| Image tag | Purpose |
| --- | --- |
| `v1.0.0-dev` | Development with editable source, all optional dependencies, tests, notebooks and documentation tools. |
| `v1.0.0-runtime` | Running imaging workflows with the installed package and CUDA/Measurement Set dependencies. |

GPU use requires a compatible NVIDIA driver and
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
on the host.

For instructions on running containers, see {ref}`Run in a Container <run-in-a-container>`.

### Build the Images

To build locally, run these commands from the repository root:

```bash
docker build --platform linux/amd64 --target development -t pyhyperri:v1.0.0-dev .
docker build --platform linux/amd64 --target runtime -t pyhyperri:v1.0.0-runtime .
```

The same Dockerfile can be built with
[Podman](https://docs.podman.io/en/latest/markdown/podman-build.1.html) by replacing
`docker build` with `podman build`.

## Verify Installation

Check that the package and top-level command-line tool are available.

```bash
hyri -h
```

<!--
## Common Issues

TBD.
-->
