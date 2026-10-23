<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/_images/logo/pyHyperRI_logo_text_white.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/_images/logo/pyHyperRI_logo_text_black.svg">
    <img src="docs/_images/logo/pyHyperRI_logo_text_black.svg" alt="pyHyperRI" height="96">
  </picture>
</p>

[![python](https://img.shields.io/badge/Python-3.11+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-enabled-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![license](https://img.shields.io/badge/license-BSD--3--Clause-brightgreen.svg)](LICENSE)
[![CI](https://github.com/ChaoTang0330/py-hyper-ri/actions/workflows/ci.yml/badge.svg)](https://github.com/ChaoTang0330/py-hyper-ri/actions/workflows/ci.yml)
[![docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://chaotang0330.github.io/py-hyper-ri/)

<!-- TODO: Add PyPI version, DOI/Zenodo, and coverage badges once those services are available. -->

pyHyperRI is a PyTorch-based platform for developing, testing, and
benchmarking radio-interferometric (RI) imaging algorithms. The current package
focuses on intensity imaging, with workflows for both monochromatic and
distributed hyperspectral reconstruction.

The package is installed as `pyHyperRI` and imported in Python as `hyri`.

## Key Features

- Modular components for RI algorithm development: measurement operators,
  regularisers, denoisers, and optimisers.
- Intensity imaging workflows for monochromatic and hyperspectral data.
- Implemented algorithms: [uSARA](https://doi.org/10.1093/mnras/stac2672),
  [AIRI](https://doi.org/10.1093/mnras/stac2672),
  [cAIRI](https://doi.org/10.1093/mnras/staf022),
  [Hyper-uSARA](https://doi.org/10.3847/1538-4365/ae3149), and
  [HyperAIRI](https://doi.org/10.3847/1538-4365/ae3149).
- PyTorch-native device support across CPU, CUDA, MPS, and XPU where available.
- Distributed hyperspectral imaging through `torch.distributed`.
- NUFFT measurement operators with FINUFFT and TorchKbNUFFT backends.

## Installation

Install PyTorch first by following the
[official PyTorch installation guide](https://pytorch.org/get-started/locally/),
because the correct build depends on the operating system and target device.

Then install pyHyperRI:

```bash
pip install pyHyperRI
```

If CUDA devices are available, install the CUDA optional dependency group:

```bash
pip install "pyHyperRI[cuda]"
```

Measurement Set dependency is available through the `ms` optional
dependency group:

```bash
pip install "pyHyperRI[ms]"
```

Use this option mainly on Linux. On other systems, the
prerequisite `casacore` usually needs to be installed separately before installing
`python-casacore`. See the documentation for detailed installation guide.

For a source install:

```bash
git clone <TODO: REPOSITORY_URL>
cd py-hyper-ri
pip install -e .
```

For development on Linux with CUDA devices, install all optional dependencies:

```bash
pip install -e ".[all]"
```

To verify the installation, run

```bash
hyri --help
```

## Minimal Example

Run an imaging job by providing the algorithm, super-resolution factor, and
measurement directory:

```bash
hyri imaging \
  --algorithm usara \
  --superresolution 1.5 \
  --data_file ./data/3c353_sim_vla_meas_hs
```

Use `hyri imaging --help` to inspect available command-line options.

## Documentation

Full documentation: [pyHyperRI documentation](https://chaotang0330.github.io/py-hyper-ri/).

The documentation covers the input data format, general usage, configuration
files, tutorials, and the API reference.

To build the documentation locally, the packages under the dependency group `docs` are required.

```bash
pip install -e ".[docs]"
make docs
```

## Citation

If you use pyHyperRI, please cite this software repository using the metadata in
[`CITATION.cff`](CITATION.cff). If you use specific implemented methods, please
also cite the relevant method papers listed in the documentation.

## License

pyHyperRI is distributed under the [BSD 3-Clause License](LICENSE).

## Dependencies

pyHyperRI builds on several external packages. The full dependency list is declared in
[`pyproject.toml`](pyproject.toml). Main dependencies and their associated references are listed below.

| Package | Reference |
| --- | --- |
| [Astropy](https://www.astropy.org/) | Astropy Collaboration et al., [The Astropy Project: Sustaining and Growing a Community-oriented Open-source Project and the Latest Major Release](https://doi.org/10.3847/1538-4357/ac7c74), *The Astrophysical Journal*, 2022. |
| [casacore](https://casacore.github.io/casacore/) and [python-casacore](https://casacore.github.io/python-casacore/) | van Diepen, [Casacore Table Data System and its use in the MeasurementSet](https://doi.org/10.1016/j.ascom.2015.06.002), *Astronomy and Computing*, 2015. |
| [FINUFFT](https://finufft.readthedocs.io/) and [cuFINUFFT](https://finufft.readthedocs.io/en/latest/c_gpu.html) | Barnett et al., [A Parallel Nonuniform Fast Fourier Transform Library Based on an Exponential of Semicircle Kernel](https://doi.org/10.1137/18M120885X), *SIAM Journal on Scientific Computing*, 2019; Shih et al., [cuFINUFFT: a load-balanced GPU library for general-purpose nonuniform FFTs](https://doi.org/10.1109/IPDPSW52791.2021.00105), *IPDPSW*, 2021. |
| [h5py](https://github.com/h5py/h5py) | Andrew Collette and contributors, [h5py](https://github.com/h5py/h5py), BSD 3-Clause licensed software, 2008. |
| [NumPy](https://numpy.org/) | Harris et al., [Array programming with NumPy](https://doi.org/10.1038/s41586-020-2649-2), *Nature*, 2020. |
| [onnx2torch](https://github.com/ENOT-AutoDL/onnx2torch) | ENOT developers et al., [onnx2torch](https://github.com/ENOT-AutoDL/onnx2torch), Apache-2.0 licensed software, 2021. |
| [PyTorch](https://pytorch.org/) | Paszke et al., [PyTorch: An Imperative Style, High-Performance Deep Learning Library](https://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library), NeurIPS 2019. |
| [PyTorch Wavelet Toolbox](https://pytorch-wavelet-toolbox.readthedocs.io/en/latest/) | Wolter et al., [ptwt - The PyTorch Wavelet Toolbox](https://www.jmlr.org/papers/v25/23-0636.html), *Journal of Machine Learning Research*, 2024. |
| [scikit-image](https://scikit-image.org/) | van der Walt et al., [scikit-image: image processing in Python](https://doi.org/10.7717/peerj.453), *PeerJ*, 2014. |
| [SciPy](https://scipy.org/) | Virtanen et al., [SciPy 1.0: fundamental algorithms for scientific computing in Python](https://doi.org/10.1038/s41592-019-0686-2), *Nature Methods*, 2020. |
| [TorchKbNUFFT](https://torchkbnufft.readthedocs.io/en/stable/) | Muckley et al., [TorchKbNufft: A High-Level, Hardware-Agnostic Non-Uniform Fast Fourier Transform](https://mmuckley.github.io/assets/publications/2020muckleytorchkbnufft.pdf), *ISMRM Workshop on Data Sampling \& Image Reconstruction*, 2020. |
