# Introduction

Radio interferometry (RI) is an astronomical imaging technique that combines
measurements from arrays of antennas to form high-resolution images of the radio
sky. Instead of observing an image directly, an interferometer samples spatial
Fourier components of the sky brightness, called visibilities. These
measurements are incomplete and noisy, so RI imaging is an inverse problem.

Modern instruments observe across wide frequency bands. Hyperspectral RI imaging
therefore aims to reconstruct an image cube, with one image per spectral
channel, from the corresponding visibility data. This is useful because radio
emission varies with frequency and can reveal physical properties of the source,
such as spectral index. This page provides a general introduction to the RI
imaging problem and the algorithms implemented in pyHyperRI.

## RI Imaging Model

Radio images can have a high dynamic range, with faint extended emission 
coexisting with bright compact features. The example below shows the radio galaxy 
3C 353.

```{image} _images/background/ri-sky-3c353.png
:width: 20%
:align: center
```

```{figure} _images/background/ri-log-colorbar.png
:width: 18%
:align: center

Example image for radio galaxy 3C 353.
```

Radio interferometers do not capture such images directly. At a given frequency,
each antenna pair measures a spatial Fourier component of the sky brightness,
called a visibility. The collection of sampled Fourier modes accumulated over
antenna pairs and time is the $uv$ coverage. The example below shows a simulated 
[Very Large Array (VLA)](https://public.nrao.edu/telescopes/vla/) $uv$
sampling pattern.

```{figure} _images/background/uv-sampling-ch1.png
:width: 20%
:align: center

Simulated VLA uv-sampling pattern, corresponding to an
observation frequency of 1.00 GHz. Coordinates are given in units of wavelength.
```

For a single spectral channel, this linear sensing process is commonly written as

$$
\boldsymbol{y}_l
= \boldsymbol{\Phi}_l \overline{\boldsymbol{x}}_l
+ \boldsymbol{n}_l,
\quad \mathrm{with}\quad
\boldsymbol{\Phi}_l = \mathbf{G}_l \mathbf{F}\mathbf{P},
$$

where $\boldsymbol{y}_l$ is the visibility vector,
$\overline{\boldsymbol{x}}_l$ is the unknown sky image,
$\boldsymbol{\Phi}_l$ is the RI measurement operator for channel $l$, and
$\boldsymbol{n}_l$ is measurement noise. The operator combines sparse
non-uniform Fourier interpolation $\mathbf{G}_l$, the Fourier transform
$\mathbf{F}$, and zero-padding and interpolation-kernel correction
$\mathbf{P}$. This follows the standard RI measurement view described by
[Thompson et al. (2017)](#ref-thompson-2017).

Applying the adjoint measurement operator to the visibility data gives the dirty
image $\mathrm{Re} \{ \boldsymbol{\Phi}_l^\dagger \boldsymbol{y}_l \}$, a
back-projected image whose structure is shaped by the incomplete Fourier coverage.

```{image} _images/background/ri-dirty-image-ch0.png
:width: 20%
:align: center
```

```{figure} _images/background/ri-dirty-colorbar-ch0.png
:width: 18%
:align: center

Dirty image associated with the example image and uv sampling pattern.
```

For hyperspectral imaging, the channel images are gathered into an image cube
$\mathbf{X} = (\boldsymbol{x}_1,\ldots,\boldsymbol{x}_L)$, the measurements
into $\mathbf{Y} = (\boldsymbol{y}_1,\ldots,\boldsymbol{y}_L)$, and the noise
into $\mathbf{N} = (\boldsymbol{n}_1,\ldots,\boldsymbol{n}_L)$. 

```{figure} _images/background/uv-sampling-ch36.png
:width: 20%
:align: center

Simulated VLA uv-sampling pattern, corresponding to an
observation frequency of 1.70 GHz. Coordinates are given in units of wavelength.
```

The sampling pattern of each spectral channel is scaled by the observation frequency 
and may be subject to different flagging patterns, so the measurement operator is
channel-specific. The hyperspectral measurement operator can then be formulated
as

$$
\Phi(\mathbf{X})
= ([\boldsymbol{\Phi}_l \boldsymbol{x}_l]_{1 \leq l \leq L}).
$$

The hyperspectral measurement model is therefore written as

$$
\mathbf{Y} = \Phi(\mathbf{X}) + \mathbf{N}.
$$

## Spectral Structure

The measurement process is separable across channels, but the sky emission is
spectrally correlated.
For many continuum radio sources, the intensity varies smoothly with frequency
in certain ranges and is often approximated by a power-law model
([Thompson et al., 2017](#ref-thompson-2017)). A common
form relates each channel image to a reference-frequency image through a
spectral index map and a spectral curvature map:

$$
\overline{x}_l =
\overline{x}_0
\left(\frac{\nu_l}{\nu_0}\right)^{\alpha + \beta \log(\nu_l / \nu_0)}.
$$

Here $\nu_l$ is the frequency of channel $l$, $\nu_0$ is a reference
frequency, $\alpha$ is the spectral index map, and $\beta$ is the spectral
curvature map. 

## Optimisation View

Because of the incomplete and noisy Fourier measurements, RI imaging is ill-posed.
Directly applying the adjoint operator to the data loses substantial
structure, as shown in the example dirty image. To solve this inverse problem,
prior information about the image is needed to regularise the solution. From the 
optimisation perspective, RI imaging can be formulated as finding the solution of
a minimisation problem of the form

$$
\min_{\mathbf{X}} f(\mathbf{X}) + \lambda r(\mathbf{X}).
$$

The data-fidelity term $f$ measures consistency with the visibility data, while
the regularisation term $r$ encodes prior information such as positivity,
sparsity, or learned image models.
The spectral correlation can also serve as prior knowledge to improve the
reconstruction quality. Some hyperspectral imaging methods choose to inject the 
power-law spectral model as prior information 
([Junklewitz et al., 2015](#ref-junklewitz-2015); 
[Ceccotti et al., 2023](#ref-ceccotti-2023); [Tang et al., 2026](#ref-tang-2026)). 
The low-rankness among the channel images is also a practical choice 
([Thouvenin et al., 2023](#ref-thouvenin-2023)). The parameter $\lambda$
controls the balance between the two.

Due to the Gaussian nature of the measurement noise, there are two common
data-fidelity formulations. The first one is the squared $\ell_2$ term:

$$
f(\mathbf{X})
= \frac{1}{2}\left\|\Phi(\mathbf{X}) - \mathbf{Y}\right\|_{\mathrm{F}}^2,
$$

where $\|\cdot\|_{\mathrm{F}}$ denotes the Frobenius norm.

The second formulation uses an indicator data-fidelity term with channel-wise
visibility-domain residual bounds:

$$
f(\mathbf{X})
= \sum_{l=1}^{L}
\iota_{\mathcal{B}_l}
\left(\boldsymbol{\Phi}_l\boldsymbol{x}_l\right),
\quad
\mathcal{B}_l
= \left\{
\boldsymbol{z}_l :
\left\|\boldsymbol{z}_l - \boldsymbol{y}_l\right\|_2
\leq \epsilon_l
\right\}.
$$

Here $\mathcal{B}_l$ is the channel-wise data-fidelity ball and $\epsilon_l$
is the residual bound associated with the noise level in channel $l$.

There are various frameworks for solving this optimisation problem.
Forward-backward algorithms apply when the objective can be split into a
differentiable term and a non-differentiable term: a gradient step handles the
differentiable term, and a proximity or denoising step handles the
non-differentiable term
([Combettes & Pesquet, 2011](#ref-combettes-2011)). Primal-dual algorithms are
useful when multiple non-differentiable terms must be handled separately,
because they introduce dual variables and split the terms across primal and dual
updates ([Condat, 2013](#ref-condat-2013)).
More generally, the CLEAN family of algorithms can also be viewed as an optimisation
method based on matching pursuit, which iteratively identifies and subtracts 
the contribution of the brightest source component from the back-projected images.

pyHyperRI follows this view through a modular design: measurement operators,
optimisers, and regularisers are implemented as separate components that can be
combined for different imaging algorithms.

(algorithms)=
## Algorithms

Currently, pyHyperRI implements the following algorithms:

| Algorithm | Optimisation backend | Joint-channel imaging | Regularisation |
| --- | --- | --- | --- |
| uSARA ([Terris et al., 2023](#ref-terris-2023)) | Forward-backward | No | $\ell_1$ sparsity in a redundant wavelet dictionary and positivity |
| AIRI ([Terris et al., 2023](#ref-terris-2023)) | Plug-and-play forward-backward | No | AIRI denoisers |
| cAIRI ([Terris et al., 2025](#ref-terris-2025)) | Plug-and-play primal-dual | No | AIRI denoisers with a data-fidelity constraint |
| Hyper-uSARA ([Tang et al., 2026](#ref-tang-2026)) | Forward-backward | Yes | $\ell_{2,1}$ sparsity in a redundant wavelet dictionary and positivity |
| HyperAIRI ([Tang et al., 2026](#ref-tang-2026)) | Plug-and-play forward-backward | Yes | HyperAIRI denoisers |

## Dependencies

pyHyperRI builds on established scientific Python and radio-astronomy
software. The full runtime and optional dependency list is declared in
{repo-file}`pyproject.toml <pyproject.toml>`. The main dependencies are
listed below.

| Dependency | Main use in pyHyperRI | Reference |
| --- | --- | --- |
| [Astropy](https://www.astropy.org/) | FITS image I/O | [Astropy Collaboration et al. (2022)](#ref-astropy-2022) |
| [casacore](https://casacore.github.io/casacore/) and [python-casacore](https://casacore.github.io/python-casacore/) | Optional Measurement Set table reading | [van Diepen (2015)](#ref-vandiepen-2015) |
| [FINUFFT](https://finufft.readthedocs.io/), [cuFINUFFT](https://finufft.readthedocs.io/en/latest/c_gpu.html), and [pytorch-finufft](https://flatironinstitute.github.io/pytorch-finufft/) | CPU and GPU non-uniform Fast Fourier transforms based on FINUFFT and cuFINUFFT | [Barnett et al. (2019)](#ref-barnett-2019); [Shih et al. (2021)](#ref-shih-2021) |
| [h5py](https://github.com/h5py/h5py) | HDF5 measurement data I/O | [Andrew Collette and contributors (2008)](#ref-h5py-2008) |
| [NumPy](https://numpy.org/) | Intermediate data structure | [Harris et al. (2020)](#ref-harris-2020) |
| [onnx2torch](https://github.com/ENOT-AutoDL/onnx2torch) | Convert networks in ONNX format to PyTorch modules | [ENOT developers et al. (2021)](#ref-onnx2torch-2021) |
| [PyTorch](https://pytorch.org/) | Tensor computation, accelerator support, and distributed execution | [Paszke et al. (2019)](#ref-paszke-2019) |
| [PyTorch Wavelet Toolbox](https://pytorch-wavelet-toolbox.readthedocs.io/en/latest/) | Wavelet transforms | [Wolter et al. (2024)](#ref-wolter-2024) |
| [scikit-image](https://scikit-image.org/) | Image operations | [van der Walt et al. (2014)](#ref-vanderwalt-2014) |
| [SciPy](https://scipy.org/) | Measurement data I/O and numerical routines | [Virtanen et al. (2020)](#ref-virtanen-2020) |
| [TorchKbNUFFT](https://torchkbnufft.readthedocs.io/en/stable/) | Purely PyTorch-based implementation for non-uniform Fast Fourier transforms | [Muckley et al. (2020)](#ref-muckley-2020) |

## Limitations

The current implementation is intended primarily as a platform for developing,
testing, and benchmarking intensity-imaging algorithms. The main current
limitations are:

- **No $w$-correction.** The implemented measurement operators do not currently
  include wide-field correction terms. Support for $w$-stacking is planned for 
  a future release.
- **Parallel execution for multiple channel on one Device.** Currently, channels 
  share a single device are processed sequentially. Exploring the proper 
  parallelisation strategy can further improve the runtime performance.
- **Intensity imaging only.** pyHyperRI currently focuses on total-intensity
  imaging. Polarisation imaging and calibrations are not yet supported.

## References

(ref-h5py-2008)=
- Andrew Collette and contributors. (2008).
  [h5py](https://github.com/h5py/h5py).
  Software, BSD 3-Clause license.

(ref-combettes-2011)=
- Combettes, P. L., & Pesquet, J.-C. (2011).
  [Proximal Splitting Methods in Signal
  Processing](https://doi.org/10.1007/978-1-4419-9569-8_10).
  In *Fixed-Point Algorithms for Inverse Problems in Science and Engineering*,
  185-212.

(ref-condat-2013)=
- Condat, L. (2013).
  [A primal-dual splitting method for convex optimization involving
  Lipschitzian, proximable and linear composite terms](https://doi.org/10.1007/s10957-012-0245-9).
  *J. Optim. Theory Appl.*, 158(2), 460-479.

(ref-vanderwalt-2014)=
- van der Walt, S., Schonberger, J. L., Nunez-Iglesias, J., Boulogne, F.,
  Warner, J. D., Yager, N., Gouillart, E., & Yu, T. (2014).
  [scikit-image: image processing in Python](https://doi.org/10.7717/peerj.453).
  *PeerJ*, 2, e453.

(ref-junklewitz-2015)=
- Junklewitz, H., Bell, M. R., & Ensslin, T. A. (2015).
  [A new approach to multifrequency synthesis in radio
  interferometry](https://doi.org/10.1051/0004-6361/201423465).
  *A&A*, 581, A59.

(ref-vandiepen-2015)=
- van Diepen, G. N. J. (2015).
  [Casacore Table Data System and its use in the
  MeasurementSet](https://doi.org/10.1016/j.ascom.2015.06.002).
  *Astronomy and Computing*, 12, 174-180.

(ref-thompson-2017)=
- Thompson, A. R., Moran, J. M., & Swenson, G. W. (2017).
  [Interferometry and Synthesis in Radio Astronomy](https://doi.org/10.1007/978-3-319-44431-4).
  Springer Nature.

(ref-barnett-2019)=
- Barnett, A. H., Magland, J., & af Klinteberg, L. (2019).
  [A parallel nonuniform fast Fourier transform library based on an
  exponential of semicircle kernel](https://doi.org/10.1137/18M120885X).
  *SIAM Journal on Scientific Computing*, 41(5), C479-C504.

(ref-paszke-2019)=
- Paszke, A., Gross, S., Massa, F., et al. (2019).
  [PyTorch: An imperative style, high-performance deep learning
  library](https://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library).
  *Advances in Neural Information Processing Systems*, 32.

(ref-harris-2020)=
- Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020).
  [Array programming with NumPy](https://doi.org/10.1038/s41586-020-2649-2).
  *Nature*, 585, 357-362.

(ref-muckley-2020)=
- Muckley, M. J., Riemenschneider, B., Radmanesh, A., Kim, S., Jeong, G.,
  Ko, J., Jun, Y., Shin, H., Hwang, D., Mostapha, M., & Knoll, F. (2020).
  [TorchKbNufft: A high-level, hardware-agnostic non-uniform fast Fourier
  transform](https://mmuckley.github.io/assets/publications/2020muckleytorchkbnufft.pdf).
  *ISMRM Workshop on Data Sampling & Image Reconstruction*.

(ref-repetti-2020)=
- Repetti, A., & Wiaux, Y. (2020).
  [A forward-backward algorithm for reweighted procedures: Application to
  radio-astronomical imaging](https://doi.org/10.1109/ICASSP40776.2020.9053284).
  *Proc. IEEE Int. Conf. Acoust., Speech Signal Process*, 1434-1438.

(ref-virtanen-2020)=
- Virtanen, P., Gommers, R., Oliphant, T. E., et al. (2020).
  [SciPy 1.0: fundamental algorithms for scientific computing in
  Python](https://doi.org/10.1038/s41592-019-0686-2).
  *Nature Methods*, 17, 261-272.

(ref-onnx2torch-2021)=
- ENOT developers, Kalgin, I., Yanchenko, A., Ivanov, P., &
  Goncharenko, A. (2021).
  [onnx2torch](https://github.com/ENOT-AutoDL/onnx2torch).
  Software, Apache-2.0 license.

(ref-shih-2021)=
- Shih, Y.-H., Wright, G. B., Anden, J., Blaschke, J., & Barnett, A. H. (2021).
  [cuFINUFFT: a load-balanced GPU library for general-purpose nonuniform
  FFTs](https://doi.org/10.1109/IPDPSW52791.2021.00105).
  *2021 IEEE International Parallel and Distributed Processing Symposium
  Workshops*, 688-697.

(ref-astropy-2022)=
- Astropy Collaboration, Price-Whelan, A. M., Lim, P. L., et al. (2022).
  [The Astropy Project: Sustaining and Growing a Community-oriented
  Open-source Project and the Latest Major Release](https://doi.org/10.3847/1538-4357/ac7c74).
  *ApJ*, 935(2), 167.

(ref-ceccotti-2023)=
- Ceccotti, E., Offringa, A. R., Koopmans, L. V. E., et al. (2023).
  [A novel radio imaging method for physical spectral index
  modelling](https://doi.org/10.1093/mnras/stad2465).
  *MNRAS*, 525(3), 3946-3962.

(ref-terris-2023)=
- Terris, M., Dabbech, A., Tang, C., & Wiaux, Y. (2023).
  [Image reconstruction algorithms in radio interferometry: From handcrafted to
  learned regularization denoisers](https://doi.org/10.1093/mnras/stac2672).
  *MNRAS*, 518(1), 604-622.

(ref-thouvenin-2023)=
- Thouvenin, P.-A., Abdulaziz, A., Dabbech, A., Repetti, A., & Wiaux, Y. (2023).
  [Parallel faceted imaging in radio interferometry via proximal splitting
  (Faceted HyperSARA): I. Algorithm and simulations](https://doi.org/10.1093/mnras/stac1521).
  *MNRAS*, 521(1), 1-19.

(ref-wolter-2024)=
- Wolter, M., Blanke, F., Garcke, J., & ten Brink, S. (2024).
  [ptwt - The PyTorch Wavelet Toolbox](https://www.jmlr.org/papers/v25/23-0636.html).
  *Journal of Machine Learning Research*, 25(80), 1-7.

(ref-terris-2025)=
- Terris, M., Tang, C., Jackson, A., & Wiaux, Y. (2025).
  [The AIRI plug-and-play algorithm for image reconstruction in
  radio-interferometry: variations and robustness](https://doi.org/10.1093/mnras/staf022).
  *MNRAS*, 537(2), 1608-1619.

(ref-tang-2026)=
- Tang, C., Dabbech, A., Jackson, A., & Wiaux, Y. (2026).
  [HyperAIRI: A Plug-and-play Algorithm for Precise Hyperspectral Image
  Reconstruction in Radio Interferometry](https://doi.org/10.3847/1538-4365/ae3149).
  *ApJS*, 283(1), 9.
