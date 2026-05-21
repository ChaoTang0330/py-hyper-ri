# Hyperspectral Imaging in Radio Interferometry

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

The example below shows the radio galaxy 3C 353. Radio images can have a high
dynamic range, with faint extended emission coexisting with bright compact
features.

```{image} _images/background/ri-sky-3c353.png
:alt: Original 3C 353 ground-truth radio image.
:width: 22%
:align: center
```

```{figure} _images/background/ri-log-colorbar.png
:alt: Logarithmic intensity colour bar for the sample radio image.
:width: 22%
:align: center

Original ground-truth image of 3C 353. The image size is
$512 \times 512$ pixels.
```

Radio interferometers do not observe this image directly. At a given frequency,
each antenna pair measures a spatial Fourier component of the sky brightness,
called a visibility. The collection of sampled Fourier modes accumulated over
baselines and time is the uv coverage. The example below shows a simulated VLA
sampling pattern for the first channel of a hyperspectral observation.

```{figure} _images/background/uv-sampling-ch1.png
:alt: Simulated VLA uv-sampling pattern for channel 1.
:width: 30%
:align: center

Simulated VLA uv-sampling pattern for channel 1, corresponding to an
observation frequency of 1.00 GHz. Coordinates are given in units of wavelength.
```

For a single spectral channel, the sensing model is commonly written as

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
image, a back-projected image whose structure is shaped by the incomplete
Fourier coverage.

```{image} _images/background/ri-dirty-image-ch0.png
:alt: Dirty image associated with channel 1 for the 3C 353 simulation.
:width: 22%
:align: center
```

```{figure} _images/background/ri-dirty-colorbar-ch0.png
:alt: Linear intensity colour bar for the dirty image.
:width: 22%
:align: center

Dirty image associated with channel 1 of the simulated 3C 353 inverse problem,
shown on a linear intensity scale.
```

For hyperspectral imaging, the channel images are gathered into an image cube
$\mathbf{X} = (\boldsymbol{x}_1,\ldots,\boldsymbol{x}_L)$, the measurements
into $\mathbf{Y} = (\boldsymbol{y}_1,\ldots,\boldsymbol{y}_L)$, and the noise
into $\mathbf{N} = (\boldsymbol{n}_1,\ldots,\boldsymbol{n}_L)$. The
hyperspectral measurement operator is defined by applying each channel-specific
operator to its corresponding channel image,

$$
\Phi(\mathbf{X})
= ([\boldsymbol{\Phi}_l \boldsymbol{x}_l]_{1 \leq l \leq L}).
$$

The cube-level measurement model is then

$$
\mathbf{Y} = \Phi(\mathbf{X}) + \mathbf{N}.
$$

The measurement process is separable across channels, but the sky emission is
spectrally correlated. Joint-channel imaging uses this spectral structure to
improve reconstruction quality compared with treating each channel independently.

## Spectral Structure

For many continuum radio sources, the intensity varies smoothly with frequency
and is often approximated by a power-law model
([Thompson et al., 2017](#ref-thompson-2017)). A common
form relates each channel image to a reference-frequency image through a
spectral index map and a spectral curvature map:

$$
\overline{x}_l =
\overline{x}_0
\left(\frac{\nu_l}{\nu_0}\right)^{-\alpha + \beta \log(\nu_l / \nu_0)}.
$$

Here $\nu_l$ is the frequency of channel $l$, $\nu_0$ is a reference
frequency, $\alpha$ is the spectral index map, and $\beta$ is the spectral
curvature map. Related hyperspectral imaging methods also use spectral
modelling as prior information ([Junklewitz et al., 2015](#ref-junklewitz-2015);
[Ceccotti et al., 2023](#ref-ceccotti-2023)).

## Optimisation View

RI imaging is ill-posed: directly applying the adjoint operator to the data
produces a dirty image, which is useful for initialisation and diagnostics but
does not by itself solve the inverse problem. A common reconstruction strategy is
to seek an image, or image cube, that balances agreement with the measurements
and prior knowledge about plausible astronomical images:

$$
\min_{\mathbf{X}} f(\mathbf{X}) + \lambda r(\mathbf{X}).
$$

The data-fidelity term $f$ measures consistency with the visibility data, while
the regularisation term $r$ encodes prior information such as positivity,
sparsity, or learned image models. The parameter $\lambda$ controls the balance
between the two.

The first data-fidelity formulation is the unconstrained squared-error term:

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

Forward-backward algorithms apply when the objective can be split into a
differentiable term and a non-differentiable term: a gradient step handles the
differentiable term, and a proximity or denoising step handles the
non-differentiable term
([Combettes & Pesquet, 2011](#ref-combettes-2011)). Primal-dual algorithms are
useful when multiple non-differentiable terms must be handled separately,
because they introduce dual variables and split the terms across primal and dual
updates ([Condat, 2013](#ref-condat-2013)).

pyHyperRI follows this view through a modular design: measurement operators,
optimisers, and regularisers are implemented as separate components that can be
combined for different imaging algorithms.

## Algorithms

Currently, pyHyperRI implements the following algorithms:

| Algorithm | Optimisation backend | Joint-channel imaging | Regularisation |
| --- | --- | --- | --- |
| [uSARA](#ref-repetti-2020) | Forward-backward | No | $\ell_1$ sparsity in a redundant wavelet dictionary and positivity |
| [AIRI](#ref-terris-2023) | Plug-and-play forward-backward | No | AIRI denoisers |
| [cAIRI](#ref-terris-2025) | Plug-and-play primal-dual | No | AIRI denoisers with a data-fidelity constraint |
| [Hyper-uSARA](#ref-tang-2026) | Forward-backward | Yes | $\ell_{2,1}$ sparsity in a redundant wavelet dictionary and positivity |
| [HyperAIRI](#ref-tang-2026) | Plug-and-play forward-backward | Yes | HyperAIRI denoisers |

## References

(ref-tang-2026)=
- Tang, C., Dabbech, A., Jackson, A., & Wiaux, Y. (2026).
  [HyperAIRI: A Plug-and-play Algorithm for Precise Hyperspectral Image
  Reconstruction in Radio Interferometry](https://doi.org/10.3847/1538-4365/ae3149).
  *ApJS*, 283(1), 9.

(ref-thompson-2017)=
- Thompson, A. R., Moran, J. M., & Swenson, G. W. (2017).
  [Interferometry and Synthesis in Radio Astronomy](https://doi.org/10.1007/978-3-319-44431-4).
  Springer Nature.

(ref-junklewitz-2015)=
- Junklewitz, H., Bell, M. R., & Ensslin, T. A. (2015).
  [A new approach to multifrequency synthesis in radio
  interferometry](https://doi.org/10.1051/0004-6361/201423465).
  *A&A*, 581, A59.

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

(ref-terris-2025)=
- Terris, M., Tang, C., Jackson, A., & Wiaux, Y. (2025).
  [The AIRI plug-and-play algorithm for image reconstruction in
  radio-interferometry: variations and robustness](https://doi.org/10.1093/mnras/staf022).
  *MNRAS*, 537(2), 1608-1619.

(ref-repetti-2020)=
- Repetti, A., & Wiaux, Y. (2020).
  [A forward-backward algorithm for reweighted procedures: Application to
  radio-astronomical imaging](https://doi.org/10.1109/ICASSP40776.2020.9053284).
  *Proc. IEEE Int. Conf. Acoust., Speech Signal Process*, 1434-1438.

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
