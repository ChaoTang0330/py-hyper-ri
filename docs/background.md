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
such as spectral index. This page follows the high-level modelling and algorithm
overview used in
[Tang et al. (2026)](#ref-tang-2026), the HyperAIRI paper.

## RI Imaging Model

For a single spectral channel, the measurement process is commonly written as

$$
y = \Phi \overline{x} + n,
$$

where $y$ is the visibility vector, $\overline{x}$ is the unknown sky image,
$\Phi$ is the RI measurement operator, and $n$ is measurement noise. In this
package, $\Phi$ models the non-uniform Fourier sampling and associated
weighting used by the imager. This follows the standard RI measurement view
described by
[Thompson et al. (2017)](#ref-thompson-2017).

For hyperspectral imaging, the channel images are gathered into an image cube
$X$, and the measurements into a visibility cube $Y$. The model becomes

$$
Y = \Phi(X) + N.
$$

The measurement operator is separable across channels, but the channels are not
independent in the sky: radio emission usually changes smoothly with frequency.
Joint-channel imaging uses this spectral structure to improve reconstruction
quality compared with treating each channel independently
([Tang et al., 2026](#ref-tang-2026)).

## Spectral Structure

For many continuum radio sources, the intensity varies smoothly with frequency
and is often approximated by a power-law model
([Thompson et al., 2017](#ref-thompson-2017)). A common
form relates each channel image to a reference-frequency image through a
spectral index map, and optionally a spectral curvature map; the curvature term
is used in multi-scale multi-frequency modelling
([Rau & Cornwell, 2011](#ref-rau-2011)) and in
the HyperAIRI hyperspectral RI model
([Tang et al., 2026](#ref-tang-2026)):

$$
\overline{x}_l =
\overline{x}_0
\left(\frac{\nu_l}{\nu_0}\right)^{-\alpha + \beta \log(\nu_l / \nu_0)}.
$$

Here $\nu_l$ is the frequency of channel $l$, $\nu_0$ is a reference
frequency, $\alpha$ is the spectral index map, and $\beta$ is the spectral
curvature map. This model is one way to express the fact that neighboring
channels carry related information about the same astronomical scene. Related
wide-band RI methods also use spectral modelling as prior information
([Junklewitz et al., 2015](#ref-junklewitz-2015);
[Ceccotti et al., 2023](#ref-ceccotti-2023)).

## Optimization View

RI imaging is ill-posed: directly applying the adjoint operator to the data
produces a dirty image, which is useful for initialization and diagnostics but
does not by itself solve the inverse problem. A common reconstruction strategy is
to seek an image, or image cube, that balances agreement with the measurements
and prior knowledge about plausible astronomical images:

$$
\min_X f(X) + \lambda r(X).
$$

The data-fidelity term $f$ measures consistency with the visibility data, while
the regularization term $r$ encodes prior information such as positivity,
sparsity, or learned denoising. The parameter $\lambda$ controls the balance
between the two.

pyHyperRI follows this view through a modular design: measurement operators,
optimisers, and regularisers are implemented as separate components that can be
combined for different imaging algorithms.

Forward-backward algorithms apply when the objective can be split into a
differentiable term and a non-differentiable term: a gradient step handles the
differentiable term, and a proximity or denoising step handles the
non-differentiable term
([Combettes & Pesquet, 2011](#ref-combettes-2011)). Primal-dual algorithms are
useful when multiple non-differentiable terms must be handled separately,
because they introduce dual variables and split the terms across primal and dual
updates ([Condat, 2013](#ref-condat-2013)).

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

(ref-rau-2011)=
- Rau, U., & Cornwell, T. J. (2011).
  [A multi-scale multi-frequency deconvolution algorithm for synthesis imaging in
  radio interferometry](https://doi.org/10.1051/0004-6361/201117104).
  *A&A*, 532, A71.

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

(ref-abdulaziz-2019)=
- Abdulaziz, A., Dabbech, A., & Wiaux, Y. (2019).
  [Wideband super-resolution imaging in radio interferometry via low rankness and
  joint average sparsity models (HyperSARA)](https://doi.org/10.1093/mnras/stz2117).
  *MNRAS*, 489(1), 1230-1248.

(ref-thouvenin-2023)=
- Thouvenin, P.-A., Abdulaziz, A., Dabbech, A., Repetti, A., & Wiaux, Y. (2023).
  [Parallel faceted imaging in radio interferometry via proximal splitting
  (Faceted HyperSARA): I. Algorithm and simulations](https://doi.org/10.1093/mnras/stac1521).
  *MNRAS*, 521(1), 1-19.
