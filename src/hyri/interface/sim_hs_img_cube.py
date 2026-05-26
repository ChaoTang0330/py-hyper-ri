"""Simulate hyperspectral image cubes from a reference intensity image.

This module provides Gaussian-field helpers and a power-law spectral model used
to generate synthetic hyperspectral radio images for package-level tutorials
and simulation experiments.
"""

import numpy as np
import scipy
from scipy.fftpack import fft2


def gauss_2d_func(sig_x, sig_y, theta):
    """Return a callable unnormalised rotated 2D Gaussian.

    Args:
        sig_x (float): Standard deviation along the local x-axis.
        sig_y (float): Standard deviation along the local y-axis.
        theta (float): Anticlockwise rotation angle in radians.

    Returns:
        Callable[[np.ndarray], np.ndarray]: Function that evaluates the
        Gaussian on an array of coordinates with shape ``(N, 2)``.
    """
    a = np.cos(theta) ** 2 / (2 * sig_x**2) + np.sin(theta) ** 2 / (2 * sig_y**2)
    b = -np.sin(2 * theta) / (4 * sig_x**2) + np.sin(2 * theta) / (4 * sig_y**2)
    c = np.sin(theta) ** 2 / (2 * sig_x**2) + np.cos(theta) ** 2 / (2 * sig_y**2)

    def rho(h):
        """Evaluate the Gaussian on coordinate samples."""
        x, y = h[:, 0], h[:, 1]
        return np.exp(-(a * x**2 + 2 * b * x * y + c * y**2))

    return rho


def gauss_2d_kernel(sig_x, sig_y, theta, kernel_size):
    """Generate a normalised rotated 2D Gaussian kernel.

    Args:
        sig_x (float): Standard deviation along the local x-axis.
        sig_y (float): Standard deviation along the local y-axis.
        theta (float): Anticlockwise rotation angle in radians.
        kernel_size (int): Width and height of the square output kernel.

    Returns:
        np.ndarray: Normalised Gaussian kernel with shape
        ``(kernel_size, kernel_size)``.
    """
    gauss_func = gauss_2d_func(sig_x, sig_y, theta)
    X, Y = np.meshgrid(
        np.linspace(-kernel_size / 2, kernel_size / 2 - 1, kernel_size),
        np.linspace(-kernel_size / 2, kernel_size / 2 - 1, kernel_size),
    )
    XY = np.concatenate(
        (X.reshape(-1, 1, order="F"), Y.reshape(-1, 1, order="F")), axis=1
    )
    kernel = np.reshape(gauss_func(XY), (kernel_size, kernel_size), order="F")
    kernel = kernel / np.sum(kernel)

    return kernel


def stationary_gaussian_process(im_size, rho):
    """Draw stationary Gaussian fields on an image grid.

    Args:
        im_size (Tuple[int, int]): Output image size as ``(height, width)``.
        rho (Callable[[np.ndarray], np.ndarray]): Covariance function evaluated
            on coordinate offsets with shape ``(N, 2)``.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Two independent
        Gaussian fields followed by the x and y grid coordinates.

    Raises:
        ValueError: If the circulant embedding has negative eigenvalues below
            the numerical tolerance.

    Notes:
        This implementation follows the block-circulant embedding used in the
        `Faceted-HyperSARA MATLAB code`_.

        .. _Faceted-HyperSARA MATLAB code:
           https://github.com/basp-group/Faceted-HyperSARA/blob/master/lib/
           generate_data/stationary_Gaussian_process.m
    """
    # Create the grid on which the field is sampled.
    tx = np.linspace(0, im_size[1] - 1, im_size[1])
    ty = np.linspace(0, im_size[0] - 1, im_size[0])
    X, Y = np.meshgrid(tx, ty)
    Rows = rho(
        np.concatenate(
            (X.reshape(-1, 1, order="F") - tx[0], Y.reshape(-1, 1, order="F") - ty[0]),
            axis=1,
        )
    )
    Cols = rho(
        np.concatenate(
            (tx[0] - X.reshape(-1, 1, order="F"), Y.reshape(-1, 1, order="F") - ty[0]),
            axis=1,
        )
    )
    Rows = Rows.reshape(*im_size, order="F")
    Cols = Cols.reshape(*im_size, order="F")

    # Create the first row of the block-circulant matrix with circulant blocks.
    BlkCirc_row = np.block(
        [[Rows, Cols[:, :0:-1]], [Cols[:0:-1, :], Rows[:0:-1, :0:-1]]]
    )
    # Compute the embedding eigenvalues.
    lam = np.real(fft2(BlkCirc_row)) / (2 * im_size[0] - 1) / (2 * im_size[1] - 1)
    if np.abs(np.min(lam[lam < 0])) > 1e-15:
        raise ValueError(
            "Could not find positive definite embedding!",
            np.abs(np.min(lam[lam < 0])),
            lam.dtype,
        )
    else:
        lam[lam < 0] = 0
        lam = np.sqrt(lam)

    # Generate complex Gaussian noise with the target covariance spectrum.
    F = fft2(
        lam
        * (
            np.random.randn(2 * im_size[0] - 1, 2 * im_size[1] - 1)
            + 1j * np.random.randn(2 * im_size[0] - 1, 2 * im_size[1] - 1)
        )
    )
    # Extract the sub-block with the desired covariance.
    F = F[0 : im_size[0], 0 : im_size[1]]
    field1 = np.real(F)
    # The real and imaginary parts are independent fields with this covariance.
    field2 = np.imag(F)

    return field1, field2, tx, ty


def log_blur(img, kernel):
    """Blur an image and map it to base-10 logarithmic scale.

    Args:
        img (np.ndarray): Two-dimensional reference image.
        kernel (np.ndarray): Convolution kernel.

    Returns:
        np.ndarray: Log-scaled blurred image.
    """
    return np.log10(
        scipy.signal.fftconvolve(img, kernel, mode="same").clip(0, 1) + 1e-15
    )


def gen_spectral_idx(im_log_blur, gauss_field):
    """Generate a spectral-index map from image structure and a random field.

    Args:
        im_log_blur (np.ndarray): Log-scaled blurred reference image.
        gauss_field (np.ndarray): Gaussian random field with the same spatial
            shape as ``im_log_blur``.

    Returns:
        np.ndarray: Spectral-index map with weak-region values set to zero.
    """
    # Normalise the Gaussian-process realisation.
    gauss_field = gauss_field / np.max(np.abs(gauss_field))

    # Bias the spectral index towards negative values and mask weak regions.
    spectral_idx = gauss_field + im_log_blur
    spectral_idx = spectral_idx - np.max(spectral_idx) - 0.3
    spectral_idx[spectral_idx < -5] = 0
    spectral_idx = np.nan_to_num(spectral_idx, posinf=0, neginf=0)

    return spectral_idx


def gen_curvature_map(gauss_field, curv_scale):
    """Generate a spectral-curvature map from a Gaussian random field.

    Args:
        gauss_field (np.ndarray): Gaussian random field.
        curv_scale (float): Maximum absolute curvature scale.

    Returns:
        np.ndarray: Curvature map with negative outliers clipped to zero.
    """
    # Rescale the field to the requested curvature amplitude.
    curv = curv_scale * gauss_field / np.max(np.abs(gauss_field))
    curv[curv < -curv_scale] = 0
    curv = np.nan_to_num(curv, posinf=0, neginf=0)

    return curv


def sim_hs_img_cube(
    img: np.ndarray,
    freqs: np.ndarray,
    sig_blur: float = 10,
    sig_gauss_field: float = 10,
    curv_scale: float = 0.5,
):
    """Simulate a hyperspectral image cube from a reference image.

    The cube follows a spatially varying power-law model with curvature,

    .. math::
        I_\nu = I_{\nu_0}
            \left(\frac{\nu}{\nu_0}\right)^{
                \alpha + \beta \log(\nu / \nu_0)
            },

    where the first frequency in ``freqs`` is used as the reference frequency
    :math:`\nu_0`.

    Args:
        img (np.ndarray): Two-dimensional reference image.
        freqs (np.ndarray): Frequency samples in Hz.
        sig_blur (float, optional): Standard deviation of the Gaussian kernel
            used to blur ``img`` before spectral-index generation. Default: 10.
        sig_gauss_field (float, optional): Standard deviation of the Gaussian
            covariance used for the random spectral fields. Default: 10.
        curv_scale (float, optional): Curvature-map amplitude scale.
            Default: 0.5.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Simulated image cube with
        frequency as the first axis, spectral-index map, and curvature map.

    References:
        Thouvenin, P.-A., Abdulaziz, A., Dabbech, A., Repetti, A., & Wiaux, Y.
        (2023). `Parallel faceted imaging in radio interferometry via proximal
        splitting (Faceted HyperSARA): I. Algorithm and simulations
        <https://doi.org/10.1093/mnras/stac1521>`_. *MNRAS*, 521(1), 1-19.
    """

    # Generate the image-dependent component of the spectral-index map.
    blur_kernel = gauss_2d_kernel(sig_blur, sig_blur, 0, 64)
    img_log_blur = log_blur(img, blur_kernel)

    # Draw two independent Gaussian fields for index and curvature maps.
    gauss_func_field = gauss_2d_func(sig_gauss_field, sig_gauss_field, 0)
    gauss_field1, gauss_field2, _, _ = stationary_gaussian_process(
        img.squeeze().shape, gauss_func_field
    )
    idx_map = gen_spectral_idx(img_log_blur, gauss_field1)
    curv_map = gen_curvature_map(gauss_field2, curv_scale)

    img_cube = img.reshape(1, *img.shape)
    freqs = freqs.flatten().reshape(-1, 1, 1)
    # Use the first channel as the spectral reference.
    freqs_ratio = freqs / freqs[0, ...]
    img_cube = img_cube * freqs_ratio ** (idx_map + curv_map * np.log(freqs_ratio))

    return img_cube, idx_map, curv_map
