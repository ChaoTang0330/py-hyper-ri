"""Fit clean beams and estimate spectral-index maps from image cubes.

This file provides the PSF-to-beam fitting step and the per-pixel spectral
index estimation used by the hyperspectral imaging pipeline.
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import convolve2d
from skimage.measure import label, regionprops


def gaussian2D(coords, sigma_x, sigma_y, theta) -> np.ndarray:
    """Evaluate an unnormalised 2D Gaussian on coordinate samples.

    Args:
        coords (np.ndarray): Array of shape ``(N, 2)`` with columns ``x`` and
            ``y``.
        sigma_x (float): Standard deviation along the x-axis.
        sigma_y (float): Standard deviation along the y-axis.
        theta (float): Counter-clockwise rotation angle in radians.

    Returns:
        np.ndarray: Gaussian values for each input coordinate.
    """
    x = coords[:, 0]
    y = coords[:, 1]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    a = ((cos_theta * x + sin_theta * y) ** 2) / (sigma_x**2)
    b = ((-sin_theta * x + cos_theta * y) ** 2) / (sigma_y**2)
    return np.exp(-0.5 * (a + b))


def fit_clean_beam(psf, verbose=True) -> tuple[np.ndarray, float, float, float]:
    """Fit an elliptical Gaussian clean beam to a point-spread function.

    The PSF is normalised before fitting, a compact PSF may fall back to a
    delta-function beam, and the fitted Gaussian kernel is normalised to unit
    sum.

    Args:
        psf (np.ndarray): Point-spread function image.
        verbose (bool, optional): Whether to print fit diagnostics.
            Default: True.

    Returns:
        Tuple[np.ndarray, float, float, float]: ``(kernel, sigma_x, sigma_y,
        theta)`` where ``kernel`` is the fitted clean-beam kernel.

    Raises:
        ValueError: If ``psf`` has no positive finite peak.

    Notes:
        The Gaussian fit uses the region where ``psf > 0.5`` to seed the
        initial guess.

    Examples:
        >>> yy, xx = np.mgrid[-8:9, -8:9]
        >>> psf = np.exp(-0.5 * ((xx / 2.5) ** 2 + (yy / 1.8) ** 2))
        >>> kernel, sigma_x, sigma_y, theta = fit_clean_beam(psf, verbose=False)
        >>> kernel.ndim
        2
        >>> np.isclose(kernel.sum(), 1.0)
        True
        >>> sigma_x > 0 and sigma_y > 0
        True
    """
    # Normalise the PSF so its maximum is 1.
    peak = np.max(psf)
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError("PSF peak must be a positive finite value.")
    psf = psf / peak

    # Seed the fit from the region where psf > 0.5.
    mask = psf > 0.5
    labeled = label(mask)
    props = regionprops(labeled)
    # Use the first connected region, which is assumed to be the beam.
    min_row, min_col, max_row, max_col = props[0].bbox
    height = max_row - min_row
    width = max_col - min_col
    if height * width < 3.0:
        if verbose:
            print(
                "INFO: PSF is very compact."
                "The clean beam is considered as a delta function."
            )
        # Use a delta kernel when the fitted beam would be numerically unstable.
        kernel = np.zeros((3, 3))
        kernel[1, 1] = 1.0
        return kernel, 0.0, 0.0, 0.0

    height = max(height, 1.1)
    width = max(width, 1.1)

    # Convert the bounding box size into an initial sigma estimate.
    sigma_x_guess = height / (2 * np.sqrt(2 * np.log(2)))
    sigma_y_guess = width / (2 * np.sqrt(2 * np.log(2)))

    # Define a region of interest around the peak.
    y0, x0 = np.unravel_index(np.argmax(psf), psf.shape)

    factor = 10
    # Use an enlarged ROI so the Gaussian fit captures the beam footprint.
    roi_x_start = int(max(0, x0 - factor * np.floor(width / 2)))
    roi_x_end = int(min(psf.shape[1], x0 + factor * np.floor(width / 2) + 1))
    roi_y_start = int(max(0, y0 - factor * np.floor(height / 2)))
    roi_y_end = int(min(psf.shape[0], y0 + factor * np.floor(height / 2) + 1))
    roi = psf[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

    # Recenter the grid on the ROI peak.
    center_y, center_x = np.unravel_index(np.argmax(roi), roi.shape)

    y_indices, x_indices = np.indices(roi.shape)
    x_grid = x_indices - center_x
    y_grid = y_indices - center_y

    # Flatten the ROI and coordinate grid for fitting.
    data = roi.ravel()
    coords = np.vstack([x_grid.ravel(), y_grid.ravel()]).T

    # Fit the Gaussian model.
    initial_guess = (sigma_x_guess, sigma_y_guess, 0)

    popt, _ = curve_fit(gaussian2D, coords, data, p0=initial_guess)
    sigma_x_fit, sigma_y_fit, theta_fit = map(float, popt)
    # Normalise theta to [-pi, pi].
    theta_fit = (theta_fit + np.pi) % (2 * np.pi) - np.pi
    # Because of symmetry, negative theta is equivalent to theta + pi.
    if theta_fit < 0:
        theta_fit = theta_fit + np.pi
    if verbose:
        print(
            "INFO: clean beam fitted with",
            f"sigma_x = {sigma_x_fit:.6f},",
            f"sigma_y = {sigma_y_fit:.6f},",
            f"theta = {theta_fit:.6f}",
        )

    # Generate the clean-beam kernel.
    half_kernel_size = int(np.ceil(max(sigma_x_fit, sigma_y_fit))) * 2
    x_range = np.arange(-half_kernel_size, half_kernel_size + 1)
    y_range = np.arange(-half_kernel_size, half_kernel_size + 1)
    X, Y = np.meshgrid(x_range, y_range)
    coords_kernel = np.vstack([X.ravel(), Y.ravel()]).T
    kernel = gaussian2D(coords_kernel, sigma_x_fit, sigma_y_fit, theta_fit)
    kernel = kernel.reshape((2 * half_kernel_size + 1, 2 * half_kernel_size + 1))
    kernel = kernel / np.sum(kernel)

    return kernel, sigma_x_fit, sigma_y_fit, theta_fit


def get_spec_idx(
    img_cube, freqs, kernel=None, threshold=1e-5, crop_roi=None
) -> np.ndarray:
    """Estimate a spectral-index map from an image cube.

    The function optionally crops the cube, optionally smooths it with a clean
    beam, and fits a log-log slope at each pixel above a threshold.

    Args:
        img_cube (np.ndarray): Image cube with frequency as the first axis.
        freqs (np.ndarray): Frequency samples corresponding to the cube.
        kernel (np.ndarray, optional): Optional convolution kernel used to
            smooth each frequency slice before fitting. Default: None.
        threshold (float, optional): Minimum per-pixel value across all
            frequencies required to fit a spectral index. Default: 1e-5.
        crop_roi (Tuple[int, int, int, int], optional): Optional crop region as
            ``(x_start, x_end, y_start, y_end)``. Default: None.

    Returns:
        np.ndarray: Spectral-index map with positive values clipped to zero and
        invalid pixels set to ``NaN``.

    Notes:
        Pixels below ``threshold`` at any frequency are excluded from fitting.

    Examples:
        >>> cube = np.ones((2, 4, 4))
        >>> alpha = get_spec_idx(cube, np.array([1.0, 2.0]), threshold=0.5)
        >>> alpha.shape
        (4, 4)
    """
    # Crop the image cube if requested.
    if crop_roi is not None:
        x_start, x_end, y_start, y_end = crop_roi
        img_cube = img_cube[:, y_start:y_end, x_start:x_end]

    # Convolve each frequency slice with the kernel if provided.
    nfreqs = np.size(freqs)
    if kernel is not None:
        for ch in range(nfreqs):
            img_cube[ch, :, :] = convolve2d(img_cube[ch, :, :], kernel, mode="same")

    # Keep only pixels whose minimum value over frequencies exceeds the threshold.
    prec_flag = np.min(img_cube, axis=0) > threshold

    # Initialise the spectral-index map.
    alpha = np.zeros(img_cube.shape[-2:])

    # Fit a log-log slope for each valid pixel.
    for i in range(img_cube.shape[-2]):
        for j in range(img_cube.shape[-1]):
            if not prec_flag[i, j].item():
                continue
            # Fit the spectral slope only for pixels that are valid across all frequencies.
            pixels = img_cube[:, i, j]
            coeffs = np.polyfit(np.log(freqs).flatten(), np.log(pixels).flatten(), 1)
            alpha[i, j] = coeffs[0]

    # Post-process the spectral-index map.
    alpha[alpha > 0] = 0
    # Mask out pixels that failed the threshold test at any frequency.
    alpha[np.logical_not(prec_flag)] = np.nan

    return alpha
