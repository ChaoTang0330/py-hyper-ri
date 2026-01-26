import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import convolve2d
from skimage.measure import label, regionprops

# 2D Gaussian model
def gaussian2D(coords, sigma_x, sigma_y, theta):
    # coords: array of shape (N,2) with columns [x, y]
    x = coords[:, 0]
    y = coords[:, 1]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    a = ((cos_theta * x + sin_theta * y) ** 2) / (sigma_x**2)
    b = ((-sin_theta * x + cos_theta * y) ** 2) / (sigma_y**2)
    return np.exp(-0.5 * (a + b))


def fit_clean_beam(psf, verbose=True):
    # Normalize the PSF so its maximum is 1.
    psf = psf / np.max(psf)

    # --- Initial guess using the bounding box of the region where psf > 0.5 ---
    # Create a binary mask for psf > 0.5 and label connected regions.
    mask = psf > 0.5
    labeled = label(mask)
    props = regionprops(labeled)
    if len(props) == 0:
        raise ValueError("No region found with psf > 0.5")
    # Using the first region (assumed to be the one of interest)
    # Note: regionprops.bbox returns (min_row, min_col, max_row, max_col)
    min_row, min_col, max_row, max_col = props[0].bbox
    height = max_row - min_row
    width = max_col - min_col
    height = max(height, 1.1)
    width = max(width, 1.1)

    # Initial guess for sigma (from FWHM = 2*sqrt(2*log(2))*sigma)
    sigma_x_guess = height / (2 * np.sqrt(2 * np.log(2)))
    sigma_y_guess = width / (2 * np.sqrt(2 * np.log(2)))
    # (theta is set to 0 initially)

    # --- Define region-of-interest (ROI) ---
    # Find the PSF's peak (assumed to be equal to 1 after normalization)
    y0, x0 = np.unravel_index(np.argmax(psf), psf.shape)

    # Define ROI boundaries (using a factor times half the bounding box width/height)
    factor = 10
    roi_x_start = int(max(0, x0 - factor * np.floor(width / 2)))
    roi_x_end = int(min(psf.shape[1], x0 + factor * np.floor(width / 2) + 1))
    roi_y_start = int(max(0, y0 - factor * np.floor(height / 2)))
    roi_y_end = int(min(psf.shape[0], y0 + factor * np.floor(height / 2) + 1))
    roi = psf[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

    # Find the peak in the ROI (to center the grid)
    center_y, center_x = np.unravel_index(np.argmax(roi), roi.shape)

    # Create a coordinate grid for the ROI and center it at the peak.
    y_indices, x_indices = np.indices(roi.shape)
    x_grid = x_indices - center_x
    y_grid = y_indices - center_y

    # Flatten the ROI and coordinates for fitting.
    data = roi.ravel()
    coords = np.vstack([x_grid.ravel(), y_grid.ravel()]).T

    # --- Fit the Gaussian ---
    # Use an initial guess based on the ROI size.
    initial_guess = (sigma_x_guess, sigma_y_guess, 0)

    popt, _ = curve_fit(gaussian2D, coords, data, p0=initial_guess)
    sigma_x_fit, sigma_y_fit, theta_fit = popt
    # Normalize theta to [-pi, pi]
    theta_fit = (theta_fit + np.pi) % (2 * np.pi) - np.pi
    if verbose:
        print(
            "INFO: Clean beam fitted with",
            f"sigma_x = {sigma_x_fit:.6f},",
            f"sigma_y = {sigma_y_fit:.6f},",
            f"theta = {theta_fit:.6f}",
        )

    # --- Generate the clean beam kernel ---
    # Determine kernel size: using 2 * ceil(max(sigma_x, sigma_y))
    half_kernel_size = int(np.ceil(max(sigma_x_fit, sigma_y_fit))) * 2
    x_range = np.arange(-half_kernel_size, half_kernel_size + 1)
    y_range = np.arange(-half_kernel_size, half_kernel_size + 1)
    X, Y = np.meshgrid(x_range, y_range)
    coords_kernel = np.vstack([X.ravel(), Y.ravel()]).T
    kernel = gaussian2D(coords_kernel, sigma_x_fit, sigma_y_fit, theta_fit)
    kernel = kernel.reshape((2 * half_kernel_size + 1, 2 * half_kernel_size + 1))
    kernel = kernel / np.sum(kernel)  # Normalize the kernel

    return kernel, sigma_x_fit, sigma_y_fit, theta_fit


def get_spec_idx(img_cube, kernel, freqs, threshold=1e-5, crop_roi=None):
    # --- Crop the image cube ---
    if crop_roi is not None:
        x_start, x_end, y_start, y_end = crop_roi
        img_cube = img_cube[:, y_start:y_end, x_start:x_end]

    # --- Convolve each frequency slice with the kernel ---
    nfreqs = np.size(freqs)
    for ch in range(nfreqs):
        # Convolve2d returns an array of the same shape with mode='same'
        img_cube[ch, :, :] = convolve2d(img_cube[ch, :, :], kernel, mode="same")

    # --- only pixels whose minimum value over frequencies exceeds threshold ---
    prec_flag = np.min(img_cube, axis=0) > threshold

    # --- Initialize the alpha array (spectral index) ---
    alpha = np.zeros(img_cube.shape[-2:])

    # --- For each pixel, fit a line to log(freqs) vs log(pixel_value) ---
    # Only do this where precFlag is True.
    for i in range(img_cube.shape[-2]):
        for j in range(img_cube.shape[-1]):
            if not prec_flag[i, j].item():
                continue

            # Extract the spectral values for the current pixel (as a 1D array)
            pixels = img_cube[:, i, j]

            # Fit a line: np.polyfit returns coefficients [slope, intercept]
            coeffs = np.polyfit(np.log(freqs).flatten(), np.log(pixels).flatten(), 1)
            alpha[i, j] = coeffs[0]

    # --- Post-process alpha ---
    # Set positive alpha values to 0.
    alpha[alpha > 0] = 0
    # Where the precision flag is False, assign NaN.
    alpha[np.logical_not(prec_flag)] = np.nan

    return alpha