"""
Unit tests for get_spec_idx in spectral_index utilities.
"""

import numpy as np

from hyri.interface.spectral_index import get_spec_idx


def _make_power_law_cube(
    num_freqs: int = 5,
    height: int = 32,
    width: int = 32,
    spec_idx: float = -2.0,
    f0: float = 1e9,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a synthetic image cube with power-law spectrum: I(f) = I0 * (f/f0)^alpha

    Args:
        num_freqs: Number of frequency channels
        height, width: Image dimensions
        spec_idx: Spectral index (typically negative for astrophysical sources)
        f0: Reference frequency

    Returns:
        img_cube: (num_freqs, height, width) array
        freqs: (num_freqs,) frequency array
    """
    freqs = np.linspace(1e9, 1.5e9, num_freqs)

    # Create image cube with a spatially varying reference intensity.
    img_cube = np.zeros((num_freqs, height, width))
    y = np.linspace(0.0, 0.5, height)
    x = np.linspace(0.0, 0.5, width)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    I0 = yy + xx

    for ch in range(num_freqs):
        # Power-law: I(f) = I0 * (f/f0)^alpha
        intensity = I0 * (freqs[ch] / f0) ** spec_idx
        img_cube[ch, :, :] = intensity

    return img_cube, freqs


def test_get_spec_idx_recovers_constant_spectral_index() -> None:
    """
    Test that spectral index is recovered correctly from synthetic power-law cubes.
    """
    spec_idx_true = -0.7
    img_cube, freqs = _make_power_law_cube(spec_idx=spec_idx_true)

    alpha = get_spec_idx(img_cube.copy(), freqs, threshold=1e-5)

    # Check output shape matches input spatial dimensions
    assert alpha.shape == img_cube.shape[1:]

    # For power-law cube with constant spectrum, alpha should be ~spec_idx_true
    # everywhere (negative after post-processing)
    valid_mask = np.isfinite(alpha)
    assert valid_mask.sum() > 0, "Expected some valid (finite) pixels"

    # Spectral indices should be close to true value (within tolerance)
    assert np.allclose(alpha[valid_mask], spec_idx_true, atol=0.05)


def test_get_spec_idx_with_threshold_creates_nan_mask() -> None:
    """Test that pixels below threshold are masked as NaN."""
    img_cube, freqs = _make_power_law_cube(num_freqs=5, height=16, width=16)

    # get_spec_idx uses min(img_cube, axis=0) > threshold, so choose a threshold
    # in the middle of the per-pixel minima range to force a mixed mask.
    per_pixel_min = np.min(img_cube, axis=0)
    threshold = 0.5 * (np.min(per_pixel_min) + np.max(per_pixel_min))
    alpha = get_spec_idx(img_cube.copy(), freqs, threshold=threshold)

    # Some pixels should be above threshold, some below
    valid_count = np.isfinite(alpha).sum()
    nan_count = (~np.isfinite(alpha)).sum()

    assert valid_count > 0, "Expected some valid pixels"
    assert nan_count > 0, "Expected some NaN pixels (below threshold)"
    assert valid_count + nan_count == alpha.size


def test_get_spec_idx_with_kernel_convolution() -> None:
    """Test that kernel convolution smooths the image before fitting."""
    img_cube, freqs = _make_power_law_cube(num_freqs=5, height=32, width=32)

    # Simple smoothing kernel (3x3 average)
    kernel = np.ones((3, 3)) / 9.0

    alpha_with_kernel = get_spec_idx(
        img_cube.copy(), freqs, kernel=kernel, threshold=1e-5
    )
    alpha_without_kernel = get_spec_idx(
        img_cube.copy(), freqs, kernel=None, threshold=1e-5
    )

    # Both should have valid values
    assert np.isfinite(alpha_with_kernel).any()
    assert np.isfinite(alpha_without_kernel).any()

    # Results should be similar but not identical (smoothing affects fitting)
    valid_mask = np.isfinite(alpha_with_kernel) & np.isfinite(alpha_without_kernel)
    assert valid_mask.any()
    # Just check they're reasonably close (within a broader tolerance)
    assert np.allclose(
        alpha_with_kernel[valid_mask], alpha_without_kernel[valid_mask], atol=0.2
    )


def test_get_spec_idx_with_cropping() -> None:
    """Test that cropping works correctly."""
    img_cube, freqs = _make_power_law_cube(num_freqs=5, height=64, width=64)

    # Define crop region
    crop_roi = (10, 50, 15, 55)  # x_start, x_end, y_start, y_end
    x_start, x_end, y_start, y_end = crop_roi
    expected_shape = (y_end - y_start, x_end - x_start)

    alpha = get_spec_idx(img_cube.copy(), freqs, crop_roi=crop_roi, threshold=1e-5)

    # Check that output shape matches cropped region
    assert alpha.shape == expected_shape


def test_get_spec_idx_positive_alpha_set_to_zero() -> None:
    """Test that positive spectral indices are set to 0 during post-processing."""
    # Create a cube where some pixels would have positive slope in log-log plot
    img_cube, freqs = _make_power_law_cube(
        num_freqs=5, height=32, width=32, spec_idx=-0.7
    )

    # Artificially set some pixels to have increasing spectrum (positive alpha)
    img_cube[:, 10:15, 10:15] = img_cube[:, 10:15, 10:15][
        ::-1, :, :
    ]  # Reverse freq order

    alpha = get_spec_idx(img_cube.copy(), freqs, threshold=1e-5)

    # All finite values should be <= 0 (negative or zero)
    valid_alpha = alpha[np.isfinite(alpha)]
    assert np.all(valid_alpha <= 0), "Expected positive spectral indices to be set to 0"


def test_get_spec_idx_output_dtype_and_finite() -> None:
    """
    Test output properties: dtype is float, contains finite and NaN values
    appropriately.
    """
    img_cube, freqs = _make_power_law_cube(num_freqs=5, height=32, width=32)

    alpha = get_spec_idx(img_cube.copy(), freqs, threshold=1e-5)

    # Check dtype
    assert alpha.dtype in [np.float32, np.float64]

    # Check that output is either finite or NaN (no inf)
    assert np.all(np.isfinite(alpha) | np.isnan(alpha))


def test_get_spec_idx_with_very_high_threshold() -> None:
    """Test behavior when threshold is so high that all pixels are masked."""
    img_cube, freqs = _make_power_law_cube(num_freqs=5, height=16, width=16)

    # Set threshold higher than max value
    threshold = np.max(img_cube) * 10.0
    alpha = get_spec_idx(img_cube.copy(), freqs, threshold=threshold)

    # All pixels should be NaN
    assert np.all(np.isnan(alpha))


def test_get_spec_idx_frequency_array_shapes() -> None:
    """Test that function works with different frequency array shapes."""
    img_cube, freqs = _make_power_law_cube(num_freqs=8, height=32, width=32)

    # Test 1D frequency array
    alpha1 = get_spec_idx(img_cube.copy(), freqs.reshape(-1), threshold=1e-5)
    assert alpha1.shape == img_cube.shape[1:]

    # Test 2D frequency array (should be flattened internally)
    alpha2 = get_spec_idx(img_cube.copy(), freqs.reshape(-1, 1), threshold=1e-5)
    assert alpha2.shape == img_cube.shape[1:]

    # Results should be identical
    assert np.allclose(alpha1, alpha2, equal_nan=True)
