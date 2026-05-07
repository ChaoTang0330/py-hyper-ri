"""
Unit tests for fit_clean_beam in spectral_index utilities.
"""

import numpy as np
import pytest

from hyri.interface.spectral_index import fit_clean_beam, gaussian2D


def _make_synthetic_psf(
    shape: tuple[int, int] = (65, 65),
    sigma_x: float = 3.2,
    sigma_y: float = 1.8,
    theta: float = 0.45,
) -> np.ndarray:
    """
    Create a normalized anisotropic Gaussian PSF centered in the image.
    """
    y_idx, x_idx = np.indices(shape)
    y0, x0 = shape[0] // 2, shape[1] // 2
    x = x_idx - x0
    y = y_idx - y0
    coords = np.vstack([x.ravel(), y.ravel()]).T
    psf = gaussian2D(coords, sigma_x, sigma_y, theta).reshape(shape)
    return psf / np.max(psf)


def test_fit_clean_beam_returns_valid_kernel_and_params() -> None:
    psf = _make_synthetic_psf()

    kernel, sigma_x_fit, sigma_y_fit, theta_fit = fit_clean_beam(psf, verbose=False)

    assert kernel.ndim == 2
    assert kernel.shape[0] == kernel.shape[1]
    assert kernel.shape[0] % 2 == 1
    assert np.isclose(np.sum(kernel), 1.0, atol=1e-10)
    assert np.isfinite(kernel).all()
    assert (kernel >= 0).all()

    assert np.isfinite(sigma_x_fit)
    assert np.isfinite(sigma_y_fit)
    assert sigma_x_fit > 0
    assert sigma_y_fit > 0
    assert np.isfinite(theta_fit)
    assert 0.0 <= theta_fit <= np.pi


def test_fit_clean_beam_matches_synthetic_psf_profile() -> None:
    sigma_x_true = 2.7
    sigma_y_true = 1.4
    theta_true = 0.60
    psf = _make_synthetic_psf(
        shape=(73, 73),
        sigma_x=sigma_x_true,
        sigma_y=sigma_y_true,
        theta=theta_true,
    )

    _, sigma_x_fit, sigma_y_fit, theta_fit = fit_clean_beam(psf, verbose=False)

    y_idx, x_idx = np.indices(psf.shape)
    y0, x0 = np.unravel_index(np.argmax(psf), psf.shape)
    coords = np.vstack([(x_idx - x0).ravel(), (y_idx - y0).ravel()]).T
    psf_fit = gaussian2D(coords, sigma_x_fit, sigma_y_fit, theta_fit).reshape(psf.shape)
    psf_fit = psf_fit / np.max(psf_fit)

    mae = np.mean(np.abs(psf - psf_fit))
    assert mae < 0.03


def test_fit_clean_beam_raises_for_invalid_psf() -> None:
    with pytest.raises(ValueError, match="PSF peak must be a positive finite value."):
        fit_clean_beam(np.zeros((17, 17), dtype=np.float64), verbose=False)
