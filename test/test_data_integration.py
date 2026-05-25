"""Integration test for hyperspectral measurement operator against reference outputs."""

import os

import numpy as np
import torch
from astropy.io import fits

from hyri.interface.load_data import load_data_to_tensor
from hyri.measurement_operator import MeasOpHyperspectral

TEST_MEAS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "3c353_sim_vla_meas_hs"
)


def test_hs_load_compute_psf_and_dirty_against_reference() -> None:
    """Load sample data, compute PSF/dirty image, and compare to references."""
    assert os.path.isdir(TEST_MEAS_PATH), f"Missing test data folder: {TEST_MEAS_PATH}"

    data = load_data_to_tensor(
        uv_file_path=TEST_MEAS_PATH,
        super_resolution=1.5,
        image_pixel_size=None,
        data_weighting=True,
        load_weight=False,
        img_size=(512, 512),
        weight_type="briggs",
        weight_robustness=0.0,
        precond_weighting=False,
        dtype=torch.float64,
        device=torch.device("cpu"),
        ch_start=1,
        ch_last=2,
        ch_effctive=2,
        verbose=False,
    )

    op = MeasOpHyperspectral(
        u=data["u"],
        v=data["v"],
        img_size=(512, 512),
        natural_weight=data["natural_weight"],
        image_weight=data["imaging_weight"],
        precond_weight=data["precond_weight"],
        real_flag=True,
        device=torch.device("cpu"),
        dtype=torch.float64,
        verbose=False,
    )

    meas_bp = op.adjoint_op(data["y"])
    psf = op.get_psf()
    psf_peak = psf.amax(dim=(2, 3), keepdim=True)
    dirty_normalised = meas_bp / psf_peak

    dirty_ref = fits.getdata(os.path.join(TEST_MEAS_PATH, "dirty_normalised.fits"))
    psf_ref = fits.getdata(os.path.join(TEST_MEAS_PATH, "psf.fits"))

    psf_out = psf.squeeze().cpu().numpy().astype(np.float32)
    dirty_out = dirty_normalised.squeeze().cpu().numpy().astype(np.float32)

    rel_psf = np.linalg.vector_norm(psf_out - psf_ref) / np.linalg.vector_norm(psf_ref)
    rel_dirty = np.linalg.vector_norm(dirty_out - dirty_ref) / np.linalg.vector_norm(
        dirty_ref
    )

    assert psf_out.shape == psf_ref.shape
    assert dirty_out.shape == dirty_ref.shape

    assert np.all(np.isfinite(psf_out))
    assert np.all(np.isfinite(dirty_out))

    assert rel_psf < 1e-6
    assert rel_dirty < 1e-6
