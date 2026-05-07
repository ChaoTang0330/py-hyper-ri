"""
Unit tests for the data loading functionality of
src.hyri.interface.load_data.load_data_to_tensor
"""

import os

import torch

from hyri.interface.load_data import load_data_to_tensor

TEST_MEAS_PATH = os.path.join(
    os.path.dirname(__file__), "data", "3c353_sim_vla_meas_hs"
)


def test_load_data_to_tensor_from_sample_meas() -> None:
    """
    Load sample measurements and verify core output structure and dimensions.
    """
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

    for key in (
        "u",
        "v",
        "w",
        "y",
        "natural_weight",
        "imaging_weight",
        "precond_weight",
        "freqs",
        "image_pixel_size",
        "max_proj_baseline",
    ):
        assert key in data

    assert isinstance(data["u"], tuple)
    assert isinstance(data["v"], tuple)
    assert isinstance(data["w"], tuple)
    assert isinstance(data["y"], tuple)
    assert len(data["u"]) == 2
    assert len(data["v"]) == 2
    assert len(data["w"]) == 2
    assert len(data["y"]) == 2

    for ch in range(2):
        u = data["u"][ch]
        v = data["v"][ch]
        w = data["w"][ch]
        y = data["y"][ch]
        nat_w = data["natural_weight"][ch]
        img_w = data["imaging_weight"][ch]

        assert u.ndim == 3
        assert u.shape[0] == 1 and u.shape[1] == 1 and u.shape[-1] > 0
        assert v.shape == u.shape
        assert w.shape == u.shape
        assert y.shape == u.shape
        assert nat_w.shape == u.shape
        assert img_w.shape == u.shape

        assert y.is_complex()
        assert torch.isfinite(u).all()
        assert torch.isfinite(v).all()
        assert torch.isfinite(w).all()
        assert torch.isfinite(nat_w).all()
        assert torch.isfinite(img_w).all()

        precond_w = data["precond_weight"][ch]
        assert precond_w.shape == (1, 1, 1)
        assert torch.allclose(precond_w, torch.ones_like(precond_w))

    freqs = data["freqs"]
    assert freqs.shape == (1, 2, 1, 1)
    assert torch.isfinite(freqs).all()
    assert (freqs > 0).all()

    assert float(data["image_pixel_size"]) > 0.0
    assert float(data["max_proj_baseline"]) > 0.0
