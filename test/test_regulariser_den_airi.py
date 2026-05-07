"""Builder utilities for DenAIRI regulariser."""

import torch

from src.hyri.regulariser.den_airi import DenAIRI

from .regulariser_helpers import (
    make_image_tensor,
    make_positive_threshold,
)


def create_den_airi(
    channels: int,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> tuple[DenAIRI, torch.Tensor]:
    """Create DenAIRI regulariser and input for common tests."""
    reg = DenAIRI(
        shelf_path="./airi_denoisers/shelf_airi_mrid.csv",
        rand_trans=False,
        rank_num_ch=channels,
        dtype=dtype,
        verbose=False,
    )

    x = make_image_tensor(channels=channels, img_size=(32, 32), dtype=dtype, seed=seed)
    ch_heuristic = make_positive_threshold(channels=channels, value=1.0e-4, dtype=dtype)
    ch_peak_est = make_positive_threshold(channels=channels, value=1.0, dtype=dtype)
    reg.update(ch_heuristic=ch_heuristic, ch_peak_est=ch_peak_est)
    return reg, x


def test_den_airi_update_handles_single_channel() -> None:
    """Single-channel update should not fail and should select one network."""
    reg, _ = create_den_airi(channels=1, dtype=torch.float32, seed=1)

    assert reg._list_network[0] is not None
    assert reg._list_network_sigma[0] > 0.0


def test_den_airi_forward_shape_dtype_finite() -> None:
    """DenAIRI forward pass should preserve shape/dtype and produce finite values."""
    reg, x = create_den_airi(channels=2, dtype=torch.float32, seed=9)
    y = reg(x)

    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert torch.isfinite(y).all()


def test_den_airi_update_selects_expected_shelf_sigma() -> None:
    """Selected shelf sigma should be the largest key not exceeding heuristic/peak."""
    channels = 2
    reg, _ = create_den_airi(channels=channels, dtype=torch.float32, seed=2)

    expected_sigma = 8.0e-5
    for sigma in reg._list_network_sigma:
        assert abs(sigma - expected_sigma) < 1.0e-12