"""Builder utilities for DenHyperAIRI regulariser."""

import torch

from src.regulariser.den_hyperairi import DenHyperAIRI

from .regulariser_helpers import (
    make_image_tensor,
    make_positive_threshold,
)


def create_den_hyperairi(
    channels: int,
    img_size: tuple[int, int] = (32, 32),
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> tuple[DenHyperAIRI, torch.Tensor]:
    """Create DenHyperAIRI regulariser and input for common tests."""
    if channels < 2:
        raise ValueError("DenHyperAIRI requires channels >= 2")

    reg = DenHyperAIRI(
        img_size=img_size,
        rank_num_ch=channels,
        rank_ch_start=0,
        ch_freq=torch.linspace(1.0, 1.1, channels, dtype=dtype),
        shelf_path="./airi_denoisers/shelf_hyperairi_mix.csv",
        rand_trans=False,
        dtype=dtype,
        verbose=False,
    )

    x = make_image_tensor(channels=channels, img_size=img_size, dtype=dtype, seed=seed)
    ch_heuristic = make_positive_threshold(channels=channels, value=1.0e-4, dtype=dtype)
    ch_peak_est = make_positive_threshold(channels=channels, value=1.0, dtype=dtype)
    reg.update(ch_heuristic=ch_heuristic, ch_peak_est=ch_peak_est)
    return reg, x


def test_den_hyperairi_update_selects_expected_shelf_sigma() -> None:
    """Model sigma selection should match heuristic/peak rule per channel."""
    reg, _ = create_den_hyperairi(channels=2, dtype=torch.float32, seed=4)

    expected_sigma = 8.0e-5
    for sigma in reg._list_network_sigma:
        assert abs(sigma - expected_sigma) < 1.0e-12


def test_den_hyperairi_forward_two_channel_shape_dtype_finite() -> None:
    """DenHyperAIRI should support the intended C=2 path with finite output."""
    reg, x = create_den_hyperairi(channels=2, dtype=torch.float32, seed=8)
    y = reg(x)

    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert torch.isfinite(y).all()


def test_den_hyperairi_spec_index_update_sets_valid_weights() -> None:
    """Spectral correction weights should keep expected shape and finite values."""
    channels = 2
    reg, _ = create_den_hyperairi(channels=channels, dtype=torch.float32, seed=6)
    # spec_idx is a 2D map shared across channels with image spatial size.
    spec_idx = torch.zeros((1, 1, *reg._img_size), dtype=torch.float32)
    reg.update(spec_idx=spec_idx)

    assert len(reg._list_spec_corr_weight) == channels
    for weight in reg._list_spec_corr_weight:
        assert weight.shape == (1, 3, *reg._img_size)
        assert torch.isfinite(weight).all()