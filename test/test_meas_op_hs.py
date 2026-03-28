"""Tests for hyperspectral measurement operator with C=1 and C=2 cases."""

import numpy as np
import pytest
import torch

from src.ri_measurement_operator import MeasOpHyperspectral

from .meas_op_helpers import create_random_uv


def build_hs_op(
    channels: int,
    n_samples: int = 128,
    img_size: tuple[int, int] = (32, 32),
    dtype: torch.dtype = torch.float32,
) -> MeasOpHyperspectral:
    """Create a hyperspectral measurement operator for a given channel count."""
    u = []
    v = []
    for ch in range(channels):
        u_ch, v_ch = create_random_uv(n_samples=n_samples, dtype=dtype, seed=10 + ch)
        u.append(u_ch)
        v.append(v_ch)

    op = MeasOpHyperspectral(
        u=tuple(u),
        v=tuple(v),
        img_size=img_size,
        natural_weight=tuple(
            torch.ones((1, 1, n_samples), dtype=dtype) for _ in range(channels)
        ),
        image_weight=tuple(
            torch.ones((1, 1, n_samples), dtype=dtype) for _ in range(channels)
        ),
        precond_weight=tuple(
            torch.ones((1, 1, n_samples), dtype=dtype) for _ in range(channels)
        ),
        real_flag=True,
        device=torch.device("cpu"),
        dtype=dtype,
        verbose=False,
    )

    return op


def create_hs_test_image(
    channels: int,
    img_size: tuple[int, int] = (32, 32),
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> torch.Tensor:
    """Create deterministic hyperspectral image tensor of shape (1, C, H, W)."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return torch.randn((1, channels, *img_size), generator=g, dtype=dtype)


@pytest.mark.parametrize("channels", [1, 2])
def test_hs_forward_and_adjoint_shapes(channels: int) -> None:
    """Hyperspectral forward/adjoint should preserve expected channel-wise shapes."""
    n_samples = 128
    img_size = (32, 32)
    op = build_hs_op(channels=channels, n_samples=n_samples, img_size=img_size)
    x = create_hs_test_image(
        channels=channels, img_size=img_size, dtype=torch.float32, seed=1
    )

    y = op.forward_op(x)
    assert isinstance(y, tuple)
    assert len(y) == channels
    for y_ch in y:
        assert y_ch.shape == (1, 1, n_samples)
        assert y_ch.dtype == torch.complex64
        assert torch.isfinite(y_ch.real).all()
        assert torch.isfinite(y_ch.imag).all()

    x_adj = op.adjoint_op(y)
    assert x_adj.shape == (1, channels, *img_size)
    assert x_adj.dtype == torch.float32
    assert torch.isfinite(x_adj).all()


@pytest.mark.parametrize("channels", [1, 2])
def test_hs_zero_input_maps_to_zero(channels: int) -> None:
    """Zero input should map to zero in both forward and adjoint paths."""
    n_samples = 96
    img_size = (24, 24)
    op = build_hs_op(channels=channels, n_samples=n_samples, img_size=img_size)

    x0 = torch.zeros((1, channels, *img_size), dtype=torch.float32)
    y0 = op.forward_op(x0)
    for y_ch in y0:
        assert torch.allclose(y_ch, torch.zeros_like(y_ch), atol=1e-6)

    y_meas_zero = tuple(
        torch.zeros((1, 1, n_samples), dtype=torch.complex64) for _ in range(channels)
    )
    x0_adj = op.adjoint_op(y_meas_zero)
    assert torch.allclose(x0_adj, torch.zeros_like(x0_adj), atol=1e-6)


@pytest.mark.parametrize("channels", [1, 2])
def test_hs_op_norm_positive(channels: int) -> None:
    """Operator norm should be finite and positive for both C=1 and C=2."""
    op = build_hs_op(channels=channels, n_samples=64, img_size=(16, 16))
    norm = op.get_op_norm(rel_tol=1e-4, max_iter=25)

    assert isinstance(norm, float)
    assert np.isfinite(norm)
    assert norm > 0.0


def test_hs_op_norm_matches_channel_wise_for_single_channel() -> None:
    """For C=1, global and channel-wise op norms should be close."""
    op = build_hs_op(channels=1, n_samples=96, img_size=(24, 24))

    torch.manual_seed(42)
    global_norm = op.get_op_norm(compute_flag=True, rel_tol=1e-6, max_iter=50)
    torch.manual_seed(42)
    ch_norm = op.get_op_norm_ch(compute_flag=True, rel_tol=1e-6, max_iter=50)

    assert ch_norm.shape == (1, 1, 1, 1)
    assert np.isfinite(global_norm)
    assert torch.isfinite(ch_norm).all()
    rel_diff = abs(global_norm - ch_norm[0].item()) / max(abs(ch_norm[0].item()), 1e-12)
    assert rel_diff < 1e-6
