"""Backend-specific tests for the FINUFFT measurement operator."""

import torch

from src.hyri.measurement_operator import MeasOpFinufft

from .meas_op_helpers import create_random_uv, create_test_image


def build_finufft_op(
    n_samples: int = 512,
    img_size: tuple[int, int] = (64, 64),
    dtype: torch.dtype = torch.float32,
) -> MeasOpFinufft:
    u, v = create_random_uv(n_samples=n_samples, dtype=dtype)

    return MeasOpFinufft(
        u=u,
        v=v,
        img_size=img_size,
        natural_weight=torch.ones((1, 1, n_samples), dtype=dtype),
        image_weight=torch.ones((1, 1, n_samples), dtype=dtype),
        precond_weight=torch.ones((1, 1, n_samples), dtype=dtype),
        real_flag=True,
        device=torch.device("cpu"),
        dtype=dtype,
    )


def test_finufft_build_and_run() -> None:
    """Constructed FINUFFT meas op should run forward and adjoint."""
    op = build_finufft_op(n_samples=128, img_size=(32, 32), dtype=torch.float32)
    x = create_test_image((32, 32), dtype=torch.float32)

    y = op.forward_op(x)
    x_adj = op.adjoint_op(y)

    assert y.shape == (1, 1, 128)
    assert y.dtype == torch.complex64
    assert x_adj.shape == (1, 1, 32, 32)
    assert x_adj.dtype == torch.float32
