"""Backend-specific tests for the torchkbnufft measurement operator."""

import pytest
import torch

from src.measurement_operator import MeasOpTkbn

from .meas_op_helpers import create_random_uv, create_test_image


def build_tkbn_op(
    mode: str = "table",
    n_samples: int = 256,
    img_size: tuple[int, int] = (64, 64),
    dtype: torch.dtype = torch.float32,
) -> MeasOpTkbn:
    u, v = create_random_uv(n_samples=n_samples, dtype=dtype)

    return MeasOpTkbn(
        u=u,
        v=v,
        img_size=img_size,
        natural_weight=torch.ones((1, 1, n_samples), dtype=dtype),
        image_weight=torch.ones((1, 1, n_samples), dtype=dtype),
        precond_weight=torch.ones((1, 1, n_samples), dtype=dtype),
        real_flag=True,
        grid_size=(img_size[0] * 2, img_size[1] * 2),
        kernel_dim=7,
        mode=mode,
        device=torch.device("cpu"),
        dtype=dtype,
    )


def test_tkbnufft_mode_table_runs() -> None:
    """Table interpolation mode should run forward/adjoint without errors."""
    op = build_tkbn_op(mode="table")

    x = create_test_image((64, 64), dtype=torch.float32)
    y = op.forward_op(x)
    x_adj = op.adjoint_op(y)

    assert y.shape == (1, 1, 256)
    assert x_adj.shape == (1, 1, 64, 64)


def test_tkbnufft_mode_matrix_runs() -> None:
    """Matrix interpolation mode should run (slower than table mode)."""
    op = build_tkbn_op(mode="matrix")

    x = create_test_image((64, 64), dtype=torch.float32)
    y = op.forward_op(x)

    assert y.shape == (1, 1, 256)


def test_tkbnufft_invalid_mode_raises() -> None:
    """Unsupported interpolation mode should raise NotImplementedError."""
    with pytest.raises(NotImplementedError, match="Unrecognised mode"):
        build_tkbn_op(mode="invalid-mode")
