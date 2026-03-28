"""Shared helpers and backend-agnostic contract tests for RI mono measurement operators."""

import numpy as np
import pytest
import torch

from .meas_op_helpers import create_test_image
from .test_meas_op_finufft import build_finufft_op
from .test_meas_op_tkbnufft import build_tkbn_op


def _build_backend_op(backend: str):
    if backend == "finufft":
        return build_finufft_op(n_samples=256, img_size=(64, 64), dtype=torch.float32)
    if backend == "tkbnufft":
        return build_tkbn_op(mode="table", n_samples=256, img_size=(64, 64), dtype=torch.float32)
    raise ValueError(f"Unknown backend: {backend}")


@pytest.mark.parametrize("backend", ["finufft", "tkbnufft"])
def test_nufft_forward_and_adjoint_shapes(backend: str) -> None:
    """Both backends should agree on expected tensor shapes."""
    op = _build_backend_op(backend)

    x = create_test_image((64, 64), dtype=torch.float32)
    y = op.forward_op(x)
    x_adj = op.adjoint_op(y)

    assert y.shape == (1, 1, 256)
    assert x_adj.shape == (1, 1, 64, 64)


@pytest.mark.parametrize("backend", ["finufft", "tkbnufft"])
def test_nufft_real_adjoint_consistency(backend: str) -> None:
    """For real_flag=True, Re(<Ax, y>) should match <x, adjoint(y)>."""
    op = _build_backend_op(backend)

    x = create_test_image((64, 64), dtype=torch.float32, seed=2)
    y = torch.randn((1, 1, 256), dtype=torch.complex64)

    Ax = op.forward_op(x)
    lhs = torch.sum(torch.conj(Ax) * y).item()

    Aty = op.adjoint_op(y)
    rhs = torch.sum(x * Aty).item()

    assert np.isclose(np.real(lhs), rhs, rtol=1e-4, atol=1e-7)


@pytest.mark.parametrize("backend", ["finufft", "tkbnufft"])
def test_nufft_zero_inputs(backend: str) -> None:
    """Both backends should map zeros to zeros in forward and adjoint paths."""
    op = _build_backend_op(backend)

    x0 = torch.zeros((1, 1, 64, 64), dtype=torch.float32)
    y0 = torch.zeros((1, 1, 256), dtype=torch.complex64)

    assert torch.allclose(op.forward_op(x0), torch.zeros((1, 1, 256), dtype=torch.complex64), atol=1e-6)
    assert torch.allclose(op.adjoint_op(y0), torch.zeros((1, 1, 64, 64), dtype=torch.float32), atol=1e-6)


@pytest.mark.parametrize("backend", ["finufft", "tkbnufft"])
def test_nufft_op_norm_is_positive(backend: str) -> None:
    """Both backends should return a finite positive operator norm."""
    op = _build_backend_op(backend)

    norm = op.get_op_norm(rel_tol=1e-4, max_iter=50)

    assert isinstance(norm, float)
    assert np.isfinite(norm)
    assert norm > 0.0


@pytest.mark.parametrize("backend", ["finufft", "tkbnufft"])
def test_nufft_float64_output_dtype(backend: str) -> None:
    """Both backends should produce complex128 measurements for float64 inputs."""
    if backend == "finufft":
        op = build_finufft_op(n_samples=256, img_size=(64, 64), dtype=torch.float64)
    else:
        op = build_tkbn_op(mode="table", n_samples=256, img_size=(64, 64), dtype=torch.float64)

    x = create_test_image((64, 64), dtype=torch.float64)
    y = op.forward_op(x)

    assert y.dtype == torch.complex128


@pytest.mark.parametrize("backend", ["finufft", "tkbnufft"])
def test_nufft_real_adjoint_consistency_float64(backend: str) -> None:
    """For real_flag=True and float64, Re(<Ax,y>) should match <x, adjoint(y)>."""
    if backend == "finufft":
        op = build_finufft_op(n_samples=256, img_size=(64, 64), dtype=torch.float64)
    else:
        op = build_tkbn_op(mode="table", n_samples=256, img_size=(64, 64), dtype=torch.float64)

    x = create_test_image((64, 64), dtype=torch.float64, seed=2)
    y = torch.randn((1, 1, 256), dtype=torch.complex128)

    Ax = op.forward_op(x)
    lhs = torch.sum(torch.conj(Ax) * y).item()

    Aty = op.adjoint_op(y)
    rhs = torch.sum(x * Aty).item()

    assert np.isclose(np.real(lhs), rhs, rtol=1e-8, atol=1e-15)


@pytest.mark.parametrize("backend", ["finufft", "tkbnufft"])
def test_nufft_op_norm_deterministic(backend: str) -> None:
    """Repeated op-norm calls should be deterministic for the same operator."""
    op = _build_backend_op(backend)

    norm1 = op.get_op_norm(rel_tol=1e-4, max_iter=100)
    norm2 = op.get_op_norm(rel_tol=1e-4, max_iter=100)

    assert np.isclose(norm1, norm2, rtol=1e-6)


@pytest.mark.parametrize("backend", ["finufft", "tkbnufft"])
def test_nufft_precond_weight_scales_forward(backend: str) -> None:
    """Changing preconditioning weights should scale forward measurements."""
    n_samples = 256
    img_size = (64, 64)

    if backend == "finufft":
        op_ref = build_finufft_op(n_samples=n_samples, img_size=img_size, dtype=torch.float32)
        op_scaled = build_finufft_op(n_samples=n_samples, img_size=img_size, dtype=torch.float32)
    else:
        op_ref = build_tkbn_op(mode="table", n_samples=n_samples, img_size=img_size, dtype=torch.float32)
        op_scaled = build_tkbn_op(mode="table", n_samples=n_samples, img_size=img_size, dtype=torch.float32)

    op_scaled.set_precond_weight(torch.full((1, 1, n_samples), 2.0, dtype=torch.complex64))

    x = create_test_image(img_size, dtype=torch.float32, seed=5)

    y_ref = op_ref.forward_op(x)
    y_scaled = op_scaled.forward_op(x)

    assert torch.allclose(y_scaled, 2.0 * y_ref, rtol=1e-2, atol=1e-6)


@pytest.mark.parametrize("backend", ["finufft", "tkbnufft"])
def test_nufft_data_weight_changes_output(backend: str) -> None:
    """Natural/image weights should affect forward output for both backends."""
    n_samples = 256
    img_size = (64, 64)

    if backend == "finufft":
        op_uniform = build_finufft_op(n_samples=n_samples, img_size=img_size, dtype=torch.float32)
    else:
        op_uniform = build_tkbn_op(mode="table", n_samples=n_samples, img_size=img_size, dtype=torch.float32)

    natural_weight = torch.ones((1, 1, n_samples), dtype=torch.float32)
    natural_weight[:, :, :64] = 0.1

    traj = op_uniform._traj

    # Rebuild backend op with changed natural weights.
    if backend == "finufft":
        from src.ri_measurement_operator import MeasOpFinufft

        assert traj.shape == (2, n_samples)  # FINUFFT trajectory shape
        v = traj[0:1, :].unsqueeze(0)
        u = traj[1:2, :].unsqueeze(0)
        op_weighted = MeasOpFinufft(
            u=u,
            v=v,
            img_size=img_size,
            natural_weight=natural_weight,
            image_weight=torch.ones((1, 1, n_samples), dtype=torch.float32),
            precond_weight=torch.ones((1, 1, n_samples), dtype=torch.float32),
            real_flag=True,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
    else:
        from src.ri_measurement_operator import MeasOpTkbn

        assert traj.shape == (1, 2, n_samples)  # KbNUFFT trajectory shape
        v = traj[:, 0:1, :]
        u = traj[:, 1:2, :]
        op_weighted = MeasOpTkbn(
            u=u,
            v=v,
            img_size=img_size,
            natural_weight=natural_weight,
            image_weight=torch.ones((1, 1, n_samples), dtype=torch.float32),
            precond_weight=torch.ones((1, 1, n_samples), dtype=torch.float32),
            real_flag=True,
            grid_size=(img_size[0] * 2, img_size[1] * 2),
            kernel_dim=7,
            mode="table",
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    x = create_test_image(img_size, dtype=torch.float32, seed=7)

    y_uniform = op_uniform.forward_op(x)
    y_weighted = op_weighted.forward_op(x)

    assert not torch.allclose(y_uniform, y_weighted, rtol=1e-5, atol=1e-7)
