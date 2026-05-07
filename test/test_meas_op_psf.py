"""Tests for the PSF-based measurement operator."""

import torch

from src.hyri.measurement_operator import MeasOpPSF

from .meas_op_helpers import create_random_uv, create_test_image
from .test_meas_op_finufft import build_finufft_op


def build_psf_op(
    n_samples: int = 512,
    img_size: tuple[int, int] = (64, 64),
    dtype: torch.dtype = torch.float32,
    normalise_psf: bool = False,
) -> MeasOpPSF:
    """Create a PSF-based measurement operator with deterministic UV sampling."""
    u, v = create_random_uv(n_samples=n_samples, dtype=dtype)

    return MeasOpPSF(
        u=u,
        v=v,
        img_size=img_size,
        natural_weight=torch.ones((1, 1, n_samples), dtype=dtype),
        image_weight=torch.ones((1, 1, n_samples), dtype=dtype),
        precond_weight=torch.ones((1, 1, n_samples), dtype=dtype),
        normalise_psf=normalise_psf,
        device=torch.device("cpu"),
        dtype=dtype,
    )


def test_psf_build_and_run() -> None:
    """Constructed PSF meas op should run forward and adjoint."""
    op = build_psf_op(n_samples=128, img_size=(32, 32), dtype=torch.float32)
    x = create_test_image((32, 32), dtype=torch.float32)

    y = op.forward_op(x)
    x_adj = op.adjoint_op(y)

    # PSF forward is identity; adjoint returns image-domain tensor.
    assert y.shape == (1, 1, 32, 32)
    assert y.dtype == torch.float32
    assert x_adj.shape == (1, 1, 32, 32)
    assert x_adj.dtype == torch.float32


def test_psf_ata_consistent_with_finufft() -> None:
    """PSF A*A should be consistent with FINUFFT A*A for same sampling/weights.

    Intentionally does not test inner-product adjointness; this checks consistency of
    adjoint(forward(x)) between PSF and FINUFFT implementations.
    """
    n_samples = 4096
    img_size = (64, 64)
    dtype = torch.float32

    op_psf = build_psf_op(n_samples=n_samples, img_size=img_size, dtype=dtype)
    op_finufft = build_finufft_op(n_samples=n_samples, img_size=img_size, dtype=dtype)

    x = create_test_image(img_size, dtype=dtype, seed=3)

    ata_psf = op_psf.adjoint_op(op_psf.forward_op(x))
    ata_finufft = op_finufft.adjoint_op(op_finufft.forward_op(x))

    assert ata_psf.shape == ata_finufft.shape

    # The PSF-based convolution is an approximation of FINUFFT A*A; allow small mismatch.
    diff_norm = torch.linalg.vector_norm(ata_psf - ata_finufft)
    ref_norm = torch.linalg.vector_norm(ata_finufft).clamp_min(1e-12)
    rel_err = (diff_norm / ref_norm).item()

    assert rel_err < 1e-5, f"PSF vs FINUFFT A*A relative error too high: {rel_err:.4f}"
