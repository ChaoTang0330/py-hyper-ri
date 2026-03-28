"""Builder utilities and specific tests for ellipsoid regulariser."""

import torch

from src.regulariser.prox_ellipsoid import ProxEllipsoid

from .regulariser_helpers import make_measurement_center_like, make_measurement_list


def create_prox_ellipsoid(
    channels: int,
    n_meas: int = 64,
    meas: list[torch.Tensor] | None = None,
    dtype: torch.dtype = torch.complex64,
    seed: int = 0,
) -> tuple[ProxEllipsoid, list[torch.Tensor]]:
    """Create ellipsoid regulariser and input for common tests.

    Ellipsoid operates in measurement domain represented as a list of channel tensors.
    """
    if meas is None:
        x = make_measurement_list(
            channels=channels,
            n_meas=n_meas,
            dtype=dtype,
            seed=seed,
        )
    else:
        x = meas
    center = [x_ch.clone() for x_ch in x]

    reg = ProxEllipsoid(
        center=center,
        precond_weight=[torch.ones(1, dtype=dtype) for _ in range(channels)],
        itr_min=1,
        itr_max=5,
        rel_var_tol=1e-6,
        dtype=dtype,
        verbose=False,
    )

    reg.update()
    return reg, x


def test_ellipsoid_keeps_feasible_point_unchanged() -> None:
    """If input is inside the l2 ball, projection should be identity."""
    channels = 2
    dtype = torch.float32
    x = make_measurement_list(channels=channels, n_meas=16, dtype=dtype, seed=5)
    center = make_measurement_center_like(x)
    radius = torch.full((channels,), 100.0, dtype=dtype)

    reg = ProxEllipsoid(
        center=center,
        precond_weight=[torch.ones(1, dtype=dtype) for _ in range(channels)],
        radius=radius,
        itr_min=1,
        itr_max=3,
        rel_var_tol=1e-8,
        dtype=dtype,
        verbose=False,
    )
    reg.update()

    y = reg([x_ch.clone() for x_ch in x])
    for x_ch, y_ch in zip(x, y):
        assert torch.allclose(y_ch, x_ch, atol=1e-5, rtol=1e-5)


def test_ellipsoid_projects_infeasible_point_inside_ball() -> None:
    """If input is outside the l2 ball, projection must satisfy norm bound."""
    dtype = torch.float32
    x = [torch.ones((1, 1, 16), dtype=dtype) * 5.0]
    center = [torch.zeros_like(x[0])]
    radius = torch.tensor([1.0], dtype=dtype)

    reg = ProxEllipsoid(
        center=center,
        precond_weight=[torch.ones(1, dtype=dtype)],
        radius=radius,
        itr_min=1,
        itr_max=6,
        rel_var_tol=1e-8,
        dtype=dtype,
        verbose=False,
    )
    reg.update()

    y = reg([x[0].clone()])
    dist_to_center = torch.linalg.vector_norm(y[0] - center[0]).item()
    assert dist_to_center <= radius[0].item() + 1.0e-5
