"""Common unit tests for regularisers across regulariser types."""

import pytest
import torch

from .regulariser_helpers import (
    assert_shape_dtype_finite_measurement_list,
    assert_shape_dtype_finite_tensor,
)
from .test_regulariser_den_airi import create_den_airi
from .test_regulariser_den_hyperairi import create_den_hyperairi
from .test_regulariser_prox_ellipsoid import create_prox_ellipsoid
from .test_regulariser_prox_positivity import create_prox_positivity
from .test_regulariser_prox_sara import create_prox_sara


REGULARISERS = ["positivity", "sara", "denairi", "ellipsoid", "denhyperairi"]
REGULARISERS_SINGLE_CHANNEL = ["positivity", "sara", "denairi", "ellipsoid"]
REGULARISERS_TWO_CHANNEL = ["positivity", "sara", "denairi", "ellipsoid", "denhyperairi"]


def _build_regulariser(
    regulariser_type: str,
    channels: int,
    dtype: torch.dtype,
    seed: int,
):
    if regulariser_type == "positivity":
        return create_prox_positivity(channels=channels, dtype=dtype, seed=seed)
    if regulariser_type == "sara":
        return create_prox_sara(channels=channels, dtype=dtype, seed=seed)
    if regulariser_type == "denairi":
        return create_den_airi(channels=channels, dtype=dtype, seed=seed)
    if regulariser_type == "ellipsoid":
        return create_prox_ellipsoid(channels=channels, dtype=dtype, seed=seed)
    if regulariser_type == "denhyperairi":
        return create_den_hyperairi(channels=channels, dtype=dtype, seed=seed)
    raise ValueError(f"Unknown regulariser type: {regulariser_type}")


@pytest.mark.parametrize("regulariser_type", REGULARISERS_SINGLE_CHANNEL)
def test_regulariser_common_shape_dtype_finite_single_channel(
    regulariser_type: str,
) -> None:
    """Common checks for C=1: shape/dtype preservation and finite output."""
    reg, x = _build_regulariser(
        regulariser_type=regulariser_type,
        channels=1,
        dtype=torch.float32,
        seed=11,
    )
    y = reg(x)

    if isinstance(x, torch.Tensor):
        assert_shape_dtype_finite_tensor(y, x)
    else:
        assert_shape_dtype_finite_measurement_list(y, x)


@pytest.mark.parametrize("regulariser_type", REGULARISERS_TWO_CHANNEL)
def test_regulariser_common_shape_dtype_finite_two_channel(
    regulariser_type: str,
) -> None:
    """Common checks for C=2: shape/dtype preservation and finite output."""
    reg, x = _build_regulariser(
        regulariser_type=regulariser_type,
        channels=2,
        dtype=torch.float32,
        seed=12,
    )
    y = reg(x)

    if isinstance(x, torch.Tensor):
        assert_shape_dtype_finite_tensor(y, x)
    else:
        assert_shape_dtype_finite_measurement_list(y, x)


@pytest.mark.parametrize("regulariser_type", REGULARISERS)
def test_regulariser_common_deterministic(
    regulariser_type: str,
) -> None:
    """Common checks: deterministic output across fresh identical initialisations."""
    reg1, x1 = _build_regulariser(
        regulariser_type=regulariser_type,
        channels=2,
        dtype=torch.float32,
        seed=99,
    )
    reg2, x2 = _build_regulariser(
        regulariser_type=regulariser_type,
        channels=2,
        dtype=torch.float32,
        seed=99,
    )

    y1 = reg1(x1)
    y2 = reg2(x2)

    if isinstance(y1, torch.Tensor):
        assert torch.allclose(y1, y2)
    else:
        assert len(y1) == len(y2)
        for a, b in zip(y1, y2):
            assert torch.allclose(a, b)
