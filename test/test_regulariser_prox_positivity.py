"""Builder utilities and specific tests for positivity regulariser."""

import torch

from src.hyri.regulariser.prox_positivity import ProxOpPositivity

from .regulariser_helpers import make_image_tensor


def create_prox_positivity(
    channels: int,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> tuple[ProxOpPositivity, torch.Tensor]:
    """Create positivity regulariser and input for common tests."""
    reg = ProxOpPositivity(dtype=dtype, verbose=False)
    x = make_image_tensor(channels=channels, img_size=(32, 32), dtype=dtype, seed=seed)
    reg.update(x)
    return reg, x


def test_positivity_clips_negative_values_to_zero() -> None:
    """Negative entries should be projected to zero."""
    reg = ProxOpPositivity(dtype=torch.float32, verbose=False)
    x = torch.tensor([[[[-1.0, 2.0], [-3.0, 4.0]]]], dtype=torch.float32)
    y = reg(x)

    assert y.min().item() >= 0.0
    assert y[0, 0, 0, 0].item() == 0.0
    assert y[0, 0, 1, 0].item() == 0.0


def test_positivity_keeps_nonnegative_values_unchanged() -> None:
    """Nonnegative input should pass through unchanged."""
    reg = ProxOpPositivity(dtype=torch.float32, verbose=False)
    x = torch.tensor([[[[0.0, 1.0], [2.5, 4.0]]]], dtype=torch.float32)
    y = reg(x)

    assert torch.allclose(y, x)
