"""Shared helper utilities for regulariser unit tests."""

import torch


def make_image_tensor(
    channels: int,
    img_size: tuple[int, int] = (32, 32),
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> torch.Tensor:
    """Create deterministic image tensor of shape (1, C, H, W)."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return torch.randn((1, channels, img_size[0], img_size[1]), generator=g, dtype=dtype)


def make_positive_threshold(
    channels: int,
    value: float,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create per-channel positive scalar tensor shaped (1, C, 1, 1)."""
    if value <= 0:
        raise ValueError("value must be positive")
    return torch.full((1, channels, 1, 1), float(value), dtype=dtype)


def make_measurement_list(
    channels: int,
    n_meas: int = 64,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> list[torch.Tensor]:
    """Create deterministic measurement-domain tensors as a channel list."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return [torch.randn((1, 1, n_meas), generator=g, dtype=dtype) for _ in range(channels)]


def make_measurement_center_like(x_list: list[torch.Tensor]) -> list[torch.Tensor]:
    """Create zero centers with the same tensor shapes as a measurement list."""
    return [torch.zeros_like(x) for x in x_list]


def assert_shape_dtype_finite_tensor(y: torch.Tensor, x: torch.Tensor) -> None:
    """Assert output tensor keeps shape/dtype and is finite."""
    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert torch.isfinite(y).all()


def assert_shape_dtype_finite_measurement_list(
    y: list[torch.Tensor],
    x: list[torch.Tensor],
) -> None:
    """Assert output measurement list keeps per-channel shape/dtype and is finite."""
    assert isinstance(y, list)
    assert len(y) == len(x)
    for y_ch, x_ch in zip(y, x):
        assert y_ch.shape == x_ch.shape
        assert y_ch.dtype == x_ch.dtype
        assert torch.isfinite(y_ch).all()
