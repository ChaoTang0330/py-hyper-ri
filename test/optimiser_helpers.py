"""Shared helper functions for optimiser build tests."""

import torch


def make_test_image(
    channels: int,
    img_size: tuple[int, int] = (32, 32),
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> torch.Tensor:
    """Create a deterministic hyperspectral test image."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return torch.randn((1, channels, *img_size), generator=g, dtype=dtype)


def make_test_meas(op, channels: int) -> tuple[torch.Tensor, ...]:
    """Generate synthetic measurements from a deterministic test image."""
    x = make_test_image(channels=channels, img_size=op.get_img_size(), dtype=op.get_data_type())
    ch_freq = torch.tensor([1e9 + 1e8 * ch for ch in range(channels)], dtype=op.get_data_type())
    return op.forward_op(x), ch_freq