"""Shared helper functions for measurement-operator tests."""

import torch

DEFAULT_DEVICE = torch.device("cpu")


def create_random_uv(
    n_samples: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device = DEFAULT_DEVICE,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create random UV coordinates in [-pi, pi] with expected shape (1, 1, N)."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    u = (
        torch.rand((1, 1, n_samples), generator=g, device=device, dtype=dtype) * 2 - 1
    ) * torch.pi
    v = (
        torch.rand((1, 1, n_samples), generator=g, device=device, dtype=dtype) * 2 - 1
    ) * torch.pi
    return u, v


def create_test_image(
    img_size: tuple[int, int],
    dtype: torch.dtype = torch.float32,
    device: torch.device = DEFAULT_DEVICE,
    seed: int = 1,
) -> torch.Tensor:
    """Create random image with expected shape (1, 1, H, W)."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return torch.randn(
        (1, 1, img_size[0], img_size[1]), generator=g, dtype=dtype, device=device
    )
