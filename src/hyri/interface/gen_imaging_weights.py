"""Generate imaging-weight arrays from UV sampling distributions.

This file provides Torch and NumPy implementations of uniform and Briggs
weighting schemes used for RI imaging.
"""

from typing import Literal

import numpy as np
import torch


def gen_imaging_weights(
    u: torch.Tensor,
    v: torch.Tensor,
    natural_weight: torch.Tensor,
    im_size: tuple[int, int],
    weight_type: str = "briggs",
    weight_gridsize: float = 2.0,
    weight_robustness: float = 0.0,
) -> torch.Tensor:
    """Generate imaging weights from UV sampling density using PyTorch.

    Args:
        u (torch.Tensor): U-coordinates in Fourier radians.
        v (torch.Tensor): V-coordinates in Fourier radians.
        natural_weight (torch.Tensor): Natural weights (square root of inverse
            variance), either scalar-like or per-visibility.
        im_size (Tuple[int, int]): Image size used to construct the UV grid.
        weight_type (str, optional): Weighting mode. Supported values are
            ``"uniform"`` and ``"briggs"``. Default: "briggs".
        weight_gridsize (float, optional): Grid oversampling factor for
            density estimation. Default: 2.0.
        weight_robustness (float, optional): Briggs robustness parameter.
            Default: 0.0.

    Returns:
        torch.Tensor: Imaging weights with shape ``(1, 1, N)``.

    Raises:
        NotImplementedError: If ``weight_type`` is unsupported.

    Notes:
        Coordinates are mirrored to the upper half-plane before gridding to
        exploit conjugate symmetry. ``natural_weight`` may be scalar-like or
        per-visibility; a scalar-like input is broadcast to all samples.

    Examples:
        Generate Briggs weights for torch tensors:

        >>> imaging_weight = gen_imaging_weights(
        ...     torch.tensor([0.0, 0.1]),
        ...     torch.tensor([0.0, -0.1]),
        ...     torch.tensor([1.0, 1.0]),
        ...     (512, 512),
        ... )
        >>> imaging_weight.shape
        torch.Size([1, 1, 2])
    """

    u = u.clone()
    v = v.clone()

    nmeas = u.numel()
    weight_grid_size = torch.tensor(
        [np.floor(i * weight_gridsize).astype(int) for i in im_size], device=u.device
    )
    natural_weight_2 = (natural_weight.view(-1).to(u.device)) ** 2

    if natural_weight_2.size(-1) == 1:
        natural_weight_2 = natural_weight_2 * torch.ones(nmeas, device=u.device)
    u[v < 0] = -u[v < 0]
    v[v < 0] = -v[v < 0]

    # Initialise gridded weights vector for UV density accumulation.
    p = (
        torch.floor((v + torch.pi) * weight_grid_size[0] / 2 / torch.pi)
        .to(torch.int64)
        .view(-1)
        - 1
    )
    q = (
        torch.floor((u + torch.pi) * weight_grid_size[1] / 2 / torch.pi)
        .to(torch.int64)
        .view(-1)
        - 1
    )

    # Handle edge cells wrapped to -1 by floor operation.
    p[p == -1] = weight_grid_size[0] - 1
    q[q == -1] = weight_grid_size[1] - 1

    gridded_weights = torch.zeros(
        torch.prod(weight_grid_size), dtype=torch.float64, device=u.device
    )
    uv_idx = p * weight_grid_size[1] + q
    if weight_type != "none":
        if weight_type == "uniform":
            values = torch.ones_like(p, dtype=torch.float64)
        elif weight_type == "briggs":
            values = natural_weight_2
        else:
            raise NotImplementedError(
                "Not implemented image weighting type: " + weight_type
            )

    # Accumulate gridded UV-density weights.
    gridded_weights.scatter_add_(0, uv_idx, values.to(torch.float64))

    # Convert gridded UV density to per-sample imaging weights.
    if weight_type != "none":
        gridded_vals = gridded_weights[uv_idx]  # Gather the values

        if weight_type == "uniform":
            imaging_weight = 1 / torch.sqrt(gridded_vals)
        elif weight_type == "briggs":
            # Compute robust scale factor
            robust_scale = (
                torch.sum(natural_weight_2) / torch.sum(gridded_weights**2)
            ) * (5 * 10 ** (-weight_robustness)) ** 2
            imaging_weight = 1 / torch.sqrt(1 + robust_scale * gridded_vals)
    return imaging_weight.view(1, 1, -1)


def gen_imaging_weights_np(
    u: np.ndarray,
    v: np.ndarray,
    natural_weight: np.ndarray | None = None,
    img_size: tuple[int, int] = (512, 512),
    weight_type: Literal["uniform", "briggs", "robust"] = "uniform",
    weight_gridsize: int = 2,
    weight_robustness: float = 0.0,
) -> np.ndarray:
    """Generate imaging weights from UV sampling density using NumPy.

    Args:
        u (np.ndarray): U-coordinates of the sampling pattern.
        v (np.ndarray): V-coordinates of the sampling pattern.
        natural_weight (np.ndarray, optional): Natural weights for robust
            weighting. Required for ``"robust"``/``"briggs"`` modes.
            Default: None.
        img_size (Tuple[int, int], optional): Image size used to construct the
            UV grid. Default: (512, 512).
        weight_type (Literal["uniform", "briggs", "robust"], optional):
            Weighting mode. ``"robust"`` and ``"briggs"`` are treated
            identically. Default: "uniform".
        weight_gridsize (int, optional): Grid oversampling factor for density
            estimation. Default: 2.
        weight_robustness (float, optional): Briggs robustness parameter.
            Default: 0.0.

    Returns:
        np.ndarray: Imaging weights with shape ``(N, 1)``.

    Raises:
        NotImplementedError: If ``weight_type`` is unsupported.

    Notes:
        Coordinates are mirrored to the upper half-plane before gridding to
        exploit conjugate symmetry. ``natural_weight`` is required for
        ``"briggs"`` and ``"robust"`` weighting.

    Examples:
        Generate uniform imaging weights for NumPy arrays:

        >>> weights = gen_imaging_weights_np(
        ...     np.array([0.0, 0.1]),
        ...     np.array([0.0, -0.1]),
        ...     img_size=(512, 512),
        ... )
        >>> weights.shape
        (2, 1)
    """
    # Flatten U/V vectors.
    u = u.reshape((-1, 1)).astype(np.double)
    v = v.reshape((-1, 1)).astype(np.double)

    # Consider only one half-plane using conjugate symmetry.
    u[v < 0] = -u[v < 0]
    v[v < 0] = -v[v < 0]

    # Initialise intermediate buffers.
    nmeas = u.size
    weight_grid_size = np.floor(
        np.array((img_size[0] * weight_gridsize, img_size[1] * weight_gridsize))
    )
    gridded_weight = np.zeros(weight_grid_size.astype(int), dtype=np.double)
    image_weight = np.ones((nmeas, 1))

    # Grid UV sample coordinates.
    q = np.floor((u + np.pi) * weight_grid_size[1] / 2.0 / np.pi).astype(int) - 1
    p = np.floor((v + np.pi) * weight_grid_size[0] / 2.0 / np.pi).astype(int) - 1

    # Handle edge cells wrapped to -1 by floor operation.
    q[q == -1] = weight_grid_size[1].astype(int) - 1
    p[p == -1] = weight_grid_size[0].astype(int) - 1

    if weight_type == "uniform":
        for idx in range(nmeas):
            gridded_weight[p[idx], q[idx]] += 1.0
        # Apply uniform weighting.
        image_weight = 1.0 / np.sqrt(gridded_weight[p, q])
    elif weight_type in ["robust", "briggs"]:
        # Use inverse-noise-variance weighting for Briggs mode.
        natural_weight = natural_weight.reshape((-1, 1)).astype(np.double)
        natural_weight2 = natural_weight**2
        if natural_weight2.size == 1:
            natural_weight2 = natural_weight2[0] * np.ones((nmeas, 1))
        for idx in range(nmeas):
            gridded_weight[p[idx], q[idx]] += natural_weight2[idx]
        # Compute robust scale factor.
        robust_scale = (np.sum(gridded_weight) / np.sum(gridded_weight**2)) * (
            5 * 10 ** (-weight_robustness)
        ) ** 2
        # Apply Briggs weighting.
        image_weight = 1.0 / np.sqrt(1.0 + robust_scale * gridded_weight[p, q])
    else:
        raise NotImplementedError(
            "Not implelemnted image weighting type: " + weight_type
        )

    return image_weight
