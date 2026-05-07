"""
Unit tests for imaging weight generation utilities.
"""

import numpy as np
import torch

from src.hyri.interface.gen_imaging_weights import gen_imaging_weights, gen_imaging_weights_np


def _sample_uvn() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a compact synthetic uv sample with positive natural weights.
    """
    # Cover multiple quadrants and include exact boundaries -pi and pi.
    u = torch.tensor([[[-torch.pi, -1.70, -0.60, 0.40, 1.30, torch.pi]]], dtype=torch.float64)
    v = torch.tensor([[[torch.pi, -2.40, 1.80, -1.20, 0.20, -torch.pi]]], dtype=torch.float64)
    natural_weight = torch.tensor([[[1.0, 0.8, 1.2, 1.1, 0.9, 1.3]]], dtype=torch.float64)
    return u, v, natural_weight


def test_gen_imaging_weights_uniform_shape_and_positive() -> None:
    u, v, natural_weight = _sample_uvn()

    imaging_weight = gen_imaging_weights(
        u,
        v,
        natural_weight,
        im_size=(32, 32),
        weight_type="uniform",
        weight_gridsize=2.0,
        weight_robustness=0.0,
    )

    assert imaging_weight.shape == u.shape
    assert torch.isfinite(imaging_weight).all()
    assert (imaging_weight > 0).all()


def test_gen_imaging_weights_briggs_shape_and_positive() -> None:
    u, v, natural_weight = _sample_uvn()

    imaging_weight = gen_imaging_weights(
        u,
        v,
        natural_weight,
        im_size=(32, 32),
        weight_type="briggs",
        weight_gridsize=2.0,
        weight_robustness=0.0,
    )

    assert imaging_weight.shape == u.shape
    assert torch.isfinite(imaging_weight).all()
    assert (imaging_weight > 0).all()


def test_gen_imaging_weights_accepts_singleton_natural_weight() -> None:
    u, v, _ = _sample_uvn()
    singleton_natural_weight = torch.tensor([[[1.0]]], dtype=torch.float64)

    imaging_weight = gen_imaging_weights(
        u,
        v,
        singleton_natural_weight,
        im_size=(32, 32),
        weight_type="briggs",
    )

    assert imaging_weight.shape == u.shape
    assert torch.isfinite(imaging_weight).all()


def test_gen_imaging_weights_matches_numpy_uniform() -> None:
    u, v, natural_weight = _sample_uvn()

    w_torch = gen_imaging_weights(
        u,
        v,
        natural_weight,
        im_size=(32, 32),
        weight_type="uniform",
        weight_gridsize=2.0,
    )

    w_numpy = gen_imaging_weights_np(
        u.squeeze().cpu().numpy(),
        v.squeeze().cpu().numpy(),
        natural_weight.squeeze().cpu().numpy(),
        img_size=(32, 32),
        weight_type="uniform",
        weight_gridsize=2,
        weight_robustness=0.0,
    )

    w_numpy = torch.from_numpy(np.asarray(w_numpy).reshape(-1)).to(dtype=torch.float64)

    assert torch.allclose(w_torch.view(-1), w_numpy, atol=1e-12, rtol=1e-10)
