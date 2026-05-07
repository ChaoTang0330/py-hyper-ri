"""Builder utilities and specific tests for Hyper-uSARA regulariser."""

import torch

from hyri.regulariser.prox_hyperusara import ProxHyperSARAPos

from .regulariser_helpers import make_image_tensor, make_positive_threshold


def create_prox_hyperusara(
    channels: int,
    img_size: tuple[int, int] = (32, 32),
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> tuple[ProxHyperSARAPos, torch.Tensor]:
    """Create Hyper-uSARA regulariser and input for common tests."""
    reg = ProxHyperSARAPos(
        img_size=img_size,
        rank_num_ch=channels,
        wl_dict=("dirac", "db1"),
        dec_lev=2,
        max_iter=3,
        obj_tol=1e-6,
        dtype=dtype,
        verbose=False,
    )
    reg.set_soft_thresholding_value(
        make_positive_threshold(1, value=1.0e-3, dtype=dtype)
    )
    reg.set_noise_floor_level(make_positive_threshold(1, value=1.0e-3, dtype=dtype))

    x = make_image_tensor(channels=channels, img_size=img_size, dtype=dtype, seed=seed)
    # Initialise reweighting with zero image as discussed.
    reg.update(torch.zeros_like(x))
    return reg, x


def test_hyper_usara_enforces_positivity_constraint() -> None:
    """Hyper-uSARA prox with positivity must output nonnegative tensor."""
    reg, _ = create_prox_hyperusara(channels=2, dtype=torch.float32, seed=3)
    x = -torch.ones((1, 2, 32, 32), dtype=torch.float32)
    y = reg(x)

    assert torch.all(y >= 0)


def test_hyper_usara_update_populates_weights() -> None:
    """Calling update should produce a finite tensor of reweighting coefficients."""
    reg, x = create_prox_hyperusara(channels=2, dtype=torch.float32, seed=7)
    reg.update(x)

    assert isinstance(reg._weights, torch.Tensor)
    assert reg._weights.numel() > 0
    assert torch.isfinite(reg._weights).all()


def test_hyper_usara_wavelet_dict_round_trip_reconstruction() -> None:
    """Forward then adjoint wavelet-dictionary transform should reconstruct input."""
    reg, x = create_prox_hyperusara(channels=2, dtype=torch.float32, seed=13)

    coeff = reg._wavedec2_dict(x)
    x_rec = reg._waverec2_dict(coeff)

    assert x_rec.shape == x.shape
    assert x_rec.dtype == x.dtype
    assert torch.allclose(x_rec, x, rtol=1e-5, atol=1e-6)
