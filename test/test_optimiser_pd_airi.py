"""Build tests for PDAIRI optimiser construction."""

from pathlib import Path

from hyri.optimiser import PDAIRI

from .optimiser_helpers import make_test_meas
from .test_meas_op_hs import build_hs_op
from .test_regulariser_den_airi import create_den_airi
from .test_regulariser_prox_ellipsoid import create_prox_ellipsoid


def test_pd_airi_build(tmp_path: Path) -> None:
    """PDAIRI should construct with minimal dual/primal regularisers."""
    channels = 2
    op = build_hs_op(channels=channels, n_samples=1024, img_size=(64, 64))
    meas, ch_freq = make_test_meas(op, channels=channels)
    reg_prime, _ = create_den_airi(channels=channels)
    reg_dual, _ = create_prox_ellipsoid(
        channels=channels,
        meas=meas,
    )

    optimiser = PDAIRI(
        meas=meas,
        ch_freq=ch_freq,
        meas_op=op,
        regulariser_prime=reg_prime,
        regulariser_dual_data=reg_dual,
        im_max_itr=2,
        save_pth=str(tmp_path),
        verbose=False,
    )

    assert optimiser._rank_ch_num == channels
    assert optimiser._model.shape == (1, channels, 64, 64)
    assert len(optimiser._dual) == channels
    assert (tmp_path / "dirty.fits").is_file()
    assert (tmp_path / "psf.fits").is_file()

    optimiser.initialisation()
    optimiser.run()
    optimiser.finalisation()
