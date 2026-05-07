"""Build tests for FBSARA optimiser construction."""

from pathlib import Path

from src.hyri.optimiser import FBSARA

from .optimiser_helpers import make_test_meas
from .test_meas_op_hs import build_hs_op
from .test_regulariser_prox_usara import create_prox_usara


def test_fb_sara_build(tmp_path: Path) -> None:
    """FBSARA should construct with a small operator and regulariser."""
    channels = 2
    op = build_hs_op(channels=channels, n_samples=1024, img_size=(64, 64))
    meas, ch_freq = make_test_meas(op, channels=channels)
    reg, _ = create_prox_usara(channels=channels, img_size=(64, 64))

    optimiser = FBSARA(
        meas=meas,
        ch_freq=ch_freq,
        meas_op=op,
        regulariser=reg,
        im_max_itr=2,
        im_max_itr_outer=1,
        save_pth=str(tmp_path),
        verbose=False,
    )

    assert optimiser._rank_ch_num == channels
    assert optimiser._model.shape == (1, channels, 64, 64)
    assert (tmp_path / "dirty.fits").is_file()
    assert (tmp_path / "psf.fits").is_file()

    optimiser.initialisation()
    optimiser.run()
    optimiser.finalisation()
