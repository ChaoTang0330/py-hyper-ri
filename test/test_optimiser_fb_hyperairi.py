"""Build tests for FBHyperAIRI optimiser construction."""

from pathlib import Path

from hyri.optimiser import FBHyperAIRI

from .optimiser_helpers import make_test_meas
from .test_meas_op_hs import build_hs_op
from .test_regulariser_den_hyperairi import create_den_hyperairi


def test_fb_hyperairi_build(tmp_path: Path) -> None:
    """FBHyperAIRI should construct with minimal valid inputs."""
    channels = 2
    op = build_hs_op(channels=channels, n_samples=1024, img_size=(64, 64))
    meas, ch_freq = make_test_meas(op, channels=channels)
    reg, _ = create_den_hyperairi(channels=channels, img_size=(64, 64))

    optimiser = FBHyperAIRI(
        meas=meas,
        ch_freq=ch_freq,
        meas_op=op,
        regulariser=reg,
        im_max_itr=2,
        spec_update=False,
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
