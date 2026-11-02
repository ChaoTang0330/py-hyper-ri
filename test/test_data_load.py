"""
Unit tests for the data loading functionality of
src.hyri.interface.load_data.load_data_to_tensor
"""

import argparse
import json
import os
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.io import loadmat, savemat

import hyri.interface.load_data as load_data_module
from hyri.interface.imaging_param import set_imaging_params_ri
from hyri.interface.io import parsing_parameters
from hyri.interface.load_data import load_data_to_tensor

TEST_MEAS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "3c353_sim_vla_meas_hs"
)


def test_load_data_to_tensor_from_sample_meas() -> None:
    """
    Load sample measurements and verify core output structure and dimensions.
    """
    assert os.path.isdir(TEST_MEAS_PATH), f"Missing test data folder: {TEST_MEAS_PATH}"

    data = load_data_to_tensor(
        uv_file_path=TEST_MEAS_PATH,
        super_resolution=1.5,
        image_pixel_size=None,
        data_weighting=True,
        load_weight=False,
        img_size=(512, 512),
        weight_type="briggs",
        weight_robustness=0.0,
        precond_weighting=False,
        dtype=torch.float64,
        device=torch.device("cpu"),
        ch_start=1,
        ch_last=2,
        ch_effctive=2,
        verbose=False,
    )

    for key in (
        "u",
        "v",
        "w",
        "y",
        "natural_weight",
        "imaging_weight",
        "precond_weight",
        "freqs",
        "image_pixel_size",
        "max_proj_baseline",
    ):
        assert key in data

    assert isinstance(data["u"], tuple)
    assert isinstance(data["v"], tuple)
    assert isinstance(data["w"], tuple)
    assert isinstance(data["y"], tuple)
    assert len(data["u"]) == 2
    assert len(data["v"]) == 2
    assert len(data["w"]) == 2
    assert len(data["y"]) == 2

    for ch in range(2):
        u = data["u"][ch]
        v = data["v"][ch]
        w = data["w"][ch]
        y = data["y"][ch]
        nat_w = data["natural_weight"][ch]
        img_w = data["imaging_weight"][ch]

        assert u.ndim == 3
        assert u.shape[0] == 1 and u.shape[1] == 1 and u.shape[-1] > 0
        assert v.shape == u.shape
        assert w.shape == u.shape
        assert y.shape == u.shape
        assert nat_w.shape == u.shape
        assert img_w.shape == u.shape

        assert y.is_complex()
        assert torch.isfinite(u).all()
        assert torch.isfinite(v).all()
        assert torch.isfinite(w).all()
        assert torch.isfinite(nat_w).all()
        assert torch.isfinite(img_w).all()

        precond_w = data["precond_weight"][ch]
        assert precond_w.shape == (1, 1, 1)
        assert torch.allclose(precond_w, torch.ones_like(precond_w))

    freqs = data["freqs"]
    assert freqs.shape == (1, 2, 1, 1)
    assert torch.isfinite(freqs).all()
    assert (freqs > 0).all()

    assert float(data["image_pixel_size"]) > 0.0
    assert float(data["max_proj_baseline"]) > 0.0


@pytest.fixture
def measurement_dir(tmp_path: Path) -> Path:
    """Write six channels with known frequencies and channel-labelled samples."""
    savemat(
        tmp_path / "msSpecs.mat",
        {
            "freqs": np.arange(1, 7, dtype=float) * 1e9,
            "uvw": np.array(
                [[1.0, 2.0, 0.0], [2.0, 3.0, 0.0], [3.0, 4.0, 0.0], [4.0, 5.0, 0.0]]
            ),
        },
    )
    for channel in range(1, 7):
        savemat(
            tmp_path / f"data_ch_{channel}.mat",
            {"data_I": np.full(4, channel, dtype=np.complex128)},
        )
    return tmp_path


@pytest.mark.parametrize("ch_start", [1, 2])
def test_packaged_default_loads_all_remaining_channels(
    measurement_dir: Path, ch_start: int
) -> None:
    """An empty last-channel setting in the bundled config reaches the loader."""
    args = argparse.Namespace(
        data_file=str(measurement_dir),
        result_path=str(measurement_dir / "results"),
        ch_start=ch_start,
    )
    params = parsing_parameters(None, args)
    meas, _, optimiser = set_imaging_params_ri(params)
    assert optimiser["ch_last"] is None

    data = load_data_to_tensor(
        optimiser["data_file"],
        super_resolution=meas["superresolution"],
        image_pixel_size=meas["im_pixel_size"],
        data_weighting=False,
        ch_start=optimiser["ch_start"],
        ch_last=optimiser["ch_last"],
        ch_effctive=optimiser["ch_effective"],
        ch_partition=optimiser["ch_partition"],
        verbose=False,
    )

    channels = list(range(ch_start, 7))
    assert len(data["y"]) == 1
    assert data["y"][0].flatten().tolist() == [
        complex(channel) for channel in channels for _ in range(4)
    ]
    assert data["freqs"].item() == pytest.approx(np.mean(channels) * 1e9)


@pytest.mark.parametrize(
    ("ch_start", "partition", "expected_freqs_ghz"),
    [
        (1, [1, 2], [1.0, 2.5]),
        (1, [3, 2, 1], [2.0, 4.5, 6.0]),
        (2, [2, 1, 2], [2.5, 4.0, 5.5]),
        (2, [2, 2], [2.5, 4.5]),
    ],
)
def test_grouped_channel_frequencies(
    measurement_dir: Path,
    ch_start: int,
    partition: list[int],
    expected_freqs_ghz: list[float],
) -> None:
    """Average the physical frequencies belonging to each consecutive group."""
    data = load_data_to_tensor(
        str(measurement_dir),
        data_weighting=False,
        ch_start=ch_start,
        ch_last=ch_start + sum(partition) - 1,
        ch_effctive=len(partition),
        ch_partition=partition,
        verbose=False,
    )

    assert data["freqs"].flatten().tolist() == pytest.approx(
        [freq * 1e9 for freq in expected_freqs_ghz]
    )


@pytest.mark.parametrize(
    ("rank", "expected_offset", "expected_channels"),
    [(0, 0, [(1,)]), (1, 1, [(2, 3), (4, 5, 6)])],
)
def test_distributed_unequal_channel_groups(
    measurement_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    rank: int,
    expected_offset: int,
    expected_channels: list[tuple[int, ...]],
) -> None:
    """Each rank concatenates the physical channels in its assigned groups."""
    monkeypatch.setattr(load_data_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(load_data_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(load_data_module.dist, "get_rank", lambda: rank)
    monkeypatch.setattr(load_data_module.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(load_data_module.dist, "get_backend", lambda: "gloo")
    monkeypatch.setattr(load_data_module.dist, "barrier", lambda: None)
    monkeypatch.setattr(load_data_module.dist, "all_reduce", lambda tensor, op: None)

    data = load_data_to_tensor(
        str(measurement_dir),
        data_weighting=False,
        ch_start=1,
        ch_last=6,
        ch_effctive=3,
        ch_partition=[1, 2, 3],
        verbose=False,
    )

    assert data["rank_eff_ch_start"] == expected_offset
    assert len(data["y"]) == len(expected_channels)
    for visibilities, channels in zip(data["y"], expected_channels, strict=True):
        assert visibilities.flatten().tolist() == [
            complex(channel) for channel in channels for _ in range(4)
        ]
    assert data["freqs"].flatten().tolist() == pytest.approx([1e9, 2.5e9, 5e9])


def _load_with_geometry(
    measurement_dir: Path,
    config_values: dict,
    cli_values: dict,
    metadata_sr: float | None = 3.0,
) -> dict:
    """Resolve CLI/config geometry and load a small measurement dataset."""
    metadata_path = measurement_dir / "msSpecs.mat"
    metadata = {
        key: value
        for key, value in loadmat(metadata_path).items()
        if not key.startswith("__")
    }
    if metadata_sr is not None:
        metadata["super_resolution"] = metadata_sr
    savemat(metadata_path, metadata)

    config = json.loads(files("hyri").joinpath("config/default.json").read_text())
    main = config[0]["main"]
    main.pop("superresolution")
    main.pop("im_pixel_size")
    main.update(
        {
            "data_file": str(measurement_dir),
            "result_path": str(measurement_dir / "results"),
            "ch_last": 1,
            **config_values,
        }
    )
    config[1]["flag"]["verbose"] = False
    config_path = measurement_dir / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    params = parsing_parameters(str(config_path), argparse.Namespace(**cli_values))
    meas_params, _, _ = set_imaging_params_ri(params)

    return load_data_to_tensor(
        str(measurement_dir),
        super_resolution=meas_params["superresolution"],
        image_pixel_size=meas_params["im_pixel_size"],
        data_weighting=False,
        ch_last=1,
        ch_effctive=1,
        verbose=False,
    )


@pytest.mark.parametrize(
    ("config_values", "cli_values", "expected_sr"),
    [
        ({"superresolution": 2.0}, {"superresolution": 4.0}, 4.0),
        ({"superresolution": 2.0}, {}, 2.0),
        ({}, {"superresolution": 4.0}, 4.0),
        ({}, {}, 3.0),
        ({"superresolution": []}, {}, 3.0),
        ({"superresolution": None}, {}, 3.0),
    ],
)
def test_super_resolution_priority(
    measurement_dir: Path,
    config_values: dict,
    cli_values: dict,
    expected_sr: float,
) -> None:
    """CLI SR overrides config SR, which overrides measurement metadata."""
    data = _load_with_geometry(measurement_dir, config_values, cli_values)

    assert data["super_resolution"] == expected_sr
    assert data["image_pixel_size"] == pytest.approx(
        np.rad2deg(1.0) * 3600 / (2 * data["max_proj_baseline"] * expected_sr)
    )


def test_super_resolution_default_without_metadata(measurement_dir: Path) -> None:
    """Use the loader default only when no source supplies an SR factor."""
    data = _load_with_geometry(measurement_dir, {}, {}, metadata_sr=None)

    assert data["super_resolution"] == 1.5


@pytest.mark.parametrize(
    ("config_values", "cli_values", "expected_pixel_size"),
    [
        (
            {"superresolution": 2.0, "im_pixel_size": 1.0},
            {"superresolution": 4.0, "im_pixel_size": 0.5},
            0.5,
        ),
        (
            {"superresolution": 2.0, "im_pixel_size": 1.0},
            {"superresolution": 4.0},
            1.0,
        ),
        ({"superresolution": 2.0}, {"im_pixel_size": 0.5}, 0.5),
    ],
)
def test_explicit_pixel_size_overrides_super_resolution(
    measurement_dir: Path,
    config_values: dict,
    cli_values: dict,
    expected_pixel_size: float,
) -> None:
    """Explicit pixel size wins over SR from the CLI, config, and metadata."""
    data = _load_with_geometry(measurement_dir, config_values, cli_values)

    assert data["image_pixel_size"] == expected_pixel_size
    assert data["super_resolution"] == pytest.approx(
        np.rad2deg(1.0) * 3600 / (2 * data["max_proj_baseline"] * expected_pixel_size)
    )
