"""
Unit tests for parsing_arguments in interface I/O utilities.
"""

import sys

import pytest

from hyri.interface.io import parsing_arguments


def test_parsing_arguments_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parser should return documented defaults when no optional args are provided."""
    monkeypatch.setattr(sys, "argv", ["run_imager.py"])

    args = parsing_arguments()

    assert args.config is None
    assert args.src_name is None
    assert args.data_file is None
    assert args.result_path is None
    assert args.algorithm is None
    assert args.im_dim_x is None
    assert args.im_dim_y is None
    assert args.dnn_shelf_path is None
    assert args.im_pixel_size is None
    assert args.superresolution is None
    assert args.ch_start is None
    assert args.ch_last is None
    assert args.ch_effective is None
    assert args.ch_partition is None
    assert args.run_id is None


def test_parsing_arguments_overrides_and_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parser should correctly read CLI overrides and apply expected types."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_imager.py",
            "-c",
            "config/example.json",
            "--src_name",
            "3c353",
            "--data_file",
            "data/sample.mat",
            "--result_path",
            "results/out",
            "--algorithm",
            "airi",
            "--im_dim_x",
            "256",
            "--im_dim_y",
            "128",
            "--dnn_shelf_path",
            "airi_denoisers/shelf_mrid.csv",
            "--im_pixel_size",
            "1.5",
            "--superresolution",
            "2.0",
            "--ch_start",
            "2",
            "--ch_last",
            "10",
            "--ch_effective",
            "5",
            "--ch_partition",
            "2",
            "4",
            "6",
            "--run_id",
            "7",
        ],
    )

    args = parsing_arguments()

    assert args.config == "config/example.json"
    assert args.src_name == "3c353"
    assert args.data_file == "data/sample.mat"
    assert args.result_path == "results/out"
    assert args.algorithm == "airi"
    assert args.im_dim_x == 256
    assert args.im_dim_y == 128
    assert args.dnn_shelf_path == "airi_denoisers/shelf_mrid.csv"
    assert args.im_pixel_size == 1.5
    assert args.superresolution == 2.0
    assert args.ch_start == 2
    assert args.ch_last == 10
    assert args.ch_effective == 5
    assert args.ch_partition == [2, 4, 6]
    assert args.run_id == 7


def test_parsing_arguments_rejects_unknown_algorithm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid algorithm names should be rejected by argparse choices."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_imager.py", "--algorithm", "clean"],
    )

    with pytest.raises(SystemExit) as exc_info:
        parsing_arguments()

    # argparse uses exit code 2 for CLI parsing errors
    assert exc_info.value.code == 2
