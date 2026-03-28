"""
Unit tests for parsing_parameters in interface I/O utilities.
"""

import argparse
import json
from pathlib import Path

import pytest

from src.interface.io import parsing_parameters


def _write_config(tmp_path: Path, algorithm: str = "airi") -> Path:
    """Create a minimal valid config JSON file for parsing_parameters tests."""
    config_obj = [
        {
            "__comment": "Main",
            "main": {
                "src_name": "base-src",
                "data_file": "data/base.mat",
                "result_path": "results/base",
                "algorithm": algorithm,
                "im_dim_x": 64,
                "im_dim_y": 64,
                "im_pixel_size": 1.5,
                "superresolution": 1.0,
                "groundtruth": "",
                "run_id": 1,
            },
        },
        {
            "__comment": "General",
            "flag": {
                "flag_comment_1": "flags",
                "flag_imaging": True,
                "flag_data_weighting": True,
                "verbose": False,
            },
            "weighting": {
                "weighting_comment_1": "weights",
                "weight_type": "briggs",
                "weight_robustness": 0,
            },
            "computing": {
                "computing_comment_1": "compute",
                "meas_device": "cpu",
                "meas_dtype": "single",
            },
            "nufft": {
                "nufft_comment_1": "nufft",
                "nufft_package": "finufft",
            },
        },
        {
            "__comment": "Denoisers",
            "airi": {
                "denoiser_comment_1": "airi",
                "heu_noise_scale": 1.0,
                "dnn_shelf_path": "airi_denoisers/from_config.csv",
            },
            "airi_default": {
                "denoiser_comment_2": "airi defaults",
                "im_min_itr": 10,
                "im_max_itr": 100,
            },
            "cairi": {
                "denoiser_comment_1": "cairi",
                "heu_noise_scale": 1.2,
                "dnn_shelf_path": "airi_denoisers/cairi_from_config.csv",
            },
            "cairi_default": {
                "denoiser_comment_2": "cairi defaults",
                "im_min_itr": 11,
                "im_max_itr": 101,
            },
            "usara": {
                "denoiser_comment_1": "usara",
                "heu_reg_param_scale": 1.0,
                "reweighting": True,
            },
            "usara_default": {
                "denoiser_comment_2": "usara defaults",
                "im_min_itr": 20,
                "im_max_itr": 200,
            },
            "hyperairi": {
                "denoiser_comment_1": "hyperairi",
                "heu_noise_scale": 0.9,
                "dnn_shelf_path": "airi_denoisers/hyper_from_config.csv",
            },
            "hyperairi_default": {
                "denoiser_comment_2": "hyperairi defaults",
                "im_min_itr": 30,
                "im_max_itr": 300,
            },
            "hyper-usara": {
                "denoiser_comment_1": "hyper-usara",
                "heu_reg_param_scale": 0.8,
                "reweighting": False,
            },
            "hyper-usara_default": {
                "denoiser_comment_2": "hyper-usara defaults",
                "im_min_itr": 40,
                "im_max_itr": 400,
            },
        },
    ]

    config_path = tmp_path / "config_test.json"
    config_path.write_text(json.dumps(config_obj), encoding="utf-8")
    return config_path


def test_parsing_parameters_merges_main_general_and_algo_sections(tmp_path: Path) -> None:
    """Config-only parse should merge main + general + algorithm-specific sections."""
    config_path = _write_config(tmp_path, algorithm="airi")

    param = parsing_parameters(str(config_path))

    # main
    assert param["algorithm"] == "airi"
    assert param["im_dim_x"] == 64

    # general
    assert param["flag_imaging"] is True
    assert param["weight_type"] == "briggs"
    assert param["meas_device"] == "cpu"
    assert param["nufft_package"] == "finufft"

    # algorithm-specific
    assert param["heu_noise_scale"] == 1.0
    assert param["im_min_itr"] == 10
    assert param["im_max_itr"] == 100
    assert param["dnn_shelf_path"] == "airi_denoisers/from_config.csv"

    # comment keys should not be propagated
    assert "flag_comment_1" not in param
    assert "weighting_comment_1" not in param
    assert "denoiser_comment_1" not in param


def test_parsing_parameters_applies_cli_overrides(tmp_path: Path) -> None:
    """CLI args should override matching main fields and dnn_shelf_path when provided."""
    config_path = _write_config(tmp_path, algorithm="airi")

    input_param = argparse.Namespace(
        config="unused.json",
        src_name="override-src",
        data_file=None,
        result_path=None,
        algorithm=None,
        im_dim_x=128,
        im_dim_y=None,
        dnn_shelf_path="airi_denoisers/from_cli.csv",
        im_pixel_size=None,
        superresolution=None,
        ch_start=None,
        ch_last=None,
        ch_effective=None,
        ch_partition=None,
        run_id=None,
    )

    param = parsing_parameters(str(config_path), input_param=input_param)

    # main overrides
    assert param["src_name"] == "override-src"
    assert param["im_dim_x"] == 128

    # non-overridden key remains from config
    assert param["algorithm"] == "airi"

    # explicit dnn_shelf_path override
    assert param["dnn_shelf_path"] == "airi_denoisers/from_cli.csv"


def test_parsing_parameters_raises_for_unknown_algorithm(tmp_path: Path) -> None:
    """Unknown algorithm should raise NotImplementedError."""
    config_path = _write_config(tmp_path, algorithm="unknown_algo")

    with pytest.raises(NotImplementedError, match="Algorithm unknown_algo not found"):
        parsing_parameters(str(config_path))
