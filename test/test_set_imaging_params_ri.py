"""
Unit tests for set_imaging_params_ri in imaging parameter utilities.
"""

from pathlib import Path

import torch

from src.hyri.interface.imaging_param import set_imaging_params_ri


def _base_param_general() -> dict:
    """Return a minimal valid parameter dict shared by tests."""
    return {
        "algorithm": "airi",
        "data_file": "data/sample_meas.mat",
        "src_name": "unit-src",
        "result_path": "results",
        "verbose": True,
        "flag_imaging": True,
    }


def test_set_imaging_params_ri_airi_branch_fields_and_prefix(tmp_path: Path) -> None:
    """AIRI-specific fields should be set and file prefix should include run id."""
    param_general = _base_param_general()
    param_general.update(
        {
            "algorithm": "airi",
            "result_path": str(tmp_path),
            "heu_noise_scale": 1.25,
            "dnn_shelf_path": "airi_denoisers/shelf_test.csv",
            "approx_meas_op": True,
            "run_id": 11,
            "im_max_itr": 50,
        }
    )

    param_measop, param_proxop, param_optimiser = set_imaging_params_ri(param_general)

    assert param_optimiser["algorithm"] == "airi"
    assert param_optimiser["heu_noise_scale"] == 1.25
    assert param_optimiser["approx_meas_op"] is True
    assert param_proxop["dnn_shelf_path"] == "airi_denoisers/shelf_test.csv"
    assert "AIRI_heuScale_1.25_" in param_optimiser["file_prefix"]
    assert "runID_11_" in param_optimiser["file_prefix"]


def test_set_imaging_params_ri_usara_branch_defaults(tmp_path: Path) -> None:
    """uSARA branch should set regularisation fields and defaults correctly."""
    param_general = _base_param_general()
    param_general.update(
        {
            "algorithm": "usara",
            "result_path": str(tmp_path),
            "heu_reg_param_scale": 2.0,
            "im_max_outer_itr": 7,
            "im_var_outer_tol": 2e-4,
        }
    )

    _, _, param_optimiser = set_imaging_params_ri(param_general)

    assert param_optimiser["algorithm"] == "usara"
    assert param_optimiser["heu_reg_param_scale"] == 2.0
    assert param_optimiser["im_max_outer_itr"] == 7
    assert param_optimiser["im_var_outer_tol"] == 2e-4
    assert "uSARA_heuRegScale_2.0_" in param_optimiser["file_prefix"]


def test_set_imaging_params_ri_channel_partition_overrides_effective(
    tmp_path: Path,
) -> None:
    """Valid ch_partition should override ch_effective to partition length."""
    param_general = _base_param_general()
    param_general.update(
        {
            "result_path": str(tmp_path),
            "ch_effective": 10,
            "ch_partition": [2, 4, 6],
        }
    )

    _, _, param_optimiser = set_imaging_params_ri(param_general)

    assert param_optimiser["ch_partition"] == (2, 4, 6)
    assert param_optimiser["ch_effective"] == 3


def test_set_imaging_params_ri_invalid_channel_partition_falls_back(
    tmp_path: Path,
) -> None:
    """Invalid ch_partition entries should clear partition and keep original ch_effective."""
    param_general = _base_param_general()
    param_general.update(
        {
            "result_path": str(tmp_path),
            "ch_effective": 5,
            "ch_partition": [1, 0, 3],
        }
    )

    _, _, param_optimiser = set_imaging_params_ri(param_general)

    assert param_optimiser["ch_partition"] == []
    assert param_optimiser["ch_effective"] == 5


def test_set_imaging_params_ri_device_fallback_cpu(tmp_path: Path) -> None:
    """Invalid device names should fall back to CPU."""
    param_general = _base_param_general()
    param_general.update(
        {
            "result_path": str(tmp_path),
            "meas_device": "not-a-device",
            "regulariser_device": "also-invalid",
        }
    )

    param_measop, param_proxop, _ = set_imaging_params_ri(param_general)

    assert param_measop["device"] == torch.device("cpu")
    assert param_proxop["device"] == torch.device("cpu")


def test_set_imaging_params_ri_dtype_fallbacks(tmp_path: Path) -> None:
    """Invalid dtype strings should fall back to default (float for meas, float for prox)."""
    param_general = _base_param_general()
    param_general.update(
        {
            "result_path": str(tmp_path),
            "meas_dtype": "invalid-dtype",
            "regulariser_dtype": "invalid-dtype",
        }
    )

    param_measop, param_proxop, _ = set_imaging_params_ri(param_general)

    # Invalid dtype falls back to defaults
    assert param_measop["dtype"] == torch.double  # Default for meas
    assert param_proxop["dtype"] == torch.float  # Default for prox


def test_set_imaging_params_ri_dist_init_method_validation(tmp_path: Path) -> None:
    """Invalid dist_init_method protocol should fall back to env://."""
    param_general = _base_param_general()
    param_general.update(
        {
            "result_path": str(tmp_path),
            "dist_init_method": "badproto://abc",
        }
    )

    _, _, param_optimiser = set_imaging_params_ri(param_general)

    assert param_optimiser["dist_init_method"] == "env://"


def test_set_imaging_params_ri_hyperairi_branch_fields(tmp_path: Path) -> None:
    """HyperAIRI-specific branch should set dnn fields and file prefix correctly."""
    param_general = _base_param_general()
    param_general.update(
        {
            "algorithm": "hyperairi",
            "result_path": str(tmp_path),
            "heu_noise_scale": 0.9,
            "dnn_shelf_path": "airi_denoisers/shelf_hyper.csv",
            "dnn_adaptive_peak": False,
            "dnn_adaptive_peak_tol_max": 0.2,
            "dnn_adaptive_peak_tol_min": 5e-4,
            "dnn_adaptive_peak_tol_step": 0.05,
            "dnn_apply_transform": False,
        }
    )

    _, param_proxop, param_optimiser = set_imaging_params_ri(param_general)

    assert param_optimiser["algorithm"] == "hyperairi"
    assert param_optimiser["heu_noise_scale"] == 0.9
    assert param_optimiser["dnn_adaptive_peak"] is False
    assert param_optimiser["dnn_adaptive_peak_tol_max"] == 0.2
    assert param_optimiser["dnn_adaptive_peak_tol_min"] == 5e-4
    assert param_optimiser["dnn_adaptive_peak_tol_step"] == 0.05
    assert param_proxop["dnn_apply_transform"] is False
    assert param_proxop["dnn_shelf_path"] == "airi_denoisers/shelf_hyper.csv"
    assert "HyperAIRI_heuScale_0.9_" in param_optimiser["file_prefix"]


def test_set_imaging_params_ri_cairi_branch_precond_flag(tmp_path: Path) -> None:
    """cAIRI branch should set precond_flag and file prefix correctly."""
    param_general = _base_param_general()
    param_general.update(
        {
            "algorithm": "cairi",
            "result_path": str(tmp_path),
            "heu_noise_scale": 0.95,
            "precond_flag": False,
            "dnn_shelf_path": "airi_denoisers/shelf_cairi.csv",
        }
    )

    _, param_proxop, param_optimiser = set_imaging_params_ri(param_general)

    assert param_optimiser["algorithm"] == "cairi"
    assert param_optimiser["heu_noise_scale"] == 0.95
    assert param_optimiser["precond_flag"] is False
    assert param_proxop["dnn_shelf_path"] == "airi_denoisers/shelf_cairi.csv"
    assert "cAIRI_heuScale_0.95_" in param_optimiser["file_prefix"]


def test_set_imaging_params_ri_minimal_params_all_defaults(tmp_path: Path) -> None:
    """Minimal param_general should produce all documented defaults."""
    param_general = {
        "algorithm": "airi",
        "data_file": "data/minimal.mat",
        "result_path": str(tmp_path),
    }

    param_measop, param_proxop, param_optimiser = set_imaging_params_ri(param_general)

    # Image size defaults
    assert param_optimiser["img_size"] == (512, 512)

    # Channel defaults
    assert param_optimiser["ch_start"] == 1
    assert param_optimiser["ch_last"] is None
    assert param_optimiser["ch_effective"] == 1
    assert param_optimiser["ch_partition"] == ()

    # Weighting defaults
    assert param_measop["weight_type"] == "briggs"
    assert param_measop["weight_robustness"] == 0.0
    assert param_measop["weight_gridsize"] == 2
    assert param_measop["flag_data_weighting"] is True

    # NUFFT defaults
    assert param_measop["nufft_package"] == "finufft"
    assert param_measop["nufft_mode"] == "table"
    assert param_measop["nufft_kb_kernel_dim"] == 7

    # Iteration defaults
    assert param_optimiser["im_max_itr"] == 2000
    assert param_optimiser["im_min_itr"] == 200
    assert param_optimiser["im_var_tol"] == 1e-5
    assert param_optimiser["itr_save"] == 2001

    # AIRI-specific defaults
    assert param_optimiser["heu_noise_scale"] == 1.0
    assert param_optimiser["dnn_adaptive_peak"] is True
    assert param_optimiser["dnn_adaptive_peak_tol_max"] == 0.1
    assert param_optimiser["dnn_adaptive_peak_tol_min"] == 1e-3
    assert param_optimiser["dnn_adaptive_peak_tol_step"] == 0.1


def test_set_imaging_params_ri_image_size_validation(tmp_path: Path) -> None:
    """Invalid image dimensions should default to (512, 512)."""
    param_general = _base_param_general()
    param_general.update(
        {
            "result_path": str(tmp_path),
            "im_dim_x": 0,  # Invalid: <= 0
            "im_dim_y": -10,  # Invalid: <= 0
        }
    )

    _, _, param_optimiser = set_imaging_params_ri(param_general)

    assert param_optimiser["img_size"] == (512, 512)


def test_set_imaging_params_ri_nufft_oversampling_and_grid_size(tmp_path: Path) -> None:
    """NUFFT oversampling factor should compute correct grid size."""
    param_general = _base_param_general()
    param_general.update(
        {
            "result_path": str(tmp_path),
            "im_dim_x": 128,
            "im_dim_y": 64,
            "nufft_oversampling_factor": (1.5, 2.0),
        }
    )

    param_measop, _, _ = set_imaging_params_ri(param_general)

    # grid_size = oversampling * img_size
    assert param_measop["nufft_grid_size"] == (192, 128)


def test_set_imaging_params_ri_nufft_package_invalid_fallback(tmp_path: Path) -> None:
    """Invalid nufft_package should fall back to finufft."""
    param_general = _base_param_general()
    param_general.update(
        {
            "result_path": str(tmp_path),
            "nufft_package": "invalid-package",
        }
    )

    param_measop, _, _ = set_imaging_params_ri(param_general)

    assert param_measop["nufft_package"] == "finufft"


def test_set_imaging_params_ri_weight_type_choices(tmp_path: Path) -> None:
    """Weight type should accept valid choices only."""
    for weight_type in ["briggs", "uniform"]:
        param_general = _base_param_general()
        param_general.update(
            {
                "result_path": str(tmp_path),
                "weight_type": weight_type,
            }
        )

        param_measop, _, _ = set_imaging_params_ri(param_general)

        assert param_measop["weight_type"] == weight_type

    # Invalid weight type falls back to briggs
    param_general = _base_param_general()
    param_general.update(
        {
            "result_path": str(tmp_path),
            "weight_type": "invalid",
        }
    )

    param_measop, _, _ = set_imaging_params_ri(param_general)

    assert param_measop["weight_type"] == "briggs"


def test_set_imaging_params_ri_result_path_creation(tmp_path: Path) -> None:
    """Result path should be created from src_name and result_path."""
    result_dir = tmp_path / "results"
    param_general = _base_param_general()
    param_general.update(
        {
            "src_name": "test-source",
            "result_path": str(result_dir),
        }
    )

    _, _, param_optimiser = set_imaging_params_ri(param_general)

    expected = str(result_dir / "test-source")
    assert param_optimiser["result_path"] == expected
    assert Path(expected).exists()


def test_set_imaging_params_ri_src_name_from_data_file_basename(tmp_path: Path) -> None:
    """src_name should be extracted from data_file basename when not provided."""
    param_general = {
        "algorithm": "airi",
        "data_file": "data/my_measurement_20240101.mat",
        "result_path": str(tmp_path),
    }

    _, _, param_optimiser = set_imaging_params_ri(param_general)

    # File prefix should contain extracted src_name without .mat extension
    assert "my_measurement_20240101" in param_optimiser["result_path"]


def test_set_imaging_params_ri_nufft_mode_validation(tmp_path: Path) -> None:
    """NUFFT mode should accept only "table" or "matrix", else default to "table"."""
    # Test valid modes
    for mode in ["table", "matrix"]:
        param_general = _base_param_general()
        param_general.update({"result_path": str(tmp_path), "nufft_mode": mode})

        param_measop, _, _ = set_imaging_params_ri(param_general)

        assert param_measop["nufft_mode"] == mode

    # Test invalid mode falls back to "table"
    param_general = _base_param_general()
    param_general.update({"result_path": str(tmp_path), "nufft_mode": "invalid-mode"})

    param_measop, _, _ = set_imaging_params_ri(param_general)

    assert param_measop["nufft_mode"] == "table"


def test_set_imaging_params_ri_image_pixel_size(tmp_path: Path) -> None:
    """Image pixel size should be set when valid and None when invalid."""
    # Valid pixel size
    param_general = _base_param_general()
    param_general.update(
        {"result_path": str(tmp_path), "im_pixel_size": 0.5}
    )

    param_measop, _, _ = set_imaging_params_ri(param_general)

    assert param_measop["im_pixel_size"] == 0.5

    # Invalid pixel size (<=0) should fall back to None
    param_general = _base_param_general()
    param_general.update(
        {"result_path": str(tmp_path), "im_pixel_size": -0.1}
    )

    param_measop, _, _ = set_imaging_params_ri(param_general)

    assert param_measop["im_pixel_size"] is None


def test_set_imaging_params_ri_superresolution_factor(tmp_path: Path) -> None:
    """Superresolution factor should be set when >=1.0 and default to 1.0 otherwise."""
    # Valid superresolution
    param_general = _base_param_general()
    param_general.update(
        {"result_path": str(tmp_path), "superresolution": 2.5}
    )

    param_measop, _, _ = set_imaging_params_ri(param_general)

    assert param_measop["superresolution"] == 2.5

    # Invalid superresolution (<1.0) should fall back to 1.0
    param_general = _base_param_general()
    param_general.update(
        {"result_path": str(tmp_path), "superresolution": 0.5}
    )

    param_measop, _, _ = set_imaging_params_ri(param_general)

    assert param_measop["superresolution"] == 1.0


def test_set_imaging_params_ri_iteration_save_override(tmp_path: Path) -> None:
    """itr_save should be overridable and default to im_max_itr + 1 when invalid."""
    # Valid itr_save
    param_general = _base_param_general()
    param_general.update(
        {"result_path": str(tmp_path), "itr_save": 500}
    )

    _, _, param_optimiser = set_imaging_params_ri(param_general)

    assert param_optimiser["itr_save"] == 500

    # Invalid itr_save should default to im_max_itr + 1
    param_general = _base_param_general()
    param_general.update(
        {"result_path": str(tmp_path), "im_max_itr": 1000, "itr_save": 0}
    )

    _, _, param_optimiser = set_imaging_params_ri(param_general)

    assert param_optimiser["itr_save"] == 1001


def test_set_imaging_params_ri_peak_estimate_adaptive_selection(tmp_path: Path) -> None:
    """im_peak_est should be set when positive and None otherwise (AIRI-specific)."""
    # Valid peak estimate
    param_general = _base_param_general()
    param_general.update(
        {
            "algorithm": "airi",
            "result_path": str(tmp_path),
            "im_peak_est": 100.0,
        }
    )

    _, _, param_optimiser = set_imaging_params_ri(param_general)

    assert param_optimiser["im_peak_est"] == 100.0

    # Invalid peak estimate (<=0) should have None
    param_general = _base_param_general()
    param_general.update(
        {
            "algorithm": "airi",
            "result_path": str(tmp_path),
            "im_peak_est": 0.0,
        }
    )

    _, _, param_optimiser = set_imaging_params_ri(param_general)

    assert param_optimiser["im_peak_est"] is None


def test_set_imaging_params_ri_usara_reweighting_save(tmp_path: Path) -> None:
    """uSARA should support reweighting_save flag (defaults to False)."""
    # Default reweighting_save for uSARA
    param_general = _base_param_general()
    param_general.update(
        {"algorithm": "usara", "result_path": str(tmp_path)}
    )

    _, _, param_optimiser = set_imaging_params_ri(param_general)

    assert param_optimiser["reweighting_save"] is False

    # Override reweighting_save for uSARA
    param_general = _base_param_general()
    param_general.update(
        {
            "algorithm": "usara",
            "result_path": str(tmp_path),
            "reweighting_save": True,
        }
    )

    _, _, param_optimiser = set_imaging_params_ri(param_general)

    assert param_optimiser["reweighting_save"] is True


def test_set_imaging_params_ri_weight_robustness_and_gridsize(tmp_path: Path) -> None:
    """Weight robustness accepts any truthy value; gridsize validates > 0.0."""
    # Valid positive values
    param_general = _base_param_general()
    param_general.update(
        {
            "result_path": str(tmp_path),
            "weight_robustness": 0.5,
            "weight_gridsize": 3.5,
        }
    )

    param_measop, _, _ = set_imaging_params_ri(param_general)

    assert param_measop["weight_robustness"] == 0.5
    assert param_measop["weight_gridsize"] == 3.5

    # Negative weight_robustness is accepted (no range validation), but gridsize falls back
    param_general = _base_param_general()
    param_general.update(
        {
            "result_path": str(tmp_path),
            "weight_robustness": -1.0,
            "weight_gridsize": -2.0,
        }
    )

    param_measop, _, _ = set_imaging_params_ri(param_general)

    assert param_measop["weight_robustness"] == -1.0  # Accepts negative (only checks truthy)
    assert param_measop["weight_gridsize"] == 2  # Falls back due to <=0 validation

    # Zero weight_robustness should fall back to default
    param_general = _base_param_general()
    param_general.update(
        {
            "result_path": str(tmp_path),
            "weight_robustness": 0.0,
        }
    )

    param_measop, _, _ = set_imaging_params_ri(param_general)

    assert param_measop["weight_robustness"] == 0.0  # Defaults when falsy (0.0)
