"""Unit tests for Measurement Set to MAT conversion helpers."""

import numpy as np
import pytest

import hyri.interface.ms2mat as ms2mat_module
from hyri.interface.ms2mat import (
    _imaging_weight_multiplier,
    _print_dry_run_summary,
    _sorted_channels,
    _stokes_i_and_weight,
    convert_ms_to_mat,
    parse_ms2mat_arguments,
)


def test_sorted_channels_orders_low_to_high_and_slices_one_based() -> None:
    """SPW channels should be grouped and sorted by observation frequency."""
    chan_freq = np.array(
        [
            [3.0, 1.0],
            [4.0, 2.0],
        ]
    )

    channels = _sorted_channels(chan_freq, ch_start=2, ch_last=3)

    assert [channel.frequency_hz for channel in channels] == [2.0, 3.0]
    assert [channel.output_index for channel in channels] == [2, 3]
    assert [channel.members for channel in channels] == [
        (ms2mat_module.MsChannelMember(spectral_window_id=1, channel_id=1),),
        (ms2mat_module.MsChannelMember(spectral_window_id=0, channel_id=0),),
    ]


def test_sorted_channels_slice_keeps_global_output_index() -> None:
    """Partial conversion should keep data_ch indices consistent with the MS."""
    channels = _sorted_channels(np.array([10.0, 20.0, 30.0]), ch_start=2)

    assert [channel.frequency_hz for channel in channels] == [20.0, 30.0]
    assert [channel.output_index for channel in channels] == [2, 3]


def test_sorted_channels_accepts_ragged_spws() -> None:
    """SPW frequency sequences should sort globally."""
    channels = _sorted_channels(
        [
            np.array([5.0, 1.0]),
            np.array([3.0]),
        ]
    )

    assert [channel.frequency_hz for channel in channels] == [1.0, 3.0, 5.0]
    assert [channel.members for channel in channels] == [
        (ms2mat_module.MsChannelMember(spectral_window_id=0, channel_id=1),),
        (ms2mat_module.MsChannelMember(spectral_window_id=1, channel_id=0),),
        (ms2mat_module.MsChannelMember(spectral_window_id=0, channel_id=0),),
    ]


def test_sorted_channels_groups_shared_frequencies() -> None:
    """Repeated SPW frequencies should become one output channel."""
    channels = _sorted_channels(
        [
            np.array([5.0, 1.0]),
            np.array([1.0, 3.0]),
        ]
    )

    assert [channel.frequency_hz for channel in channels] == [1.0, 3.0, 5.0]
    assert channels[0].members == (
        ms2mat_module.MsChannelMember(spectral_window_id=0, channel_id=1),
        ms2mat_module.MsChannelMember(spectral_window_id=1, channel_id=0),
    )


def test_sorted_channels_groups_with_frequency_tolerance() -> None:
    """Optional tolerance should merge nearly equal frequencies after sorting."""
    channels = _sorted_channels(
        [
            np.array([1.0, 2.0]),
            np.array([1.01]),
        ],
        freq_tol_hz=0.02,
    )

    assert len(channels) == 2
    assert channels[0].frequency_hz == pytest.approx(1.005)
    assert channels[0].members == (
        ms2mat_module.MsChannelMember(spectral_window_id=0, channel_id=0),
        ms2mat_module.MsChannelMember(spectral_window_id=1, channel_id=0),
    )


def test_sorted_channels_caps_tolerance_below_same_spw_spacing() -> None:
    """A large tolerance should not merge neighbouring channels within one SPW."""
    channels = _sorted_channels(np.array([1.0, 1.01]), freq_tol_hz=0.02)

    assert [channel.frequency_hz for channel in channels] == [1.0, 1.01]
    assert [channel.members for channel in channels] == [
        (ms2mat_module.MsChannelMember(spectral_window_id=0, channel_id=0),),
        (ms2mat_module.MsChannelMember(spectral_window_id=0, channel_id=1),),
    ]


def test_sorted_channels_rejects_invalid_range() -> None:
    """Channel ranges should be 1-based and inclusive."""
    with pytest.raises(ValueError, match="ch_start"):
        _sorted_channels(np.array([1.0]), ch_start=0)
    with pytest.raises(ValueError, match="ch_last"):
        _sorted_channels(np.array([1.0]), ch_start=2, ch_last=1)
    with pytest.raises(ValueError, match="exceeds"):
        _sorted_channels(np.array([1.0]), ch_start=1, ch_last=2)


def test_stokes_i_and_weight_uses_first_and_last_correlations() -> None:
    """Stokes I should use the weighted average of edge correlations."""
    value, weight, invalid = _stokes_i_and_weight(
        np.array([[1.0 + 1.0j, 100.0 + 0.0j, 3.0 + 5.0j]]),
        np.array([[2.0, 10.0, 6.0]]),
    )

    assert invalid[0] == np.False_
    assert value[0] == pytest.approx((2.0 * (1.0 + 1.0j) + 6.0 * (3.0 + 5.0j)) / 8.0)
    assert weight[0] == pytest.approx(np.sqrt(8.0))


def test_stokes_i_and_weight_flags_nonpositive_weight() -> None:
    """Invalid weights should produce a flagged result."""
    _, weight, invalid = _stokes_i_and_weight(
        np.array([[1.0 + 0.0j, 2.0 + 0.0j]]),
        np.array([[1.0, -1.0]]),
    )

    assert invalid[0] == np.True_
    assert weight[0] == 0.0


def test_imaging_weight_multiplier_selects_frequency_without_corr_sum() -> None:
    """IMAGING_WEIGHT rows should select the channel value directly."""
    weight = _imaging_weight_multiplier(np.array([4.0, 9.0]), channel_id=1)

    assert weight == pytest.approx(3.0)


def test_imaging_weight_multiplier_averages_spectrum_edge_correlations() -> None:
    """IMAGING_WEIGHT_SPECTRUM should not sum first/last correlations."""
    weight = _imaging_weight_multiplier(
        np.array(
            [
                [100.0, 100.0, 100.0],
                [4.0, 100.0, 16.0],
            ]
        ),
        channel_id=1,
    )

    assert weight == pytest.approx(np.sqrt(10.0))


def test_parse_ms2mat_arguments_defaults() -> None:
    """CLI parser should expose the documented converter defaults."""
    args = parse_ms2mat_arguments(["input.ms"])

    assert args.ms_path == "input.ms"
    assert args.output_path == "data"
    assert args.src_name is None
    assert args.src_id == 0
    assert args.ch_start == 1
    assert args.ch_last is None
    assert args.min_uv == pytest.approx(1e-6)
    assert args.freq_tol_hz == pytest.approx(1e-5)
    assert args.data_column == "CORRECTED_DATA"
    assert args.write_imaging_weight is True
    assert args.dry_run is False


def test_parse_ms2mat_arguments_overrides() -> None:
    """CLI parser should read converter overrides with expected types."""
    args = parse_ms2mat_arguments(
        [
            "input.ms",
            "--output_path",
            "out",
            "--src_name",
            "target",
            "--src_id",
            "2",
            "--ch_start",
            "3",
            "--ch_last",
            "5",
            "--min_uv",
            "12.5",
            "--freq_tol_hz",
            "0.1",
            "--data_column",
            "DATA",
            "--write_imaging_weight",
            "false",
            "--dry_run",
        ]
    )

    assert args.output_path == "out"
    assert args.src_name == "target"
    assert args.src_id == 2
    assert args.ch_start == 3
    assert args.ch_last == 5
    assert args.min_uv == pytest.approx(12.5)
    assert args.freq_tol_hz == pytest.approx(0.1)
    assert args.data_column == "DATA"
    assert args.write_imaging_weight is False
    assert args.dry_run is True


def test_print_dry_run_summary_reports_physical_and_grouped_channels(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Dry-run output should show raw channels and grouped output channels."""
    channel_frequencies = [np.array([2.0, 1.0]), np.array([1.0])]
    channels = _sorted_channels(channel_frequencies)

    _print_dry_run_summary(channel_frequencies, channels, requested_freq_tol_hz=1e-5)

    output = capsys.readouterr().out
    assert "physical channels: 3" in output
    assert "SPW 0, channel 1: 1.00000000000 Hz" in output
    assert "data_ch_1: 1.00000000000 Hz (SPW 0/ch 1, SPW 1/ch 0)" in output


def test_missing_casacore_has_friendly_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Optional casacore dependency failures should explain the extra."""
    monkeypatch.setattr(ms2mat_module, "casa_table", None)

    with pytest.raises(ImportError, match="python-casacore is required"):
        convert_ms_to_mat("missing.ms")
