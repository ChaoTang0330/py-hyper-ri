"""Convert Measurement Set data into pyHyperRI MAT files.

This module provides the ``hyri-ms2mat`` command-line implementation and helper
functions for turning Measurement Set rows into the MAT layout consumed by the
pyHyperRI data loader.
"""

import argparse
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import savemat

try:
    from casacore.tables import table as casa_table
except ImportError as exc:
    casa_table = None
    CASA_IMPORT_ERROR = exc
else:
    CASA_IMPORT_ERROR = None

CASA_IMPORT_MESSAGE = (
    "python-casacore is required for Measurement Set conversion.\n"
    "Check the installation instructions for details:\n"
    "<TODO: add documentation link>. "
)


@dataclass(frozen=True)
class MsChannelMember:
    """Describe one physical MS channel that contributes to an output channel.

    The pair stores the spectral-window row index and channel index used to
    retrieve data from the Measurement Set.
    """

    spectral_window_id: int
    channel_id: int


@dataclass(frozen=True)
class MsChannel:
    """Describe one sorted output channel.

    One output channel may merge same-frequency channels from different spectral
    windows. The output index is global after frequency sorting, so partial
    conversion keeps filenames consistent with the full Measurement Set.
    """

    output_index: int
    frequency_hz: float
    members: tuple[MsChannelMember, ...]


@dataclass(frozen=True)
class MsChannelData:
    """Store per-output-channel arrays aligned to the selected MS rows.

    Each list has one entry per selected output channel. The arrays use the same
    row order as ``msSpecs.uvw`` before per-channel flags remove invalid samples.
    """

    data_i: list[np.ndarray]
    weights_nat: list[np.ndarray]
    flags: list[np.ndarray]
    weights_img: list[np.ndarray] | None


def parse_ms2mat_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the MS-to-MAT converter.

    Args:
        argv (list[str] | None, optional): Command-line arguments without the
            program name. Default: None, which reads from ``sys.argv``.

    Returns:
        argparse.Namespace: Parsed converter arguments.
    """
    parser = argparse.ArgumentParser(
        description="Convert a Measurement Set into pyHyperRI MAT files."
    )
    parser.add_argument("ms_path", help="Path to the input Measurement Set.")
    parser.add_argument(
        "--output_path",
        default="data",
        help="Root output path. Files are written to output_path/src_name.",
    )
    parser.add_argument(
        "--src_name",
        default=None,
        help="Source/output folder name. Defaults to the MS filename without .ms.",
    )
    parser.add_argument(
        "--src_id",
        type=int,
        default=0,
        help="FIELD_ID to convert. Default: 0.",
    )
    parser.add_argument(
        "--ch_start",
        type=int,
        default=1,
        help="First sorted frequency channel to convert, 1-based. Default: 1.",
    )
    parser.add_argument(
        "--ch_last",
        type=int,
        default=None,
        help="Last sorted frequency channel to convert, 1-based and inclusive.",
    )
    parser.add_argument(
        "--min_uv",
        type=float,
        default=1e-6,
        help="Minimum projected baseline in metres. Default: 1e-6.",
    )
    parser.add_argument(
        "--freq_tol_hz",
        type=float,
        default=1e-5,
        help=(
            "Frequency tolerance in Hz for merging channels from different SPWs. "
            "Default: 1e-5."
        ),
    )
    parser.add_argument(
        "--data_column",
        default="CORRECTED_DATA",
        help="Visibility column to read, falling back to DATA if unavailable.",
    )
    parser.add_argument(
        "--write_imaging_weight",
        type=_bool_argument,
        default=True,
        help="Write MS imaging weights when available. Default: true.",
    )
    parser.add_argument(
        "--flag_zero_data",
        type=_bool_argument,
        default=True,
        help="Flag rows whose selected correlations are exactly zero. Default: true.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help=(
            "Inspect MS channel frequencies and the grouped conversion plan "
            "without writing MAT files."
        ),
    )
    return parser.parse_args(argv)


def convert_ms_to_mat(
    ms_path: str | Path,
    output_path: str | Path = "data",
    src_name: str | None = None,
    src_id: int = 0,
    ch_start: int = 1,
    ch_last: int | None = None,
    min_uv: float = 1e-6,
    data_column: str = "CORRECTED_DATA",
    write_imaging_weight: bool = True,
    flag_zero_data: bool = True,
    freq_tol_hz: float = 1e-5,
    dry_run: bool = False,
) -> Path:
    """Convert a Measurement Set into pyHyperRI's MAT measurement layout.

    Args:
        ms_path (str | Path): Path to the input Measurement Set directory.
        output_path (str | Path, optional): Root output directory. Files are
            written under ``output_path/src_name``. Default: "data".
        src_name (str | None, optional): Output source folder name. If None, the
            Measurement Set filename is used with a trailing ``.ms`` removed.
            Default: None.
        src_id (int, optional): ``FIELD_ID`` value to convert. Default: 0.
        ch_start (int, optional): First global frequency-sorted channel to
            convert, using one-based indexing. Default: 1.
        ch_last (int | None, optional): Last global frequency-sorted channel to
            convert, inclusive. If None, converts through the last channel.
            Default: None.
        min_uv (float, optional): Minimum projected baseline in metres. Rows
            with ``sqrt(u**2 + v**2) <= min_uv`` are flagged. Default: 1e-6.
        data_column (str, optional): Preferred visibility column. If unavailable,
            the converter falls back to ``DATA``. Default: "CORRECTED_DATA".
        write_imaging_weight (bool, optional): Whether to write ``weightsImg``
            when MS imaging weights are available. Default: True.
        flag_zero_data (bool, optional): Whether to flag selected correlations
            that are exactly zero. Default: True.
        freq_tol_hz (float, optional): Frequency tolerance in Hz for grouping
            matching channels across spectral windows. Default: 1e-5.
        dry_run (bool, optional): Whether to print the channel plan without
            writing MAT files. Default: False.

    Returns:
        Path: Output source directory path.

    Raises:
        ImportError: If python-casacore cannot be imported.
        FileNotFoundError: If a required Measurement Set subtables is missing.
        ValueError: If required columns, channel ranges, or valid visibilities
            are missing.

    Examples:
        >>> convert_ms_to_mat("target.ms", output_path="data", dry_run=True)
        PosixPath('data/target')
    """
    if casa_table is None:
        raise ImportError(CASA_IMPORT_MESSAGE) from CASA_IMPORT_ERROR

    ms_path = Path(ms_path).expanduser()
    if src_name is None:
        src_name = ms_path.stem if ms_path.suffix.lower() == ".ms" else ms_path.name
    out_dir = Path(output_path).expanduser() / src_name

    channel_frequencies = _read_channel_frequencies(ms_path)
    all_channels = _sorted_channels(
        channel_frequencies,
        freq_tol_hz=freq_tol_hz,
    )
    channels = _select_channel_range(
        all_channels,
        ch_start=ch_start,
        ch_last=ch_last,
    )
    all_freqs = np.asarray([channel.frequency_hz for channel in all_channels])
    if dry_run:
        _print_dry_run_summary(channel_frequencies, channels, freq_tol_hz)
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    spw_ids_by_desc = _read_data_description_spws(ms_path)
    with closing(casa_table(str(ms_path), readonly=True, ack=False)) as tab:
        colnames = set(tab.colnames())
        _validate_required_columns(colnames)
        data_column_name = _resolve_data_column(colnames, data_column)

        field_ids = np.asarray(tab.getcol("FIELD_ID"), dtype=np.int64)
        data_desc_ids = np.asarray(tab.getcol("DATA_DESC_ID"), dtype=np.int64)
        row_spw_ids = spw_ids_by_desc[data_desc_ids]
        # Keep all rows for the selected field so msSpecs.uvw is stable under
        # changes to ch_start/ch_last. SPW/channel membership is represented by
        # per-channel flags in data_ch_<index>.mat.
        base_row_mask = field_ids == src_id
        base_rows = np.flatnonzero(base_row_mask)
        if base_rows.size == 0:
            raise ValueError(f"No rows found for FIELD_ID={src_id}.")

        uvw = np.asarray(tab.getcol("UVW"), dtype=np.float64)[base_rows, :]
        flag_row = np.asarray(tab.getcol("FLAG_ROW"), dtype=bool)[base_rows]
        baseline = np.sqrt(uvw[:, 0] ** 2 + uvw[:, 1] ** 2)
        base_flag = flag_row | (baseline <= min_uv)
        base_row_spw_ids = row_spw_ids[base_rows]
        channel_data = _read_selected_channels(
            tab=tab,
            colnames=colnames,
            data_column_name=data_column_name,
            channels=channels,
            base_rows=base_rows,
            base_row_spw_ids=base_row_spw_ids,
            base_flag=base_flag,
            flag_zero_data=flag_zero_data,
            write_imaging_weight=write_imaging_weight,
        )

    if write_imaging_weight and channel_data.weights_img is None:
        print(
            "INFO: no MS imaging weight column found; weightsImg will be skipped.",
            flush=True,
        )

    _write_ms_specs(out_dir, all_freqs, uvw)

    for channel_idx, channel in enumerate(channels):
        # The list index is local to this conversion; ``channel.output_index`` is
        # the global filename/index preserved across partial conversions.
        valid = ~channel_data.flags[channel_idx]
        if not np.any(valid):
            raise ValueError(
                f"Channel {channel.output_index} "
                f"(frequency {channel.frequency_hz:g} Hz) "
                "has no valid visibilities after flagging."
            )

        _write_channel_data(
            out_dir,
            channel,
            channel_data.data_i[channel_idx][valid],
            channel_data.weights_nat[channel_idx][valid],
            channel_data.flags[channel_idx],
            (
                None
                if channel_data.weights_img is None
                else channel_data.weights_img[channel_idx][valid]
            ),
        )
        print(
            f"INFO: wrote data_ch_{channel.output_index}.mat "
            f"with {int(np.count_nonzero(valid))} valid samples at "
            f"{channel.frequency_hz:#.8g} Hz from "
            f"{len(channel.members)} MS channel(s).",
            flush=True,
        )

    print(f"INFO: wrote MS conversion output to {out_dir}", flush=True)
    return out_dir


def ms2mat(argv: list[str] | None = None) -> Path:
    """Run the MS-to-MAT converter from command-line arguments.

    Args:
        argv (list[str] | None, optional): Command-line arguments without the
            program name. Default: None, which reads from ``sys.argv``.

    Returns:
        Path: Output source directory path.

    Examples:
        >>> ms2mat(["target.ms", "--dry_run"])
        PosixPath('data/target')
    """
    args = parse_ms2mat_arguments(argv)
    return convert_ms_to_mat(
        ms_path=args.ms_path,
        output_path=args.output_path,
        src_name=args.src_name,
        src_id=args.src_id,
        ch_start=args.ch_start,
        ch_last=args.ch_last,
        min_uv=args.min_uv,
        data_column=args.data_column,
        write_imaging_weight=args.write_imaging_weight,
        flag_zero_data=args.flag_zero_data,
        freq_tol_hz=args.freq_tol_hz,
        dry_run=args.dry_run,
    )


def _bool_argument(text: str) -> bool:
    """Parse a boolean command-line value.

    Args:
        text (str): Text value supplied by argparse.

    Returns:
        bool: Parsed boolean value.

    Raises:
        argparse.ArgumentTypeError: If `text` is not a recognised boolean value.
    """
    value = text.lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected true/false or yes/no")


def _validate_required_columns(colnames: set[str]) -> None:
    """Validate required main-table Measurement Set columns.

    Args:
        colnames (set[str]): Available column names in the main MS table.

    Raises:
        ValueError: If a required column is absent.
    """
    for name in ("FIELD_ID", "DATA_DESC_ID", "UVW", "FLAG_ROW", "FLAG"):
        if name not in colnames:
            raise ValueError(f"MS table is missing {name}.")


def _resolve_data_column(colnames: set[str], data_column: str) -> str:
    """Return the visibility column to read.

    Args:
        colnames (set[str]): Available column names in the main MS table.
        data_column (str): Preferred visibility column name.

    Returns:
        str: Resolved visibility column name.

    Raises:
        ValueError: If neither the requested column nor ``DATA`` is available.
    """
    preferred_data_column = data_column.upper()
    if preferred_data_column in colnames:
        return preferred_data_column
    if "DATA" in colnames:
        print(
            f"INFO: {preferred_data_column} not found, reading DATA instead.",
            flush=True,
        )
        return "DATA"
    raise ValueError("MS table has neither requested data column nor DATA.")


def _read_data_description_spws(ms_path: Path) -> np.ndarray:
    """Read spectral-window IDs from the ``DATA_DESCRIPTION`` subtable.

    Args:
        ms_path (Path): Path to the Measurement Set directory.

    Returns:
        np.ndarray: Integer array mapping ``DATA_DESC_ID`` to
        ``SPECTRAL_WINDOW_ID``.

    Raises:
        FileNotFoundError: If the ``DATA_DESCRIPTION`` subtable is missing.
    """
    dd_path = ms_path / "DATA_DESCRIPTION"
    if not dd_path.exists():
        raise FileNotFoundError(f"Missing DATA_DESCRIPTION table: {dd_path}")

    with closing(casa_table(str(dd_path), readonly=True, ack=False)) as dd_tab:
        return np.asarray(dd_tab.getcol("SPECTRAL_WINDOW_ID"), dtype=np.int64)


def _read_channel_frequencies(ms_path: Path) -> list[np.ndarray]:
    """Read channel frequencies for every spectral window.

    Args:
        ms_path (Path): Path to the Measurement Set directory.

    Returns:
        list[np.ndarray]: One frequency vector per spectral window, in Hz.
    """
    with closing(
        casa_table(str(ms_path / "SPECTRAL_WINDOW"), readonly=True, ack=False)
    ) as spw_tab:
        return [
            np.asarray(spw_tab.getcell("CHAN_FREQ", row), dtype=np.float64).reshape(-1)
            for row in range(spw_tab.nrows())
        ]


def _print_dry_run_summary(
    channel_frequencies: Sequence[np.ndarray],
    channels: list[MsChannel],
    requested_freq_tol_hz: float,
) -> None:
    """Print channel-frequency inspection output for ``--dry_run``.

    Args:
        channel_frequencies (Sequence[np.ndarray]): Raw per-SPW channel
            frequency arrays.
        channels (list[MsChannel]): Selected grouped output channels.
        requested_freq_tol_hz (float): User-requested grouping tolerance in Hz.
    """
    physical_channel_count = sum(len(freqs) for freqs in channel_frequencies)
    effective_freq_tol_hz = _safe_frequency_grouping_tolerance(
        channel_frequencies, requested_freq_tol_hz
    )
    print("MS channel inspection", flush=True)
    print(f"  spectral windows: {len(channel_frequencies)}", flush=True)
    print(f"  physical channels: {physical_channel_count}", flush=True)
    print(f"  requested frequency tolerance: {requested_freq_tol_hz:g} Hz", flush=True)
    print(f"  effective frequency tolerance: {effective_freq_tol_hz:g} Hz", flush=True)
    print("Physical channels:", flush=True)
    for spw_id, freqs in enumerate(channel_frequencies):
        for channel_id, freq_hz in enumerate(np.asarray(freqs).reshape(-1)):
            print(
                f"  SPW {spw_id}, channel {channel_id}: " f"{float(freq_hz):#.12g} Hz",
                flush=True,
            )

    print("Grouped output channels:", flush=True)
    for channel in channels:
        members = ", ".join(
            f"SPW {member.spectral_window_id}/ch {member.channel_id}"
            for member in channel.members
        )
        print(
            f"  data_ch_{channel.output_index}: {channel.frequency_hz:#.12g} Hz "
            f"({members})",
            flush=True,
        )


def _sorted_channels(
    chan_freq: np.ndarray | Sequence[np.ndarray],
    ch_start: int = 1,
    ch_last: int | None = None,
    freq_tol_hz: float = 0.0,
) -> list[MsChannel]:
    """Return 1-based output channels sorted by frequency.

    Equal frequencies are merged into one output channel. A positive
    ``freq_tol_hz`` also merges adjacent sorted frequencies within that tolerance.

    Args:
        chan_freq (np.ndarray | Sequence[np.ndarray]): Channel frequencies in Hz.
            A 1D array represents one SPW; a 2D array represents
            ``(spw, channel)``; a sequence allows ragged SPWs.
        ch_start (int, optional): First one-based output channel to return.
            Default: 1.
        ch_last (int | None, optional): Last one-based output channel to return,
            inclusive. Default: None.
        freq_tol_hz (float, optional): Requested grouping tolerance in Hz.
            Default: 0.0.

    Returns:
        list[MsChannel]: Frequency-sorted channels with global one-based
        ``output_index`` values.

    Raises:
        ValueError: If the channel range is invalid, frequencies are non-finite,
            or grouping would merge channels from the same SPW.
    """
    if freq_tol_hz < 0:
        raise ValueError("freq_tol_hz must be >= 0.")

    spw_freqs = _normalise_channel_frequencies(chan_freq)
    grouping_tolerance = _safe_frequency_grouping_tolerance(spw_freqs, freq_tol_hz)
    all_channels = [
        (float(freq), spw_id, ch_id)
        for spw_id, freqs_for_spw in enumerate(spw_freqs)
        for ch_id, freq in enumerate(freqs_for_spw)
    ]
    all_channels.sort(key=lambda item: item[0])
    grouped_channels = _group_channels_by_frequency(all_channels, grouping_tolerance)

    channels = [
        MsChannel(
            output_index=idx,
            frequency_hz=freq,
            members=tuple(
                MsChannelMember(spectral_window_id=spw_id, channel_id=ch_id)
                for spw_id, ch_id in members
            ),
        )
        for idx, (freq, members) in enumerate(grouped_channels, start=1)
    ]
    return _select_channel_range(channels, ch_start=ch_start, ch_last=ch_last)


def _normalise_channel_frequencies(
    chan_freq: np.ndarray | Sequence[np.ndarray],
) -> list[np.ndarray]:
    """Return channel frequencies as one 1D array per spectral window.

    Args:
        chan_freq (np.ndarray | Sequence[np.ndarray]): Channel-frequency input.

    Returns:
        list[np.ndarray]: Normalised per-SPW frequency arrays.

    Raises:
        ValueError: If an ndarray input is not 1D or 2D.
    """
    if isinstance(chan_freq, np.ndarray):
        freqs = np.asarray(chan_freq, dtype=np.float64)
        if freqs.ndim == 1:
            return [freqs.reshape(-1)]
        if freqs.ndim == 2:
            return [freqs[spw_id, :].reshape(-1) for spw_id in range(freqs.shape[0])]
        raise ValueError("CHAN_FREQ must be a 1D or 2D array.")
    return [np.asarray(freq, dtype=np.float64).reshape(-1) for freq in chan_freq]


def _select_channel_range(
    channels: list[MsChannel],
    ch_start: int = 1,
    ch_last: int | None = None,
) -> list[MsChannel]:
    """Return an inclusive one-based channel range without renumbering.

    Args:
        channels (list[MsChannel]): Full grouped channel list.
        ch_start (int, optional): First one-based channel to keep. Default: 1.
        ch_last (int | None, optional): Last one-based channel to keep,
            inclusive. Default: None.

    Returns:
        list[MsChannel]: Selected channel objects with original ``output_index``
        values preserved.

    Raises:
        ValueError: If the requested range is invalid.
    """
    if ch_start < 1:
        raise ValueError("ch_start must be >= 1.")
    if ch_last is None:
        ch_last = len(channels)
    if ch_last < ch_start:
        raise ValueError("ch_last must be >= ch_start.")
    if ch_last > len(channels):
        raise ValueError(
            f"ch_last={ch_last} exceeds available channel count {len(channels)}."
        )
    return channels[ch_start - 1 : ch_last]


def _safe_frequency_grouping_tolerance(
    spw_freqs: Sequence[np.ndarray],
    requested_tolerance: float,
) -> float:
    """Cap grouping tolerance below within-SPW channel spacing.

    Args:
        spw_freqs (Sequence[np.ndarray]): Per-SPW channel frequency arrays.
        requested_tolerance (float): User-requested frequency tolerance in Hz.

    Returns:
        float: Effective grouping tolerance in Hz.

    Notes:
        The cap prevents neighbouring channels in the same SPW from being
        grouped when the user supplies a large tolerance.
    """
    min_spw_spacing = np.inf
    for freqs in spw_freqs:
        sorted_freqs = np.sort(np.asarray(freqs, dtype=np.float64).reshape(-1))
        if sorted_freqs.size < 2:
            continue
        spacings = np.diff(sorted_freqs)
        positive_spacings = spacings[spacings > 0]
        if positive_spacings.size:
            min_spw_spacing = min(min_spw_spacing, float(np.min(positive_spacings)))

    if np.isinf(min_spw_spacing) or requested_tolerance < min_spw_spacing:
        return requested_tolerance
    return float(np.nextafter(min_spw_spacing, 0.0))


def _group_channels_by_frequency(
    sorted_channels: list[tuple[float, int, int]],
    freq_tol_hz: float,
) -> list[tuple[float, list[tuple[int, int]]]]:
    """Group sorted physical channels by observation frequency.

    Args:
        sorted_channels (list[tuple[float, int, int]]): Tuples of
            ``(frequency_hz, spectral_window_id, channel_id)`` sorted by
            frequency.
        freq_tol_hz (float): Effective grouping tolerance in Hz.

    Returns:
        list[tuple[float, list[tuple[int, int]]]]: Grouped frequency and member
        channel IDs.

    Raises:
        ValueError: If a frequency is non-finite or a group contains repeated
            SPW IDs.
    """
    grouped_channels: list[tuple[float, list[tuple[int, int]]]] = []
    for freq, spw_id, ch_id in sorted_channels:
        if not np.isfinite(freq):
            raise ValueError("CHAN_FREQ contains a non-finite frequency.")
        if grouped_channels and abs(freq - grouped_channels[-1][0]) <= freq_tol_hz:
            group_freq, members = grouped_channels[-1]
            members.append((spw_id, ch_id))
            group_size = len(members)
            grouped_channels[-1] = (
                group_freq + (freq - group_freq) / group_size,
                members,
            )
        else:
            grouped_channels.append((freq, [(spw_id, ch_id)]))
    _validate_unique_spw_per_group(grouped_channels)
    return grouped_channels


def _validate_unique_spw_per_group(
    grouped_channels: list[tuple[float, list[tuple[int, int]]]],
) -> None:
    """Validate that a grouped frequency has at most one channel per SPW.

    Args:
        grouped_channels (list[tuple[float, list[tuple[int, int]]]]): Grouped
            channel descriptors from :func:`_group_channels_by_frequency`.

    Raises:
        ValueError: If a group contains multiple channels from one SPW.
    """
    for freq, members in grouped_channels:
        spw_ids = [spw_id for spw_id, _ in members]
        if len(set(spw_ids)) != len(spw_ids):
            raise ValueError(
                "Frequency grouping found multiple channels from the same SPW "
                f"near {freq:g} Hz. This converter assumes each SPW has unique "
                "channel frequencies; reduce --freq_tol_hz or inspect CHAN_FREQ."
            )


def _read_selected_channels(
    tab,
    colnames: set[str],
    data_column_name: str,
    channels: list[MsChannel],
    base_rows: np.ndarray,
    base_row_spw_ids: np.ndarray,
    base_flag: np.ndarray,
    flag_zero_data: bool,
    write_imaging_weight: bool,
) -> MsChannelData:
    """Read selected channel data row-by-row from the main MS table.

    Row-wise ``getcell`` calls are used instead of whole-column ``getcol`` calls
    so selected spectral windows may have different channel counts.

    Args:
        tab: Open casacore main-table object.
        colnames (set[str]): Available column names in the main MS table.
        data_column_name (str): Resolved visibility column name.
        channels (list[MsChannel]): Selected output channels to read.
        base_rows (np.ndarray): Main-table row indices selected by field and SPW.
        base_row_spw_ids (np.ndarray): SPW ID for each selected row.
        base_flag (np.ndarray): Row-level invalid mask from ``FLAG_ROW`` and
            ``min_uv`` filtering.
        flag_zero_data (bool): Whether exact zero selected correlations are
            invalid.
        write_imaging_weight (bool): Whether optional imaging weights should be
            read.

    Returns:
        MsChannelData: Channel arrays aligned to `base_rows`.

    Raises:
        ValueError: If required weight columns or channel axes are invalid.
    """
    row_count = base_rows.size
    data_by_channel = [np.zeros(row_count, dtype=np.complex128) for _ in channels]
    weights_nat_by_channel = [np.zeros(row_count, dtype=np.float64) for _ in channels]
    flags = [np.ones(row_count, dtype=bool) for _ in channels]
    imaging_weight_column = _resolve_imaging_weight_column(colnames)
    weights_img = (
        [np.zeros(row_count, dtype=np.float64) for _ in channels]
        if write_imaging_weight and imaging_weight_column is not None
        else None
    )
    channels_by_spw = _channels_by_spw(channels)

    for local_row, ms_row in enumerate(base_rows):
        # A selected MS row belongs to one SPW, but that SPW may contribute to
        # several output channels if partial conversion spans several channels.
        row_channels = channels_by_spw.get(int(base_row_spw_ids[local_row]), ())
        if not row_channels:
            continue

        ms_row = int(ms_row)
        data_row = np.asarray(tab.getcell(data_column_name, ms_row))
        flag_row = np.asarray(tab.getcell("FLAG", ms_row), dtype=bool)
        natural_weight_row = _read_natural_weight_cell(tab, colnames, ms_row)
        imaging_weight_row = (
            None
            if weights_img is None
            else np.asarray(tab.getcell(imaging_weight_column, ms_row))
        )

        for channel_idx, member in row_channels:
            # ``channel_idx`` indexes local output arrays; ``member.channel_id``
            # indexes the physical channel inside the row's SPW.
            data_ch = _select_channel_correlations(data_row, member.channel_id)
            flag_ch = _select_channel_correlations(flag_row, member.channel_id)
            natural_weight_ch = _select_weight_correlations(
                natural_weight_row, member.channel_id
            )
            data_i_ch, weights_nat_ch, invalid_weight = _stokes_i_and_weight(
                data_ch.reshape(1, -1), natural_weight_ch.reshape(1, -1)
            )
            invalid_data = _invalid_correlations(
                data_ch.reshape(1, -1), flag_ch.reshape(1, -1), flag_zero_data
            )
            # The MAT flag convention follows load_data.py: True means invalid.
            invalid = bool(base_flag[local_row] | invalid_data[0] | invalid_weight[0])

            data_by_channel[channel_idx][local_row] = data_i_ch[0]
            weights_nat_by_channel[channel_idx][local_row] = weights_nat_ch[0]
            flags[channel_idx][local_row] = invalid

            if weights_img is not None and imaging_weight_row is not None:
                weights_img[channel_idx][local_row] = _imaging_weight_multiplier(
                    imaging_weight_row,
                    member.channel_id,
                )

    return MsChannelData(
        data_i=data_by_channel,
        weights_nat=weights_nat_by_channel,
        flags=flags,
        weights_img=weights_img,
    )


def _channels_by_spw(
    channels: list[MsChannel],
) -> dict[int, tuple[tuple[int, MsChannelMember], ...]]:
    """Group selected output-channel members by spectral window.

    Args:
        channels (list[MsChannel]): Selected output channels.

    Returns:
        dict[int, tuple[tuple[int, MsChannelMember], ...]]: Mapping from SPW ID
        to local output-channel indices and physical-channel descriptors.
    """
    grouped: dict[int, list[tuple[int, MsChannelMember]]] = {}
    for channel_idx, channel in enumerate(channels):
        for member in channel.members:
            grouped.setdefault(member.spectral_window_id, []).append(
                (channel_idx, member)
            )
    return {spw_id: tuple(members) for spw_id, members in grouped.items()}


def _read_natural_weight_cell(tab, colnames: set[str], row: int) -> np.ndarray:
    """Read natural weights for one main-table row.

    Args:
        tab: Open casacore main-table object.
        colnames (set[str]): Available column names in the main MS table.
        row (int): Main-table row index.

    Returns:
        np.ndarray: Natural weight values from ``WEIGHT_SPECTRUM`` or ``WEIGHT``.

    Raises:
        ValueError: If no supported natural-weight column is available.
    """
    if "WEIGHT_SPECTRUM" in colnames:
        return np.asarray(tab.getcell("WEIGHT_SPECTRUM", row), dtype=np.float64)
    if "WEIGHT" in colnames:
        return np.asarray(tab.getcell("WEIGHT", row), dtype=np.float64)
    raise ValueError("MS table is missing both WEIGHT_SPECTRUM and WEIGHT.")


def _resolve_imaging_weight_column(colnames: set[str]) -> str | None:
    """Return the preferred optional imaging-weight column.

    Args:
        colnames (set[str]): Available column names in the main MS table.

    Returns:
        str | None: ``IMAGING_WEIGHT``, ``IMAGING_WEIGHT_SPECTRUM``, or None.
    """
    if "IMAGING_WEIGHT" in colnames:
        return "IMAGING_WEIGHT"
    if "IMAGING_WEIGHT_SPECTRUM" in colnames:
        return "IMAGING_WEIGHT_SPECTRUM"
    return None


def _select_channel_correlations(values: np.ndarray, channel_id: int) -> np.ndarray:
    """Select one channel's correlation vector from an MS row.

    Args:
        values (np.ndarray): Row values with shape ``(channel, correlation)`` or
            ``(correlation,)`` for single-channel rows.
        channel_id (int): Channel index to select.

    Returns:
        np.ndarray: Correlation vector for the selected channel.

    Raises:
        ValueError: If `values` has an unsupported shape.
    """
    array = np.asarray(values)
    if array.ndim == 1:
        if channel_id != 0:
            raise ValueError(
                f"Cannot select channel {channel_id} from a single-channel row."
            )
        return array
    if array.ndim == 2:
        return array[channel_id, :]
    raise ValueError(f"Expected a 1D or 2D channel row, got shape {array.shape}.")


def _select_weight_correlations(weight: np.ndarray, channel_id: int) -> np.ndarray:
    """Select natural-weight correlations for one channel.

    Args:
        weight (np.ndarray): Weight row with shape ``(channel, correlation)`` or
            ``(correlation,)``.
        channel_id (int): Channel index to select when a spectrum is present.

    Returns:
        np.ndarray: Natural-weight correlation vector.

    Raises:
        ValueError: If `weight` has an unsupported shape.
    """
    array = np.asarray(weight, dtype=np.float64)
    if array.ndim == 1:
        return array
    if array.ndim == 2:
        return array[channel_id, :]
    raise ValueError(f"Expected a 1D or 2D weight row, got shape {array.shape}.")


def _imaging_weight_multiplier(imaging_weight: np.ndarray, channel_id: int) -> float:
    """Return the square-root imaging-weight multiplier for one channel.

    Args:
        imaging_weight (np.ndarray): Imaging-weight row. Supported shapes are
            scalar, ``(channel,)``, and ``(channel, correlation)``.
        channel_id (int): Channel index to select.

    Returns:
        float: Non-negative square-root imaging-weight multiplier. Invalid or
        non-positive source weights return 0.0.

    Raises:
        ValueError: If `imaging_weight` has an unsupported shape.

    Notes:
        ``IMAGING_WEIGHT`` is selected directly per channel. For
        ``IMAGING_WEIGHT_SPECTRUM``, the first and last correlation entries are
        averaged before taking the square root.
    """
    array = np.asarray(imaging_weight, dtype=np.float64)
    if array.ndim == 0:
        weight = float(array)
    elif array.ndim == 1:
        weight = float(array[channel_id])
    elif array.ndim == 2:
        corr_weights = array[channel_id, :]
        if corr_weights.size == 1:
            weight = float(corr_weights[0])
        else:
            weight = float(0.5 * (corr_weights[0] + corr_weights[-1]))
    else:
        raise ValueError(
            f"Expected a scalar, 1D, or 2D imaging-weight row, got shape {array.shape}."
        )

    if weight <= 0 or not np.isfinite(weight):
        return 0.0
    return float(np.sqrt(weight))


def _correlation_weight_sum(corr_weights: np.ndarray) -> np.ndarray:
    """Sum first and last correlation weights.

    Args:
        corr_weights (np.ndarray): Natural-weight array with shape
            ``(row, correlation)`` or ``(correlation,)``.

    Returns:
        np.ndarray: Per-row first-plus-last correlation weight.

    Raises:
        ValueError: If no correlation axis is present.
    """
    weights = np.asarray(corr_weights, dtype=np.float64)
    if weights.ndim == 1:
        weights = weights.reshape(-1, 1)
    if weights.shape[1] == 0:
        raise ValueError("Weight arrays must have correlations.")
    if weights.shape[1] == 1:
        return weights[:, 0].astype(np.float64)
    return weights[:, 0] + weights[:, -1]


def _stokes_i_and_weight(
    corr_values: np.ndarray,
    corr_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine first and last correlations into Stokes I and whitening weights.

    Args:
        corr_values (np.ndarray): Complex visibility values with shape
            ``(row, correlation)``.
        corr_weights (np.ndarray): Natural weights with shape
            ``(row, correlation)``.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Stokes I values,
        ``sqrt(w_first + w_last)`` whitening weights, and invalid-row mask.

    Raises:
        ValueError: If visibility and weight row counts differ or correlation
            axes are empty.
    """
    values = np.asarray(corr_values)
    weights = np.asarray(corr_weights, dtype=np.float64)

    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if weights.ndim == 1:
        weights = weights.reshape(-1, 1)
    if values.shape[0] != weights.shape[0]:
        raise ValueError(
            f"Visibility and weight row counts differ: "
            f"{values.shape[0]} and {weights.shape[0]}."
        )
    if values.shape[1] == 0 or weights.shape[1] == 0:
        raise ValueError("Visibility and weight arrays must have correlations.")

    if values.shape[1] == 1 or weights.shape[1] == 1:
        value = values[:, 0].astype(np.complex128)
        weight_sum = _correlation_weight_sum(weights)
    else:
        last = min(values.shape[1], weights.shape[1]) - 1
        weight_sum = _correlation_weight_sum(weights[:, [0, last]])
        numerator = weights[:, 0] * values[:, 0] + weights[:, last] * values[:, last]
        value = np.zeros(values.shape[0], dtype=np.complex128)
        np.divide(numerator, weight_sum, out=value, where=weight_sum > 0)

    invalid = (
        (weight_sum <= 0)
        | ~np.isfinite(weight_sum)
        | ~np.isfinite(value.real)
        | ~np.isfinite(value.imag)
    )
    whitening_weight = np.zeros_like(weight_sum, dtype=np.float64)
    np.sqrt(weight_sum, out=whitening_weight, where=~invalid)
    return value, whitening_weight, invalid


def _invalid_correlations(
    data_values: np.ndarray,
    flags: np.ndarray,
    flag_zero_data: bool,
) -> np.ndarray:
    """Return invalid rows from selected data and flag correlations.

    Args:
        data_values (np.ndarray): Complex visibility values with shape
            ``(row, correlation)``.
        flags (np.ndarray): Boolean MS flags with shape ``(row, correlation)``.
        flag_zero_data (bool): Whether exact zero selected correlations are
            invalid.

    Returns:
        np.ndarray: Boolean invalid-row mask.
    """
    values = np.asarray(data_values)
    flag_values = np.asarray(flags, dtype=bool)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if flag_values.ndim == 1:
        flag_values = flag_values.reshape(-1, 1)

    if values.shape[1] == 1:
        selected_values = values[:, :1]
        selected_flags = flag_values[:, :1]
    else:
        selected_values = values[:, [0, -1]]
        selected_flags = flag_values[:, [0, -1]]

    invalid = np.any(selected_flags, axis=1)
    invalid |= np.any(~np.isfinite(selected_values.real), axis=1)
    invalid |= np.any(~np.isfinite(selected_values.imag), axis=1)
    if flag_zero_data:
        invalid |= np.any(selected_values == 0, axis=1)
    return invalid


def _write_ms_specs(output_dir: Path, freqs: np.ndarray, uvw: np.ndarray) -> None:
    """Write shared Measurement Set metadata to ``msSpecs.mat``.

    Args:
        output_dir (Path): Output source directory.
        freqs (np.ndarray): Full grouped frequency list in Hz.
        uvw (np.ndarray): Selected MS UVW coordinates in metres.
    """
    savemat(
        output_dir / "msSpecs.mat",
        {"freqs": np.asarray(freqs, dtype=np.float64), "uvw": np.asarray(uvw)},
        appendmat=False,
        oned_as="column",
    )


def _write_channel_data(
    output_dir: Path,
    channel: MsChannel,
    data_i: np.ndarray,
    weights_nat: np.ndarray,
    flag: np.ndarray,
    weights_img: np.ndarray | None = None,
) -> None:
    """Write one output channel MAT file.

    Args:
        output_dir (Path): Output source directory.
        channel (MsChannel): Output channel descriptor.
        data_i (np.ndarray): Valid Stokes I visibility values.
        weights_nat (np.ndarray): Natural whitening weights for valid samples.
        flag (np.ndarray): Full invalid-sample mask aligned to ``msSpecs.uvw``.
        weights_img (np.ndarray | None, optional): Optional imaging-weight
            multipliers for valid samples. Default: None.
    """
    mat_data = {
        "data_I": np.asarray(data_i, dtype=np.complex128).reshape(-1, 1),
        "weightsNat": np.asarray(weights_nat, dtype=np.float64).reshape(-1, 1),
        "flag": np.asarray(flag, dtype=bool).reshape(-1, 1),
    }
    if weights_img is not None:
        mat_data["weightsImg"] = np.asarray(weights_img, dtype=np.float64).reshape(
            -1, 1
        )

    savemat(
        output_dir / f"data_ch_{channel.output_index}.mat",
        mat_data,
        appendmat=False,
        oned_as="column",
    )
