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
    # DATA_DESC_ID is mapped to SPECTRAL_WINDOW_ID through the MS
    # DATA_DESCRIPTION subtable; FIELD_ID remains the source selector.
    spw_ids_by_desc = _read_data_description_spws(ms_path)
    with closing(casa_table(str(ms_path), readonly=True, ack=False)) as tab:
        colnames = set(tab.colnames())
        _validate_required_columns(colnames)
        data_column_name = _resolve_data_column(colnames, data_column)
        natural_weight_column = _resolve_natural_weight_column(colnames)
        imaging_weight_column = _resolve_imaging_weight_column(colnames)
        if write_imaging_weight and imaging_weight_column is None:
            print(
                "INFO: no MS imaging weight column found; "
                "weightsImg will not be written.",
                flush=True,
            )

        with closing(tab.query(f"FIELD_ID == {int(src_id)}")) as field_tab:
            row_count = field_tab.nrows()
            if row_count == 0:
                raise ValueError(f"No rows found for FIELD_ID={src_id}.")

            data_desc_ids = np.asarray(field_tab.getcol("DATA_DESC_ID"), dtype=np.int64)
            uvw = np.asarray(field_tab.getcol("UVW"), dtype=np.float64)
            flag_row = np.asarray(field_tab.getcol("FLAG_ROW"), dtype=bool)
            baseline = np.sqrt(uvw[:, 0] ** 2 + uvw[:, 1] ** 2)
            base_flag = flag_row | (baseline <= min_uv)
            _write_ms_specs(out_dir, all_freqs, uvw)
            print(
                f"INFO: wrote msSpecs.mat with {row_count} rows for "
                f"FIELD_ID={src_id}.",
                flush=True,
            )

            for channel in channels:
                print(
                    f"INFO: reading data_ch_{channel.output_index}.mat "
                    f"at {channel.frequency_hz:#.8g} Hz from "
                    f"{len(channel.members)} MS channel(s).",
                    flush=True,
                )
                data_i = np.zeros(row_count, dtype=np.complex128)
                weights_nat = np.zeros(row_count, dtype=np.float64)
                # Keep arrays aligned to msSpecs.uvw until the final write so
                # non-contributing SPWs remain represented by per-channel flags.
                flag = np.ones(row_count, dtype=bool)
                weights_img = (
                    np.zeros(row_count, dtype=np.float64)
                    if write_imaging_weight and imaging_weight_column is not None
                    else None
                )

                for member in channel.members:
                    member_data_desc_ids = np.flatnonzero(
                        spw_ids_by_desc == member.spectral_window_id
                    )
                    for data_desc_id in member_data_desc_ids:
                        row_positions = np.flatnonzero(data_desc_ids == data_desc_id)
                        if row_positions.size == 0:
                            continue
                        with closing(
                            field_tab.query(f"DATA_DESC_ID == {int(data_desc_id)}")
                        ) as dd_tab:
                            if dd_tab.nrows() != row_positions.size:
                                raise ValueError(
                                    f"DATA_DESC_ID={data_desc_id} query returned "
                                    f"{dd_tab.nrows()} rows, expected "
                                    f"{row_positions.size}."
                                )
                            (
                                data_i_part,
                                weights_nat_part,
                                flag_part,
                                weights_img_part,
                            ) = _read_single_channel(
                                tab=dd_tab,
                                data_column_name=data_column_name,
                                natural_weight_column=natural_weight_column,
                                imaging_weight_column=imaging_weight_column,
                                channel=channel,
                                member=member,
                                channel_count=np.asarray(
                                    channel_frequencies[member.spectral_window_id]
                                ).size,
                                base_flag=base_flag[row_positions],
                                flag_zero_data=flag_zero_data,
                                write_imaging_weight=write_imaging_weight,
                            )

                        data_i[row_positions] = data_i_part
                        weights_nat[row_positions] = weights_nat_part
                        flag[row_positions] = flag_part
                        if weights_img is not None and weights_img_part is not None:
                            weights_img[row_positions] = weights_img_part

                valid = ~flag
                if not np.any(valid):
                    raise ValueError(
                        f"Channel {channel.output_index} "
                        f"(frequency {channel.frequency_hz:g} Hz) "
                        "has no valid visibilities after flagging."
                    )

                _write_channel_data(
                    out_dir,
                    channel,
                    data_i[valid],
                    weights_nat[valid],
                    flag,
                    None if weights_img is None else weights_img[valid],
                )
                print(
                    f"INFO: wrote data_ch_{channel.output_index}.mat "
                    f"with {int(np.count_nonzero(valid))} valid samples at "
                    f"{channel.frequency_hz:#.8g} Hz.",
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


def _resolve_natural_weight_column(colnames: set[str]) -> str:
    """Return the MS column used for natural visibility weights.

    Args:
        colnames (set[str]): Available column names in the main MS table.

    Returns:
        str: ``"WEIGHT_SPECTRUM"`` when available, otherwise ``"WEIGHT"``.

    Raises:
        ValueError: If neither ``WEIGHT_SPECTRUM`` nor ``WEIGHT`` is present.

    Notes:
        ``WEIGHT_SPECTRUM`` is channel-dependent and follows the selected
        physical channel. ``WEIGHT`` is channel-independent and is used for all
        channels in the row.
    """
    if "WEIGHT_SPECTRUM" in colnames:
        return "WEIGHT_SPECTRUM"
    if "WEIGHT" in colnames:
        return "WEIGHT"
    raise ValueError("MS table is missing both WEIGHT_SPECTRUM and WEIGHT.")


def _read_single_channel(
    tab,
    data_column_name: str,
    natural_weight_column: str,
    imaging_weight_column: str | None,
    channel: MsChannel,
    member: MsChannelMember,
    channel_count: int,
    base_flag: np.ndarray,
    flag_zero_data: bool,
    write_imaging_weight: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Read one physical channel from a selected MS row subset.

    Args:
        tab: Casacore table view containing one ``FIELD_ID`` and one
            ``DATA_DESC_ID``.
        data_column_name (str): Visibility column to read, usually ``DATA`` or
            ``CORRECTED_DATA``.
        natural_weight_column (str): Natural-weight column, either
            ``WEIGHT_SPECTRUM`` or ``WEIGHT``.
        imaging_weight_column (str | None): Optional imaging-weight column.
        channel (MsChannel): Output channel descriptor used for logging.
        member (MsChannelMember): Physical SPW/channel member to read.
        channel_count (int): Number of channels in the member's SPW.
        base_flag (np.ndarray): Row-level invalid mask aligned to ``tab``.
        flag_zero_data (bool): Whether zero-valued selected correlations are
            invalid.
        write_imaging_weight (bool): Whether optional imaging weights should be
            returned.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]: Stokes I
        values, natural whitening weights, full invalid-row flags, and optional
        imaging-weight multipliers. All arrays are aligned to ``tab`` rows.

    Raises:
        ValueError: If a column shape does not match the supported MS
            convention.
    """
    row_count = int(base_flag.size)
    data_i = np.zeros(row_count, dtype=np.complex128)
    weights_nat = np.zeros(row_count, dtype=np.float64)
    flag = np.ones(row_count, dtype=bool)
    weights_img = (
        np.zeros(row_count, dtype=np.float64)
        if write_imaging_weight and imaging_weight_column is not None
        else None
    )

    data_ch = _read_channel_correlations(
        tab, data_column_name, member.channel_id, channel_count
    )
    corr_count = data_ch.shape[1]
    flag_ch = _read_channel_correlations(
        tab, "FLAG", member.channel_id, channel_count, corr_count
    )
    if natural_weight_column == "WEIGHT_SPECTRUM":
        natural_weight_ch = _read_channel_correlations(
            tab, natural_weight_column, member.channel_id, channel_count, corr_count
        )
    else:
        # In the MS convention, WEIGHT is per row/correlation and does not vary
        # with channel. Only WEIGHT_SPECTRUM is sliced by channel.
        natural_weight_ch = np.asarray(
            tab.getcol(natural_weight_column), dtype=np.float64
        )
        if natural_weight_ch.ndim == 1:
            natural_weight_ch = natural_weight_ch.reshape(-1, 1)
        elif natural_weight_ch.ndim != 2:
            raise ValueError(
                f"Expected a 1D or 2D channel-independent {natural_weight_column} "
                f"column, got {natural_weight_ch.shape}."
            )
        if natural_weight_ch.shape[0] != row_count:
            raise ValueError(
                f"{natural_weight_column} row count {natural_weight_ch.shape[0]} "
                f"does not match selected row count {row_count}."
            )
        if natural_weight_ch.shape[1] not in (1, corr_count):
            raise ValueError(
                f"Expected {natural_weight_column} to have 1 or {corr_count} "
                f"correlation weights per row, got {natural_weight_ch.shape[1]}."
            )

    data_i_ch, weights_nat_ch, invalid_weight = _stokes_i_and_weight(
        data_ch, natural_weight_ch
    )
    invalid_data = _invalid_correlations(data_ch, flag_ch, flag_zero_data)
    flag = base_flag | invalid_data | invalid_weight
    data_i[:] = data_i_ch
    weights_nat[:] = weights_nat_ch

    if weights_img is not None and imaging_weight_column is not None:
        if imaging_weight_column == "IMAGING_WEIGHT_SPECTRUM":
            img_corr = _read_channel_correlations(
                tab, imaging_weight_column, member.channel_id, channel_count, corr_count
            )
            img_weight = (
                img_corr[:, 0]
                if img_corr.shape[1] == 1
                else 0.5 * (img_corr[:, 0] + img_corr[:, -1])
            )
        else:
            # Follow the WSClean convention: IMAGING_WEIGHT is per row/channel,
            # while IMAGING_WEIGHT_SPECTRUM carries correlation information.
            img_values = np.asarray(tab.getcol(imaging_weight_column), dtype=np.float64)
            if img_values.ndim == 1:
                img_weight = img_values
            elif img_values.ndim == 2:
                if img_values.shape[1] != channel_count:
                    raise ValueError(
                        f"Expected {imaging_weight_column} to have "
                        f"{channel_count} channel weights per row, got "
                        f"{img_values.shape[1]}."
                    )
                img_weight = img_values[:, member.channel_id]
            else:
                raise ValueError(
                    f"Expected a 1D or 2D WSClean-style {imaging_weight_column} "
                    f"column, got {img_values.shape}."
                )
            if img_weight.shape[0] != row_count:
                raise ValueError(
                    f"{imaging_weight_column} row count {img_weight.shape[0]} "
                    f"does not match selected row count {row_count}."
                )
        valid_weight = (img_weight > 0) & np.isfinite(img_weight)
        np.sqrt(img_weight, out=weights_img, where=valid_weight)

    print(
        f"INFO: data_ch_{channel.output_index}.mat processed {row_count} rows.",
        flush=True,
    )

    return data_i, weights_nat, flag, weights_img


def _read_channel_correlations(
    tab,
    column_name: str,
    channel_id: int,
    channel_count: int,
    expected_corr_count: int | None = None,
) -> np.ndarray:
    """Read all correlations for one physical channel with ``getcolslice``.

    Args:
        tab: Casacore table view to read.
        column_name (str): Array column name, such as ``DATA``, ``FLAG``, or
            ``WEIGHT_SPECTRUM``.
        channel_id (int): Zero-based channel index inside the SPW.
        channel_count (int): Number of channels in the SPW.
        expected_corr_count (int | None, optional): Expected number of
            correlations. If given, the selected column must match it. Default:
            None.

    Returns:
        np.ndarray: Array with shape ``(row, correlation)``.

    Raises:
        ValueError: If the column is not shaped as ``(channel, correlation)``
            per MS row, or if the correlation count does not match
            ``expected_corr_count``.

    Notes:
        This converter intentionally supports one in-memory MS convention:
        python-casacore exposes spectrum-like cells as
        ``(channel, correlation)``. Tables exposing another axis order should
        fail here instead of producing silently transposed data.
    """
    sample = np.asarray(tab.getcell(column_name, 0))
    if sample.ndim == 1:
        if channel_count != 1 or channel_id != 0:
            raise ValueError(
                f"Cannot select channel {channel_id} from 1D {column_name} "
                f"cell with {channel_count} channels."
            )
        corr_count = sample.shape[0]
        values = np.asarray(tab.getcolslice(column_name, [0], [corr_count - 1]))
        if values.ndim == 1:
            values = values.reshape(-1, 1)
    elif sample.ndim == 2:
        if sample.shape[0] != channel_count:
            raise ValueError(
                f"Expected {column_name} cells to use shape "
                f"(channel, correlation) with {channel_count} channels, got "
                f"{sample.shape}."
            )
        corr_count = sample.shape[1]
        values = np.asarray(
            tab.getcolslice(
                column_name,
                [channel_id, 0],
                [channel_id, corr_count - 1],
            )
        )
        if values.ndim == 3 and values.shape[1] == 1:
            values = values[:, 0, :]
    else:
        raise ValueError(f"Expected a 1D or 2D {column_name} cell, got {sample.shape}.")

    if expected_corr_count is not None and corr_count != expected_corr_count:
        raise ValueError(
            f"{column_name} has {corr_count} correlations, expected "
            f"{expected_corr_count}."
        )
    if values.ndim == 2 and values.shape[1] == corr_count:
        return values
    raise ValueError(
        f"Expected getcolslice({column_name!r}) to return shape "
        f"(row, {corr_count}), got {values.shape}."
    )


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
    """Write one output channel MAT file in the loader-compatible layout.

    Args:
        output_dir (Path): Output source directory.
        channel (MsChannel): Output channel descriptor.
        data_i (np.ndarray): Stokes I visibility values for unflagged rows.
        weights_nat (np.ndarray): Natural whitening weights for unflagged rows.
        flag (np.ndarray): Full invalid-sample mask aligned to ``msSpecs.uvw``.
        weights_img (np.ndarray | None, optional): Optional imaging-weight
            multipliers for unflagged rows. Default: None.

    Raises:
        ValueError: If the unflagged row count implied by ``flag`` does not
            match ``data_i``, ``weights_nat``, or ``weights_img``.

    Notes:
        ``load_data.py`` expects ``flag`` to keep the full UVW length, while
        ``data_I`` and weights contain only the unflagged samples.
    """
    valid_count = int(np.count_nonzero(~np.asarray(flag, dtype=bool)))
    if data_i.size != valid_count or weights_nat.size != valid_count:
        raise ValueError(
            f"data_ch_{channel.output_index} has {valid_count} unflagged UVW "
            f"rows, but data_I has {data_i.size} values and weightsNat has "
            f"{weights_nat.size} values."
        )
    if weights_img is not None and weights_img.size != valid_count:
        raise ValueError(
            f"data_ch_{channel.output_index} has {valid_count} unflagged UVW "
            f"rows, but weightsImg has {weights_img.size} values."
        )

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
