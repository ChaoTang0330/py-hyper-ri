"""Unit tests for the HYRI configuration creator."""

import copy
import json
from pathlib import Path

import pytest

import hyri.interface.create_config as cc


def test_parsing_config_arguments_defaults() -> None:
    """Parser should return documented defaults without touching sys.argv."""
    args = cc.parsing_config_arguments([])

    assert args.default is False
    assert args.output is None
    assert args.overwrite is False


def test_parsing_config_arguments_overrides() -> None:
    """Parser should read default-copy, output, and overwrite options."""
    args = cc.parsing_config_arguments(
        ["--default", "--output", "generated.json", "--overwrite"]
    )

    assert args.default is True
    assert args.output == "generated.json"
    assert args.overwrite is True


def test_create_config_default_writes_bundled_template(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Default mode should write the bundled template without prompting."""
    output_path = tmp_path / "nested" / "hyri_config.json"

    cc.create_config(["--default", "--output", str(output_path)])

    assert (
        json.loads(output_path.read_text(encoding="utf-8"))
        == cc.load_default_config()
    )
    captured = capsys.readouterr().out
    assert f"Wrote {output_path}" in captured
    assert f"hyri-imaging --config {output_path} --data_file <path>" in captured


def test_prepare_output_path_rejects_existing_file_without_overwrite(
    tmp_path: Path,
) -> None:
    """Existing output files should be protected unless overwrite is enabled."""
    output_path = tmp_path / "existing.json"
    output_path.write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit, match="Use --overwrite to replace it"):
        cc.prepare_output_path(str(output_path))


def test_prepare_output_path_allows_existing_file_with_overwrite(
    tmp_path: Path,
) -> None:
    """Overwrite mode should allow an existing output path."""
    output_path = tmp_path / "existing.json"
    output_path.write_text("{}", encoding="utf-8")

    cc.prepare_output_path(str(output_path), overwrite=True)

    assert output_path.exists()


def test_write_config_serialises_pretty_json(tmp_path: Path) -> None:
    """Configuration writing should use indented JSON with a trailing newline."""
    output_path = tmp_path / "config.json"
    config = [{"main": {"algorithm": "usara"}}]

    cc.write_config(config, str(output_path))

    text = output_path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert json.loads(text) == config


def test_apply_config_answers_updates_copy_and_prunes_algorithms() -> None:
    """Wizard answers should update a copy and keep only the selected algorithm."""
    config = cc.load_default_config()
    original = copy.deepcopy(config)
    answers = {
        "updates": [
            ("main", "algorithm", "airi"),
            ("main", "im_dim_x", 128),
            ("airi", "heu_noise_scale", 1.5),
        ]
    }

    updated = cc.apply_config_answers(config, answers)
    sections = cc.config_sections(updated)

    assert config == original
    assert sections["main"]["algorithm"] == "airi"
    assert sections["main"]["im_dim_x"] == 128
    assert sections["airi"]["heu_noise_scale"] == 1.5
    assert "airi_default" in sections
    assert "usara" not in sections
    assert "hyperairi" not in sections


def test_ask_until_valid_reprompts_after_invalid_input(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Invalid parser values should print an error and ask again."""
    responses = iter(["bad", "3"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(responses))

    value = cc.ask_until_valid("Iterations", 1, cc.parse_int)

    assert value == 3
    assert "Invalid value: enter an integer" in capsys.readouterr().out


def test_ask_until_valid_returns_deep_copy_of_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty input should return a copy of mutable default values."""
    default = [1, 2]
    monkeypatch.setattr("builtins.input", lambda prompt: "")

    value = cc.ask_until_valid("Partition", default, cc.parse_optional_int_list)

    assert value == default
    assert value is not default


def test_ask_until_valid_raises_runtime_error_on_eof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Closed stdin should be reported as an incomplete interactive session."""

    def raise_eof(prompt: str) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", raise_eof)

    with pytest.raises(RuntimeError, match="Interactive input ended"):
        cc.ask_until_valid("Value", "", cc.parse_str)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("yes", True),
        ("Y", True),
        ("true", True),
        ("1", True),
        ("no", False),
        ("N", False),
        ("false", False),
        ("0", False),
    ],
)
def test_parse_bool_accepts_common_boolean_values(text: str, expected: bool) -> None:
    """Boolean parser should accept common true and false spellings."""
    assert cc.parse_bool(text) is expected


@pytest.mark.parametrize("text", ["maybe", "2", ""])
def test_parse_bool_rejects_unknown_values(text: str) -> None:
    """Boolean parser should reject ambiguous values."""
    with pytest.raises(ValueError, match="enter yes/no or true/false"):
        cc.parse_bool(text)


def test_parse_numeric_values_and_empty_sentinels() -> None:
    """Numeric parsers should enforce bounds and preserve empty sentinels."""
    assert cc.parse_int("4", minimum=1) == 4
    assert cc.parse_float("1.25", minimum=0.0) == 1.25
    assert cc.parse_optional_int("none") == []
    assert cc.parse_optional_float("[]") == []
    assert cc.parse_optional_int_list("1, 2 3", minimum=1) == [1, 2, 3]

    with pytest.raises(ValueError, match=">= 5"):
        cc.parse_int("4", minimum=5)
    with pytest.raises(ValueError, match=">= 2.0"):
        cc.parse_float("1.5", minimum=2.0)


def test_parse_choice_and_spectral_index_initialisation() -> None:
    """Choice parsers should accept only configured options."""
    assert cc.parse_choice("airi", cc.ALGORITHMS) == "airi"
    assert cc.parse_spec_idx_init("zero") == "zero"
    assert cc.parse_spec_idx_init("custom.fits") == "custom.fits"

    with pytest.raises(ValueError, match="choose one of"):
        cc.parse_choice("clean", cc.ALGORITHMS)
    with pytest.raises(ValueError, match="FITS file path"):
        cc.parse_spec_idx_init("custom.txt")


def test_config_sections_ignores_comment_blocks() -> None:
    """Section mapping should omit JSON comment keys."""
    sections = cc.config_sections(
        [{"__comment": "Main", "main": {"algorithm": "usara"}}]
    )

    assert sections == {"main": {"algorithm": "usara"}}
