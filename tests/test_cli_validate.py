"""Tests for the validate CLI command."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from ams02wb.cli.main import cli


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_VALID_RECORD = {
    "energy_centre_gev": 10.0,
    "energy_low_gev": 5.0,
    "energy_high_gev": 15.0,
    "flux": 1.23,
    "flux_err_stat_lo": 0.01,
    "flux_err_stat_hi": 0.02,
}

_INVALID_RECORD = {
    "energy_centre_gev": -1.0,
    "energy_low_gev": 5.0,
    "energy_high_gev": 15.0,
    "flux": 1.23,
    "flux_err_stat_lo": 0.01,
    "flux_err_stat_hi": 0.02,
}


def _write_json(path: Path, data: object) -> Path:
    path.write_text(json.dumps(data))
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_validate_all_valid_files_prints_pass(tmp_path: Path) -> None:
    """All valid files should produce PASS status and exit 0."""
    _write_json(tmp_path / "a.json", [_VALID_RECORD])
    _write_json(tmp_path / "b.json", [_VALID_RECORD])

    runner = CliRunner()
    result = runner.invoke(cli, ["validate", "--input-dir", str(tmp_path)])

    assert result.exit_code == 0
    assert "PASS" in result.output
    # No FAIL lines expected
    assert "FAIL" not in result.output


def test_validate_invalid_file_prints_fail_with_field_name(tmp_path: Path) -> None:
    """Invalid records should produce FAIL and show the offending field name."""
    _write_json(tmp_path / "bad.json", [_INVALID_RECORD])

    runner = CliRunner()
    result = runner.invoke(cli, ["validate", "--input-dir", str(tmp_path)])

    assert result.exit_code == 1
    assert "FAIL" in result.output
    # energy_centre_gev is -1.0, below minimum — field name should appear
    assert "energy_centre_gev" in result.output


def test_validate_mixed_dir_reports_per_file_status(tmp_path: Path) -> None:
    """A directory with valid and invalid files reports status for each."""
    _write_json(tmp_path / "good.json", [_VALID_RECORD])
    _write_json(tmp_path / "bad.json", [_INVALID_RECORD])

    runner = CliRunner()
    result = runner.invoke(cli, ["validate", "--input-dir", str(tmp_path)])

    assert result.exit_code == 1
    assert "PASS" in result.output
    assert "FAIL" in result.output


def test_validate_invalid_file_error_includes_offending_value(
    tmp_path: Path,
) -> None:
    """Validation output should include the actual offending value."""
    _write_json(tmp_path / "bad.json", [_INVALID_RECORD])

    runner = CliRunner()
    result = runner.invoke(cli, ["validate", "--input-dir", str(tmp_path)])

    assert result.exit_code == 1
    # The offending value -1.0 should appear in the output
    assert "-1.0" in result.output


def test_validate_empty_dir_exits_with_code_two(tmp_path: Path) -> None:
    """An empty directory (no JSON files) should exit with code 2."""
    runner = CliRunner()
    result = runner.invoke(cli, ["validate", "--input-dir", str(tmp_path)])

    assert result.exit_code == 2
    assert "No JSON files found" in result.output


def test_validate_nonexistent_dir_exits_nonzero(tmp_path: Path) -> None:
    """A nonexistent directory should exit non-zero with an error message."""
    runner = CliRunner()
    result = runner.invoke(
        cli, ["validate", "--input-dir", str(tmp_path / "no_such_dir")]
    )

    assert result.exit_code != 0
    assert "does not exist" in result.output
