"""Tests for the export-dataset CLI command in main.py.

Verifies that export-dataset dispatches correctly to each exporter
and fails clearly on unsupported formats.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from ams02wb.cli.main import cli


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def dataset_file(tmp_path: Path) -> Path:
    """Write a minimal JSON dataset file for CLI tests."""
    data = {
        "measurements": [{"x": 1.0, "y": 2.0}],
        "provenance": {"source": "test"},
    }
    path = tmp_path / "input.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_export_dataset_help_shows_options(runner: CliRunner) -> None:
    """Help output must list --dataset, --format, and --output."""
    result = runner.invoke(cli, ["export-dataset", "--help"])
    assert result.exit_code == 0
    assert "--dataset" in result.output
    assert "--format" in result.output
    assert "--output" in result.output


def test_export_dataset_parquet_dispatches(
    runner: CliRunner, dataset_file: Path, tmp_path: Path
) -> None:
    """--format parquet delegates to export_parquet from the exports module."""
    out_file = tmp_path / "out.parquet"

    with patch("ams02wb.exports.parquet.export_parquet") as mock_exp:
        result = runner.invoke(cli, [
            "export-dataset",
            "--dataset", str(dataset_file),
            "--format", "parquet",
            "--output", str(out_file),
        ])

    assert result.exit_code == 0, result.output
    mock_exp.assert_called_once()
    # Verify the first arg contains a DataFrame under 'data' key.
    call_args = mock_exp.call_args[0]
    assert "data" in call_args[0]


def test_export_dataset_csv_dispatches(
    runner: CliRunner, dataset_file: Path, tmp_path: Path
) -> None:
    """--format csv delegates to export_csv with Measurement objects."""
    out_file = tmp_path / "out.csv"

    with patch("ams02wb.exports.csv_export.export_csv") as mock_csv:
        result = runner.invoke(cli, [
            "export-dataset",
            "--dataset", str(dataset_file),
            "--format", "csv",
            "--output", str(out_file),
        ])

    assert result.exit_code == 0, result.output
    mock_csv.assert_called_once()


def test_export_dataset_json_dispatches(
    runner: CliRunner, dataset_file: Path, tmp_path: Path
) -> None:
    """--format json delegates to export_json."""
    out_file = tmp_path / "out.json"

    with patch("ams02wb.exports.json_export.export_json") as mock_json:
        result = runner.invoke(cli, [
            "export-dataset",
            "--dataset", str(dataset_file),
            "--format", "json",
            "--output", str(out_file),
        ])

    assert result.exit_code == 0, result.output
    mock_json.assert_called_once()


def test_export_dataset_usine_writes_file(
    runner: CliRunner, tmp_path: Path
) -> None:
    """--format usine calls export_usine and writes the returned text to disk."""
    dataset_file = tmp_path / "input.json"
    dataset_file.write_text(json.dumps([]), encoding="utf-8")
    out_file = tmp_path / "out.usine"

    with patch("ams02wb.exports.usine_export.export_usine", return_value="# USINE v3\n") as mock_usine:
        result = runner.invoke(cli, [
            "export-dataset",
            "--dataset", str(dataset_file),
            "--format", "usine",
            "--output", str(out_file),
        ])

    assert result.exit_code == 0, result.output
    mock_usine.assert_called_once()
    assert out_file.read_text(encoding="utf-8") == "# USINE v3\n"


def test_export_dataset_unknown_format_exits_nonzero(
    runner: CliRunner, tmp_path: Path
) -> None:
    """An invalid --format value causes a non-zero exit with a clear message.

    Click's Choice type rejects the value before the command body runs,
    so exit code is 2 (usage error) and the bad value appears in output.
    """
    dataset_file = tmp_path / "input.json"
    dataset_file.write_text(json.dumps([]), encoding="utf-8")

    result = runner.invoke(cli, [
        "export-dataset",
        "--dataset", str(dataset_file),
        "--format", "xlsx",
        "--output", str(tmp_path / "out.bin"),
    ])

    assert result.exit_code != 0
    # Click mentions the invalid value in the error output.
    assert "xlsx" in result.output.lower()


def test_export_dataset_missing_dataset_exits_nonzero(
    runner: CliRunner, tmp_path: Path
) -> None:
    """--dataset pointing to a nonexistent file must fail with non-zero exit."""
    result = runner.invoke(cli, [
        "export-dataset",
        "--dataset", str(tmp_path / "nonexistent.json"),
        "--format", "csv",
        "--output", str(tmp_path / "out.csv"),
    ])

    assert result.exit_code != 0
