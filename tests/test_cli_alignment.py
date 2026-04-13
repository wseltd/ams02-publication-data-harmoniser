"""Tests for align-time-series and export-dataset CLI commands."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from click.testing import CliRunner

from ams02wb.cli.main import cli
from ams02wb.alignment.daily import DailyAlignedResult


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def sample_daily_result() -> DailyAlignedResult:
    """A minimal DailyAlignedResult for mocking align_daily_series."""
    import datetime

    df = pd.DataFrame({
        "time_start": [datetime.date(2020, 1, 1), datetime.date(2020, 1, 2)],
        "y_value_proton": [1.0, 2.0],
        "y_value_helium": [3.0, 4.0],
    })
    return DailyAlignedResult(
        aligned_df=df,
        missing_counts={"proton": 0, "helium": 1},
        join_type="inner",
        date_range=(datetime.date(2020, 1, 1), datetime.date(2020, 1, 2)),
    )


# --- align-time-series tests ---


def test_align_time_series_help_shows_options(runner: CliRunner) -> None:
    """Help output must list --species, --join, --cadence, --output."""
    result = runner.invoke(cli, ["align-time-series", "--help"])
    assert result.exit_code == 0
    assert "--species" in result.output
    assert "--join" in result.output
    assert "--cadence" in result.output
    assert "--output" in result.output


def test_align_time_series_daily_intersection(
    runner: CliRunner, sample_daily_result: DailyAlignedResult, tmp_path: Path
) -> None:
    """Daily cadence with intersection join delegates to align_daily_series with join='inner'."""
    out_file = tmp_path / "aligned.parquet"

    with patch("ams02wb.cli.main._load_harmonised_dataframe") as mock_load, \
         patch("ams02wb.cli.main.align_daily_series", return_value=sample_daily_result) as mock_align:
        mock_load.side_effect = lambda sp, input_dir=None: pd.DataFrame({
            "time_start": [pd.Timestamp("2020-01-01")],
            "y_value": [1.0],
        })

        result = runner.invoke(cli, [
            "align-time-series",
            "--species", "proton",
            "--species", "helium",
            "--join", "intersection",
            "--cadence", "daily",
            "--output", str(out_file),
        ])

    assert result.exit_code == 0, result.output
    mock_align.assert_called_once()
    call_kwargs = mock_align.call_args
    assert call_kwargs.kwargs["join"] == "inner"
    assert out_file.exists()


def test_align_time_series_bartels_cadence(
    runner: CliRunner, tmp_path: Path
) -> None:
    """Bartels cadence delegates to align_bartels_rotation."""
    out_file = tmp_path / "bartels.parquet"

    bartels_result = {
        2700: pd.DataFrame({"time_start": ["2020-01-01"], "y_value": [1.0]}),
        2701: pd.DataFrame({"time_start": ["2020-01-28"], "y_value": [2.0]}),
    }

    with patch("ams02wb.cli.main._load_harmonised_dataframe") as mock_load, \
         patch("ams02wb.cli.main.align_bartels_rotation", return_value=bartels_result) as mock_bartels:
        mock_load.return_value = pd.DataFrame({
            "time_start": ["2020-01-01", "2020-01-28"],
            "y_value": [1.0, 2.0],
        })

        result = runner.invoke(cli, [
            "align-time-series",
            "--species", "proton",
            "--cadence", "bartels",
            "--output", str(out_file),
        ])

    assert result.exit_code == 0, result.output
    mock_bartels.assert_called_once()
    assert out_file.exists()


def test_align_time_series_logs_dropped_dates(
    runner: CliRunner, sample_daily_result: DailyAlignedResult, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """When outer join introduces missing dates, diagnostics are logged to stderr at INFO."""
    out_file = tmp_path / "aligned.parquet"

    with caplog.at_level(logging.INFO, logger="ams02wb.cli.main"), \
         patch("ams02wb.cli.main._load_harmonised_dataframe") as mock_load, \
         patch("ams02wb.cli.main.align_daily_series", return_value=sample_daily_result):
        mock_load.return_value = pd.DataFrame({
            "time_start": [pd.Timestamp("2020-01-01")],
            "y_value": [1.0],
        })

        result = runner.invoke(cli, [
            "align-time-series",
            "--species", "proton",
            "--species", "helium",
            "--join", "union",
            "--cadence", "daily",
            "--output", str(out_file),
        ])

    assert result.exit_code == 0
    # helium has missing_counts=1, so a diagnostic log record should exist.
    log_text = caplog.text.lower()
    assert "helium" in log_text
    assert "1" in caplog.text


# --- export-dataset tests ---


def test_export_dataset_help_shows_options(runner: CliRunner) -> None:
    """Help output must list --dataset, --format, --output."""
    result = runner.invoke(cli, ["export-dataset", "--help"])
    assert result.exit_code == 0
    assert "--dataset" in result.output
    assert "--format" in result.output
    assert "--output" in result.output


def test_export_dataset_parquet(runner: CliRunner, tmp_path: Path) -> None:
    """Parquet export delegates to export_parquet."""
    dataset_file = tmp_path / "input.json"
    dataset_file.write_text(json.dumps({
        "measurements": [{"x": 1.0, "y": 2.0}],
        "provenance": {"source": "test"},
    }))
    out_file = tmp_path / "out.parquet"

    with patch("ams02wb.exports.parquet.export_parquet", return_value=out_file) as mock_exp:
        result = runner.invoke(cli, [
            "export-dataset",
            "--dataset", str(dataset_file),
            "--format", "parquet",
            "--output", str(out_file),
        ])

    assert result.exit_code == 0, result.output
    mock_exp.assert_called_once()


def test_export_dataset_csv(runner: CliRunner, tmp_path: Path) -> None:
    """CSV export delegates to export_csv with Measurement objects."""
    dataset_file = tmp_path / "input.json"
    dataset_file.write_text(json.dumps([]))
    out_file = tmp_path / "out.csv"

    with patch("ams02wb.exports.csv_export.export_csv_from_dicts", return_value=out_file) as mock_csv:
        result = runner.invoke(cli, [
            "export-dataset",
            "--dataset", str(dataset_file),
            "--format", "csv",
            "--output", str(out_file),
        ])

    assert result.exit_code == 0, result.output
    mock_csv.assert_called_once()


def test_export_dataset_usine(runner: CliRunner, tmp_path: Path) -> None:
    """USINE export calls export_usine and writes result to file."""
    dataset_file = tmp_path / "input.json"
    dataset_file.write_text(json.dumps([]))
    out_file = tmp_path / "out.usine"

    with patch("ams02wb.exports.usine_export.export_usine", return_value="# USINE header\n") as mock_usine:
        result = runner.invoke(cli, [
            "export-dataset",
            "--dataset", str(dataset_file),
            "--format", "usine",
            "--output", str(out_file),
        ])

    assert result.exit_code == 0, result.output
    mock_usine.assert_called_once()
    assert out_file.read_text() == "# USINE header\n"


def test_export_dataset_unknown_format_exits_error(runner: CliRunner, tmp_path: Path) -> None:
    """An unsupported format must exit non-zero with a clear error.

    Click's Choice type rejects invalid values before the command body runs,
    so the exit code is 2 (usage error) and the error message names the bad value.
    """
    dataset_file = tmp_path / "input.json"
    dataset_file.write_text(json.dumps([]))
    out_file = tmp_path / "out.bin"

    result = runner.invoke(cli, [
        "export-dataset",
        "--dataset", str(dataset_file),
        "--format", "xlsx",
        "--output", str(out_file),
    ])

    assert result.exit_code != 0
    assert "xlsx" in result.output.lower()
