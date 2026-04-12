"""CLI integration tests exercising each subcommand via CliRunner."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from click.testing import CliRunner

from ams02wb.cli.main import cli
from tests.conftest import make_fixture_record


def _make_runner() -> CliRunner:
    return CliRunner()


def _write_publication_index(directory: Path) -> Path:
    """Write a tiny JSON publication index with 2 entries."""
    index = [
        {
            "publication_id": "pub-001",
            "title": "Proton Flux Measurement",
            "source_url": "https://ams02.space/papers/001/",
            "attachments": [{"filename": "data.csv", "url": "https://ams02.space/001/data.csv"}],
        },
        {
            "publication_id": "pub-002",
            "title": "Helium Flux Measurement",
            "source_url": "https://ams02.space/papers/002/",
            "attachments": [],
        },
    ]
    path = directory / "publication_index.json"
    path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return path


def _write_harmonised_parquet(path: Path, n_rows: int = 5) -> Path:
    """Write a small parquet file with harmonised records."""
    records = [make_fixture_record(
        x_min=1.0 + i,
        x_max=2.0 + i,
        x_centre=1.5 + i,
        y_value=100.0 + i * 20,
    ) for i in range(n_rows)]
    df = pd.DataFrame(records)
    df.to_parquet(path, index=False)
    return path


def _write_ingested_dataset(directory: Path, filename: str) -> Path:
    """Write a minimal ingested-format JSON dataset file."""
    dataset = {
        "publication_id": "test-001",
        "title": "Test Publication",
        "source_url": "https://example.com/test",
        "measurements": [
            {
                "energy_low": 1.0,
                "energy_high": 2.0,
                "energy_mid": 1.5,
                "value": 150.0,
                "unit": "GeV",
                "axis_type": "kinetic_energy_per_nucleon",
                "species": "PROTON",
            },
        ],
        "provenance": {
            "source_url": "https://example.com/test",
            "content_hash": "sha256:abc123",
            "parse_method": "csv_table_extraction",
            "ingested_at": "2025-01-01T00:00:00+00:00",
        },
    }
    path = directory / filename
    path.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    return path


def _write_export_dataset(path: Path) -> Path:
    """Write a minimal JSON dataset file for export commands."""
    dataset = {
        "measurements": [
            {
                "energy_low": 1.0,
                "energy_high": 2.0,
                "energy_mid": 1.5,
                "value": 150.0,
                "unit": "GeV",
                "axis_type": "kinetic_energy_per_nucleon",
                "species": "PROTON",
            },
        ],
        "provenance": {
            "source_url": "https://example.com/test",
        },
    }
    path.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_cli_main_help():
    """Main CLI --help exits 0 and lists subcommands."""
    runner = _make_runner()
    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0, result.output
    assert "AMS-02" in result.output
    # All registered subcommands should appear in help
    assert "index-publications" in result.output
    assert "harmonise" in result.output


@patch("ams02wb.cli.index_publications.requests.Session")
@patch("ams02wb.cli.index_publications.load_publication_index")
def test_index_publications_happy_path(mock_load, mock_session_cls, tmp_path):
    """index-publications writes a JSON index file from the fetched entries."""
    from ams02wb.crawler.index_loader import PublicationEntry

    mock_load.return_value = [
        PublicationEntry(paper_id="101", title="Proton", url="https://ams02.space/101/"),
        PublicationEntry(paper_id="202", title="Helium", url="https://ams02.space/202/"),
    ]

    runner = _make_runner()
    out_dir = str(tmp_path / "output")
    result = runner.invoke(cli, ["index-publications", "--output-dir", out_dir])

    assert result.exit_code == 0, result.output
    index_file = Path(out_dir) / "publication_index.json"
    assert index_file.exists()
    data = json.loads(index_file.read_text(encoding="utf-8"))
    assert len(data) == 2
    assert data[0]["paper_id"] == "101"


@patch("ams02wb.cli.ingest_publication.requests.Session")
@patch("ams02wb.cli.ingest_publication.load_publication_index")
@patch("ams02wb.cli.ingest_publication.fetch_publication_page")
@patch("ams02wb.cli.ingest_publication.resolve_attachments")
def test_ingest_publication_happy_path(
    mock_resolve, mock_fetch, mock_load, mock_session_cls, tmp_path
):
    """ingest-publication fetches a single paper and writes dataset JSON."""
    from ams02wb.crawler.index_loader import PublicationEntry
    from ams02wb.crawler.page_fetcher import FetchedPage

    entry = PublicationEntry(paper_id="101", title="Proton", url="https://ams02.space/101/")
    mock_load.return_value = [entry]
    mock_fetch.return_value = FetchedPage(html="<html></html>", base_url="https://ams02.space/101/")
    mock_resolve.return_value = []

    runner = _make_runner()
    out_dir = str(tmp_path / "output")
    result = runner.invoke(cli, [
        "ingest-publication",
        "--publication-id", "101",
        "--output-dir", out_dir,
    ])

    assert result.exit_code == 0, result.output
    dataset_file = Path(out_dir) / "101.json"
    assert dataset_file.exists()
    data = json.loads(dataset_file.read_text(encoding="utf-8"))
    assert data["publication_id"] == "101"


@patch("ams02wb.cli.ingest_all.requests.Session")
@patch("ams02wb.cli.ingest_all.load_publication_index")
@patch("ams02wb.cli.ingest_publication.fetch_publication_page")
@patch("ams02wb.cli.ingest_publication.resolve_attachments")
def test_ingest_all_happy_path(
    mock_resolve, mock_fetch, mock_load, mock_session_cls, tmp_path
):
    """ingest-all processes all entries and writes dataset files."""
    from ams02wb.crawler.index_loader import PublicationEntry
    from ams02wb.crawler.page_fetcher import FetchedPage

    entries = [
        PublicationEntry(paper_id="101", title="Proton", url="https://ams02.space/101/"),
        PublicationEntry(paper_id="202", title="Helium", url="https://ams02.space/202/"),
    ]
    mock_load.return_value = entries
    mock_fetch.return_value = FetchedPage(html="<html></html>", base_url="https://ams02.space/")
    mock_resolve.return_value = []

    runner = _make_runner()
    out_dir = str(tmp_path / "output")
    result = runner.invoke(cli, ["ingest-all", "--output-dir", out_dir])

    assert result.exit_code == 0, result.output
    assert (Path(out_dir) / "101.json").exists()
    assert (Path(out_dir) / "202.json").exists()


@patch("ams02wb.cli.validate.validate_dataset_dir")
def test_validate_happy_path(mock_validate, tmp_path):
    """validate reports PASS for well-formed datasets."""
    mock_validate.return_value = [
        {"file_path": "test.json", "status": "PASS", "findings": []},
    ]

    runner = _make_runner()
    input_dir = tmp_path / "datasets"
    input_dir.mkdir()
    _write_ingested_dataset(input_dir, "test.json")

    result = runner.invoke(cli, ["validate", "--input-dir", str(input_dir)])

    assert result.exit_code == 0, result.output
    assert "PASS" in result.output


@patch("ams02wb.cli.harmonise.run_harmonisation_pipeline")
def test_harmonise_happy_path(mock_pipeline, tmp_path):
    """harmonise processes input JSON and writes harmonised output."""
    mock_pipeline.return_value = [make_fixture_record()]

    runner = _make_runner()
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    _write_ingested_dataset(input_dir, "pub.json")

    result = runner.invoke(cli, [
        "harmonise",
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
    ])

    assert result.exit_code == 0, result.output
    assert "Harmonised" in result.output
    output_files = list(output_dir.glob("*.json"))
    assert len(output_files) == 1


def test_build_likelihood_happy_path(tmp_path):
    """build-likelihood reads a parquet dataset and writes fit-ready output."""
    dataset_path = tmp_path / "harmonised.parquet"
    _write_harmonised_parquet(dataset_path)

    output_path = tmp_path / "fit.parquet"

    runner = _make_runner()
    result = runner.invoke(cli, [
        "build-likelihood",
        "--dataset", str(dataset_path),
        "--mode", "diag",
        "--output", str(output_path),
    ])

    assert result.exit_code == 0, result.output
    assert output_path.exists()
    # Sidecar JSON should also be written
    sidecar = output_path.with_suffix(".json")
    assert sidecar.exists()
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    assert meta["mode"] == "diagonal"


@patch("ams02wb.cli.main._load_harmonised_dataframe")
@patch("ams02wb.cli.main.align_daily_series")
def test_align_time_series_happy_path(mock_align, mock_load, tmp_path):
    """align-time-series aligns species and writes a parquet file."""
    import datetime

    from ams02wb.alignment.daily import DailyAlignedResult

    aligned_df = pd.DataFrame({
        "time_start": [datetime.date(2020, 1, 1)],
        "y_value_proton": [1.0],
    })
    mock_align.return_value = DailyAlignedResult(
        aligned_df=aligned_df,
        missing_counts={"proton": 0},
        join_type="inner",
        date_range=(datetime.date(2020, 1, 1), datetime.date(2020, 1, 1)),
    )
    mock_load.return_value = pd.DataFrame({"y_value": [1.0]})

    output_path = tmp_path / "aligned.parquet"

    runner = _make_runner()
    result = runner.invoke(cli, [
        "align-time-series",
        "--species", "proton",
        "--output", str(output_path),
    ])

    assert result.exit_code == 0, result.output
    assert output_path.exists()
    df = pd.read_parquet(output_path)
    assert len(df) == 1


def test_export_dataset_parquet_happy_path(tmp_path):
    """export-dataset --format parquet reads JSON and writes a parquet file."""
    dataset_path = tmp_path / "dataset.json"
    _write_export_dataset(dataset_path)
    output_path = tmp_path / "exported.parquet"

    runner = _make_runner()
    result = runner.invoke(cli, [
        "export-dataset",
        "--dataset", str(dataset_path),
        "--format", "parquet",
        "--output", str(output_path),
    ])

    assert result.exit_code == 0, result.output
    assert output_path.exists()
    df = pd.read_parquet(output_path)
    assert len(df) >= 1


def test_subcommand_help():
    """Every registered subcommand responds to --help without errors."""
    runner = _make_runner()
    subcommands = [
        "index-publications",
        "ingest-publication",
        "ingest-all",
        "validate",
        "harmonise",
        "build-likelihood",
        "align-time-series",
        "export-dataset",
    ]
    for subcmd in subcommands:
        result = runner.invoke(cli, [subcmd, "--help"])
        assert result.exit_code == 0, f"{subcmd} --help failed: {result.output}"
        # Help text should contain the subcommand's options or description
        assert "--help" in result.output, f"{subcmd} --help missing --help in output"
