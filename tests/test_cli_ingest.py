"""Tests for index-publications and ingest-publication CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from click.testing import CliRunner

from ams02wb.cli.main import cli
from ams02wb.crawler import CrawlerError
from ams02wb.crawler.index_loader import PublicationEntry
from ams02wb.crawler.page_fetcher import FetchedPage


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_FAKE_ENTRIES = [
    PublicationEntry(paper_id="101", title="Proton Flux", url="https://ams02.space/papers/101/"),
    PublicationEntry(paper_id="202", title="Helium Flux", url="https://ams02.space/papers/202/"),
]

_FAKE_PAGE = FetchedPage(
    html='<html><body><a href="data.csv">CSV</a></body></html>',
    base_url="https://ams02.space/papers/101/",
)

_FAKE_CSV_BYTES = b"energy_low,energy_high,value\n1.0,2.0,100.5\n"


def _mock_session_for_ingest() -> MagicMock:
    """Build a mock session that responds to attachment downloads."""
    session = MagicMock()
    resp = MagicMock()
    resp.content = _FAKE_CSV_BYTES
    resp.raise_for_status = MagicMock()
    session.get.return_value = resp
    return session


# ---------------------------------------------------------------------------
# index-publications tests
# ---------------------------------------------------------------------------


@patch("ams02wb.cli.index_publications.requests.Session")
@patch("ams02wb.cli.index_publications.load_publication_index")
def test_index_publications_writes_json_index_to_output_dir(
    mock_load: MagicMock, mock_session_cls: MagicMock, tmp_path: Path
) -> None:
    """index-publications must write a JSON file to the output directory."""
    mock_load.return_value = _FAKE_ENTRIES

    runner = CliRunner()
    result = runner.invoke(cli, ["index-publications", "--output-dir", str(tmp_path)])

    assert result.exit_code == 0, result.output
    index_file = tmp_path / "publication_index.json"
    assert index_file.exists()

    data = json.loads(index_file.read_text())
    assert isinstance(data, list)
    assert len(data) == 2


@patch("ams02wb.cli.index_publications.requests.Session")
@patch("ams02wb.cli.index_publications.load_publication_index")
def test_index_publications_index_contains_publication_ids(
    mock_load: MagicMock, mock_session_cls: MagicMock, tmp_path: Path
) -> None:
    """Each entry in the written index must include its paper_id."""
    mock_load.return_value = _FAKE_ENTRIES

    runner = CliRunner()
    result = runner.invoke(cli, ["index-publications", "--output-dir", str(tmp_path)])

    assert result.exit_code == 0, result.output
    data = json.loads((tmp_path / "publication_index.json").read_text())
    ids = {entry["paper_id"] for entry in data}
    assert ids == {"101", "202"}


# ---------------------------------------------------------------------------
# ingest-publication tests
# ---------------------------------------------------------------------------


@patch("ams02wb.cli.ingest_publication.requests.Session")
@patch("ams02wb.cli.ingest_publication.load_publication_index")
@patch("ams02wb.cli.ingest_publication.fetch_publication_page")
def test_ingest_publication_produces_schema_conformant_dataset(
    mock_fetch: MagicMock,
    mock_load: MagicMock,
    mock_session_cls: MagicMock,
    tmp_path: Path,
) -> None:
    """Ingested dataset must have the required top-level keys."""
    mock_load.return_value = _FAKE_ENTRIES
    mock_fetch.return_value = _FAKE_PAGE
    mock_session_cls.return_value = _mock_session_for_ingest()

    runner = CliRunner()
    result = runner.invoke(
        cli, ["ingest-publication", "--publication-id", "101", "--output-dir", str(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    dataset = json.loads((tmp_path / "101.json").read_text())

    required_keys = {"publication_id", "title", "source_url", "attachments", "provenance"}
    assert required_keys.issubset(dataset.keys())
    assert dataset["publication_id"] == "101"


@patch("ams02wb.cli.ingest_publication.requests.Session")
@patch("ams02wb.cli.ingest_publication.load_publication_index")
@patch("ams02wb.cli.ingest_publication.fetch_publication_page")
def test_ingest_publication_provenance_contains_source_url(
    mock_fetch: MagicMock,
    mock_load: MagicMock,
    mock_session_cls: MagicMock,
    tmp_path: Path,
) -> None:
    """Provenance record must trace back to the publication URL."""
    mock_load.return_value = _FAKE_ENTRIES
    mock_fetch.return_value = _FAKE_PAGE
    mock_session_cls.return_value = _mock_session_for_ingest()

    runner = CliRunner()
    result = runner.invoke(
        cli, ["ingest-publication", "--publication-id", "101", "--output-dir", str(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    provenance = json.loads((tmp_path / "101.json").read_text())["provenance"]
    assert provenance["source_url"] == "https://ams02.space/papers/101/"


@patch("ams02wb.cli.ingest_publication.requests.Session")
@patch("ams02wb.cli.ingest_publication.load_publication_index")
@patch("ams02wb.cli.ingest_publication.fetch_publication_page")
def test_ingest_publication_provenance_contains_content_hash(
    mock_fetch: MagicMock,
    mock_load: MagicMock,
    mock_session_cls: MagicMock,
    tmp_path: Path,
) -> None:
    """Provenance must include a SHA-256 content hash."""
    mock_load.return_value = _FAKE_ENTRIES
    mock_fetch.return_value = _FAKE_PAGE
    mock_session_cls.return_value = _mock_session_for_ingest()

    runner = CliRunner()
    result = runner.invoke(
        cli, ["ingest-publication", "--publication-id", "101", "--output-dir", str(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    provenance = json.loads((tmp_path / "101.json").read_text())["provenance"]
    assert provenance["content_hash"].startswith("sha256:")
    # Hash hex portion must be 64 chars (SHA-256).
    hex_part = provenance["content_hash"].split(":", 1)[1]
    assert len(hex_part) == 64


@patch("ams02wb.cli.ingest_publication.requests.Session")
@patch("ams02wb.cli.ingest_publication.load_publication_index")
@patch("ams02wb.cli.ingest_publication.fetch_publication_page")
def test_ingest_publication_provenance_contains_parse_method(
    mock_fetch: MagicMock,
    mock_load: MagicMock,
    mock_session_cls: MagicMock,
    tmp_path: Path,
) -> None:
    """Provenance must declare which parse method was used."""
    mock_load.return_value = _FAKE_ENTRIES
    mock_fetch.return_value = _FAKE_PAGE
    mock_session_cls.return_value = _mock_session_for_ingest()

    runner = CliRunner()
    result = runner.invoke(
        cli, ["ingest-publication", "--publication-id", "101", "--output-dir", str(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    provenance = json.loads((tmp_path / "101.json").read_text())["provenance"]
    assert "parse_method" in provenance
    assert isinstance(provenance["parse_method"], str)
    assert len(provenance["parse_method"]) > 0


@patch("ams02wb.cli.ingest_publication.requests.Session")
@patch("ams02wb.cli.ingest_publication.load_publication_index")
def test_ingest_publication_network_error_exits_nonzero(
    mock_load: MagicMock,
    mock_session_cls: MagicMock,
    tmp_path: Path,
) -> None:
    """A network error during index fetch must cause a non-zero exit."""
    mock_load.side_effect = CrawlerError("Connection refused", status_code=None)

    runner = CliRunner()
    result = runner.invoke(
        cli, ["ingest-publication", "--publication-id", "101", "--output-dir", str(tmp_path)]
    )

    assert result.exit_code != 0


@patch("ams02wb.cli.ingest_publication.requests.Session")
@patch("ams02wb.cli.ingest_publication.load_publication_index")
@patch("ams02wb.cli.ingest_publication.fetch_publication_page")
def test_ingest_publication_parse_error_exits_nonzero(
    mock_fetch: MagicMock,
    mock_load: MagicMock,
    mock_session_cls: MagicMock,
    tmp_path: Path,
) -> None:
    """A CrawlerError when fetching the publication page must exit non-zero."""
    mock_load.return_value = _FAKE_ENTRIES
    mock_fetch.side_effect = CrawlerError("HTTP 500 Internal Server Error", status_code=500)

    runner = CliRunner()
    result = runner.invoke(
        cli, ["ingest-publication", "--publication-id", "101", "--output-dir", str(tmp_path)]
    )

    assert result.exit_code != 0


@patch("ams02wb.cli.ingest_publication.requests.Session")
@patch("ams02wb.cli.ingest_publication.load_publication_index")
def test_ingest_publication_missing_id_exits_nonzero(
    mock_load: MagicMock,
    mock_session_cls: MagicMock,
    tmp_path: Path,
) -> None:
    """Requesting a publication ID not in the index must exit non-zero."""
    mock_load.return_value = _FAKE_ENTRIES  # IDs 101 and 202 only

    runner = CliRunner()
    result = runner.invoke(
        cli, ["ingest-publication", "--publication-id", "999", "--output-dir", str(tmp_path)]
    )

    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# ingest-all tests
# ---------------------------------------------------------------------------


@patch("ams02wb.cli.ingest_all.requests.Session")
@patch("ams02wb.cli.ingest_all.load_publication_index")
@patch("ams02wb.cli.ingest_all.run_single_ingest")
def test_ingest_all_iterates_full_index(
    mock_run: MagicMock,
    mock_load: MagicMock,
    mock_session_cls: MagicMock,
    tmp_path: Path,
) -> None:
    """ingest-all must call run_single_ingest once per index entry."""
    mock_load.return_value = _FAKE_ENTRIES
    mock_run.return_value = {}

    runner = CliRunner()
    runner.invoke(cli, ["ingest-all", "--output-dir", str(tmp_path)])

    assert mock_run.call_count == len(_FAKE_ENTRIES)
    ingested_ids = [call.args[1].paper_id for call in mock_run.call_args_list]
    assert ingested_ids == ["101", "202"]


@patch("ams02wb.cli.ingest_all.requests.Session")
@patch("ams02wb.cli.ingest_all.load_publication_index")
@patch("ams02wb.cli.ingest_publication.fetch_publication_page")
def test_ingest_all_writes_datasets_to_output_dir(
    mock_fetch: MagicMock,
    mock_load: MagicMock,
    mock_session_cls: MagicMock,
    tmp_path: Path,
) -> None:
    """ingest-all must write one JSON file per publication to output-dir."""
    mock_load.return_value = _FAKE_ENTRIES
    mock_fetch.return_value = _FAKE_PAGE
    mock_session_cls.return_value = _mock_session_for_ingest()

    runner = CliRunner()
    runner.invoke(cli, ["ingest-all", "--output-dir", str(tmp_path)])

    written_files = sorted(p.name for p in tmp_path.glob("*.json"))
    assert "101.json" in written_files
    assert "202.json" in written_files

    dataset = json.loads((tmp_path / "101.json").read_text())
    assert dataset["publication_id"] == "101"


@patch("ams02wb.cli.ingest_all.requests.Session")
@patch("ams02wb.cli.ingest_all.load_publication_index")
@patch("ams02wb.cli.ingest_all.run_single_ingest")
def test_ingest_all_single_failure_does_not_abort_batch(
    mock_run: MagicMock,
    mock_load: MagicMock,
    mock_session_cls: MagicMock,
    tmp_path: Path,
) -> None:
    """A failure for one publication must not prevent ingestion of the rest."""
    mock_load.return_value = _FAKE_ENTRIES

    # First call raises, second succeeds.
    mock_run.side_effect = [
        CrawlerError("HTTP 500", status_code=500),
        {"publication_id": "202"},
    ]

    runner = CliRunner()
    runner.invoke(cli, ["ingest-all", "--output-dir", str(tmp_path)])

    # Both entries were attempted despite the first failure.
    assert mock_run.call_count == 2


@patch("ams02wb.cli.ingest_all.requests.Session")
@patch("ams02wb.cli.ingest_all.load_publication_index")
@patch("ams02wb.cli.ingest_all.run_single_ingest")
def test_ingest_all_summary_lists_failed_publication_ids(
    mock_run: MagicMock,
    mock_load: MagicMock,
    mock_session_cls: MagicMock,
    tmp_path: Path,
) -> None:
    """The failure summary must list the paper_id of each failed publication."""
    mock_load.return_value = _FAKE_ENTRIES

    mock_run.side_effect = [
        CrawlerError("HTTP 500", status_code=500),
        {"publication_id": "202"},
    ]

    runner = CliRunner()
    invocation = runner.invoke(cli, ["ingest-all", "--output-dir", str(tmp_path)])

    # The failed ID must appear in stderr output (mixed_stderr=False not set,
    # so Click merges stderr into output by default in CliRunner).
    assert "101" in invocation.output


@patch("ams02wb.cli.ingest_all.requests.Session")
@patch("ams02wb.cli.ingest_all.load_publication_index")
@patch("ams02wb.cli.ingest_all.run_single_ingest")
def test_ingest_all_exits_nonzero_when_any_failed(
    mock_run: MagicMock,
    mock_load: MagicMock,
    mock_session_cls: MagicMock,
    tmp_path: Path,
) -> None:
    """ingest-all must exit non-zero when at least one publication failed."""
    mock_load.return_value = _FAKE_ENTRIES

    mock_run.side_effect = [
        CrawlerError("HTTP 500", status_code=500),
        {"publication_id": "202"},
    ]

    runner = CliRunner()
    invocation = runner.invoke(cli, ["ingest-all", "--output-dir", str(tmp_path)])

    assert invocation.exit_code != 0
