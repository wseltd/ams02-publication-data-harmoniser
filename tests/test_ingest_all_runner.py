"""Tests for the batch ingestion orchestrator (run_ingest_all)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from ams02wb.cli.ingest_all_runner import run_ingest_all, IngestSummary
from ams02wb.crawler import CrawlerError
from ams02wb.crawler.index_loader import PublicationEntry
from ams02wb.crawler.page_fetcher import FetchedPage


_FAKE_ENTRIES = [
    {"paper_id": "101", "title": "Proton Flux", "url": "https://ams02.space/papers/101/"},
    {"paper_id": "202", "title": "Helium Flux", "url": "https://ams02.space/papers/202/"},
    {"paper_id": "303", "title": "Electron Flux", "url": "https://ams02.space/papers/303/"},
]

_FAKE_PAGE = FetchedPage(
    html='<html><body><a href="data.csv">CSV</a></body></html>',
    base_url="https://ams02.space/papers/101/",
)

_FAKE_CSV_BYTES = b"energy_low,energy_high,value\n1.0,2.0,100.5\n"


def _write_index(tmp_path: Path, entries: list[dict] | None = None) -> Path:
    """Write a publication_index.json into tmp_path and return the directory."""
    index_file = tmp_path / "publication_index.json"
    payload = _FAKE_ENTRIES if entries is None else entries
    index_file.write_text(json.dumps(payload), encoding="utf-8")
    return tmp_path


def _mock_session() -> MagicMock:
    session = MagicMock()
    resp = MagicMock()
    resp.content = _FAKE_CSV_BYTES
    resp.raise_for_status = MagicMock()
    session.get.return_value = resp
    return session


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@patch("ams02wb.cli.ingest_all_runner.requests.Session")
@patch("ams02wb.cli.ingest_all_runner.run_single_ingest")
def test_run_ingest_all_calls_single_ingest_for_each_entry(
    mock_run: MagicMock, mock_session_cls: MagicMock, tmp_path: Path
) -> None:
    """run_ingest_all must invoke run_single_ingest once per index entry."""
    index_dir = _write_index(tmp_path)
    out_dir = tmp_path / "output"
    out_dir.mkdir()

    mock_run.return_value = {}

    summary = run_ingest_all(index_dir, out_dir)

    assert mock_run.call_count == len(_FAKE_ENTRIES)
    assert summary.total == 3
    assert len(summary.succeeded) == 3
    assert summary.succeeded == ["101", "202", "303"]


@patch("ams02wb.cli.ingest_all_runner.requests.Session")
@patch("ams02wb.cli.ingest_all_runner.run_single_ingest")
def test_run_ingest_all_writes_summary_json(
    mock_run: MagicMock, mock_session_cls: MagicMock, tmp_path: Path
) -> None:
    """run_ingest_all must write ingest-summary.json to output_dir."""
    index_dir = _write_index(tmp_path)
    out_dir = tmp_path / "output"
    out_dir.mkdir()

    mock_run.return_value = {}

    run_ingest_all(index_dir, out_dir)

    summary_file = out_dir / "ingest-summary.json"
    assert summary_file.exists()

    data = json.loads(summary_file.read_text())
    assert data["total"] == 3
    assert data["succeeded"] == ["101", "202", "303"]
    assert data["failed"] == []
    assert "completed_at" in data


@patch("ams02wb.cli.ingest_all_runner.requests.Session")
@patch("ams02wb.cli.ingest_all_runner.run_single_ingest")
def test_run_ingest_all_summary_completed_at_is_iso8601(
    mock_run: MagicMock, mock_session_cls: MagicMock, tmp_path: Path
) -> None:
    """completed_at in the summary must be a valid ISO-8601 timestamp."""
    index_dir = _write_index(tmp_path)
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    mock_run.return_value = {}

    run_ingest_all(index_dir, out_dir)

    data = json.loads((out_dir / "ingest-summary.json").read_text())
    # Will raise ValueError if not valid ISO format.
    from datetime import datetime
    parsed = datetime.fromisoformat(data["completed_at"])
    assert parsed is not None


# ---------------------------------------------------------------------------
# Failure handling — the core risk surface
# ---------------------------------------------------------------------------


@patch("ams02wb.cli.ingest_all_runner.requests.Session")
@patch("ams02wb.cli.ingest_all_runner.run_single_ingest")
def test_single_failure_does_not_abort_remaining_entries(
    mock_run: MagicMock, mock_session_cls: MagicMock, tmp_path: Path
) -> None:
    """If one publication fails, the rest must still be attempted."""
    index_dir = _write_index(tmp_path)
    out_dir = tmp_path / "output"
    out_dir.mkdir()

    mock_run.side_effect = [
        CrawlerError("HTTP 500", status_code=500),
        {},
        {},
    ]

    summary = run_ingest_all(index_dir, out_dir)

    assert mock_run.call_count == 3
    assert len(summary.succeeded) == 2
    assert len(summary.failed) == 1


@patch("ams02wb.cli.ingest_all_runner.requests.Session")
@patch("ams02wb.cli.ingest_all_runner.run_single_ingest")
def test_failed_list_contains_publication_id_and_reason(
    mock_run: MagicMock, mock_session_cls: MagicMock, tmp_path: Path
) -> None:
    """Each failure record must include the publication_id and a reason string."""
    index_dir = _write_index(tmp_path)
    out_dir = tmp_path / "output"
    out_dir.mkdir()

    mock_run.side_effect = [
        CrawlerError("HTTP 500", status_code=500),
        {},
        RuntimeError("parse error"),
    ]

    summary = run_ingest_all(index_dir, out_dir)

    assert len(summary.failed) == 2
    failed_ids = {f["publication_id"] for f in summary.failed}
    assert failed_ids == {"101", "303"}

    # Reason strings must contain the original exception messages.
    reasons = {f["publication_id"]: f["reason"] for f in summary.failed}
    assert "500" in reasons["101"]
    assert "parse error" in reasons["303"]


@patch("ams02wb.cli.ingest_all_runner.requests.Session")
@patch("ams02wb.cli.ingest_all_runner.run_single_ingest")
def test_all_failures_still_writes_summary(
    mock_run: MagicMock, mock_session_cls: MagicMock, tmp_path: Path
) -> None:
    """Even if every entry fails, ingest-summary.json must still be written."""
    index_dir = _write_index(tmp_path)
    out_dir = tmp_path / "output"
    out_dir.mkdir()

    mock_run.side_effect = RuntimeError("boom")

    summary = run_ingest_all(index_dir, out_dir)

    assert summary.total == 3
    assert len(summary.succeeded) == 0
    assert len(summary.failed) == 3

    data = json.loads((out_dir / "ingest-summary.json").read_text())
    assert data["total"] == 3
    assert len(data["failed"]) == 3


@patch("ams02wb.cli.ingest_all_runner.requests.Session")
@patch("ams02wb.cli.ingest_all_runner.run_single_ingest")
def test_summary_json_failed_entries_match_return_value(
    mock_run: MagicMock, mock_session_cls: MagicMock, tmp_path: Path
) -> None:
    """The on-disk summary must match the returned IngestSummary."""
    index_dir = _write_index(tmp_path)
    out_dir = tmp_path / "output"
    out_dir.mkdir()

    mock_run.side_effect = [
        {},
        CrawlerError("timeout", status_code=None),
        {},
    ]

    summary = run_ingest_all(index_dir, out_dir)

    data = json.loads((out_dir / "ingest-summary.json").read_text())
    assert data["succeeded"] == summary.succeeded
    assert data["failed"] == summary.failed
    assert data["total"] == summary.total


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@patch("ams02wb.cli.ingest_all_runner.requests.Session")
@patch("ams02wb.cli.ingest_all_runner.run_single_ingest")
def test_empty_index_produces_empty_summary(
    mock_run: MagicMock, mock_session_cls: MagicMock, tmp_path: Path
) -> None:
    """An empty publication index must produce a valid summary with total=0."""
    index_dir = _write_index(tmp_path, entries=[])
    out_dir = tmp_path / "output"
    out_dir.mkdir()

    summary = run_ingest_all(index_dir, out_dir)

    assert summary.total == 0
    assert summary.succeeded == []
    assert summary.failed == []
    assert mock_run.call_count == 0

    data = json.loads((out_dir / "ingest-summary.json").read_text())
    assert data["total"] == 0


def test_missing_index_file_raises_file_not_found(tmp_path: Path) -> None:
    """run_ingest_all must fail clearly if publication_index.json does not exist."""
    out_dir = tmp_path / "output"
    out_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        run_ingest_all(tmp_path, out_dir)
