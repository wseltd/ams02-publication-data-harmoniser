"""Tests for ams02wb.crawler.index_loader."""

from __future__ import annotations

import dataclasses
import logging
from unittest.mock import MagicMock

import pytest

from ams02wb.crawler.index_loader import PublicationEntry, load_publication_index

# --- fixtures ---

_INDEX_HTML = """\
<html><body>
<a href="/papers/101/report.html">First measurement of X</a>
<a href="/papers/202/">Cosmic ray helium flux</a>
</body></html>
"""

_EMPTY_HTML = "<html><body><p>No publications yet.</p></body></html>"

_MALFORMED_HTML = """\
<html><body>
<a href="/papers/301/">Valid entry</a>
<a href="">Empty href</a>
<a href="/no-id-here/">Missing numeric ID</a>
<a href="/papers/302/">Another valid</a>
</body></html>
"""


def _mock_session(html: str) -> MagicMock:
    """Create a mock HTTP session that returns the given HTML."""
    response = MagicMock()
    response.text = html
    response.raise_for_status = MagicMock()
    session = MagicMock()
    session.get.return_value = response
    return session


# --- tests ---


def test_load_index_returns_publication_entry_list() -> None:
    """Loading a page with two valid links returns two PublicationEntry items."""
    session = _mock_session(_INDEX_HTML)
    result = load_publication_index(session, "https://ams02.example.org/pubs")

    assert len(result) == 2
    assert all(isinstance(entry, PublicationEntry) for entry in result)
    assert result[0].title == "First measurement of X"
    assert result[1].title == "Cosmic ray helium flux"
    session.get.assert_called_once_with("https://ams02.example.org/pubs")


def test_load_index_extracts_paper_id_from_url_path() -> None:
    """Paper IDs are extracted from the numeric segment in the URL path."""
    session = _mock_session(_INDEX_HTML)
    result = load_publication_index(session, "https://example.org/index")

    assert result[0].paper_id == "101"
    assert "/papers/101/report.html" in result[0].url
    assert result[1].paper_id == "202"
    assert "/papers/202/" in result[1].url


def test_load_index_handles_empty_page_returns_empty_list(caplog: pytest.LogCaptureFixture) -> None:
    """A page with no paper links returns an empty list and logs a warning."""
    session = _mock_session(_EMPTY_HTML)

    with caplog.at_level(logging.WARNING):
        result = load_publication_index(session, "https://example.org/empty")

    assert result == []
    assert any("No paper entries found" in msg for msg in caplog.messages)


def test_load_index_handles_malformed_entry_skips_gracefully() -> None:
    """Malformed entries (empty href, no numeric ID) are skipped; valid ones kept."""
    session = _mock_session(_MALFORMED_HTML)
    result = load_publication_index(session, "https://example.org/mixed")

    assert len(result) == 2
    assert result[0].paper_id == "301"
    assert result[1].paper_id == "302"


def test_publication_entry_fields_are_typed() -> None:
    """PublicationEntry is a frozen dataclass with exactly three str fields."""
    fields = {f.name: f.type for f in dataclasses.fields(PublicationEntry)}
    assert fields == {"paper_id": "str", "title": "str", "url": "str"}

    # frozen — assignment raises
    entry = PublicationEntry(paper_id="1", title="t", url="u")
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.paper_id = "2"  # type: ignore[misc]
