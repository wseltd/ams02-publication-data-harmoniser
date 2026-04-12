"""Tests for ams02wb.crawler.page_fetcher."""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock

from ams02wb.crawler import CrawlerError
from ams02wb.crawler.page_fetcher import FetchedPage, fetch_publication_page


# --- helpers ---


def _mock_session(
    html: str,
    final_url: str,
    status_code: int = 200,
) -> MagicMock:
    """Create a mock HTTP session returning a response with the given attributes."""
    response = MagicMock()
    response.text = html
    response.url = final_url
    response.status_code = status_code
    response.raise_for_status = MagicMock()
    session = MagicMock()
    session.get.return_value = response
    return session


def _mock_error_session(status_code: int, page_url: str) -> MagicMock:
    """Create a mock session whose response raises on raise_for_status."""
    response = MagicMock()
    response.status_code = status_code
    response.url = page_url
    response.raise_for_status.side_effect = Exception(
        f"HTTP {status_code}"
    )
    session = MagicMock()
    session.get.return_value = response
    return session


# --- tests ---


def test_fetch_returns_html_and_base_url() -> None:
    """Fetching a page returns a FetchedPage with html body and base_url."""
    html = "<html><body>AMS-02 proton flux</body></html>"
    url = "https://ams02.example.org/papers/101/"
    session = _mock_session(html=html, final_url=url)

    result = fetch_publication_page(session, url)

    assert isinstance(result, FetchedPage)
    assert result.html == html
    assert result.base_url == url
    # FetchedPage has exactly two fields
    fields = {f.name: f.type for f in dataclasses.fields(FetchedPage)}
    assert fields == {"html": "str", "base_url": "str"}


def test_fetch_resolves_base_url_after_redirect() -> None:
    """base_url must reflect the final URL after redirect, not the original request URL."""
    original_url = "https://ams02.example.org/old/101/"
    redirected_url = "https://ams02.example.org/papers/101/report.html"
    session = _mock_session(
        html="<html>redirected</html>",
        final_url=redirected_url,
    )

    result = fetch_publication_page(session, original_url)

    # base_url comes from response.url (post-redirect), not the request URL
    assert result.base_url == redirected_url
    assert result.base_url != original_url
    session.get.assert_called_once_with(original_url)


def test_fetch_raises_crawler_error_on_404() -> None:
    """A 404 response raises CrawlerError with the status code."""
    url = "https://ams02.example.org/papers/999/"
    session = _mock_error_session(status_code=404, page_url=url)

    try:
        fetch_publication_page(session, url)
        assert False, "Expected CrawlerError to be raised"
    except CrawlerError as exc:
        assert exc.status_code == 404
        assert "404" in str(exc)
        assert url in str(exc)


def test_fetch_raises_crawler_error_on_500() -> None:
    """A 500 response raises CrawlerError with the status code."""
    url = "https://ams02.example.org/papers/101/"
    session = _mock_error_session(status_code=500, page_url=url)

    try:
        fetch_publication_page(session, url)
        assert False, "Expected CrawlerError to be raised"
    except CrawlerError as exc:
        assert exc.status_code == 500
        assert "500" in str(exc)
        assert url in str(exc)
