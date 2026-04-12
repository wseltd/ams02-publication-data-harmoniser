"""Fetch a single AMS-02 publication page and return its HTML with resolved base URL."""

from __future__ import annotations

import dataclasses
import logging

from ams02wb.crawler import CrawlerError

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class FetchedPage:
    """Raw result of fetching a publication page.

    Attributes:
        html: The response body as a string.
        base_url: The final resolved URL after any redirects. Used by downstream
            modules to resolve relative attachment links.
    """

    html: str
    base_url: str


def fetch_publication_page(session: object, page_url: str) -> FetchedPage:
    """Fetch a publication page and return its HTML with the final resolved URL.

    Args:
        session: An HTTP session with a ``get(url)`` method (e.g. requests.Session).
        page_url: URL of the publication page to fetch.

    Returns:
        FetchedPage with the response body and the final URL after redirects.

    Raises:
        CrawlerError: On HTTP 4xx/5xx responses.
    """
    response = session.get(page_url)  # type: ignore[attr-defined]

    try:
        response.raise_for_status()  # type: ignore[attr-defined]
    except Exception as exc:
        status = getattr(response, "status_code", None)  # type: ignore[attr-defined]
        raise CrawlerError(
            f"HTTP {status} fetching {page_url}",
            status_code=status,
        ) from exc

    # response.url is the final URL after any redirects — critical for
    # resolving relative attachment hrefs in downstream modules (T011).
    final_url = str(response.url)  # type: ignore[attr-defined]

    return FetchedPage(html=response.text, base_url=final_url)  # type: ignore[attr-defined]
