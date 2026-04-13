"""Load the AMS-02 publication index page and extract publication entries."""

from __future__ import annotations

import dataclasses
import logging
import re
from typing import List

from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class PublicationEntry:
    """A single publication from the AMS-02 index page."""

    paper_id: str
    title: str
    url: str


# Extract numeric paper_id from URL paths like /papers/123/, /papers/123/report.html
_PAPER_ID_PATTERN = re.compile(r"/(\d+)(?:/[^/]*)?$")


def _extract_paper_id(url: str) -> str | None:
    """Extract a numeric paper ID from the last path segment of a URL.

    Returns None if no numeric ID is found.
    """
    match = _PAPER_ID_PATTERN.search(url)
    if match:
        return match.group(1)
    return None


def load_publication_index(
    session: object,
    index_url: str,
) -> List[PublicationEntry]:
    """Fetch the publication index page and return a list of PublicationEntry.

    Args:
        session: An HTTP session with a `get(url)` method (e.g. requests.Session).
        index_url: URL of the AMS-02 publication index page.

    Returns:
        List of PublicationEntry extracted from the page. Empty list if no
        entries are found (with a WARNING log).
    """
    response = session.get(index_url)  # type: ignore[attr-defined]
    response.raise_for_status()  # type: ignore[attr-defined]

    soup = BeautifulSoup(response.text, "html.parser")  # type: ignore[attr-defined]

    entries: List[PublicationEntry] = []

    for link in soup.find_all("a", href=True):
        if not isinstance(link, Tag):
            continue

        href = str(link["href"])
        title_text = link.get_text(strip=True)

        if not title_text or not href:
            continue

        paper_id = _extract_paper_id(href)
        if paper_id is None:
            logger.debug("Skipping link with no extractable paper_id: %s", href)
            continue

        entries.append(
            PublicationEntry(paper_id=paper_id, title=title_text, url=href)
        )

    if not entries:
        logger.warning(
            "No paper entries found on index page: %s", index_url
        )

    return entries
