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


_PUB_LINK_PATTERN = re.compile(r"/(?:publications|papers)/(\d+)")


def load_publication_index(
    session: object,
    index_url: str,
) -> List[PublicationEntry]:
    """Fetch the publication index page and return a list of PublicationEntry.

    Args:
        session: An HTTP session with a ``get(url)`` method (e.g. requests.Session).
        index_url: URL of the AMS-02 publication index page.

    Returns:
        List of PublicationEntry extracted from the page.  Empty list if no
        entries are found (with a WARNING log).
    """
    response = session.get(index_url)  # type: ignore[attr-defined]
    response.raise_for_status()  # type: ignore[attr-defined]

    soup = BeautifulSoup(response.text, "html.parser")  # type: ignore[attr-defined]

    # Resolve base URL for relative hrefs
    base_url = index_url.rstrip("/").rsplit("/publications", 1)[0]

    entries: List[PublicationEntry] = []
    seen_ids: set[str] = set()

    # Strategy 1: Parse structured AMS publication rows.
    # Each row has a title div (class ams-publication-row-title) and a
    # sibling <a> linking to /publications/XXXXXX.
    for row in soup.find_all("div", class_="ams-publication-row"):
        if not isinstance(row, Tag):
            continue

        # Extract title from the dedicated title div
        title_div = row.find("div", class_="ams-publication-row-title")
        title = title_div.get_text(strip=True) if title_div else ""

        # Find the /publications/XXXXXX link
        for link in row.find_all("a", href=True):
            href = str(link["href"])
            match = _PUB_LINK_PATTERN.search(href)
            if match:
                paper_id = match.group(1)
                if paper_id in seen_ids:
                    continue
                seen_ids.add(paper_id)

                # Build absolute URL
                if href.startswith("/"):
                    href = base_url + href

                entries.append(
                    PublicationEntry(
                        paper_id=paper_id,
                        title=title or f"Publication {paper_id}",
                        url=href,
                    )
                )
                break  # one entry per row

    # Strategy 2: Fallback — scan all <a> tags for /publications/XXXXXX
    # links if strategy 1 found nothing (e.g. different page layout).
    if not entries:
        for link in soup.find_all("a", href=True):
            if not isinstance(link, Tag):
                continue
            href = str(link["href"])
            match = _PUB_LINK_PATTERN.search(href)
            if not match:
                continue

            paper_id = match.group(1)
            if paper_id in seen_ids:
                continue
            seen_ids.add(paper_id)

            if href.startswith("/"):
                href = base_url + href

            link_text = link.get_text(strip=True)
            entries.append(
                PublicationEntry(
                    paper_id=paper_id,
                    title=link_text or f"Publication {paper_id}",
                    url=href,
                )
            )

    if not entries:
        logger.warning("No paper entries found on index page: %s", index_url)

    return entries
