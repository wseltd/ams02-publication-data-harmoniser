"""Resolve downloadable CSV and PDF attachments from an AMS-02 publication page."""

from __future__ import annotations

import dataclasses
import posixpath
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

# Extensions we recognise as downloadable attachments.
_SUPPORTED_EXTENSIONS = {".csv": "csv", ".pdf": "pdf"}


@dataclasses.dataclass
class Attachment:
    """A downloadable file linked from a publication page.

    Attributes:
        url: Fully-resolved URL to the file.
        filename: Basename extracted from the URL path (query params stripped).
        content_type: Either ``'csv'`` or ``'pdf'``, inferred from extension.
    """

    url: str
    filename: str
    content_type: str


def resolve_attachments(html: str, base_url: str) -> list[Attachment]:
    """Extract CSV and PDF attachment links from *html*.

    Args:
        html: Raw HTML of the publication page.
        base_url: Base URL used to resolve relative ``href`` values
                  via :func:`urllib.parse.urljoin`.

    Returns:
        List of :class:`Attachment` objects for every ``<a>`` whose ``href``
        ends in ``.csv`` or ``.pdf`` (case-insensitive).  Links without a
        recognised extension are silently ignored.
    """
    soup = BeautifulSoup(html, "html.parser")
    attachments: list[Attachment] = []

    for anchor in soup.find_all("a", href=True):
        href: str = anchor["href"]
        absolute_url = urljoin(base_url, href)

        # Strip query string / fragment to get the true filename.
        path = urlparse(absolute_url).path
        filename = posixpath.basename(path)

        _, ext = posixpath.splitext(filename)
        content_type = _SUPPORTED_EXTENSIONS.get(ext.lower())
        if content_type is None:
            continue

        attachments.append(
            Attachment(url=absolute_url, filename=filename, content_type=content_type)
        )

    return attachments
