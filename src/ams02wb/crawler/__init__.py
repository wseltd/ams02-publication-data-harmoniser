"""Crawler package for fetching AMS-02 publication data."""

from __future__ import annotations

# Canonical URL for the AMS-02 publication index page.
INDEX_URL = "https://ams02.space/publications"


class CrawlerError(Exception):
    """Raised when a crawler HTTP request fails (4xx/5xx)."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code

    def __repr__(self) -> str:
        return f"CrawlerError({self.args[0]!r}, status_code={self.status_code!r})"
