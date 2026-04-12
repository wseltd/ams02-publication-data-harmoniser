"""CLI command: ingest-publication — fetch, parse, and store a single publication."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import click
import requests  # type: ignore[import-untyped]

from ams02wb.crawler import CrawlerError, INDEX_URL
from ams02wb.crawler.attachment_resolver import resolve_attachments
from ams02wb.crawler.index_loader import PublicationEntry, load_publication_index
from ams02wb.crawler.page_fetcher import fetch_publication_page

logger = logging.getLogger(__name__)


def run_single_ingest(
    session: requests.Session,
    entry: PublicationEntry,
    output_dir: Path,
) -> dict:
    """Ingest one publication: fetch page, resolve attachments, download, write JSON.

    Args:
        session: HTTP session for all requests.
        entry: The publication entry from the index.
        output_dir: Directory to write the dataset JSON into.

    Returns:
        The assembled dataset dict.

    Raises:
        CrawlerError: On network or parse failures from the crawler layer.
        requests.RequestException: On attachment download failures.
    """
    page = fetch_publication_page(session, entry.url)

    attachments = resolve_attachments(page.html, page.base_url)

    attachment_records: list[dict[str, str]] = []
    content_parts: list[bytes] = []

    for att in attachments:
        resp = session.get(att.url)
        resp.raise_for_status()

        raw = resp.content
        content_parts.append(raw)
        attachment_records.append({
            "filename": att.filename,
            "url": att.url,
            "content_type": att.content_type,
            "size_bytes": str(len(raw)),
        })

    hasher = hashlib.sha256()
    for part in content_parts:
        hasher.update(part)
    # Fall back to page HTML when no attachments were downloaded.
    if not content_parts:
        hasher.update(page.html.encode("utf-8"))

    dataset = {
        "publication_id": entry.paper_id,
        "title": entry.title,
        "source_url": entry.url,
        "attachments": attachment_records,
        "provenance": {
            "source_url": entry.url,
            "content_hash": f"sha256:{hasher.hexdigest()}",
            "parse_method": "csv_table_extraction",
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / f"{entry.paper_id}.json"
    dest.write_text(json.dumps(dataset, indent=2), encoding="utf-8")

    logger.info("Ingested publication %s -> %s", entry.paper_id, dest)
    return dataset


@click.command("ingest-publication")
@click.option(
    "--publication-id",
    required=True,
    help="Numeric paper ID to ingest (e.g. '123').",
)
@click.option(
    "--output-dir",
    default="./ams02wb-data/",
    type=click.Path(),
    help="Directory to write ingested dataset JSON into.",
)
def ingest_publication(publication_id: str, output_dir: str) -> None:
    """Ingest a single AMS-02 publication by ID."""
    session = requests.Session()

    try:
        entries = load_publication_index(session, INDEX_URL)
    except CrawlerError as exc:
        logger.error("Network error fetching index: %s", exc)
        sys.exit(1)

    matching = [e for e in entries if e.paper_id == publication_id]
    if not matching:
        logger.error(
            "Publication ID %r not found in index (%d entries checked)",
            publication_id,
            len(entries),
        )
        sys.exit(1)

    entry = matching[0]

    try:
        run_single_ingest(session, entry, Path(output_dir))
    except (CrawlerError, requests.RequestException) as exc:
        logger.error("Ingestion failed for %s: %s", publication_id, exc)
        sys.exit(1)
