"""CLI command: ingest-all — batch-ingest every publication in the index."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import requests  # type: ignore[import-untyped]

from ams02wb.crawler import CrawlerError, INDEX_URL
from ams02wb.crawler.index_loader import load_publication_index
from ams02wb.cli.ingest_publication import run_single_ingest

logger = logging.getLogger(__name__)


@click.command("ingest-all")
@click.option(
    "--output-dir",
    default="./ams02wb-data/",
    type=click.Path(),
    help="Directory to write ingested dataset JSON files into.",
)
def ingest_all(output_dir: str) -> None:
    """Ingest all AMS-02 publications listed in the index.

    Each publication is ingested independently — a failure for one paper
    does not abort the batch.  A summary of failures is printed at the end
    and the command exits non-zero if any publication failed.
    """
    session = requests.Session()

    try:
        entries = load_publication_index(session, INDEX_URL)
    except CrawlerError as exc:
        logger.error("Failed to load publication index: %s", exc)
        sys.exit(1)

    out_path = Path(output_dir)
    failed_ids: list[str] = []

    for entry in entries:
        try:
            run_single_ingest(session, entry, out_path)
        except Exception as exc:
            logger.warning(
                "Ingestion failed for publication %s: %s", entry.paper_id, exc
            )
            failed_ids.append(entry.paper_id)

    if failed_ids:
        click.echo(f"Failed publications: {', '.join(failed_ids)}", err=True)
        sys.exit(1)

    click.echo(f"Ingested {len(entries)} publications successfully.")
