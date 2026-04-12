"""CLI command: index-publications — fetch and save the AMS-02 publication index."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import click
import requests  # type: ignore[import-untyped]

from ams02wb.crawler import CrawlerError, INDEX_URL
from ams02wb.crawler.index_loader import load_publication_index

logger = logging.getLogger(__name__)


@click.command("index-publications")
@click.option(
    "--output-dir",
    default="./ams02wb-data/",
    type=click.Path(),
    help="Directory to write publication_index.json into.",
)
def index_publications(output_dir: str) -> None:
    """Fetch the AMS-02 publication index and write it to JSON."""
    try:
        session = requests.Session()
        entries = load_publication_index(session, INDEX_URL)

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        index_file = out_path / "publication_index.json"
        payload = [asdict(e) for e in entries]
        index_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        logger.info(
            "Wrote %d publications to %s", len(entries), index_file
        )
    except CrawlerError as exc:
        logger.error("Failed to fetch publication index: %s", exc)
        sys.exit(1)
