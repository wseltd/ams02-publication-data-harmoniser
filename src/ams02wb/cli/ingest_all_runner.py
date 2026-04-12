"""Batch ingestion orchestrator — iterate every publication in a local index and ingest."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

import requests  # type: ignore[import-untyped]

from ams02wb.crawler.index_loader import PublicationEntry
from ams02wb.cli.ingest_publication import run_single_ingest

logger = logging.getLogger(__name__)


@dataclass
class IngestSummary:
    """Result of a batch ingestion run."""

    succeeded: list[str] = field(default_factory=list)
    failed: list[dict] = field(default_factory=list)
    total: int = 0


def run_ingest_all(index_path: Path, output_dir: Path) -> IngestSummary:
    """Ingest every publication listed in a local publication_index.json.

    Args:
        index_path: Directory containing ``publication_index.json``.
        output_dir: Directory to write per-publication dataset JSON files
            and the final ``ingest-summary.json``.

    Returns:
        IngestSummary with lists of succeeded/failed publication IDs.
    """
    index_file = index_path / "publication_index.json"
    raw_entries = json.loads(index_file.read_text(encoding="utf-8"))

    entries = [
        PublicationEntry(
            paper_id=e["paper_id"],
            title=e["title"],
            url=e["url"],
        )
        for e in raw_entries
    ]

    summary = IngestSummary(total=len(entries))
    session = requests.Session()

    for entry in entries:
        try:
            run_single_ingest(session, entry, output_dir)
            summary.succeeded.append(entry.paper_id)
        except Exception as exc:
            logger.warning(
                "Ingestion failed for publication %s: %s", entry.paper_id, exc
            )
            summary.failed.append({
                "publication_id": entry.paper_id,
                "reason": str(exc),
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "succeeded": summary.succeeded,
        "failed": summary.failed,
        "total": summary.total,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    summary_file = output_dir / "ingest-summary.json"
    summary_file.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    return summary
