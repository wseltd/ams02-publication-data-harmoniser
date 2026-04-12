"""Provenance builder helpers for CLI ingestion commands.

Pure functions that construct, hash, and attach provenance metadata to
ingested records.  No side effects — all state flows through arguments
and return values.
"""

from __future__ import annotations

import hashlib
import json


def build_provenance_json(
    publication_id: str,
    publication_url: str,
    table_id: str,
    source_file_url: str,
    source_file_format: str,
    parse_method: str,
    parse_version: str,
    retrieval_timestamp: str,
    content_hash: str,
) -> dict[str, str]:
    """Return a dict containing all nine provenance fields.

    Args:
        publication_id: Identifier for the parent publication.
        publication_url: Canonical URL of the publication page.
        table_id: Identifier for the specific table within the publication.
        source_file_url: URL of the downloaded source file.
        source_file_format: Format of the source file (e.g. 'csv', 'pdf').
        parse_method: Name of the parser used to extract data.
        parse_version: Version string of the parser.
        retrieval_timestamp: ISO-8601 timestamp of when data was fetched.
        content_hash: Pre-computed hash of the raw source bytes.

    Returns:
        Dict with exactly nine string-valued provenance fields.
    """
    return {
        "publication_id": publication_id,
        "publication_url": publication_url,
        "table_id": table_id,
        "source_file_url": source_file_url,
        "source_file_format": source_file_format,
        "parse_method": parse_method,
        "parse_version": parse_version,
        "retrieval_timestamp": retrieval_timestamp,
        "content_hash": content_hash,
    }


def compute_content_hash(data: bytes) -> str:
    """Return the SHA-256 hex digest of *data*.

    Args:
        data: Raw bytes to hash.

    Returns:
        Lowercase hex string (64 characters for SHA-256).
    """
    return hashlib.sha256(data).hexdigest()


def attach_provenance(records: list[dict], provenance: dict) -> list[dict]:
    """Inject a JSON-serialised provenance field into every record.

    Each record gains a ``provenance_json`` key whose value is the
    JSON-serialised *provenance* dict.  Records are modified in-place
    **and** returned for convenience.

    Args:
        records: List of dicts representing data rows.
        provenance: Provenance dict (typically from :func:`build_provenance_json`).

    Returns:
        The same *records* list, with ``provenance_json`` added to each entry.
    """
    serialised = json.dumps(provenance, sort_keys=True)
    for record in records:
        record["provenance_json"] = serialised
    return records
