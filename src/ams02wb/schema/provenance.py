"""In-memory provenance registry keyed by dataset_id.

Design choice: plain dict, no persistence, no thread safety.  This is a
build-time bookkeeping structure populated during ingestion and serialised
once at export.  Dict lookup is O(1) and serialisation via model_dump
enables direct JSON/parquet metadata embedding.

Rejected alternatives:
- SQLite / on-disk store: unnecessary for a build-time index that fits in
  memory and is never queried concurrently.
- Singleton pattern: would hide dependency and complicate testing.
"""

from __future__ import annotations

from ams02wb.schema.models import ProvenanceRecord


class ProvenanceRegistry:
    """Collects ProvenanceRecord instances keyed by dataset_id.

    Args: None — the registry starts empty.

    Raises:
        ValueError: from ``add()`` if a dataset_id is already registered.
    """

    def __init__(self) -> None:
        self._records: dict[str, ProvenanceRecord] = {}

    def add(self, dataset_id: str, record: ProvenanceRecord) -> None:
        """Register *record* under *dataset_id*.

        Raises:
            ValueError: If *dataset_id* is already present.  Silent
                overwrites would destroy provenance integrity, so
                collisions are hard errors.
        """
        if dataset_id in self._records:
            raise ValueError(
                f"Duplicate dataset_id '{dataset_id}': already registered. "
                "Remove the existing entry before re-adding, or use a "
                "distinct dataset_id."
            )
        self._records[dataset_id] = record

    def get(self, dataset_id: str) -> ProvenanceRecord | None:
        """Return the record for *dataset_id*, or ``None`` if absent."""
        return self._records.get(dataset_id)

    def list_ids(self) -> list[str]:
        """Return all registered dataset_ids."""
        return list(self._records)

    def to_dict(self) -> dict[str, dict]:
        """Serialise all entries to ``{dataset_id: dict}``."""
        return {did: rec.model_dump() for did, rec in self._records.items()}
