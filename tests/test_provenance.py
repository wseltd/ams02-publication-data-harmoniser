"""Tests for ProvenanceRegistry — focused on duplicate-key detection and
serialisation correctness, which are the highest-risk surfaces.

Weight: 3 basic correctness (add/get/missing-key), 3 higher-risk
(serialisation round-trip, duplicate-key, multi-record integrity).
"""

import pytest

from ams02wb.schema.models import ProvenanceRecord
from ams02wb.schema.provenance import ProvenanceRegistry

# Shared defaults — avoids repeating field values across tests while keeping
# each test self-contained (fresh registry per test, no shared state).
_DEFAULTS = {
    "paper_doi": "10.1103/PhysRevLett.122.041102",
    "table_id": "table_1",
    "file_url": "https://example.com/data.csv",
    "ingested_at": "2024-06-15T10:00:00",
    "source_type": "csv",
}


def _make_record(**overrides):
    """Build a ProvenanceRecord with sensible defaults, overridable per-field."""
    fields = {**_DEFAULTS, **overrides}
    return ProvenanceRecord(**fields)


# --- Basic correctness (3 tests) ---


def test_add_and_get_record():
    """Adding a record and retrieving it by dataset_id returns the same object."""
    reg = ProvenanceRegistry()
    rec = _make_record()
    reg.add("ds1", rec)

    retrieved = reg.get("ds1")
    assert retrieved is rec
    assert retrieved.paper_doi == "10.1103/PhysRevLett.122.041102"
    assert retrieved.table_id == "table_1"
    assert retrieved.file_url == "https://example.com/data.csv"


def test_get_missing_key_raises():
    """Querying an absent dataset_id returns None — a distinct sentinel,
    not an empty record or a default-constructed ProvenanceRecord."""
    reg = ProvenanceRegistry()
    # Add one record so the registry is not empty
    reg.add("present", _make_record())

    result = reg.get("nonexistent")

    # Value assertion: must be exactly None
    assert result is None
    # Verify the absent key is not listed among registered ids
    assert "nonexistent" not in reg.list_ids()
    # Verify the present key is still accessible (registry not corrupted)
    assert reg.get("present").paper_doi == "10.1103/PhysRevLett.122.041102"


def test_add_duplicate_key_raises():
    """Adding a second record with the same dataset_id must raise ValueError.

    Silent overwrites would destroy provenance integrity — this is the
    single most important safety invariant of the registry.
    """
    reg = ProvenanceRegistry()
    reg.add("ds1", _make_record())

    with pytest.raises(ValueError, match="ds1"):
        reg.add("ds1", _make_record(paper_doi="10.9999/other"))

    # Original record must survive the failed insert
    assert reg.get("ds1").paper_doi == "10.1103/PhysRevLett.122.041102"


# --- Higher-risk tests (3 tests) ---


def test_serialisation_round_trip():
    """to_dict must produce a plain dict with field-level equality on all
    5 provenance fields.  This is the highest-risk surface: silent
    corruption here means provenance data loss at export.
    """
    reg = ProvenanceRegistry()
    rec = _make_record(
        paper_doi="10.1/roundtrip",
        table_id="t99",
        file_url="https://example.com/rt.csv",
        ingested_at="2025-01-01T00:00:00",
        source_type="pdf",
    )
    reg.add("rt", rec)

    serialised = reg.to_dict()

    # Must be a plain dict, not a model instance
    entry = serialised["rt"]
    assert not isinstance(entry, ProvenanceRecord)

    # Field-level equality on every provenance field
    assert entry["paper_doi"] == "10.1/roundtrip"
    assert entry["table_id"] == "t99"
    assert entry["file_url"] == "https://example.com/rt.csv"
    assert entry["ingested_at"] == "2025-01-01T00:00:00"
    assert entry["source_type"] == "pdf"

    # Reconstruct from serialised dict and verify equivalence
    reconstructed = ProvenanceRecord(**entry)
    assert reconstructed.paper_doi == rec.paper_doi
    assert reconstructed.table_id == rec.table_id
    assert reconstructed.file_url == rec.file_url
    assert reconstructed.ingested_at == rec.ingested_at
    assert reconstructed.source_type == rec.source_type


def test_serialisation_preserves_timestamp():
    """ingested_at is a string timestamp that must survive serialisation
    without truncation, reformatting, or type coercion.

    This is the focus test: datetime-like strings are a common corruption
    vector when serialisation layers try to be clever about parsing.
    """
    # Deliberately use an ISO-8601 timestamp with sub-second precision
    # to catch truncation or reformatting.
    timestamp = "2024-12-31T23:59:59.123456"

    reg = ProvenanceRegistry()
    rec = _make_record(ingested_at=timestamp)
    reg.add("ts", rec)

    serialised = reg.to_dict()
    assert serialised["ts"]["ingested_at"] == timestamp

    # Round-trip through reconstruction
    reconstructed = ProvenanceRecord(**serialised["ts"])
    assert reconstructed.ingested_at == timestamp


def test_multi_record_integrity():
    """Multiple records with distinct dataset_ids must all survive
    serialisation without cross-contamination or data loss."""
    reg = ProvenanceRegistry()

    records = {
        "alpha": _make_record(paper_doi="10.1/alpha", table_id="t1"),
        "beta": _make_record(paper_doi="10.1/beta", table_id="t2"),
        "gamma": _make_record(paper_doi="10.1/gamma", table_id="t3"),
    }
    for did, rec in records.items():
        reg.add(did, rec)

    serialised = reg.to_dict()

    # All three present with correct field values
    assert len(serialised) == 3
    for did, rec in records.items():
        assert did in serialised
        assert serialised[did]["paper_doi"] == rec.paper_doi
        assert serialised[did]["table_id"] == rec.table_id

    # No cross-contamination: each entry has its own distinct doi
    dois = [serialised[did]["paper_doi"] for did in serialised]
    assert len(set(dois)) == 3
