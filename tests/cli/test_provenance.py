"""Tests for CLI provenance builder helpers.

Focus areas:
- build_provenance_json: field completeness and value fidelity
- compute_content_hash: correctness against known vectors, empty input
- attach_provenance: injection into records, JSON validity, mutation semantics
"""

from __future__ import annotations

import hashlib
import json

from ams02wb.cli.provenance import (
    attach_provenance,
    build_provenance_json,
    compute_content_hash,
)


# ---------------------------------------------------------------------------
# build_provenance_json
# ---------------------------------------------------------------------------

_SAMPLE_ARGS = {
    "publication_id": "101",
    "publication_url": "https://ams02.space/papers/101/",
    "table_id": "table_3",
    "source_file_url": "https://ams02.space/papers/101/data.csv",
    "source_file_format": "csv",
    "parse_method": "csv_table_extraction",
    "parse_version": "1.0.0",
    "retrieval_timestamp": "2025-01-15T08:30:00Z",
    "content_hash": "abc123def456",
}


def test_build_provenance_json_returns_all_nine_fields():
    """Every provenance field must be present in the returned dict."""
    result = build_provenance_json(**_SAMPLE_ARGS)
    assert len(result) == 9
    for key in _SAMPLE_ARGS:
        assert key in result


def test_build_provenance_json_preserves_values():
    """Values must be passed through unchanged — no normalisation or mangling."""
    result = build_provenance_json(**_SAMPLE_ARGS)
    for key, expected in _SAMPLE_ARGS.items():
        assert result[key] == expected, f"Mismatch on {key}"


def test_build_provenance_json_all_values_are_strings():
    """Return type contract: every value must be a string."""
    result = build_provenance_json(**_SAMPLE_ARGS)
    for key, value in result.items():
        assert isinstance(value, str), f"{key} is {type(value).__name__}, not str"


def test_build_provenance_json_with_empty_strings():
    """Empty strings are valid — the function must not reject them."""
    args = {k: "" for k in _SAMPLE_ARGS}
    result = build_provenance_json(**args)
    assert all(v == "" for v in result.values())
    assert len(result) == 9


# ---------------------------------------------------------------------------
# compute_content_hash
# ---------------------------------------------------------------------------


def test_compute_content_hash_known_vector():
    """Verify against a known SHA-256 digest to catch algorithm drift."""
    data = b"hello world"
    expected = hashlib.sha256(data).hexdigest()
    assert compute_content_hash(data) == expected


def test_compute_content_hash_empty_bytes():
    """The empty-input hash is well-defined; must not raise or return empty."""
    result = compute_content_hash(b"")
    # SHA-256 of empty input is a known constant.
    assert result == hashlib.sha256(b"").hexdigest()
    assert len(result) == 64


def test_compute_content_hash_returns_lowercase_hex():
    """Hex digest must be lowercase, 64-char — no prefix, no encoding noise."""
    result = compute_content_hash(b"\x00\xff" * 100)
    assert result == result.lower()
    assert len(result) == 64
    # Must be valid hex.
    int(result, 16)


def test_compute_content_hash_different_inputs_differ():
    """Distinct inputs must produce distinct hashes (collision resistance)."""
    h1 = compute_content_hash(b"alpha")
    h2 = compute_content_hash(b"beta")
    assert h1 != h2


# ---------------------------------------------------------------------------
# attach_provenance
# ---------------------------------------------------------------------------


def test_attach_provenance_injects_provenance_json_key():
    """Each record must gain a 'provenance_json' field."""
    records = [{"x": 1}, {"x": 2}]
    prov = build_provenance_json(**_SAMPLE_ARGS)
    result = attach_provenance(records, prov)

    for rec in result:
        assert "provenance_json" in rec


def test_attach_provenance_json_is_valid():
    """The provenance_json value must be valid JSON that round-trips to the original dict."""
    prov = build_provenance_json(**_SAMPLE_ARGS)
    records = [{"val": 42}]
    attach_provenance(records, prov)

    decoded = json.loads(records[0]["provenance_json"])
    assert decoded == prov


def test_attach_provenance_returns_same_list():
    """Return value must be the same list object — no unnecessary copies."""
    records = [{"a": 1}]
    prov = {"k": "v"}
    result = attach_provenance(records, prov)
    assert result is records


def test_attach_provenance_empty_records():
    """An empty record list must not raise; return the same empty list."""
    records: list[dict] = []
    result = attach_provenance(records, {"k": "v"})
    assert result == []
    assert result is records


def test_attach_provenance_uniform_value_across_records():
    """All records must receive the identical provenance_json string."""
    records = [{"i": i} for i in range(5)]
    prov = build_provenance_json(**_SAMPLE_ARGS)
    attach_provenance(records, prov)

    values = [r["provenance_json"] for r in records]
    assert len(set(values)) == 1, "All records should share the same provenance string"


def test_attach_provenance_does_not_remove_existing_fields():
    """Existing record fields must survive injection — attach is additive."""
    records = [{"energy": 10.5, "flux": 0.3}]
    attach_provenance(records, {"source": "test"})

    assert records[0]["energy"] == 10.5
    assert records[0]["flux"] == 0.3
    assert "provenance_json" in records[0]
