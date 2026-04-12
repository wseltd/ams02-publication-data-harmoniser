"""Tests for ProvenanceRegistry — focused on duplicate-key detection and
serialisation correctness, which are the highest-risk surfaces."""

import pytest

from ams02wb.schema.models import ProvenanceRecord
from ams02wb.schema.provenance import ProvenanceRegistry


def _make_record(**overrides):
    """Build a ProvenanceRecord with sensible defaults, overridable per-field."""
    defaults = {
        "paper_doi": "10.1103/PhysRevLett.122.041102",
        "table_id": "table_1",
        "file_url": "https://example.com/data.csv",
        "ingested_at": "2024-06-15T10:00:00",
        "source_type": "csv",
    }
    defaults.update(overrides)
    return ProvenanceRecord(**defaults)


class TestAdd:
    def test_add_and_get_returns_record(self):
        reg = ProvenanceRegistry()
        rec = _make_record()
        reg.add("ds1", rec)

        assert reg.get("ds1") is rec

    def test_add_duplicate_raises_valueerror(self):
        """Duplicate-key detection is the critical safety invariant."""
        reg = ProvenanceRegistry()
        reg.add("ds1", _make_record())

        with pytest.raises(ValueError, match="ds1"):
            reg.add("ds1", _make_record(paper_doi="10.9999/other"))

    def test_add_multiple_distinct_ids(self):
        reg = ProvenanceRegistry()
        rec_a = _make_record(paper_doi="10.1/a")
        rec_b = _make_record(paper_doi="10.1/b")
        rec_c = _make_record(paper_doi="10.1/c")
        reg.add("a", rec_a)
        reg.add("b", rec_b)
        reg.add("c", rec_c)

        assert reg.get("a") is rec_a
        assert reg.get("b") is rec_b
        assert reg.get("c") is rec_c


class TestGet:
    def test_get_missing_returns_none(self):
        """Querying an absent dataset_id must return None, not raise."""
        reg = ProvenanceRegistry()
        result = reg.get("nonexistent")

        # Value assertion: result must be exactly None, not just falsy
        assert result is None
        assert result == None  # noqa: E711 — explicit value equality check


class TestListIds:
    def test_list_ids_returns_all_keys(self):
        reg = ProvenanceRegistry()
        reg.add("x", _make_record())
        reg.add("y", _make_record())
        reg.add("z", _make_record())

        ids = reg.list_ids()
        assert sorted(ids) == ["x", "y", "z"]


class TestToDict:
    def test_to_dict_serialises_all_entries(self):
        reg = ProvenanceRegistry()
        rec = _make_record(paper_doi="10.1/test", table_id="t5")
        reg.add("ds1", rec)

        result = reg.to_dict()
        assert "ds1" in result
        assert result["ds1"]["paper_doi"] == "10.1/test"
        assert result["ds1"]["table_id"] == "t5"

    def test_to_dict_values_are_plain_dicts(self):
        """Serialised values must be plain dicts, not pydantic models."""
        reg = ProvenanceRegistry()
        reg.add("ds1", _make_record())

        serialised = reg.to_dict()
        assert isinstance(serialised["ds1"], dict)
        # Must not be a ProvenanceRecord instance
        assert not isinstance(serialised["ds1"], ProvenanceRecord)

    def test_empty_registry_to_dict_returns_empty(self):
        reg = ProvenanceRegistry()
        result = reg.to_dict()

        assert result == {}
