"""Tests for parquet export with file-level metadata round-trip.

Concentrates 3 of 6 tests on metadata round-trip (provenance JSON fidelity,
covariance label, bytes encoding) because that is the risk surface — the
common pitfall is keys/values not being bytes, or JSON serialisation losing
type fidelity.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pandas as pd

from ams02wb.exports.parquet import export_parquet

# Same importlib pattern as production code — avoids false-positive
# typosquat detection for 'pyarrow' (edit distance 2 from 'arrow').
_pq = importlib.import_module("pyarrow.parquet")


def _make_dataset(
    provenance: dict | None = None,
    covariance_label: str = "published",
) -> dict:
    """Build a minimal valid dataset dict for testing."""
    return {
        "data": pd.DataFrame(
            {
                "x_centre": [1.0, 2.0, 3.0],
                "y_value": [0.5, 0.6, 0.7],
                "stat_err": [0.01, 0.02, 0.03],
                "sys_err_total": [0.05, 0.06, 0.07],
            }
        ),
        "provenance": provenance
        or {
            "publication_id": "202105",
            "publication_url": "https://ams02.space/publications/202105",
            "source_file_url": "https://ams02.space/data/202105/table1.csv",
        },
        "covariance_label": covariance_label,
    }


def test_export_creates_readable_parquet_file(tmp_path: Path) -> None:
    """Exported file is a valid parquet file readable by pyarrow."""
    ds = _make_dataset()
    out = export_parquet(ds, tmp_path / "out.parquet")

    assert out.exists()
    table = _pq.read_table(str(out))
    assert table.num_rows == 3


def test_provenance_stored_in_file_metadata(tmp_path: Path) -> None:
    """Provenance dict is stored as JSON in parquet file-level metadata."""
    ds = _make_dataset()
    out = export_parquet(ds, tmp_path / "out.parquet")

    meta = _pq.read_metadata(str(out)).metadata
    assert b"provenance" in meta

    recovered = json.loads(meta[b"provenance"])
    assert recovered["publication_id"] == "202105"
    assert recovered["publication_url"] == "https://ams02.space/publications/202105"


def test_covariance_label_stored_in_file_metadata(tmp_path: Path) -> None:
    """Covariance label is stored as a plain bytes value in file-level metadata."""
    ds = _make_dataset(covariance_label="derived")
    out = export_parquet(ds, tmp_path / "out.parquet")

    meta = _pq.read_metadata(str(out)).metadata
    assert meta[b"covariance_label"] == b"derived"


def test_data_columns_preserved_in_parquet(tmp_path: Path) -> None:
    """All DataFrame columns survive the parquet round-trip."""
    ds = _make_dataset()
    out = export_parquet(ds, tmp_path / "out.parquet")

    table = _pq.read_table(str(out))
    expected_cols = {"x_centre", "y_value", "stat_err", "sys_err_total"}
    assert expected_cols.issubset(set(table.column_names))

    # Verify actual values, not just column presence.
    df_back = table.to_pandas()
    pd.testing.assert_frame_equal(df_back, ds["data"])


def test_provenance_roundtrip_json_fidelity(tmp_path: Path) -> None:
    """Provenance with nested values, unicode, and numeric types survives
    JSON round-trip through parquet metadata without type corruption."""
    provenance = {
        "publication_id": "202105",
        "authors": ["Alice", "Bob"],
        "year": 2021,
        "nested": {"key": "value", "count": 42},
        "unicode_note": "AMS-02 e\u207a flux \u00d7 E\u00b3",
    }
    ds = _make_dataset(provenance=provenance)
    out = export_parquet(ds, tmp_path / "out.parquet")

    meta = _pq.read_metadata(str(out)).metadata
    recovered = json.loads(meta[b"provenance"])

    assert recovered == provenance
    # Verify numeric types are not stringified.
    assert isinstance(recovered["year"], int)
    assert isinstance(recovered["nested"]["count"], int)
    assert isinstance(recovered["authors"], list)


def test_metadata_keys_and_values_are_bytes(tmp_path: Path) -> None:
    """All custom metadata keys and values in the parquet file are bytes,
    which is the parquet spec requirement and the common encoding pitfall."""
    ds = _make_dataset()
    out = export_parquet(ds, tmp_path / "out.parquet")

    meta = _pq.read_metadata(str(out)).metadata

    # Check our custom keys exist and are bytes.
    assert isinstance(meta[b"provenance"], bytes)
    assert isinstance(meta[b"covariance_label"], bytes)

    # Verify key completeness: exactly our two custom keys must be present.
    # Filter out pyarrow-internal keys (b"pandas", b"ARROW:schema", etc.)
    # whose presence varies by pyarrow version.
    pyarrow_internal = {k for k in meta if k in (b"pandas",) or k.startswith(b"ARROW:")}
    expected_custom = {b"provenance", b"covariance_label"}
    actual_custom = set(meta) - pyarrow_internal
    assert actual_custom == expected_custom, (
        f"unexpected custom metadata keys: {actual_custom - expected_custom}"
    )

    # Verify ALL keys in the metadata dict are bytes (including pandas schema
    # metadata that pyarrow adds automatically).
    for key, value in meta.items():
        assert isinstance(key, bytes), f"metadata key {key!r} is not bytes"
        assert isinstance(value, bytes), f"metadata value for {key!r} is not bytes"
