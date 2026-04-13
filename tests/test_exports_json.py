"""Tests for the JSON export function and AMS02Encoder."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np

from ams02wb.exports.json_export import export_json


def test_export_json_writes_valid_json_file(tmp_path: Path) -> None:
    """export_json produces a file that json.loads can parse without error."""
    data = {
        "covariance": np.eye(2),
        "mode": "diag",
        "uncertainty_label": "published",
        "measurements": [{"value": 1.0}],
        "metadata": {"source": "test"},
    }
    out = export_json(data, tmp_path / "out.json")
    parsed = json.loads(out.read_text(encoding="utf-8"))

    assert isinstance(parsed, dict)
    assert parsed["mode"] == "diag"
    assert len(parsed["measurements"]) == 1


def test_export_json_covariance_ndarray_becomes_nested_list(
    tmp_path: Path,
) -> None:
    """A 2-D numpy ndarray is serialised as a nested list of lists."""
    cov = np.array([[1.0, 0.5], [0.5, 2.0]])
    data = {"covariance": cov, "measurements": [], "metadata": {}}
    out = export_json(data, tmp_path / "cov.json")
    parsed = json.loads(out.read_text(encoding="utf-8"))

    assert parsed["covariance"] == [[1.0, 0.5], [0.5, 2.0]]
    # Verify structure: outer is list of lists, not flat
    assert isinstance(parsed["covariance"][0], list)


def test_export_json_covariance_preserves_float64_precision(
    tmp_path: Path,
) -> None:
    """Float64 values survive the round-trip without precision loss."""
    # Use a value that would lose precision under float32
    precise_val = 1.2345678901234567
    cov = np.array([[precise_val]], dtype=np.float64)
    data = {"covariance": cov, "measurements": [], "metadata": {}}
    out = export_json(data, tmp_path / "precise.json")

    raw = out.read_text(encoding="utf-8")
    parsed = json.loads(raw)
    # Python float round-trip preserves 15-17 significant digits
    assert parsed["covariance"][0][0] == precise_val


def test_export_json_handles_numpy_scalar_types(tmp_path: Path) -> None:
    """Numpy integer and floating scalars are converted to native Python types."""
    data = {
        "count": np.int64(42),
        "ratio": np.float64(3.14),
        "index": np.int32(7),
        "small": np.float32(2.5),
        "measurements": [],
        "metadata": {},
    }
    out = export_json(data, tmp_path / "scalars.json")
    parsed = json.loads(out.read_text(encoding="utf-8"))

    assert parsed["count"] == 42
    assert isinstance(parsed["count"], int)
    assert parsed["ratio"] == 3.14
    assert isinstance(parsed["ratio"], float)
    assert parsed["index"] == 7
    assert parsed["small"] == 2.5


def test_export_json_handles_datetime_as_iso8601(tmp_path: Path) -> None:
    """datetime.datetime and datetime.date are serialised as ISO 8601 strings."""
    dt = datetime.datetime(2024, 3, 15, 10, 30, 0)
    d = datetime.date(2024, 3, 15)
    data = {
        "timestamp": dt,
        "date_only": d,
        "measurements": [],
        "metadata": {},
    }
    out = export_json(data, tmp_path / "dates.json")
    parsed = json.loads(out.read_text(encoding="utf-8"))

    assert parsed["timestamp"] == "2024-03-15T10:30:00"
    assert parsed["date_only"] == "2024-03-15"


def test_export_json_includes_mode_and_uncertainty_label(
    tmp_path: Path,
) -> None:
    """mode and uncertainty_label fields appear in the output."""
    data = {
        "mode": "kernel_covariance",
        "uncertainty_label": "derived",
        "measurements": [],
        "metadata": {},
    }
    out = export_json(data, tmp_path / "mode.json")
    parsed = json.loads(out.read_text(encoding="utf-8"))

    assert parsed["mode"] == "kernel_covariance"
    assert parsed["uncertainty_label"] == "derived"


def test_export_json_includes_provenance_metadata(tmp_path: Path) -> None:
    """Provenance fields in metadata survive serialisation."""
    provenance = {
        "paper_doi": "10.1103/PhysRevLett.123.456",
        "table_id": "T3",
        "file_url": "https://example.com/data.csv",
        "source_type": "csv",
    }
    data = {
        "measurements": [],
        "metadata": provenance,
    }
    out = export_json(data, tmp_path / "prov.json")
    parsed = json.loads(out.read_text(encoding="utf-8"))

    assert parsed["metadata"]["paper_doi"] == "10.1103/PhysRevLett.123.456"
    assert parsed["metadata"]["table_id"] == "T3"
    assert parsed["metadata"]["file_url"] == "https://example.com/data.csv"
    assert parsed["metadata"]["source_type"] == "csv"


def test_export_json_indent_parameter_controls_formatting(
    tmp_path: Path,
) -> None:
    """indent=None produces compact single-line JSON; indent=4 adds whitespace."""
    data = {"a": 1, "b": 2}

    compact_path = export_json(data, tmp_path / "compact.json", indent=None)
    compact_text = compact_path.read_text(encoding="utf-8")
    assert "\n" not in compact_text.strip()

    indented_path = export_json(data, tmp_path / "indented.json", indent=4)
    indented_text = indented_path.read_text(encoding="utf-8")
    # indent=4 produces multi-line output with 4-space indentation
    assert "\n" in indented_text
    assert "    " in indented_text


def test_export_json_empty_measurements_list_allowed(tmp_path: Path) -> None:
    """An empty measurements list is valid and serialised as []."""
    data = {
        "covariance": np.eye(3),
        "mode": "diag",
        "uncertainty_label": "published",
        "measurements": [],
        "metadata": {},
    }
    out = export_json(data, tmp_path / "empty.json")
    parsed = json.loads(out.read_text(encoding="utf-8"))

    assert parsed["measurements"] == []
    # Covariance still serialised correctly even with empty measurements
    assert parsed["covariance"] == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
