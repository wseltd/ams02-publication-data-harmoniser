"""Tests for the CSV export function."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from ams02wb.exports.csv_export import CANONICAL_FIELDS, export_csv
from ams02wb.schema.models import Measurement, UncertaintyLabel


def _make_measurement(**overrides: object) -> Measurement:
    """Build a Measurement with sensible defaults, applying typed overrides."""
    defaults: dict[str, object] = {
        "energy_low": 1.0,
        "energy_high": 2.0,
        "energy_mid": 1.5,
        "value": 42.0,
        "unit": "GeV",
        "axis_type": "kinetic_energy_per_nucleon",
        "species": "PROTON",
    }
    defaults.update(overrides)
    return Measurement.model_validate(defaults)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Return (fieldnames, rows) from a CSV file."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def test_export_csv_writes_all_canonical_fields(tmp_path: Path) -> None:
    """Every canonical field appears as a column in the output CSV."""
    m = _make_measurement()
    out = export_csv([m], tmp_path / "out.csv")
    fields, rows = _read_csv(out)

    assert fields == CANONICAL_FIELDS
    assert len(rows) == 1
    # Every canonical field must be present as a key in the row
    for field in CANONICAL_FIELDS:
        assert field in rows[0], f"missing canonical field: {field}"


def test_export_csv_column_order_matches_canonical_schema(tmp_path: Path) -> None:
    """Column order in the header must exactly match CANONICAL_FIELDS."""
    m = _make_measurement()
    out = export_csv([m], tmp_path / "out.csv")

    with open(out, encoding="utf-8") as fh:
        header_line = fh.readline().strip()

    expected_header = ",".join(CANONICAL_FIELDS)
    assert header_line == expected_header


def test_export_csv_serialises_sys_err_components_as_json_string(
    tmp_path: Path,
) -> None:
    """sys_err_components column must be a valid JSON string with error fields."""
    m = _make_measurement(
        sys_err_pos=0.05,
        sys_err_neg=-0.03,
        stat_err_pos=0.01,
        stat_err_neg=-0.01,
    )
    out = export_csv([m], tmp_path / "out.csv")
    _, rows = _read_csv(out)

    raw = rows[0]["sys_err_components"]
    parsed = json.loads(raw)
    assert isinstance(parsed, dict)
    assert parsed["sys_err_pos"] == 0.05
    assert parsed["sys_err_neg"] == -0.03
    assert parsed["stat_err_pos"] == 0.01
    assert parsed["stat_err_neg"] == -0.01


def test_export_csv_serialises_provenance_json_as_json_string(
    tmp_path: Path,
) -> None:
    """provenance_json column must be a valid JSON string."""
    m = _make_measurement(
        stat_err_label=UncertaintyLabel.PUBLISHED,
        sys_err_label=UncertaintyLabel.DERIVED,
    )
    out = export_csv([m], tmp_path / "out.csv")
    _, rows = _read_csv(out)

    raw = rows[0]["provenance_json"]
    parsed = json.loads(raw)
    assert isinstance(parsed, dict)
    assert parsed["stat_err_label"] == "published"
    assert parsed["sys_err_label"] == "derived"


def test_export_csv_preserves_uncertainty_label_in_provenance(
    tmp_path: Path,
) -> None:
    """Uncertainty labels survive the CSV round-trip inside provenance_json."""
    m = _make_measurement(
        stat_err_label=UncertaintyLabel.ASSUMED,
        sys_err_label=UncertaintyLabel.PUBLISHED,
    )
    out = export_csv([m], tmp_path / "out.csv")
    _, rows = _read_csv(out)

    provenance = json.loads(rows[0]["provenance_json"])
    assert provenance["stat_err_label"] == "assumed"
    assert provenance["sys_err_label"] == "published"


def test_export_csv_handles_empty_measurement_list(tmp_path: Path) -> None:
    """An empty measurement list produces a CSV with only the header row."""
    out = export_csv([], tmp_path / "empty.csv")
    fields, rows = _read_csv(out)

    assert fields == CANONICAL_FIELDS
    assert rows == []


def test_export_csv_none_nested_fields_serialize_correctly(tmp_path: Path) -> None:
    """When nested fields are None/default, JSON cells contain null values, not empty strings."""
    m = _make_measurement()  # No error fields, no labels set
    out = export_csv([m], tmp_path / "none.csv")
    _, rows = _read_csv(out)

    sys_comp = json.loads(rows[0]["sys_err_components"])
    # All error sub-fields should be None (JSON null) when not set
    assert sys_comp["stat_err_pos"] is None
    assert sys_comp["stat_err_neg"] is None
    assert sys_comp["sys_err_pos"] is None
    assert sys_comp["sys_err_neg"] is None
    assert sys_comp["stat_error_low"] is None
    assert sys_comp["sys_error_high"] is None

    prov = json.loads(rows[0]["provenance_json"])
    assert prov["stat_err_label"] is None
    assert prov["sys_err_label"] is None
    assert prov["time_start_utc"] is None
    assert prov["time_end_utc"] is None

    meta = json.loads(rows[0]["metadata_json"])
    assert meta == {}


def test_export_csv_round_trip_nested_fields_recoverable(tmp_path: Path) -> None:
    """json.loads on sys_err_components and provenance_json yields dicts."""
    m = _make_measurement(
        sys_err_pos=0.1,
        stat_err_pos=0.02,
        stat_err_label=UncertaintyLabel.DERIVED,
    )
    out = export_csv([m], tmp_path / "rt.csv")
    _, rows = _read_csv(out)

    sys_comp = json.loads(rows[0]["sys_err_components"])
    assert isinstance(sys_comp, dict)
    assert sys_comp["sys_err_pos"] == 0.1
    assert sys_comp["stat_err_pos"] == 0.02

    prov = json.loads(rows[0]["provenance_json"])
    assert isinstance(prov, dict)
    assert prov["stat_err_label"] == "derived"
