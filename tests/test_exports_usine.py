"""Tests for the USINE-format exporter."""

from __future__ import annotations

import pytest

from ams02wb.exports.usine_export import (
    AXIS_TO_USINE,
    MEASUREMENT_TYPE_TO_USINE,
    build_usine_quantity_string,
    export_usine,
)
from ams02wb.schema.models import Measurement


def _make_measurement(**overrides: object) -> Measurement:
    """Build a Measurement with sensible defaults."""
    defaults: dict[str, object] = {
        "energy_low": 1.0,
        "energy_high": 2.0,
        "energy_mid": 1.5,
        "value": 42.0,
        "unit": "GV",
        "axis_type": "rigidity",
        "species": "H",
        "stat_error_high": 0.5,
        "sys_error_high": 0.3,
    }
    defaults.update(overrides)
    return Measurement.model_validate(defaults)


# --- Header tests -----------------------------------------------------


def test_export_usine_writes_header_block_with_comment_prefix() -> None:
    """Every header line must start with '# '."""
    m = _make_measurement()
    output = export_usine([m], species_num="H", x_axis_type="rigidity")

    header_lines = [ln for ln in output.splitlines() if ln.startswith("# ")]
    assert len(header_lines) >= 3, "Expected at least 3 comment-prefixed header lines"
    for line in header_lines:
        assert line.startswith("# ")


def test_export_usine_header_contains_experiment_ams02() -> None:
    """Header must contain the experiment name 'AMS-02'."""
    m = _make_measurement()
    output = export_usine([m], species_num="H", x_axis_type="rigidity")

    assert "# experiment AMS-02" in output


# --- Axis mapping tests -----------------------------------------------


def test_export_usine_maps_rigidity_to_R() -> None:
    """rigidity axis type must map to USINE code 'R'."""
    assert AXIS_TO_USINE["rigidity"] == "R"

    m = _make_measurement()
    output = export_usine([m], species_num="H", x_axis_type="rigidity")
    assert "# energy_type R" in output


def test_export_usine_maps_kinetic_energy_per_nucleon_to_Ekn() -> None:
    """kinetic_energy_per_nucleon axis type must map to USINE code 'Ekn'."""
    assert AXIS_TO_USINE["kinetic_energy_per_nucleon"] == "Ekn"

    m = _make_measurement(axis_type="kinetic_energy_per_nucleon")
    output = export_usine(
        [m], species_num="H", x_axis_type="kinetic_energy_per_nucleon"
    )
    assert "# energy_type Ekn" in output


def test_export_usine_maps_kinetic_energy_to_Ek() -> None:
    """kinetic_energy axis type must map to USINE code 'Ek'."""
    assert AXIS_TO_USINE["kinetic_energy"] == "Ek"


# --- Quantity string tests --------------------------------------------


def test_export_usine_maps_species_num_den_to_quantity_string() -> None:
    """Single-species flux produces bare species name as quantity."""
    assert build_usine_quantity_string("H", None) == "H"

    m = _make_measurement()
    output = export_usine([m], species_num="H", x_axis_type="rigidity")
    assert "# quantity H" in output


def test_export_usine_ratio_quantity_format() -> None:
    """Ratio measurement produces 'NUM/DEN' quantity string."""
    assert build_usine_quantity_string("B", "C") == "B/C"

    m = _make_measurement()
    output = export_usine(
        [m], species_num="B", species_den="C", x_axis_type="rigidity"
    )
    assert "# quantity B/C" in output


# --- Data format tests ------------------------------------------------


def test_export_usine_fixed_width_columns_aligned() -> None:
    """Data columns must be fixed-width aligned (14 chars per column)."""
    m = _make_measurement()
    output = export_usine([m], species_num="H", x_axis_type="rigidity")

    # Find the column header line (starts with "# e_low")
    lines = output.splitlines()
    col_header = [ln for ln in lines if ln.startswith("# e_low")]
    assert len(col_header) == 1, "Expected exactly one column header line"

    # Data lines are non-comment, non-empty
    data_lines = [ln for ln in lines if ln and not ln.startswith("#")]
    assert len(data_lines) >= 1

    # Each data value occupies exactly 14 characters (6 columns × 14 = 84)
    for line in data_lines:
        # Columns are left-justified in 14-char slots
        assert len(line) == 14 * 6, f"Line width {len(line)} != {14 * 6}: {line!r}"


def test_export_usine_data_rows_match_column_count() -> None:
    """Each data row must have the same number of whitespace-separated fields as columns."""
    measurements = [
        _make_measurement(energy_low=1.0, energy_high=2.0, energy_mid=1.5, value=10.0),
        _make_measurement(energy_low=2.0, energy_high=3.0, energy_mid=2.5, value=20.0),
    ]
    output = export_usine(measurements, species_num="H", x_axis_type="rigidity")

    data_lines = [
        ln for ln in output.splitlines() if ln.strip() and not ln.startswith("#")
    ]
    assert len(data_lines) == 2

    expected_col_count = 6  # e_low, e_high, e_mid, value, stat_err, sys_err
    for line in data_lines:
        fields = line.split()
        assert len(fields) == expected_col_count, (
            f"Expected {expected_col_count} columns, got {len(fields)}: {line!r}"
        )


# --- Optional fields tests -------------------------------------------


def test_export_usine_includes_bibcode_when_provided() -> None:
    """Bibcode must appear in header when supplied."""
    m = _make_measurement()
    output = export_usine(
        [m],
        species_num="H",
        x_axis_type="rigidity",
        bibcode="2021PhRvL.126d1104A",
    )
    assert "# bibcode 2021PhRvL.126d1104A" in output


def test_export_usine_omits_bibcode_when_not_provided() -> None:
    """When bibcode is None, no bibcode line should appear."""
    m = _make_measurement()
    output = export_usine([m], species_num="H", x_axis_type="rigidity")
    assert "bibcode" not in output


# --- Flux data output test --------------------------------------------


def test_export_usine_flux_data_output() -> None:
    """Flux data values must appear in the correct columns."""
    m = _make_measurement(
        energy_low=10.0,
        energy_high=20.0,
        energy_mid=15.0,
        value=3.14,
        stat_error_high=0.1,
        sys_error_high=0.05,
    )
    output = export_usine([m], species_num="H", x_axis_type="rigidity")

    data_lines = [
        ln for ln in output.splitlines() if ln.strip() and not ln.startswith("#")
    ]
    assert len(data_lines) == 1

    fields = data_lines[0].split()
    assert fields[0] == "10.0"
    assert fields[1] == "20.0"
    assert fields[2] == "15.0"
    assert fields[3] == "3.14"
    assert fields[4] == "0.1"
    assert fields[5] == "0.05"


# --- Error handling tests ---------------------------------------------


def test_export_usine_rejects_unknown_axis_type() -> None:
    """Unknown axis type must raise ValueError with a helpful message."""
    m = _make_measurement()
    with pytest.raises(ValueError, match="Unknown x_axis_type 'banana'") as exc_info:
        export_usine([m], species_num="H", x_axis_type="banana")

    assert "expected one of" in str(exc_info.value)


# --- Mapping dict completeness ----------------------------------------


def test_measurement_type_to_usine_contains_flux_and_ratio() -> None:
    """MEASUREMENT_TYPE_TO_USINE must map at least flux and ratio."""
    assert MEASUREMENT_TYPE_TO_USINE["flux"] == "flux"
    assert MEASUREMENT_TYPE_TO_USINE["ratio"] == "ratio"
