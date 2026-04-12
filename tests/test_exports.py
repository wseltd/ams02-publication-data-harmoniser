"""Tests for all four exporters: parquet, CSV, JSON, USINE.

Eight cases covering round-trip parquet fidelity, CSV field order and
precision, JSON covariance shape and symmetry, and USINE header/column
format compliance.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pandas as pd

from ams02wb.exports.csv_export import CANONICAL_FIELDS, export_csv
from ams02wb.exports.json_export import export_json
from ams02wb.exports.parquet import export_parquet
from ams02wb.exports.usine_export import export_usine
from ams02wb.schema.models import Measurement


# --- Helpers for constructing minimal test data --------------------------

def _make_fit_ready_dataframe(n: int = 5) -> pd.DataFrame:
    """Build a minimal fit-ready DataFrame with canonical columns."""
    return pd.DataFrame({
        "dataset_id": [f"ds_{i}" for i in range(n)],
        "publication_id": [f"pub_{i}" for i in range(n)],
        "x_centre": np.linspace(1.0, 100.0, n, dtype=np.float64),
        "y_value": np.logspace(-3, 1, n, dtype=np.float64),
        "stat_err": np.full(n, 0.01, dtype=np.float64),
        "sys_err_total": np.full(n, 0.005, dtype=np.float64),
        "covariance_label": ["published"] * n,
        "provenance_json": [json.dumps({"source": f"table_{i}"}) for i in range(n)],
    })


def _make_fit_ready_dataset(n: int = 5) -> dict:
    """Build a dataset dict accepted by export_parquet."""
    return {
        "data": _make_fit_ready_dataframe(n),
        "provenance": {"paper_doi": "10.1103/test", "table_id": "T1"},
        "covariance_label": "published",
    }


def _make_measurements(n: int = 4) -> list[Measurement]:
    """Build a list of Measurement objects for CSV/USINE tests."""
    measurements = []
    for i in range(n):
        measurements.append(Measurement(
            energy_low=float(i),
            energy_high=float(i + 1),
            energy_mid=float(i) + 0.5,
            value=1.23456789012345e-3 * (i + 1),
            stat_err_pos=0.001 * (i + 1),
            stat_err_neg=0.001 * (i + 1),
            sys_err_pos=0.0005 * (i + 1),
            sys_err_neg=0.0005 * (i + 1),
            species="PROTON",
            unit="GV",
            axis_type="rigidity",
        ))
    return measurements


# --- Parquet tests -------------------------------------------------------

def test_parquet_roundtrip_fields_preserved(tmp_path):
    """Parquet round-trip preserves all column names from the source DataFrame."""
    dataset = _make_fit_ready_dataset(n=3)
    original_columns = list(dataset["data"].columns)

    out = export_parquet(dataset, tmp_path / "test.parquet")
    roundtrip = pd.read_parquet(out)

    assert list(roundtrip.columns) == original_columns


def test_parquet_roundtrip_values_exact(tmp_path):
    """Float64 values survive parquet round-trip without precision loss."""
    dataset = _make_fit_ready_dataset(n=6)
    original_df = dataset["data"].copy()

    out = export_parquet(dataset, tmp_path / "test.parquet")
    roundtrip = pd.read_parquet(out)

    for col in ["x_centre", "y_value", "stat_err", "sys_err_total"]:
        original_vals = original_df[col].values
        roundtrip_vals = roundtrip[col].values
        # Exact equality — float64 in parquet should be bit-identical
        np.testing.assert_array_equal(
            roundtrip_vals, original_vals,
            err_msg=f"Column {col!r} lost precision in parquet round-trip",
        )
        assert roundtrip_vals.dtype == np.float64, (
            f"Column {col!r} dtype changed from float64 to {roundtrip_vals.dtype}"
        )


# --- CSV tests -----------------------------------------------------------

def test_csv_field_order_matches_canonical(tmp_path):
    """CSV header column order matches the CANONICAL_FIELDS ordering exactly."""
    measurements = _make_measurements(n=2)
    out = export_csv(measurements, tmp_path / "test.csv")

    with open(out, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)

    assert header == CANONICAL_FIELDS


def test_csv_numeric_precision_preserved(tmp_path):
    """Numeric values in CSV are not silently truncated below float64 useful digits."""
    # Use a value with many significant digits
    m = Measurement(
        energy_low=0.0,
        energy_high=1.0,
        energy_mid=0.123456789012345,
        value=9.87654321098765e-4,
        stat_err_pos=1.23456789e-5,
        sys_err_pos=9.87654321e-6,
        species="PROTON",
        unit="GV",
        axis_type="rigidity",
    )
    out = export_csv([m], tmp_path / "precision.csv")

    with open(out, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        row = next(reader)

    # x_centre maps from energy_mid, y_value maps from value
    x_centre_str = row["x_centre"]
    y_value_str = row["y_value"]

    # Recovered floats must match originals within float64 representable precision
    assert float(x_centre_str) == m.energy_mid, (
        f"x_centre truncated: {x_centre_str!r} != {m.energy_mid}"
    )
    assert float(y_value_str) == m.value, (
        f"y_value truncated: {y_value_str!r} != {m.value}"
    )


# --- JSON tests ----------------------------------------------------------

def test_json_covariance_shape_square(tmp_path):
    """Covariance matrix serialises to JSON as a nested list with shape (n, n)."""
    n = 5
    cov = np.diag(np.ones(n, dtype=np.float64) * 0.01)
    data = {
        "dataset_id": "ds_test",
        "covariance_matrix": cov,
        "n_points": n,
    }

    out = export_json(data, tmp_path / "cov.json")
    with open(out, encoding="utf-8") as fh:
        loaded = json.load(fh)

    matrix = loaded["covariance_matrix"]
    assert len(matrix) == n, f"Expected {n} rows, got {len(matrix)}"
    for i, row in enumerate(matrix):
        assert len(row) == n, f"Row {i} has {len(row)} columns, expected {n}"


def test_json_covariance_symmetric(tmp_path):
    """Covariance matrix remains symmetric after JSON round-trip."""
    n = 4
    # Build a symmetric positive-definite matrix
    rng = np.random.default_rng(42)
    a = rng.standard_normal((n, n))
    cov = a @ a.T  # guaranteed symmetric
    assert np.allclose(cov, cov.T), "Test setup: matrix should be symmetric"

    data = {"covariance_matrix": cov}
    out = export_json(data, tmp_path / "sym.json")

    with open(out, encoding="utf-8") as fh:
        loaded = json.load(fh)

    matrix = loaded["covariance_matrix"]
    for i in range(n):
        for j in range(n):
            assert matrix[i][j] == matrix[j][i], (
                f"Symmetry broken at [{i}][{j}]: "
                f"{matrix[i][j]} != {matrix[j][i]}"
            )


# --- USINE tests ---------------------------------------------------------

_REQUIRED_HEADER_KEYS = {"experiment", "quantity", "energy_type"}


def test_usine_header_format_valid(tmp_path):
    """USINE header lines start with # and contain required metadata keys."""
    measurements = _make_measurements(n=3)
    text = export_usine(
        measurements,
        species_num="H",
        x_axis_type="rigidity",
        experiment="AMS-02",
        bibcode="2021PhRvL.127A1102A",
    )

    lines = text.strip().splitlines()
    header_lines = [ln for ln in lines if ln.startswith("#")]

    assert len(header_lines) >= 3, (
        f"Expected at least 3 header lines, got {len(header_lines)}"
    )

    # Every header comment line must start with '# '
    for ln in header_lines:
        assert ln.startswith("# "), f"Header line missing '# ' prefix: {ln!r}"

    # Extract keys from comment lines (format: "# key value...")
    found_keys = set()
    for ln in header_lines:
        parts = ln.lstrip("# ").split(None, 1)
        if parts:
            found_keys.add(parts[0])

    missing = _REQUIRED_HEADER_KEYS - found_keys
    assert not missing, f"Missing required header keys: {missing}"


def test_usine_column_names_match_spec(tmp_path):
    """Data column header matches USINE expected names: e_low, e_high, e_mid, value, stat_err, sys_err."""
    measurements = _make_measurements(n=2)
    text = export_usine(
        measurements,
        species_num="H",
        x_axis_type="kinetic_energy_per_nucleon",
    )

    expected_columns = ["e_low", "e_high", "e_mid", "value", "stat_err", "sys_err"]

    lines = text.strip().splitlines()
    # The column header is the last comment line before data rows (starts with "# ")
    # and contains the column names
    column_header_line = None
    for ln in lines:
        if ln.startswith("# "):
            # Check if this looks like the column header (contains data column names)
            stripped = ln.lstrip("# ").strip()
            if stripped.startswith("e_low"):
                column_header_line = stripped
                break

    assert column_header_line is not None, (
        "Could not find column header line starting with 'e_low' in USINE output"
    )

    actual_columns = column_header_line.split()
    assert actual_columns == expected_columns, (
        f"Column names mismatch: {actual_columns} != {expected_columns}"
    )
