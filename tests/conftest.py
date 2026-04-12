"""PDF test fixtures — generated at test time using fpdf2.

No binary PDF files are committed to the repo.  Each fixture creates a
temporary PDF via fpdf2 and returns a pathlib.Path to it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fpdf import FPDF

# Column headers matching the canonical schema fields.
_TABLE_HEADERS = ["x_min", "x_max", "y_value", "stat_err", "sys_err_total"]

# Shared layout constants for table rendering.
_COL_WIDTH = 30
_ROW_HEIGHT = 7


def _write_table_row(pdf: FPDF, cells: list[str]) -> None:
    """Write one row of cells with borders, then advance to next line."""
    for cell in cells:
        pdf.cell(_COL_WIDTH, _ROW_HEIGHT, cell, border=1)
    pdf.ln()


# Five data rows used across fixtures.  Values are realistic AMS-02-style
# measurements (rigidity bin edges, flux, stat/sys errors).
_SINGLE_TABLE_DATA = [
    ["1.00", "1.16", "2.15e+04", "1.20e+02", "8.50e+01"],
    ["1.16", "1.33", "1.98e+04", "1.10e+02", "7.90e+01"],
    ["1.33", "1.51", "1.72e+04", "9.80e+01", "6.80e+01"],
    ["1.51", "1.71", "1.45e+04", "8.50e+01", "5.70e+01"],
    ["1.71", "1.92", "1.21e+04", "7.20e+01", "4.80e+01"],
]

# Nine data rows for multi-page fixture — more than fit on one page at the
# chosen y-position.  Header is written once on page 1 only.
_MULTI_PAGE_DATA = [
    ["1.00", "1.16", "2.15e+04", "1.20e+02", "8.50e+01"],
    ["1.16", "1.33", "1.98e+04", "1.10e+02", "7.90e+01"],
    ["1.33", "1.51", "1.72e+04", "9.80e+01", "6.80e+01"],
    ["1.51", "1.71", "1.45e+04", "8.50e+01", "5.70e+01"],
    ["1.71", "1.92", "1.21e+04", "7.20e+01", "4.80e+01"],
    ["1.92", "2.15", "9.85e+03", "6.10e+01", "3.90e+01"],
    ["2.15", "2.40", "7.96e+03", "5.20e+01", "3.15e+01"],
    ["2.40", "2.67", "6.35e+03", "4.30e+01", "2.50e+01"],
    ["2.67", "2.97", "5.01e+03", "3.60e+01", "1.98e+01"],
]


@pytest.fixture
def single_table_pdf(tmp_path: Path) -> Path:
    """One-page PDF with a single table: header + 5 numeric data rows."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=9)

    _write_table_row(pdf, _TABLE_HEADERS)
    for row in _SINGLE_TABLE_DATA:
        _write_table_row(pdf, row)

    out = tmp_path / "single_table.pdf"
    pdf.output(str(out))
    return out


@pytest.fixture
def multi_page_table_pdf(tmp_path: Path) -> Path:
    """PDF where one table spans 2 pages.

    Header row on page 1, continuation rows on page 2.  Header is NOT
    repeated — this is the key property tests assert on.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=9)

    _write_table_row(pdf, _TABLE_HEADERS)

    # Write first 4 rows on page 1, remaining rows on page 2.
    for i, row in enumerate(_MULTI_PAGE_DATA):
        if i == 4:
            pdf.add_page()
            pdf.set_font("Helvetica", size=9)
        _write_table_row(pdf, row)

    out = tmp_path / "multi_page_table.pdf"
    pdf.output(str(out))
    return out


@pytest.fixture
def no_table_pdf(tmp_path: Path) -> Path:
    """Text-only PDF with paragraphs but no tabular structure."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)

    paragraphs = [
        (
            "The Alpha Magnetic Spectrometer (AMS-02) is a particle physics "
            "experiment module mounted on the International Space Station."
        ),
        (
            "It measures cosmic ray fluxes with unprecedented precision over "
            "extended energy ranges and observation periods."
        ),
        (
            "This document contains no tabular data and serves as a control "
            "fixture for table extraction tests."
        ),
    ]

    for paragraph in paragraphs:
        pdf.multi_cell(0, 10, paragraph)
        pdf.ln(5)

    out = tmp_path / "no_table.pdf"
    pdf.output(str(out))
    return out


def make_fixture_record(**overrides: object) -> dict:
    """Build a minimal valid harmonised record with canonical schema fields.

    Returns a dict with all fields expected by the canonical schema.
    Callers can override any field via keyword arguments.
    """
    record: dict[str, object] = {
        "species_num": "PROTON",
        "species_den": "",
        "x_axis_type": "kinetic_energy_per_nucleon",
        "x_axis_unit": "GeV",
        "x_min": 1.0,
        "x_max": 2.0,
        "x_centre": 1.5,
        "y_value": 150.0,
        "y_unit": "m^-2 s^-1 sr^-1 GeV^-1",
        "stat_err": 5.0,
        "sys_err_total": 3.0,
        "sys_err_components": "{}",
        "time_start": "2015-05-19T00:00:00+00:00",
        "time_stop": "2015-11-19T00:00:00+00:00",
        "provenance_json": '{"harmonisation_metadata": {"source": "test"}}',
    }
    record.update(overrides)
    return record
