"""Tests for ams02wb.parsers.pdf_parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from ams02wb.parsers.pdf_parser import (
    _build_column_mapping,
    _is_numeric_row,
    extract_first_numeric_table,
    extract_tables,
    map_columns,
    map_pdf_rows_to_measurements,
)


# ---------------------------------------------------------------------------
# _is_numeric_row tests
# ---------------------------------------------------------------------------

class TestIsNumericRow:
    """Tests for the _is_numeric_row helper."""

    def test_is_numeric_row_with_mixed_cells(self) -> None:
        """Row with 3/4 numeric cells at default threshold (0.5) passes."""
        row = ["1.23", "hello", "4.56e-2", "+7"]
        assert _is_numeric_row(row) is True

    def test_is_numeric_row_rejects_text_only(self) -> None:
        """Row with zero numeric cells is rejected."""
        row = ["alpha", "beta", "gamma", "delta"]
        assert _is_numeric_row(row) is False

    def test_is_numeric_row_empty_row(self) -> None:
        """Empty row returns False (no cells to evaluate)."""
        assert _is_numeric_row([]) is False

    def test_is_numeric_row_all_numeric(self) -> None:
        """Row where every cell is numeric passes any threshold."""
        row = ["1", "2.5", "-3.14", "1e10"]
        assert _is_numeric_row(row, threshold=1.0) is True

    def test_is_numeric_row_at_exact_threshold(self) -> None:
        """Boundary: fraction exactly equal to threshold passes."""
        # 2 of 4 cells numeric = 0.5 fraction, threshold 0.5 -> True
        row = ["1.0", "text", "2.0", "more text"]
        assert _is_numeric_row(row, threshold=0.5) is True

    def test_is_numeric_row_just_below_threshold(self) -> None:
        """Boundary: fraction just below threshold fails."""
        # 1 of 4 cells numeric = 0.25, threshold 0.5 -> False
        row = ["42", "alpha", "beta", "gamma"]
        assert _is_numeric_row(row, threshold=0.5) is False

    def test_is_numeric_row_scientific_notation(self) -> None:
        """Scientific notation variants are recognised as numeric."""
        row = ["1.23e-4", "5.67E+8", "-9.01e2", "+3.45E-1"]
        assert _is_numeric_row(row, threshold=1.0) is True

    def test_is_numeric_row_sign_prefixed(self) -> None:
        """Plus/minus prefixed numbers are recognised."""
        row = ["+100", "-200", "+0.5", "-0.001"]
        assert _is_numeric_row(row, threshold=1.0) is True

    def test_is_numeric_row_whitespace_padding(self) -> None:
        """Leading/trailing whitespace does not prevent numeric detection."""
        row = ["  1.5 ", " -2 ", "  3e4  "]
        assert _is_numeric_row(row, threshold=1.0) is True

    def test_is_numeric_row_decimal_only(self) -> None:
        """Numbers like '.5' (no leading digit) are numeric."""
        row = [".5", ".123", "text"]
        # 2/3 numeric ≈ 0.667 > 0.5
        assert _is_numeric_row(row) is True

    def test_is_numeric_row_high_threshold(self) -> None:
        """Custom high threshold rejects rows with too much text."""
        row = ["1", "2", "text", "4"]
        # 3/4 = 0.75, threshold 0.9 -> False
        assert _is_numeric_row(row, threshold=0.9) is False


# ---------------------------------------------------------------------------
# extract_first_numeric_table tests
# ---------------------------------------------------------------------------

class TestExtractFirstNumericTable:
    """Tests for the extract_first_numeric_table function."""

    def test_extract_numeric_table_returns_list_of_lists(self) -> None:
        """Returned table is a list of rows (list of lists of strings)."""
        table = [["x", "y"], ["1.0", "2.0"], ["3.0", "4.0"]]
        result = extract_first_numeric_table([table])
        assert result is not None
        assert isinstance(result, list)
        assert all(isinstance(row, list) for row in result)

    def test_extract_numeric_table_skips_header_only_tables(self) -> None:
        """A table with only text rows is skipped."""
        text_table = [["Name", "Species"], ["Proton", "p"], ["Helium", "He"]]
        numeric_table = [["R [GV]", "Flux"], ["1.0", "0.5"], ["2.0", "0.8"]]
        result = extract_first_numeric_table([text_table, numeric_table])
        assert result is numeric_table

    def test_extract_numeric_table_returns_none_for_no_tables(self) -> None:
        """Empty input list returns None."""
        result = extract_first_numeric_table([])
        assert result is None
        assert not isinstance(result, list)

    def test_extract_numeric_table_selects_first_numeric_table(self) -> None:
        """When multiple numeric tables exist, the first one is returned."""
        table_a = [["1.0", "2.0"], ["3.0", "4.0"]]
        table_b = [["5.0", "6.0"], ["7.0", "8.0"]]
        result = extract_first_numeric_table([table_a, table_b])
        assert result is table_a

    def test_extract_numeric_table_with_empty_pdf(self) -> None:
        """Tables list containing only empty tables returns None."""
        result = extract_first_numeric_table([[], [], []])
        assert result is None
        assert not isinstance(result, list)

    def test_extract_all_text_tables_returns_none(self) -> None:
        """Multiple tables with only text all get skipped."""
        t1 = [["header", "col"], ["a", "b"]]
        t2 = [["name", "type"], ["x", "y"]]
        result = extract_first_numeric_table([t1, t2])
        assert result is None
        assert not isinstance(result, list)

    def test_extract_preserves_full_table(self) -> None:
        """The entire table is returned, not just the numeric rows."""
        table = [["R [GV]", "Flux"], ["1.5", "0.42"], ["text", "note"]]
        result = extract_first_numeric_table([table])
        # Should return full table since row ["1.5", "0.42"] is numeric
        assert result is table
        assert len(result) == 3


# ---------------------------------------------------------------------------
# _build_column_mapping tests
# ---------------------------------------------------------------------------

class TestBuildColumnMapping:
    """Tests for header-to-field mapping via regex patterns."""

    def test_build_column_mapping_bracket_units(self) -> None:
        """Bracket-style unit delimiters: 'R [GV]', 'Kinetic Energy [GeV/n]'."""
        header = ["R [GV]", "Flux", "stat", "sys"]
        mapping = _build_column_mapping(header)
        assert mapping[0] == "energy"
        assert mapping[1] == "value"
        assert mapping[2] == "stat_error"
        assert mapping[3] == "sys_error"

    def test_build_column_mapping_parenthetical_units(self) -> None:
        """Parenthetical unit delimiters: 'R (GV)', 'Ek (GeV/n)'."""
        header = ["Ek (GeV/n)", "Flux", "stat error"]
        mapping = _build_column_mapping(header)
        assert mapping[0] == "energy"
        assert mapping[1] == "value"
        assert mapping[2] == "stat_error"

    def test_build_column_mapping_unknown_header_ignored(self) -> None:
        """Columns with unrecognised headers are not included in mapping."""
        header = ["Bin #", "R [GV]", "Flux", "Notes"]
        mapping = _build_column_mapping(header)
        assert 0 not in mapping  # "Bin #" not mapped
        assert 3 not in mapping  # "Notes" not mapped
        assert mapping[1] == "energy"
        assert mapping[2] == "value"

    def test_rigidity_verbose_header(self) -> None:
        """'Rigidity [GV]' (verbose form) maps to energy."""
        mapping = _build_column_mapping(["Rigidity [GV]", "Flux"])
        assert mapping[0] == "energy"

    def test_kinetic_energy_underscore_variant(self) -> None:
        """'E_k [GeV/n]' (underscore variant) maps to energy."""
        mapping = _build_column_mapping(["E_k [GeV/n]", "Flux"])
        assert mapping[0] == "energy"

    def test_sys_error_verbose(self) -> None:
        """'sys error' (verbose form) maps to sys_error."""
        mapping = _build_column_mapping(["Flux", "sys error"])
        assert mapping[1] == "sys_error"

    def test_empty_header_returns_empty_mapping(self) -> None:
        """Empty header row produces empty mapping."""
        assert _build_column_mapping([]) == {}

    def test_case_insensitive_matching(self) -> None:
        """Header matching is case-insensitive."""
        mapping = _build_column_mapping(["FLUX", "STAT", "SYS"])
        assert mapping[0] == "value"
        assert mapping[1] == "stat_error"
        assert mapping[2] == "sys_error"


# ---------------------------------------------------------------------------
# map_pdf_rows_to_measurements tests
# ---------------------------------------------------------------------------

class TestMapPdfRowsToMeasurements:
    """Tests for full row-to-measurement conversion."""

    def test_map_rows_rigidity_header(self) -> None:
        """Rigidity header produces energy_axis='rigidity', energy_unit='GV'."""
        rows = [
            ["R [GV]", "Flux"],
            ["1.0", "0.5"],
        ]
        result = map_pdf_rows_to_measurements(rows)
        assert len(result) == 1
        assert result[0]["energy_axis"] == "rigidity"
        assert result[0]["energy_unit"] == "GV"
        assert result[0]["energy"] == 1.0
        assert result[0]["value"] == 0.5

    def test_map_rows_kinetic_energy_header(self) -> None:
        """Kinetic energy header produces energy_axis='kinetic_energy'."""
        rows = [
            ["Ek (GeV/n)", "Flux"],
            ["2.5", "0.3"],
        ]
        result = map_pdf_rows_to_measurements(rows)
        assert len(result) == 1
        assert result[0]["energy_axis"] == "kinetic_energy"
        assert result[0]["energy_unit"] == "GeV/n"

    def test_map_rows_missing_flux_column_raises(self) -> None:
        """Table without a flux column raises ValueError."""
        rows = [
            ["R [GV]", "Notes"],
            ["1.0", "some text"],
        ]
        with pytest.raises(ValueError, match="No flux column") as exc_info:
            map_pdf_rows_to_measurements(rows)
        assert exc_info.type is ValueError

    def test_map_rows_multiple_data_rows(self) -> None:
        """Multiple data rows each produce one measurement dict."""
        rows = [
            ["R [GV]", "Flux"],
            ["1.0", "0.5"],
            ["2.0", "0.8"],
            ["5.0", "1.2"],
        ]
        result = map_pdf_rows_to_measurements(rows)
        assert len(result) == 3
        assert result[0]["energy"] == 1.0
        assert result[1]["energy"] == 2.0
        assert result[2]["value"] == 1.2

    def test_map_rows_stat_sys_error_columns(self) -> None:
        """Statistical and systematic error columns are mapped correctly."""
        rows = [
            ["R [GV]", "Flux", "stat", "sys"],
            ["1.0", "0.5", "0.01", "0.02"],
        ]
        result = map_pdf_rows_to_measurements(rows)
        assert result[0]["stat_error"] == 0.01
        assert result[0]["sys_error"] == 0.02

    def test_map_rows_output_has_measurement_fields(self) -> None:
        """Each output dict contains the expected canonical field names."""
        rows = [
            ["R [GV]", "Flux", "stat", "sys"],
            ["1.0", "0.5", "0.01", "0.02"],
        ]
        result = map_pdf_rows_to_measurements(rows)
        record = result[0]
        assert "value" in record
        assert "energy" in record
        assert "energy_axis" in record
        assert "energy_unit" in record
        # Values are floats, not strings
        assert isinstance(record["value"], float)
        assert isinstance(record["energy"], float)

    def test_map_rows_empty_input(self) -> None:
        """Rows with only a header (no data) returns empty list."""
        rows = [["R [GV]", "Flux"]]
        assert map_pdf_rows_to_measurements(rows) == []

    def test_map_rows_completely_empty(self) -> None:
        """Empty rows list returns empty list."""
        assert map_pdf_rows_to_measurements([]) == []

    def test_map_rows_non_numeric_cell_in_data(self) -> None:
        """Non-numeric cell in data row is stored as-is (not silently dropped)."""
        rows = [
            ["R [GV]", "Flux"],
            ["N/A", "0.5"],
        ]
        result = map_pdf_rows_to_measurements(rows)
        assert result[0]["energy"] == "N/A"
        assert result[0]["value"] == 0.5


# ---------------------------------------------------------------------------
# extract_tables — PDF file-level extraction (uses fixtures from conftest.py)
# ---------------------------------------------------------------------------


def test_extract_tables_single_table(single_table_pdf: Path) -> None:
    """A one-page PDF with one bordered table yields at least one table."""
    tables = extract_tables(single_table_pdf)
    assert len(tables) >= 1
    # The table should contain the header row + 5 data rows.
    all_rows = [row for table in tables for row in table]
    assert len(all_rows) >= 6  # header + 5 data rows


def test_extract_tables_multi_page_continuous(
    multi_page_table_pdf: Path,
) -> None:
    """A table spanning 2 pages is extracted (tables from both pages present)."""
    tables = extract_tables(multi_page_table_pdf)
    # pdfplumber extracts per-page; we expect tables from both pages.
    assert len(tables) >= 2
    total_rows = sum(len(t) for t in tables)
    # Header (1) + 9 data rows = 10 minimum rows across all tables.
    assert total_rows >= 10


def test_extract_tables_no_table_returns_empty(no_table_pdf: Path) -> None:
    """A text-only PDF returns an empty list — no tables detected."""
    tables = extract_tables(no_table_pdf)
    assert tables == []


def test_extract_tables_header_not_duplicated_across_pages(
    multi_page_table_pdf: Path,
) -> None:
    """Header row appears only in the first page's table, not repeated."""
    tables = extract_tables(multi_page_table_pdf)
    assert len(tables) >= 2

    # Count how many tables start with a row containing "x_min" (header).
    header_tables = [
        t for t in tables
        if any("x_min" in cell for cell in t[0])
    ]
    assert len(header_tables) == 1, (
        "Header row should appear in exactly one table (page 1 only)"
    )


def test_extract_tables_numeric_cells_preserved(
    single_table_pdf: Path,
) -> None:
    """Numeric cell values survive PDF round-trip without corruption."""
    tables = extract_tables(single_table_pdf)
    assert len(tables) >= 1

    # Flatten all rows, skip header, check that numeric values parse.
    all_rows = [row for table in tables for row in table]
    # Find data rows (not the header)
    data_rows = [
        row for row in all_rows
        if not any("x_min" in cell for cell in row)
    ]
    assert len(data_rows) >= 5

    # Every cell in data rows should be parseable as float.
    for row in data_rows:
        for cell in row:
            stripped = cell.strip()
            if stripped:
                float(stripped)  # raises ValueError if corrupted


# ---------------------------------------------------------------------------
# map_columns — header-to-schema-field mapping
# ---------------------------------------------------------------------------


def test_map_columns_known_headers() -> None:
    """All known schema headers map to their canonical field names."""
    headers = ["x_min", "x_max", "y_value", "stat_err", "sys_err_total"]
    result = map_columns(headers)
    assert len(result) == 5
    for h in headers:
        assert result[h] == h


def test_map_columns_unknown_header_raises() -> None:
    """Entirely unrecognised headers raise ValueError."""
    with pytest.raises(ValueError, match="No recognised schema fields") as exc_info:
        map_columns(["foo", "bar", "baz"])
    assert exc_info.type is ValueError


def test_map_columns_partial_match() -> None:
    """Mix of known and unknown headers: only known ones appear in mapping."""
    headers = ["x_min", "notes", "y_value", "comments"]
    result = map_columns(headers)
    assert "x_min" in result
    assert "y_value" in result
    assert "notes" not in result
    assert "comments" not in result
    assert len(result) == 2
