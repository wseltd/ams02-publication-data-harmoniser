"""Tests for ams02wb.parsers.pdf_parser."""

from __future__ import annotations

from ams02wb.parsers.pdf_parser import _is_numeric_row, extract_first_numeric_table


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
        assert extract_first_numeric_table([]) is None

    def test_extract_numeric_table_selects_first_numeric_table(self) -> None:
        """When multiple numeric tables exist, the first one is returned."""
        table_a = [["1.0", "2.0"], ["3.0", "4.0"]]
        table_b = [["5.0", "6.0"], ["7.0", "8.0"]]
        result = extract_first_numeric_table([table_a, table_b])
        assert result is table_a

    def test_extract_numeric_table_with_empty_pdf(self) -> None:
        """Tables list containing only empty tables returns None."""
        assert extract_first_numeric_table([[], [], []]) is None

    def test_extract_all_text_tables_returns_none(self) -> None:
        """Multiple tables with only text all get skipped."""
        t1 = [["header", "col"], ["a", "b"]]
        t2 = [["name", "type"], ["x", "y"]]
        assert extract_first_numeric_table([t1, t2]) is None

    def test_extract_preserves_full_table(self) -> None:
        """The entire table is returned, not just the numeric rows."""
        table = [["R [GV]", "Flux"], ["1.5", "0.42"], ["text", "note"]]
        result = extract_first_numeric_table([table])
        # Should return full table since row ["1.5", "0.42"] is numeric
        assert result is table
        assert len(result) == 3
