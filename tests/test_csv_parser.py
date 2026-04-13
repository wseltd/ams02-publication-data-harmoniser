"""Tests for parse_csv from ams02wb.parsers.csv_parser.

Uses io.StringIO for all fixtures — no external files committed.
Weighted toward malformed-input cases because parser resilience is
the risky logic.
"""

from __future__ import annotations

import io

import pytest

from ams02wb.parsers.csv_parser import REQUIRED_CSV_COLUMNS, parse_csv


# -- Canonical column names used across fixtures ----------------------------
# Defined once to avoid typos; mirrors the schema contract.
_ALL_CANONICAL = ("x_min", "x_max", "x_centre", "y_value", "stat_err", "sys_err_total")


def _csv(text: str) -> io.StringIO:
    """Wrap a raw CSV string in a StringIO for parse_csv."""
    return io.StringIO(text)


# ===== Positive cases (4) ==================================================

def test_parse_csv_good_file() -> None:
    """A well-formed CSV with required columns returns one dict per data row."""
    rows = parse_csv(_csv(
        "x_min,x_max,y_value,stat_err,sys_err_total\n"
        "1.0,2.0,100.5,0.1,0.3\n"
        "2.0,3.0,200.5,0.5,0.7\n"
    ))

    assert len(rows) == 2
    assert rows[0]["x_min"] == 1.0
    assert rows[0]["x_max"] == 2.0
    assert rows[0]["y_value"] == 100.5
    assert rows[1]["y_value"] == 200.5


def test_parse_csv_good_file_semicolon_delimited() -> None:
    """Semicolon-delimited CSV is auto-detected and parsed correctly."""
    rows = parse_csv(_csv(
        "x_min;x_max;y_value;stat_err\n"
        "1.0;2.0;50.0;0.01\n"
    ))

    assert len(rows) == 1
    assert rows[0]["x_min"] == 1.0
    assert rows[0]["y_value"] == 50.0
    assert rows[0]["stat_err"] == 0.01


def test_parse_csv_good_file_all_columns_present() -> None:
    """When all six canonical columns appear, every key is in the output dict."""
    header = ",".join(_ALL_CANONICAL)
    data = "1.0,2.0,1.5,100.0,0.1,0.3"
    rows = parse_csv(_csv(f"{header}\n{data}\n"))

    assert len(rows) == 1
    for col in _ALL_CANONICAL:
        assert col in rows[0], f"Missing canonical column: {col}"


def test_parse_csv_good_file_numeric_values() -> None:
    """Numeric fields are floats, including scientific notation."""
    rows = parse_csv(_csv(
        "x_min,x_max,y_value,stat_err\n"
        "1.23e1,2.5e1,3.14159,1e-3\n"
    ))

    m = rows[0]
    assert isinstance(m["x_min"], float)
    assert isinstance(m["y_value"], float)
    assert m["x_min"] == pytest.approx(12.3)
    assert m["y_value"] == pytest.approx(3.14159)
    assert m["stat_err"] == pytest.approx(0.001)


# ===== Negative / edge cases (6) ===========================================

def test_parse_csv_missing_required_column_raises() -> None:
    """A CSV whose header lacks one required column raises ValueError
    naming the missing column — not a generic 'parse failed' message.
    """
    # Header has x_min and x_max but no y_value
    with pytest.raises(ValueError, match="y_value") as exc_info:
        parse_csv(_csv("x_min,x_max,stat_err\n1,2,0.1\n"))

    # Singular form for a single missing column
    assert "missing required column:" in str(exc_info.value).lower()


def test_parse_csv_missing_multiple_columns_raises() -> None:
    """When multiple required columns are absent the error lists all of them,
    so the caller can fix everything in one pass.
    """
    # Header has none of the required columns
    with pytest.raises(ValueError, match="missing required columns") as exc_info:
        parse_csv(_csv("stat_err,sys_err_total\n0.1,0.2\n"))

    msg = str(exc_info.value)
    # All three required columns should be named
    for col in sorted(REQUIRED_CSV_COLUMNS):
        assert col in msg, f"Error should mention missing column {col!r}"


def test_parse_csv_extra_column_preserved() -> None:
    """Columns beyond the canonical set are kept in the output dict so that
    downstream code can decide whether to use them.
    """
    rows = parse_csv(_csv(
        "x_min,x_max,y_value,custom_quality_flag\n"
        "1.0,2.0,50.0,A\n"
    ))

    assert "custom_quality_flag" in rows[0]
    assert rows[0]["custom_quality_flag"] == "A"
    # Standard columns still present
    assert rows[0]["y_value"] == 50.0


def test_parse_csv_comment_header_skipped() -> None:
    """Lines starting with '#' are treated as comments and skipped.
    The first non-comment line is the header.
    """
    rows = parse_csv(_csv(
        "# AMS-02 proton flux, published 2021\n"
        "# Units: GV, m-2 sr-1 s-1 GV-1\n"
        "x_min,x_max,y_value\n"
        "1.0,2.0,100.0\n"
    ))

    assert len(rows) == 1
    assert rows[0]["x_min"] == 1.0
    assert rows[0]["y_value"] == 100.0


def test_parse_csv_empty_file_raises() -> None:
    """A completely empty source raises ValueError with 'empty' in the
    message — distinct from a missing-column error.
    """
    with pytest.raises(ValueError, match="(?i)empty") as exc_info:
        parse_csv(_csv(""))

    assert exc_info.type is ValueError
    assert "empty" in str(exc_info.value).lower()

    # Also test whitespace-only content
    with pytest.raises(ValueError, match="(?i)empty") as exc_info:
        parse_csv(_csv("   \n\n  \n"))

    assert exc_info.type is ValueError
    assert "empty" in str(exc_info.value).lower()


def test_parse_csv_whitespace_trimmed() -> None:
    """Leading/trailing whitespace in headers and values is stripped so that
    sloppy formatting does not break column lookup or numeric conversion.
    """
    rows = parse_csv(_csv(
        "  x_min , x_max , y_value , stat_err \n"
        "  1.0 , 2.0 , 99.9 , 0.05 \n"
    ))

    assert len(rows) == 1
    # Headers trimmed — lookup by canonical name works
    assert "x_min" in rows[0]
    assert "y_value" in rows[0]
    # Values trimmed and converted to float
    assert rows[0]["x_min"] == 1.0
    assert rows[0]["y_value"] == pytest.approx(99.9)
    assert rows[0]["stat_err"] == pytest.approx(0.05)
