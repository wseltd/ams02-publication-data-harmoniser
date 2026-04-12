"""Tests for ams02wb.parsers.csv_parser."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from ams02wb.parsers.csv_parser import ParsedTable, parse_csv_table


def _write_csv(tmp_path: Path, name: str, lines: list[str]) -> Path:
    """Write lines to a CSV file and return its path."""
    p = tmp_path / name
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


# Header used by most tests — uses common AMS column names.
_STANDARD_HEADER = (
    "Rigidity [GV],Rigidity High [GV],Flux,Stat err+,Stat err-,Sys err+,Sys err-"
)


def _make_clean_csv(tmp_path: Path) -> Path:
    """Create a simple well-formed CSV with two data rows."""
    return _write_csv(tmp_path, "clean.csv", [
        _STANDARD_HEADER,
        "1.0,2.0,100.5,0.1,0.2,0.3,0.4",
        "2.0,3.0,200.5,0.5,0.6,0.7,0.8",
    ])


def test_parse_clean_csv_returns_measurements(tmp_path: Path) -> None:
    """A well-formed CSV with a recognised header yields Measurement records."""
    path = _make_clean_csv(tmp_path)
    result = parse_csv_table(path, paper_id="paper-01", file_url="https://example.com/a.csv")

    assert isinstance(result, ParsedTable)
    assert len(result.measurements) == 2
    m0 = result.measurements[0]
    assert m0.rigidity_gv_low == 1.0
    assert m0.rigidity_gv_high == 2.0
    assert m0.value == 100.5
    assert m0.stat_err_plus == 0.1


def test_parse_detects_header_after_metadata_lines(tmp_path: Path) -> None:
    """Header detection skips leading metadata/comment lines."""
    path = _write_csv(tmp_path, "meta.csv", [
        "# AMS-02 proton flux",
        "# Published 2021",
        "",
        _STANDARD_HEADER,
        "1.0,2.0,50.0,0.1,0.1,0.1,0.1",
    ])
    result = parse_csv_table(path, paper_id="p2", file_url="https://example.com/b.csv")

    assert result.header_row_index == 3
    assert len(result.measurements) == 1
    assert result.measurements[0].value == 50.0


def test_parse_maps_known_column_variants(tmp_path: Path) -> None:
    """Alternative column names map to the same canonical fields."""
    # Use different variant names than the standard header
    path = _write_csv(tmp_path, "variants.csv", [
        "R [GV],R High [GV],Flux value,Stat error +,Stat error -,Sys error +,Sys error -",
        "5.0,10.0,300.0,1.0,1.1,2.0,2.1",
    ])
    result = parse_csv_table(path, paper_id="p3", file_url="https://example.com/c.csv")

    assert len(result.measurements) == 1
    m = result.measurements[0]
    assert m.rigidity_gv_low == 5.0
    assert m.value == 300.0
    assert m.stat_err_plus == 1.0
    assert m.sys_err_minus == 2.1


def test_parse_skips_unmapped_columns_with_warning(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unmapped columns are logged at WARNING and their data is ignored."""
    path = _write_csv(tmp_path, "extra.csv", [
        "Rigidity [GV],Rigidity High [GV],Flux,Stat err+,Stat err-,Sys err+,Sys err-,FooBar",
        "1.0,2.0,99.0,0.1,0.1,0.1,0.1,junk",
    ])
    with caplog.at_level(logging.WARNING, logger="ams02wb.parsers.csv_parser"):
        result = parse_csv_table(path, paper_id="p4", file_url="https://example.com/d.csv")

    # The unmapped column should trigger a warning
    assert any("FooBar" in rec.message for rec in caplog.records)
    # Data is still parsed for mapped columns
    assert len(result.measurements) == 1
    assert result.measurements[0].value == 99.0


def test_parse_handles_csv_with_no_matching_header(tmp_path: Path) -> None:
    """A CSV with no recognisable header raises ValueError."""
    path = _write_csv(tmp_path, "noheader.csv", [
        "alpha,beta,gamma",
        "1,2,3",
    ])
    with pytest.raises(ValueError, match="No header row found"):
        parse_csv_table(path, paper_id="p5", file_url="https://example.com/e.csv")


def test_parse_empty_csv_raises_value_error(tmp_path: Path) -> None:
    """An empty CSV file raises ValueError with a clear message."""
    path = _write_csv(tmp_path, "empty.csv", [""])
    with pytest.raises(ValueError, match="empty"):
        parse_csv_table(path, paper_id="p6", file_url="https://example.com/f.csv")


def test_parse_provenance_tracks_paper_id_and_url(tmp_path: Path) -> None:
    """Provenance records the paper_id and file_url passed to parse_csv_table."""
    path = _make_clean_csv(tmp_path)
    result = parse_csv_table(
        path,
        paper_id="AMS-02/2021/proton",
        file_url="https://ams02.space/data/proton.csv",
    )

    assert result.provenance.paper_id == "AMS-02/2021/proton"
    assert result.provenance.file_url == "https://ams02.space/data/proton.csv"


def test_parse_numeric_conversion_of_measurement_values(tmp_path: Path) -> None:
    """All numeric fields are converted to float, not left as strings."""
    path = _write_csv(tmp_path, "numeric.csv", [
        _STANDARD_HEADER,
        "1.23e1,2.5e1,3.14159,1e-3,2e-3,3e-3,4e-3",
    ])
    result = parse_csv_table(path, paper_id="p8", file_url="https://example.com/h.csv")

    assert len(result.measurements) == 1
    m = result.measurements[0]
    assert isinstance(m.value, float)
    assert isinstance(m.stat_err_plus, float)
    assert isinstance(m.sys_err_minus, float)
    # Verify actual scientific notation parsing
    assert m.rigidity_gv_low == pytest.approx(12.3)
    assert m.value == pytest.approx(3.14159)
    assert m.stat_err_plus == pytest.approx(0.001)
