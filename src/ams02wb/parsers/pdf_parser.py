"""PDF table extraction for AMS-02 publication data.

Extracts numeric tables from pre-parsed PDF table data.  The actual PDF
reading is handled upstream (e.g. by pdfplumber or camelot); this module
operates on already-extracted row/column lists and maps header columns
to canonical Measurement field names.

Chose regex over float() casting: scientific notation with +/- prefixes
needs explicit pattern matching, and regex handles all variants in one pass.
Chose regex for header matching too: AMS papers use inconsistent delimiter
styles (brackets vs parentheses, spacing) so exact string matching would
require an unwieldy lookup table that breaks on each new variant.
"""

from __future__ import annotations

import re
from typing import Any, NamedTuple

# Pattern matching integers, floats, and scientific notation, optionally
# prefixed with + or -.  Examples: "42", "-3.14", "1.23e-4", "+0.5E10".
# Kept permissive on whitespace: cells are stripped before matching.
_NUMERIC_RE = re.compile(
    r"^[+-]?"           # optional sign
    r"(?:\d+\.?\d*"     # integer or float (digits before optional decimal)
    r"|\.\d+)"          # or float starting with decimal point (.5)
    r"(?:[eE][+-]?\d+)?"  # optional exponent
    r"$"
)

# ---------------------------------------------------------------------------
# Header pattern matching
# ---------------------------------------------------------------------------
# AMS-02 papers use varying delimiters for units in column headers:
#   "Rigidity [GV]", "R (GV)", "R [GV]"       -> rigidity axis
#   "Kinetic Energy [GeV/n]", "Ek (GeV/n)"    -> kinetic_energy axis
#   "Flux", "stat", "sys"                      -> value / uncertainties
# Patterns are compiled once at module level (rule: expensive resources
# built once, not per-request).


class _HeaderPattern(NamedTuple):
    """Maps a regex to the canonical field name it identifies."""

    pattern: re.Pattern[str]
    field: str

    def __repr__(self) -> str:
        return (
            f"_HeaderPattern(pattern={self.pattern.pattern!r}, "
            f"field={self.field!r})"
        )


# Rigidity headers: "Rigidity [GV]", "R (GV)", "R [GV]", etc.
_RIGIDITY_RE = re.compile(
    r"(?:rigidity|r)\s*[\[\(]\s*gv\s*[\]\)]",
    re.IGNORECASE,
)

# Kinetic energy headers: "Kinetic Energy [GeV/n]", "Ek (GeV/n)", "E_k [GeV/n]"
_KINETIC_ENERGY_RE = re.compile(
    r"(?:kinetic\s*energy|e_?k)\s*[\[\(]\s*gev/n\s*[\]\)]",
    re.IGNORECASE,
)

# Flux column
_FLUX_RE = re.compile(r"^flux$", re.IGNORECASE)

# Statistical error variants: "stat", "σ_stat", "stat error", "stat err"
_STAT_ERR_RE = re.compile(
    r"(?:σ_?)?stat(?:\s*err(?:or)?)?",
    re.IGNORECASE,
)

# Systematic error variants: "sys", "σ_sys", "sys error", "sys err"
_SYS_ERR_RE = re.compile(
    r"(?:σ_?)?sys(?:\s*err(?:or)?)?",
    re.IGNORECASE,
)

# Ordered: checked first-to-last; first match wins per column.
_HEADER_PATTERNS: list[_HeaderPattern] = [
    _HeaderPattern(_RIGIDITY_RE, "energy"),
    _HeaderPattern(_KINETIC_ENERGY_RE, "energy"),
    _HeaderPattern(_FLUX_RE, "value"),
    _HeaderPattern(_STAT_ERR_RE, "stat_error"),
    _HeaderPattern(_SYS_ERR_RE, "sys_error"),
]

# Energy axis metadata inferred from which energy pattern matched.
_ENERGY_AXIS_INFO: dict[re.Pattern[str], dict[str, str]] = {
    _RIGIDITY_RE: {"energy_axis": "rigidity", "energy_unit": "GV"},
    _KINETIC_ENERGY_RE: {"energy_axis": "kinetic_energy", "energy_unit": "GeV/n"},
}


def _is_numeric_row(row: list[str], threshold: float = 0.5) -> bool:
    """Determine whether a row is predominantly numeric.

    A row is considered numeric when the fraction of cells matching a
    numeric pattern (int, float, scientific notation, +/- prefixed)
    meets or exceeds *threshold*.

    Parameters
    ----------
    row : list[str]
        Cell values from a single table row.
    threshold : float
        Minimum fraction of numeric cells required.  Defaults to 0.5.

    Returns
    -------
    bool
        True if the numeric fraction >= threshold.
    """
    if not row:
        return False

    numeric_count = sum(
        1 for cell in row if _NUMERIC_RE.match(cell.strip())
    )
    return numeric_count / len(row) >= threshold


def extract_first_numeric_table(
    tables: list[list[list[str]]],
) -> list[list[str]] | None:
    """Return the first table that contains at least one numeric row.

    Iterates over *tables* (each a list of rows, each row a list of cell
    strings) and returns the first table where any row passes
    ``_is_numeric_row``.  Returns None if no qualifying table is found.

    This is a heuristic filter, not a guarantee: tables with mixed
    text/numeric content will pass if any single row crosses the threshold.

    Parameters
    ----------
    tables : list[list[list[str]]]
        Pre-extracted tables, e.g. from pdfplumber or camelot.

    Returns
    -------
    list[list[str]] | None
        The first numeric table, or None.
    """
    for table in tables:
        for row in table:
            if _is_numeric_row(row):
                return table
    return None


def _build_column_mapping(header_row: list[str]) -> dict[int, str]:
    """Map column indices to canonical Measurement field names.

    Scans each header cell against compiled regex patterns for known AMS-02
    column header variants.  Unknown headers are silently skipped — AMS tables
    often contain auxiliary columns (bin number, notes) that have no canonical
    mapping.

    Parameters
    ----------
    header_row : list[str]
        Cell values from the header row of a PDF table.

    Returns
    -------
    dict[int, str]
        Mapping from column index to canonical field name.
        Possible field names: 'energy', 'value', 'stat_error', 'sys_error'.
    """
    mapping: dict[int, str] = {}
    for col_idx, cell in enumerate(header_row):
        text = cell.strip()
        for hp in _HEADER_PATTERNS:
            if hp.pattern.search(text):
                mapping[col_idx] = hp.field
                break
    return mapping


def _infer_energy_axis(header_row: list[str]) -> dict[str, str]:
    """Determine energy axis type and unit from the header row.

    Returns a dict with 'energy_axis' and 'energy_unit' keys based on
    which energy pattern matched, or empty dict if no energy column found.
    """
    for cell in header_row:
        text = cell.strip()
        for energy_re, axis_info in _ENERGY_AXIS_INFO.items():
            if energy_re.search(text):
                return dict(axis_info)
    return {}


def map_pdf_rows_to_measurements(
    rows: list[list[str]],
) -> list[dict[str, Any]]:
    """Convert pre-extracted PDF table rows to canonical Measurement dicts.

    First row is treated as the header; remaining rows are data.  Each data
    row becomes one dict with fields mapped from the header via
    ``_build_column_mapping``.

    Parameters
    ----------
    rows : list[list[str]]
        Table rows where rows[0] is the header and rows[1:] are data.

    Returns
    -------
    list[dict[str, Any]]
        One dict per data row with canonical field names as keys and
        float-parsed cell values.  Energy axis metadata ('energy_axis',
        'energy_unit') is included when an energy column is recognised.

    Raises
    ------
    ValueError
        If no 'value' (flux) column is found in the header — a table
        without flux data cannot produce valid measurements.
    """
    if len(rows) < 2:
        return []

    header_row = rows[0]
    col_mapping = _build_column_mapping(header_row)

    # A table without a flux column cannot produce measurements.
    if "value" not in col_mapping.values():
        raise ValueError(
            f"No flux column found in header: {header_row}. "
            "Expected a column matching 'Flux' (case-insensitive)."
        )

    axis_info = _infer_energy_axis(header_row)

    measurements: list[dict[str, Any]] = []
    for row in rows[1:]:
        record: dict[str, Any] = {}
        for col_idx, field_name in col_mapping.items():
            if col_idx < len(row):
                try:
                    record[field_name] = float(row[col_idx])
                except ValueError:
                    # Non-numeric cell in a data row — store as-is.
                    record[field_name] = row[col_idx]
        record.update(axis_info)
        measurements.append(record)

    return measurements
