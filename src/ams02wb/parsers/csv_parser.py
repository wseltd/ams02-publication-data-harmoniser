"""CSV table parser for AMS-02 publication data.

Reads CSV files exported from AMS-02 publication pages, detects the
header row by keyword matching, maps columns to canonical field names,
and returns structured Measurement records with provenance.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column-mapping constants (T007 / T012)
# ---------------------------------------------------------------------------
# Keywords that identify a header row when at least 2 appear in a single line.
# Kept as a frozenset for O(1) membership checks.
KNOWN_COLUMN_KEYWORDS: frozenset[str] = frozenset({
    "rigidity", "flux", "energy", "error", "stat", "sys",
    "kinetic", "momentum", "ratio", "bin",
})

# Maps raw AMS CSV header strings (lowercased) to canonical Measurement
# field names.  At least 3 variants per canonical field where AMS papers
# are known to differ.
# Chose a flat dict over regex: the alias set is finite and exact lookup
# is simpler to maintain and debug.
COLUMN_MAP: dict[str, str] = {
    # Rigidity bin low edge
    "rigidity [gv]": "rigidity_gv_low",
    "r [gv]": "rigidity_gv_low",
    "rigidity(gv)": "rigidity_gv_low",
    "rigidity low [gv]": "rigidity_gv_low",
    "r low [gv]": "rigidity_gv_low",
    "rigidity_low": "rigidity_gv_low",
    "rig low (gv)": "rigidity_gv_low",
    # Rigidity bin high edge
    "rigidity high [gv]": "rigidity_gv_high",
    "r high [gv]": "rigidity_gv_high",
    "rigidity_high": "rigidity_gv_high",
    "rig high (gv)": "rigidity_gv_high",
    # Flux / value
    "flux": "value",
    "flux [m-2 sr-1 s-1 gv-1]": "value",
    "flux value": "value",
    "value": "value",
    # Statistical error (positive)
    "stat err+": "stat_err_plus",
    "stat_err_plus": "stat_err_plus",
    "stat error +": "stat_err_plus",
    "stat+": "stat_err_plus",
    "stat err +": "stat_err_plus",
    # Statistical error (negative)
    "stat err-": "stat_err_minus",
    "stat_err_minus": "stat_err_minus",
    "stat error -": "stat_err_minus",
    "stat-": "stat_err_minus",
    "stat err -": "stat_err_minus",
    # Systematic error (positive)
    "sys err+": "sys_err_plus",
    "sys_err_plus": "sys_err_plus",
    "sys error +": "sys_err_plus",
    "sys+": "sys_err_plus",
    "sys err +": "sys_err_plus",
    # Systematic error (negative)
    "sys err-": "sys_err_minus",
    "sys_err_minus": "sys_err_minus",
    "sys error -": "sys_err_minus",
    "sys-": "sys_err_minus",
    "sys err -": "sys_err_minus",
    # Species
    "species": "species",
    "particle": "species",
    "particle type": "species",
}

# Maximum number of lines to scan for a header row.
_HEADER_SCAN_LIMIT = 20


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Provenance:
    """Tracks where a parsed table came from."""

    paper_id: str
    file_url: str


@dataclass(frozen=True)
class Measurement:
    """A single row of harmonised AMS-02 measurement data."""

    species: str
    rigidity_gv_low: float
    rigidity_gv_high: float
    value: float
    stat_err_plus: float
    stat_err_minus: float
    sys_err_plus: float
    sys_err_minus: float


@dataclass
class ParsedTable:
    """Result of parsing a single CSV table."""

    measurements: List[Measurement]
    source_columns: List[str]
    header_row_index: int
    provenance: Provenance


# ---------------------------------------------------------------------------
# Header detection
# ---------------------------------------------------------------------------
def _count_keyword_hits(cells: list[str]) -> int:
    """Count how many cells contain a known column keyword."""
    hits = 0
    for cell in cells:
        tokens = cell.lower().split()
        if any(tok in KNOWN_COLUMN_KEYWORDS for tok in tokens):
            hits += 1
    return hits


def _find_header_row(
    rows: list[list[str]],
) -> tuple[int, list[str]] | None:
    """Scan the first _HEADER_SCAN_LIMIT rows for the best header candidate.

    Returns (row_index, cells) for the row with the most keyword hits
    (minimum 2), or None if no header is found.
    """
    best_index = -1
    best_hits = 1  # require at least 2 hits
    best_cells: list[str] = []

    limit = min(len(rows), _HEADER_SCAN_LIMIT)
    for i in range(limit):
        hits = _count_keyword_hits(rows[i])
        if hits > best_hits:
            best_hits = hits
            best_index = i
            best_cells = rows[i]

    if best_index < 0:
        return None
    return best_index, best_cells


# ---------------------------------------------------------------------------
# Column resolution
# ---------------------------------------------------------------------------
def _resolve_columns(
    header_cells: list[str],
) -> dict[int, str]:
    """Map column indices to canonical field names.

    Unmapped columns are logged at WARNING and skipped.
    Returns {column_index: canonical_field_name}.
    """
    mapping: dict[int, str] = {}
    for i, cell in enumerate(header_cells):
        key = cell.strip().lower()
        if key in COLUMN_MAP:
            mapping[i] = COLUMN_MAP[key]
        elif key:
            logger.warning("Unmapped column at index %d: %r", i, cell.strip())
    return mapping


# ---------------------------------------------------------------------------
# Row parsing
# ---------------------------------------------------------------------------
_MEASUREMENT_FIELDS = {
    "species", "rigidity_gv_low", "rigidity_gv_high", "value",
    "stat_err_plus", "stat_err_minus", "sys_err_plus", "sys_err_minus",
}

_NUMERIC_FIELDS = _MEASUREMENT_FIELDS - {"species"}


def _parse_data_row(
    row: list[str],
    col_mapping: dict[int, str],
) -> Measurement | None:
    """Parse a single data row into a Measurement, or None on failure."""
    values: dict[str, str] = {}
    for col_idx, field_name in col_mapping.items():
        if col_idx < len(row):
            values[field_name] = row[col_idx].strip()

    # Need at minimum the value field to produce a measurement
    if "value" not in values or not values["value"]:
        return None

    converted: dict[str, object] = {}
    for fname in _MEASUREMENT_FIELDS:
        raw = values.get(fname, "")
        if fname == "species":
            converted[fname] = raw
            continue
        try:
            converted[fname] = float(raw) if raw else 0.0
        except ValueError:
            logger.warning("Non-numeric value for %s: %r", fname, raw)
            return None

    return Measurement(
        species=str(converted["species"]),
        rigidity_gv_low=float(converted["rigidity_gv_low"]),
        rigidity_gv_high=float(converted["rigidity_gv_high"]),
        value=float(converted["value"]),
        stat_err_plus=float(converted["stat_err_plus"]),
        stat_err_minus=float(converted["stat_err_minus"]),
        sys_err_plus=float(converted["sys_err_plus"]),
        sys_err_minus=float(converted["sys_err_minus"]),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def parse_csv_table(
    file_path: Path,
    paper_id: str,
    file_url: str,
) -> ParsedTable:
    """Parse an AMS-02 CSV file into a ParsedTable.

    Scans the first 20 lines to detect a header row by matching cell
    contents against KNOWN_COLUMN_KEYWORDS.  Maps header strings to
    canonical Measurement fields via COLUMN_MAP.  Data rows below the
    header are converted to Measurement instances.

    Parameters
    ----------
    file_path : Path
        Path to the CSV file on disk.
    paper_id : str
        Identifier for the source paper (stored in provenance).
    file_url : str
        URL of the original file (stored in provenance).

    Returns
    -------
    ParsedTable
        Parsed measurements with provenance and column metadata.

    Raises
    ------
    ValueError
        If the file is empty or no header row can be detected.
    """
    path = Path(file_path)

    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        all_rows = list(reader)

    if not all_rows:
        raise ValueError(f"CSV file is empty: {path}")

    result = _find_header_row(all_rows)
    if result is None:
        raise ValueError(
            f"No header row found in the first {_HEADER_SCAN_LIMIT} lines of {path}"
        )

    header_index, header_cells = result
    col_mapping = _resolve_columns(header_cells)
    source_columns = [cell.strip() for cell in header_cells]

    measurements: list[Measurement] = []
    for row in all_rows[header_index + 1:]:
        m = _parse_data_row(row, col_mapping)
        if m is not None:
            measurements.append(m)

    provenance = Provenance(paper_id=paper_id, file_url=file_url)

    return ParsedTable(
        measurements=measurements,
        source_columns=source_columns,
        header_row_index=header_index,
        provenance=provenance,
    )
