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
from pathlib import Path
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

# Canonical schema fields for AMS-02 measurement tables.
# Defined once here; used by map_columns for validation.
SCHEMA_FIELDS: frozenset[str] = frozenset({
    "x_min", "x_max", "y_value", "stat_err", "sys_err_total",
})

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


# ---------------------------------------------------------------------------
# PDF file-level extraction
# ---------------------------------------------------------------------------


def extract_tables(pdf_path: Path | str) -> list[list[list[str]]]:
    """Extract all tables from a PDF file using pdfplumber.

    Reads every page and collects tables detected by pdfplumber's
    line-based table finder.  Each table is a list of rows; each row
    is a list of cell strings.  None cells (pdfplumber uses None for
    empty cells) are normalised to empty strings.

    Parameters
    ----------
    pdf_path : Path | str
        Filesystem path to the PDF.

    Returns
    -------
    list[list[list[str]]]
        All extracted tables.  Empty list if no tables found.
    """
    import pdfplumber

    tables: list[list[list[str]]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_tables = page.extract_tables()
            for table in page_tables:
                if table:
                    cleaned = [
                        [cell if cell is not None else "" for cell in row]
                        for row in table
                    ]
                    tables.append(cleaned)

    # Fallback: if pdfplumber found no gridline tables, try text extraction
    # and convert the result to the same list-of-rows format.
    if not tables:
        text_rows = extract_tables_from_text(pdf_path)
        if text_rows:
            # Build a single table from all text-extracted rows
            all_keys = list(text_rows[0].keys())
            header = all_keys
            data = [[str(row.get(k, "")) for k in all_keys] for row in text_rows]
            tables.append([header] + data)

    return tables


# ---------------------------------------------------------------------------
# Text-based extraction fallback for AMS space-delimited PDFs
# ---------------------------------------------------------------------------

# AMS scientific notation: (1.082 0.001 ...)×10 −1  or  ×100
_AMS_EXPONENT_RE = re.compile(
    r"\(([^)]+)\)\s*[×x]\s*10\s*[−\-]?\s*(\d+)"
)

# Unicode minus (U+2212) → ASCII minus
_UNICODE_MINUS = "\u2212"


def _parse_ams_line(line: str) -> list[float] | None:
    """Parse a line of AMS-format data, handling (...)×10^N notation.

    Returns a flat list of floats, or None if the line is not data.
    """
    line = line.replace(_UNICODE_MINUS, "-")

    # Check for (...)×10^N pattern
    m = _AMS_EXPONENT_RE.search(line)
    if m:
        prefix = line[: m.start()].strip()
        inner = m.group(1).strip()
        exp_str = m.group(2)

        # Determine sign: check if there's a minus before the exponent digit
        exp_segment = line[m.start() : m.end()]
        if "-" in exp_segment.split("10")[-1]:
            exponent = -(int(exp_str))
        else:
            exponent = int(exp_str)

        multiplier = 10.0**exponent

        # Parse prefix (rigidity range like "31.1 - 33.5")
        prefix_nums = []
        if prefix:
            for token in re.split(r"[\s\-]+", prefix):
                token = token.strip()
                if not token:
                    continue
                try:
                    prefix_nums.append(float(token))
                except ValueError:
                    pass

        # Parse inner values and apply multiplier
        inner_nums = []
        for token in inner.split():
            token = token.strip()
            if not token:
                continue
            try:
                inner_nums.append(float(token) * multiplier)
            except ValueError:
                pass

        if inner_nums:
            return prefix_nums + inner_nums

    # Fallback: plain space-separated numbers
    tokens = line.replace(_UNICODE_MINUS, "-").split()
    nums = []
    for t in tokens:
        t = t.strip().rstrip(",;")
        try:
            nums.append(float(t))
        except ValueError:
            pass

    if len(nums) >= 3:
        return nums

    return None


def _detect_header_line(line: str) -> list[str] | None:
    """Check if a line looks like an AMS table header.

    Returns column names if detected, None otherwise.
    """
    line_lower = line.lower().replace(_UNICODE_MINUS, "-")

    # Must contain "rigidity" or "energy" plus at least one error keyword
    has_axis = any(kw in line_lower for kw in ("rigidity", "energy", "ek"))
    has_err = any(kw in line_lower for kw in ("stat", "sys", "syst", "flux"))

    if has_axis and has_err:
        # Split on multiple spaces to get column names
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) >= 3:
            return parts

    return None


def extract_tables_from_text(
    pdf_path: Path | str,
    max_pages: int | None = None,
) -> list[dict[str, Any]]:
    """Extract tables from PDF using text-line parsing.

    Fallback for AMS PDFs where pdfplumber's table detection fails
    because tables are space-delimited rather than gridline-based.

    Parameters
    ----------
    pdf_path : Path | str
        Filesystem path to the PDF.
    max_pages : int | None
        Maximum number of pages to process.  None means all pages.

    Returns
    -------
    list[dict[str, Any]]
        One dict per data row with keys:
        'rigidity_min', 'rigidity_max', 'flux', and error columns
        as detected from the header.
    """
    import pdfplumber

    all_rows: list[dict[str, Any]] = []
    column_names: list[str] = []

    with pdfplumber.open(pdf_path) as pdf:
        pages = pdf.pages[:max_pages] if max_pages else pdf.pages

        for page in pages:
            text = page.extract_text()
            if not text:
                continue

            for line in text.split("\n"):
                line = line.strip()
                if not line:
                    continue

                # Try detecting header
                header = _detect_header_line(line)
                if header and not column_names:
                    # Build canonical column names from header
                    column_names = []
                    for h in header:
                        h_lower = h.strip().lower()
                        if "rigidity" in h_lower or h_lower in ("r", "ek"):
                            column_names.append("rigidity")
                        elif "flux" in h_lower or h_lower == "φ" or "φ" in h_lower:
                            column_names.append("flux")
                        elif "stat" in h_lower:
                            column_names.append("stat_err")
                        elif "trig" in h_lower:
                            column_names.append("trig_err")
                        elif "acc" in h_lower:
                            column_names.append("acc_err")
                        elif "unf" in h_lower:
                            column_names.append("unf_err")
                        elif "scale" in h_lower:
                            column_names.append("scale_err")
                        elif "syst" in h_lower:
                            column_names.append("sys_err_total")
                        else:
                            column_names.append(h.strip())
                    continue

                # Try parsing as data row
                nums = _parse_ams_line(line)
                if nums is None:
                    continue

                # Map numbers to columns
                # For AMS tables: first 2 nums are rigidity range, rest are flux+errors
                if len(nums) >= 3:
                    row: dict[str, Any] = {
                        "rigidity_min": nums[0],
                        "rigidity_max": nums[1],
                    }

                    # Map remaining values to column names (skip rigidity)
                    value_cols = [c for c in column_names if c != "rigidity"]
                    for i, val in enumerate(nums[2:]):
                        if i < len(value_cols):
                            row[value_cols[i]] = val
                        else:
                            row[f"col_{i + 2}"] = val

                    all_rows.append(row)

    return all_rows


def map_columns(headers: list[str]) -> dict[str, str]:
    """Map PDF table headers to canonical schema field names.

    Checks each header against ``SCHEMA_FIELDS``.  Only recognised
    headers appear in the returned mapping.

    Raises ValueError when *no* header matches any known field — a table
    with entirely unrecognised columns cannot be processed.

    Parameters
    ----------
    headers : list[str]
        Column header strings from a PDF table row.

    Returns
    -------
    dict[str, str]
        Mapping from original header text to canonical field name.
        Only includes recognised headers; unrecognised ones are omitted.

    Raises
    ------
    ValueError
        If none of the headers match a known schema field.
    """
    mapping: dict[str, str] = {}
    for header in headers:
        normalised = header.strip().lower()
        if normalised in SCHEMA_FIELDS:
            mapping[header] = normalised

    if not mapping:
        raise ValueError(
            f"No recognised schema fields in headers: {headers}. "
            f"Expected one or more of: {sorted(SCHEMA_FIELDS)}"
        )

    return mapping
