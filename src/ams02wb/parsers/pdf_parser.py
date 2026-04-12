"""PDF table extraction for AMS-02 publication data.

Extracts numeric tables from pre-parsed PDF table data.  The actual PDF
reading is handled upstream (e.g. by pdfplumber or camelot); this module
operates on already-extracted row/column lists and selects the first table
whose rows are predominantly numeric.

Chose regex over float() casting: scientific notation with +/- prefixes
needs explicit pattern matching, and regex handles all variants in one pass.
"""

from __future__ import annotations

import re

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
