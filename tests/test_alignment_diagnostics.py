"""Tests for join diagnostics on aligned species DataFrames.

Concentrates test effort on gap histogram edge cases (consecutive runs,
leading/trailing gaps, single-day gaps) because that is the trickiest
logic — coverage and overlap are arithmetic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ams02wb.alignment.diagnostics import compute_join_diagnostics


def _make_aligned_df(species_values: dict[str, list]) -> pd.DataFrame:
    """Build a minimal aligned DataFrame from species value lists."""
    n = len(next(iter(species_values.values())))
    data: dict[str, object] = {
        "time_start": pd.date_range("2020-01-01", periods=n),
    }
    for species, values in species_values.items():
        data[f"y_value_{species}"] = values
    return pd.DataFrame(data)


# --- Coverage fraction ---


def test_full_coverage_returns_fraction_one():
    """All rows valid → coverage = 1.0 for each species."""
    df = _make_aligned_df({
        "proton": [1.0, 2.0, 3.0, 4.0],
        "helium": [5.0, 6.0, 7.0, 8.0],
    })
    result = compute_join_diagnostics(df, ["proton", "helium"])
    assert result.coverage_fraction["proton"] == 1.0
    assert result.coverage_fraction["helium"] == 1.0


def test_half_missing_returns_fraction_point_five():
    """Half NaN rows → coverage = 0.5."""
    df = _make_aligned_df({
        "proton": [1.0, np.nan, 3.0, np.nan],
    })
    result = compute_join_diagnostics(df, ["proton"])
    assert abs(result.coverage_fraction["proton"] - 0.5) < 1e-9


# --- Gap histogram (3 tests — highest risk surface) ---


def test_gap_histogram_consecutive_nans_counted():
    """Two separate NaN runs (length 2 and length 1) listed in occurrence order."""
    #                    gap1(2)        gap2(1)
    df = _make_aligned_df({
        "proton": [1, np.nan, np.nan, 4, np.nan, 6, 7, 8, 9, 10],
    })
    result = compute_join_diagnostics(df, ["proton"])
    assert result.gap_histogram["proton"] == [2, 1]


def test_gap_histogram_leading_and_trailing_gaps():
    """NaN at start and end of series are counted as gaps in occurrence order."""
    #       leading(2)                 trailing(1)
    df = _make_aligned_df({
        "proton": [np.nan, np.nan, 3, 4, 5, np.nan],
    })
    result = compute_join_diagnostics(df, ["proton"])
    # Leading gap of 2 appears first, trailing gap of 1 second.
    assert result.gap_histogram["proton"] == [2, 1]


# --- Overlap count ---


def test_overlap_count_all_species_present():
    """Rows where both species have valid data are counted."""
    df = _make_aligned_df({
        "proton": [1, 2, np.nan, np.nan, 5, 6, 7, np.nan, 9, 10],
        "helium": [1, 2, 3, 4, 5, np.nan, np.nan, np.nan, 9, 10],
    })
    # Both valid at indices: 0, 1, 4, 8, 9 → 5 rows.
    result = compute_join_diagnostics(df, ["proton", "helium"])
    assert result.overlap_count == 5


def test_overlap_count_zero_when_no_common_valid_rows():
    """Completely disjoint valid rows → overlap = 0."""
    df = _make_aligned_df({
        "proton": [1.0, np.nan, 3.0, np.nan],
        "helium": [np.nan, 2.0, np.nan, 4.0],
    })
    result = compute_join_diagnostics(df, ["proton", "helium"])
    assert result.overlap_count == 0


def test_single_species_overlap_equals_nonnan_count():
    """With one species, overlap equals the count of non-NaN rows."""
    df = _make_aligned_df({
        "proton": [1.0, np.nan, 3.0, 4.0, np.nan],
    })
    result = compute_join_diagnostics(df, ["proton"])
    assert result.overlap_count == 3
