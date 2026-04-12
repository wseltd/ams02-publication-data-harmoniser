"""Tests for join diagnostics: coverage, gaps, overlap, and edge cases.

Exercises compute_join_diagnostics (T027) with synthetic aligned DataFrames.
Does NOT test alignment correctness (covered by T050).
"""

from __future__ import annotations

import datetime

import pandas as pd
import pytest

from ams02wb.alignment.daily import align_daily_series
from ams02wb.alignment.diagnostics import compute_join_diagnostics


def _make_daily_df(dates: list[datetime.date], species: str) -> pd.DataFrame:
    """Build a minimal daily DataFrame for one species."""
    return pd.DataFrame({
        "time_start": dates,
        "y_value": [1.0] * len(dates),
    })


# ── Case 1: Full coverage, no gaps ──────────────────────────────────────

def test_diagnostics_full_coverage_no_gaps():
    """When all species share the same dates (inner join), every row is
    covered and there are no NaN gaps."""
    dates = [datetime.date(2020, 1, d) for d in range(1, 11)]  # 10 days
    result = align_daily_series(
        {"proton": _make_daily_df(dates, "proton"),
         "helium": _make_daily_df(dates, "helium")},
        join="inner",
    )
    diag = compute_join_diagnostics(result.aligned_df, ["proton", "helium"])

    assert diag.coverage_fraction["proton"] == 1.0
    assert diag.coverage_fraction["helium"] == 1.0
    assert diag.gap_histogram["proton"] == []
    assert diag.gap_histogram["helium"] == []
    assert diag.overlap_count == 10


# ── Case 2: Partial gaps reported ────────────────────────────────────────

def test_diagnostics_partial_gaps_reported():
    """Outer join of species with partially overlapping dates produces
    coverage_fraction < 1.0 and non-empty gap histogram."""
    shared = [datetime.date(2020, 1, d) for d in range(1, 6)]  # days 1-5
    extra = [datetime.date(2020, 1, d) for d in range(6, 11)]  # days 6-10

    result = align_daily_series(
        {"proton": _make_daily_df(shared + extra, "proton"),
         "helium": _make_daily_df(shared, "helium")},
        join="outer",
    )
    diag = compute_join_diagnostics(result.aligned_df, ["proton", "helium"])

    assert diag.coverage_fraction["proton"] == 1.0
    assert diag.coverage_fraction["helium"] == pytest.approx(0.5)
    # Helium has a single trailing gap of 5 consecutive NaN days
    assert diag.gap_histogram["helium"] == [5]
    assert diag.gap_histogram["proton"] == []
    assert diag.overlap_count == 5


# ── Case 3: Zero-overlap edge case ──────────────────────────────────────

def test_diagnostics_zero_overlap_edge_case():
    """When two species share NO dates, an inner join produces an empty
    DataFrame.  Diagnostics must report coverage_fraction=0.0 for all
    species, not silently return an empty or misleading result."""
    jan_dates = [datetime.date(2020, 1, d) for d in range(1, 6)]
    feb_dates = [datetime.date(2020, 2, d) for d in range(1, 6)]

    result = align_daily_series(
        {"proton": _make_daily_df(jan_dates, "proton"),
         "helium": _make_daily_df(feb_dates, "helium")},
        join="inner",
    )
    # Inner join of disjoint date ranges → empty aligned_df
    assert len(result.aligned_df) == 0

    diag = compute_join_diagnostics(result.aligned_df, ["proton", "helium"])

    # Zero rows → zero coverage, zero overlap — the caller gets a clear signal
    assert diag.coverage_fraction["proton"] == 0.0
    assert diag.coverage_fraction["helium"] == 0.0
    assert diag.overlap_count == 0
    assert diag.gap_histogram["proton"] == []
    assert diag.gap_histogram["helium"] == []


# ── Case 4: Gap histogram matches expected run lengths exactly ───────────

def test_diagnostics_dropped_dates_exact():
    """Gap histogram must capture the exact pattern of consecutive NaN runs,
    not just their total count.  Construct a species with two distinct gaps
    of different lengths and verify both runs appear in order."""
    # Proton: days 1-15 continuous
    all_dates = [datetime.date(2020, 3, d) for d in range(1, 16)]
    # Helium: days 1-3, skip 4-5 (gap=2), days 6-10, skip 11-13 (gap=3), days 14-15
    helium_dates = (
        [datetime.date(2020, 3, d) for d in range(1, 4)]
        + [datetime.date(2020, 3, d) for d in range(6, 11)]
        + [datetime.date(2020, 3, d) for d in range(14, 16)]
    )

    result = align_daily_series(
        {"proton": _make_daily_df(all_dates, "proton"),
         "helium": _make_daily_df(helium_dates, "helium")},
        join="outer",
    )
    diag = compute_join_diagnostics(result.aligned_df, ["proton", "helium"])

    # Two gaps in order: first a 2-day gap, then a 3-day gap
    assert diag.gap_histogram["helium"] == [2, 3]
    assert diag.gap_histogram["proton"] == []
    # 10 out of 15 days present for helium
    assert diag.coverage_fraction["helium"] == pytest.approx(10 / 15)


# ── Case 5: Overlap count matches expected joint coverage ────────────────

def test_diagnostics_contributing_tables_listed():
    """overlap_count must equal the number of rows where ALL species have
    valid (non-NaN) data — the joint coverage, not the union."""
    # Three species with staggered coverage: 5 days shared by all three
    base = [datetime.date(2020, 6, d) for d in range(1, 16)]     # days 1-15
    mid = [datetime.date(2020, 6, d) for d in range(5, 16)]      # days 5-15
    short = [datetime.date(2020, 6, d) for d in range(5, 10)]    # days 5-9

    result = align_daily_series(
        {"proton": _make_daily_df(base, "proton"),
         "helium": _make_daily_df(mid, "helium"),
         "electron": _make_daily_df(short, "electron")},
        join="outer",
    )
    diag = compute_join_diagnostics(
        result.aligned_df, ["proton", "helium", "electron"],
    )

    # Only days 5-9 have all three species → 5 overlap rows
    assert diag.overlap_count == 5
    assert diag.coverage_fraction["proton"] == 1.0
    assert diag.coverage_fraction["helium"] == pytest.approx(11 / 15)
    assert diag.coverage_fraction["electron"] == pytest.approx(5 / 15)


# ── Case 6: Inner (exact) vs outer (approximate) join behaviour ──────────

def test_diagnostics_approximate_vs_exact_flag():
    """Inner join keeps only exact date matches (no NaN), while outer join
    introduces NaN-filled rows for missing dates.  Diagnostics must
    reflect this: inner produces full coverage, outer shows gaps."""
    shared = [datetime.date(2020, 7, d) for d in range(1, 6)]    # days 1-5
    extended = [datetime.date(2020, 7, d) for d in range(1, 11)]  # days 1-10

    frames = {
        "proton": _make_daily_df(extended, "proton"),
        "helium": _make_daily_df(shared, "helium"),
    }

    # Inner join: exact date matching — only shared dates survive
    inner_result = align_daily_series(dict(frames), join="inner")
    inner_diag = compute_join_diagnostics(
        inner_result.aligned_df, ["proton", "helium"],
    )

    assert inner_result.join_type == "inner"
    assert inner_diag.coverage_fraction["proton"] == 1.0
    assert inner_diag.coverage_fraction["helium"] == 1.0
    assert inner_diag.gap_histogram["helium"] == []
    assert inner_diag.overlap_count == 5

    # Outer join: approximate — keeps all dates, fills gaps with NaN
    outer_result = align_daily_series(dict(frames), join="outer")
    outer_diag = compute_join_diagnostics(
        outer_result.aligned_df, ["proton", "helium"],
    )

    assert outer_result.join_type == "outer"
    assert outer_diag.coverage_fraction["proton"] == 1.0
    assert outer_diag.coverage_fraction["helium"] == pytest.approx(0.5)
    assert outer_diag.gap_histogram["helium"] == [5]
    assert outer_diag.overlap_count == 5
