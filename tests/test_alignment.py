"""Tests for daily-series and Bartels-rotation aligners.

Exercises align_daily_series (daily aligner) and align_bartels_rotation
(Bartels aligner) with synthetic in-memory DataFrames.  Daily path gets
8 tests (higher risk due to cardinality mismatches), Bartels gets 4.
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd

from ams02wb.alignment.bartels import align_bartels_rotation, BARTELS_EPOCH
from ams02wb.alignment.daily import align_daily_series


# ---------------------------------------------------------------------------
# Daily aligner tests (8 cases)
# ---------------------------------------------------------------------------


def test_daily_inner_join_exact_overlap():
    """Inner join on two species with identical date ranges keeps all rows."""
    dates = [datetime.date(2020, 1, d) for d in range(1, 6)]
    proton = pd.DataFrame({"time_start": dates, "flux": [1.0, 2.0, 3.0, 4.0, 5.0]})
    helium = pd.DataFrame({"time_start": dates, "flux": [10.0, 20.0, 30.0, 40.0, 50.0]})

    result = align_daily_series({"proton": proton, "helium": helium}, join="inner")

    assert len(result.aligned_df) == 5
    assert set(result.aligned_df["time_start"]) == set(dates)
    assert result.missing_counts["proton"] == 0
    assert result.missing_counts["helium"] == 0
    assert result.join_type == "inner"


def test_daily_inner_join_missing_days_dropped():
    """Inner join drops dates present in only one species."""
    shared = [datetime.date(2020, 3, 10), datetime.date(2020, 3, 12)]
    proton_dates = shared + [datetime.date(2020, 3, 14)]
    helium_dates = [datetime.date(2020, 3, 8)] + shared

    proton = pd.DataFrame({"time_start": proton_dates, "flux": [1.0, 2.0, 3.0]})
    helium = pd.DataFrame({"time_start": helium_dates, "flux": [10.0, 20.0, 30.0]})

    result = align_daily_series({"proton": proton, "helium": helium}, join="inner")

    assert len(result.aligned_df) == 2
    assert set(result.aligned_df["time_start"]) == set(shared)


def test_daily_outer_join_fills_missing():
    """Outer join keeps all dates and fills absent species-date pairs with NaN."""
    proton_dates = [datetime.date(2020, 6, 1), datetime.date(2020, 6, 2)]
    helium_dates = [datetime.date(2020, 6, 2), datetime.date(2020, 6, 3)]

    proton = pd.DataFrame({"time_start": proton_dates, "flux": [1.0, 2.0]})
    helium = pd.DataFrame({"time_start": helium_dates, "flux": [20.0, 30.0]})

    result = align_daily_series({"proton": proton, "helium": helium}, join="outer")

    all_dates = {datetime.date(2020, 6, 1), datetime.date(2020, 6, 2), datetime.date(2020, 6, 3)}
    assert len(result.aligned_df) == 3
    assert set(result.aligned_df["time_start"]) == all_dates

    # Proton missing on June 3 => NaN in flux_proton
    row_june3 = result.aligned_df[result.aligned_df["time_start"] == datetime.date(2020, 6, 3)]
    assert np.isnan(row_june3["flux_proton"].iloc[0])

    # Helium missing on June 1 => NaN in flux_helium
    row_june1 = result.aligned_df[result.aligned_df["time_start"] == datetime.date(2020, 6, 1)]
    assert np.isnan(row_june1["flux_helium"].iloc[0])

    # Missing counts reflect the NaN-filled rows
    assert result.missing_counts["proton"] == 1
    assert result.missing_counts["helium"] == 1


def test_daily_single_species_passthrough():
    """Single-species input passes through with suffixed columns and zero missing."""
    dates = [datetime.date(2021, 7, d) for d in range(1, 4)]
    electron = pd.DataFrame({"time_start": dates, "flux": [0.1, 0.2, 0.3]})

    result = align_daily_series({"electron": electron}, join="inner")

    assert len(result.aligned_df) == 3
    assert "flux_electron" in result.aligned_df.columns
    assert result.missing_counts["electron"] == 0
    assert result.date_range == (dates[0], dates[-1])


def test_daily_mismatched_cardinalities():
    """Mimics real AMS mismatch (e.g. 2824 vs 3300 days) — inner join trims to overlap."""
    # Proton: 50 days starting Jan 1 2019
    proton_dates = [datetime.date(2019, 1, 1) + datetime.timedelta(days=i) for i in range(50)]
    # Helium: 30 days starting Jan 11 2019 — overlap is Jan 11–Jan 31 (20 days shifted by 1)
    helium_start = datetime.date(2019, 1, 11)
    helium_dates = [helium_start + datetime.timedelta(days=i) for i in range(30)]

    proton = pd.DataFrame({"time_start": proton_dates, "flux": range(50)})
    helium = pd.DataFrame({"time_start": helium_dates, "flux": range(30)})

    overlap = set(proton_dates) & set(helium_dates)

    result = align_daily_series({"proton": proton, "helium": helium}, join="inner")

    assert len(result.aligned_df) == len(overlap)
    assert set(result.aligned_df["time_start"]) == overlap


def test_daily_period_boundary_start():
    """First date in the shared range is correctly included in inner join."""
    boundary_date = datetime.date(2020, 1, 1)
    proton_dates = [boundary_date, datetime.date(2020, 1, 2), datetime.date(2020, 1, 3)]
    helium_dates = [boundary_date, datetime.date(2020, 1, 2)]

    proton = pd.DataFrame({"time_start": proton_dates, "flux": [1.0, 2.0, 3.0]})
    helium = pd.DataFrame({"time_start": helium_dates, "flux": [10.0, 20.0]})

    result = align_daily_series({"proton": proton, "helium": helium}, join="inner")

    assert boundary_date in set(result.aligned_df["time_start"])
    assert result.date_range[0] == boundary_date


def test_daily_period_boundary_end():
    """Last date in the shared range is correctly included in inner join."""
    end_date = datetime.date(2020, 12, 31)
    proton_dates = [datetime.date(2020, 12, 29), datetime.date(2020, 12, 30), end_date]
    helium_dates = [datetime.date(2020, 12, 30), end_date]

    proton = pd.DataFrame({"time_start": proton_dates, "flux": [1.0, 2.0, 3.0]})
    helium = pd.DataFrame({"time_start": helium_dates, "flux": [20.0, 30.0]})

    result = align_daily_series({"proton": proton, "helium": helium}, join="inner")

    assert end_date in set(result.aligned_df["time_start"])
    assert result.date_range[1] == end_date


def test_daily_records_dropped_dates():
    """Alignment metadata missing_counts records how many dates each species lost."""
    # Proton has 5 dates, helium has 3 — inner join keeps the 2 that overlap.
    proton_dates = [datetime.date(2022, 4, d) for d in range(1, 6)]
    helium_dates = [datetime.date(2022, 4, 3), datetime.date(2022, 4, 4), datetime.date(2022, 4, 7)]

    proton = pd.DataFrame({"time_start": proton_dates, "flux": range(5)})
    helium = pd.DataFrame({"time_start": helium_dates, "flux": range(3)})

    result = align_daily_series({"proton": proton, "helium": helium}, join="inner")

    # Only Apr 3 and Apr 4 overlap
    assert len(result.aligned_df) == 2
    # In an inner join, missing_counts tracks dates in the aligned set but not
    # in the original — for inner join that is always 0 (no NaN rows created).
    assert result.missing_counts["proton"] == 0
    assert result.missing_counts["helium"] == 0

    # Outer join: all 6 unique dates kept, missing_counts reflect NaN-fills.
    result_outer = align_daily_series(
        {"proton": proton, "helium": helium}, join="outer"
    )
    all_dates = set(proton_dates) | set(helium_dates)
    assert len(result_outer.aligned_df) == len(all_dates)
    # Proton missing Apr 7 => 1 missing
    assert result_outer.missing_counts["proton"] == 1
    # Helium missing Apr 1, 2, 5 => 3 missing
    assert result_outer.missing_counts["helium"] == 3


# ---------------------------------------------------------------------------
# Bartels rotation aligner tests (4 cases)
# ---------------------------------------------------------------------------


def test_bartels_inner_join_aligned():
    """Measurements on dates within the same 27-day rotation group together."""
    # Pick two dates in the same Bartels rotation
    base = BARTELS_EPOCH + datetime.timedelta(days=27 * 100)  # Rotation 101
    dates = [base, base + datetime.timedelta(days=5), base + datetime.timedelta(days=20)]
    df = pd.DataFrame({
        "time_start": [d.isoformat() for d in dates],
        "flux": [1.0, 2.0, 3.0],
    })

    grouped = align_bartels_rotation(df)

    assert len(grouped) == 1
    rotation_num = list(grouped.keys())[0]
    assert rotation_num == 101
    assert len(grouped[rotation_num]) == 3


def test_bartels_period_boundary_27day():
    """Dates on the exact 27-day boundary belong to the new rotation."""
    # Day 0 of rotation 1 is the epoch itself
    day_26 = BARTELS_EPOCH + datetime.timedelta(days=26)  # Last day of rotation 1
    day_27 = BARTELS_EPOCH + datetime.timedelta(days=27)  # First day of rotation 2

    df = pd.DataFrame({
        "time_start": [day_26.isoformat(), day_27.isoformat()],
        "flux": [1.0, 2.0],
    })

    grouped = align_bartels_rotation(df)

    assert len(grouped) == 2
    assert 1 in grouped  # day_26 => rotation 1
    assert 2 in grouped  # day_27 => rotation 2
    assert len(grouped[1]) == 1
    assert len(grouped[2]) == 1


def test_bartels_multi_species_merge():
    """Multiple species across partially overlapping rotations group correctly."""
    # Rotation 200 starts at epoch + 27*199 days
    rot_200_start = BARTELS_EPOCH + datetime.timedelta(days=27 * 199)
    rot_201_start = BARTELS_EPOCH + datetime.timedelta(days=27 * 200)
    rot_202_start = BARTELS_EPOCH + datetime.timedelta(days=27 * 201)

    # Proton covers rotations 200 and 201
    proton_dates = [rot_200_start, rot_200_start + datetime.timedelta(days=10), rot_201_start]
    # Helium covers rotations 201 and 202
    helium_dates = [rot_201_start + datetime.timedelta(days=5), rot_202_start]

    combined_df = pd.DataFrame({
        "time_start": [d.isoformat() for d in proton_dates + helium_dates],
        "flux": [1.0, 2.0, 3.0, 4.0, 5.0],
        "species": ["proton", "proton", "proton", "helium", "helium"],
    })

    grouped = align_bartels_rotation(combined_df)

    assert set(grouped.keys()) == {200, 201, 202}
    assert len(grouped[200]) == 2  # 2 proton measurements
    assert len(grouped[201]) == 2  # 1 proton + 1 helium
    assert len(grouped[202]) == 1  # 1 helium


def test_bartels_records_contributing_tables():
    """Grouped result tracks which rotation numbers received data."""
    rot_50_start = BARTELS_EPOCH + datetime.timedelta(days=27 * 49)
    rot_51_start = BARTELS_EPOCH + datetime.timedelta(days=27 * 50)
    rot_53_start = BARTELS_EPOCH + datetime.timedelta(days=27 * 52)

    dates = [rot_50_start, rot_51_start, rot_53_start]
    df = pd.DataFrame({
        "time_start": [d.isoformat() for d in dates],
        "flux": [1.0, 2.0, 3.0],
    })

    grouped = align_bartels_rotation(df)

    # contributing_tables = the rotation keys in the result dict
    contributing_rotations = set(grouped.keys())
    assert contributing_rotations == {50, 51, 53}
    # Rotation 52 is absent — no data in that 27-day window
    assert 52 not in contributing_rotations
