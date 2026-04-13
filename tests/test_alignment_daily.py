"""Tests for daily time-series alignment (ams02wb.alignment.daily)."""

from __future__ import annotations

import datetime

import pandas as pd

from ams02wb.alignment.daily import DailyAlignedResult, align_daily_series


def _make_df(
    dates: list[datetime.date],
    value_col: str = "y_value",
    values: list | None = None,
    extra_cols: dict[str, list] | None = None,
) -> pd.DataFrame:
    """Helper: build a simple species DataFrame with time_start + values."""
    data: dict[str, list] = {
        "time_start": dates,
        value_col: values if values is not None else list(range(len(dates))),
    }
    if extra_cols:
        data.update(extra_cols)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Join edge cases (3x concentration here — this is the tricky part)
# ---------------------------------------------------------------------------


class TestJoinEdgeCases:
    """Inner/outer join semantics and boundary conditions."""

    def test_inner_join_overlapping_dates_returns_intersection(self) -> None:
        """Inner join keeps only the 3 overlapping dates."""
        dates_a = [datetime.date(2020, 1, d) for d in range(1, 6)]  # Jan 1-5
        dates_b = [datetime.date(2020, 1, d) for d in range(3, 8)]  # Jan 3-7

        result = align_daily_series(
            {"proton": _make_df(dates_a), "helium": _make_df(dates_b)},
            join="inner",
        )

        assert isinstance(result, DailyAlignedResult)
        assert result.join_type == "inner"
        aligned_dates = sorted(result.aligned_df["time_start"])
        expected = [datetime.date(2020, 1, d) for d in (3, 4, 5)]
        assert aligned_dates == expected
        # Inner join: no NaN rows for either species.
        assert result.missing_counts["proton"] == 0
        assert result.missing_counts["helium"] == 0

    def test_outer_join_fills_missing_with_nan(self) -> None:
        """Outer join keeps all 7 dates; non-overlapping cells are NaN."""
        dates_a = [datetime.date(2020, 1, d) for d in range(1, 6)]  # Jan 1-5
        dates_b = [datetime.date(2020, 1, d) for d in range(3, 8)]  # Jan 3-7

        result = align_daily_series(
            {"proton": _make_df(dates_a), "helium": _make_df(dates_b)},
            join="outer",
        )

        assert result.join_type == "outer"
        assert len(result.aligned_df) == 7
        # Helium missing Jan 1-2, proton missing Jan 6-7.
        assert result.missing_counts["proton"] == 2
        assert result.missing_counts["helium"] == 2
        # Verify actual NaN presence in non-overlapping cells.
        df = result.aligned_df
        jan1_row = df[df["time_start"] == datetime.date(2020, 1, 1)]
        assert pd.isna(jan1_row["y_value_helium"].iloc[0])
        jan7_row = df[df["time_start"] == datetime.date(2020, 1, 7)]
        assert pd.isna(jan7_row["y_value_proton"].iloc[0])

    def test_disjoint_dates_inner_join_returns_empty(self) -> None:
        """Inner join on non-overlapping date ranges produces empty result."""
        dates_a = [datetime.date(2020, 1, d) for d in range(1, 4)]  # Jan 1-3
        dates_b = [datetime.date(2020, 6, d) for d in range(1, 4)]  # Jun 1-3

        result = align_daily_series(
            {"proton": _make_df(dates_a), "helium": _make_df(dates_b)},
            join="inner",
        )

        assert len(result.aligned_df) == 0
        assert result.join_type == "inner"

    def test_identical_date_ranges_produce_zero_missing(self) -> None:
        """When all species share the same dates, missing_counts are all 0."""
        dates = [datetime.date(2020, 1, d) for d in range(1, 6)]

        result = align_daily_series(
            {"proton": _make_df(dates), "helium": _make_df(dates)},
            join="outer",
        )

        assert result.missing_counts["proton"] == 0
        assert result.missing_counts["helium"] == 0
        assert len(result.aligned_df) == 5


# ---------------------------------------------------------------------------
# Remaining functional tests
# ---------------------------------------------------------------------------


def test_missing_counts_reflect_nan_rows_from_outer_join() -> None:
    """Missing counts exactly equal the number of NaN-introducing rows."""
    dates_a = [datetime.date(2020, 1, d) for d in range(1, 11)]  # 10 days
    dates_b = [datetime.date(2020, 1, d) for d in range(6, 11)]  # 5 days

    result = align_daily_series(
        {"alpha": _make_df(dates_a), "beta": _make_df(dates_b)},
        join="outer",
    )

    # beta is missing 5 days (Jan 1-5); alpha has full coverage.
    assert result.missing_counts["alpha"] == 0
    assert result.missing_counts["beta"] == 5
    # Verify NaN count matches missing_counts for beta.
    nan_count = result.aligned_df["y_value_beta"].isna().sum()
    assert nan_count == result.missing_counts["beta"]


def test_single_species_passthrough_unchanged() -> None:
    """A single species passes through with suffixed columns, no data loss."""
    dates = [datetime.date(2020, 1, d) for d in range(1, 4)]
    values = [10.0, 20.0, 30.0]
    df_in = _make_df(dates, values=values)

    result = align_daily_series({"proton": df_in}, join="inner")

    assert "y_value_proton" in result.aligned_df.columns
    assert list(result.aligned_df["y_value_proton"]) == values
    assert result.missing_counts["proton"] == 0
    assert len(result.aligned_df) == 3


def test_date_range_tuple_matches_aligned_extent() -> None:
    """date_range reflects the actual min/max dates in the aligned output."""
    dates_a = [datetime.date(2020, 1, d) for d in range(1, 6)]
    dates_b = [datetime.date(2020, 1, d) for d in range(3, 8)]

    result_inner = align_daily_series(
        {"p": _make_df(dates_a), "h": _make_df(dates_b)}, join="inner"
    )
    assert result_inner.date_range == (
        datetime.date(2020, 1, 3),
        datetime.date(2020, 1, 5),
    )

    result_outer = align_daily_series(
        {"p": _make_df(dates_a), "h": _make_df(dates_b)}, join="outer"
    )
    assert result_outer.date_range == (
        datetime.date(2020, 1, 1),
        datetime.date(2020, 1, 7),
    )


def test_provenance_columns_preserved_after_join() -> None:
    """Extra columns (e.g. species_num) survive the join with proper suffix."""
    dates = [datetime.date(2020, 1, d) for d in range(1, 4)]
    df_a = _make_df(dates, extra_cols={"species_num": ["proton"] * 3})
    df_b = _make_df(dates, extra_cols={"species_num": ["helium"] * 3})

    result = align_daily_series({"proton": df_a, "helium": df_b}, join="inner")

    assert "species_num_proton" in result.aligned_df.columns
    assert "species_num_helium" in result.aligned_df.columns
    assert list(result.aligned_df["species_num_proton"]) == ["proton"] * 3
    assert list(result.aligned_df["species_num_helium"]) == ["helium"] * 3
