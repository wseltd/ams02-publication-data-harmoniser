"""Daily time-series alignment for AMS-02 species data.

Aligns multiple species DataFrames on a shared daily ``time_start`` date
column, producing a wide-form result with species-suffixed value columns
and diagnostics about missing data introduced by the join.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Literal

import pandas as pd


@dataclass(frozen=True)
class DailyAlignedResult:
    """Result of aligning daily species time-series.

    Attributes:
        aligned_df: Wide-form DataFrame with a shared ``time_start`` date
            column and species-suffixed value columns.
        missing_counts: Mapping of species name to count of NaN-filled rows
            introduced by the join (rows present in the aligned result but
            absent in the original species data).
        join_type: The join strategy used (``'inner'`` or ``'outer'``).
        date_range: ``(min_date, max_date)`` extent of the aligned result.
    """

    aligned_df: pd.DataFrame
    missing_counts: dict[str, int]
    join_type: str
    date_range: tuple[datetime.date, datetime.date]


def _align_single_species(
    name: str,
    species_frames: dict[str, pd.DataFrame],
    join: Literal["inner", "outer"],
) -> DailyAlignedResult:
    """Return an aligned result for a single-species input (no merge needed)."""
    df = species_frames[name].copy()
    value_cols = [c for c in df.columns if c != "time_start"]
    rename_map = {c: f"{c}_{name}" for c in value_cols}
    df = df.rename(columns=rename_map)
    missing_counts: dict[str, int] = {name: 0}
    dates = df["time_start"]
    date_range = (min(dates), max(dates))
    return DailyAlignedResult(
        aligned_df=df,
        missing_counts=missing_counts,
        join_type=join,
        date_range=date_range,
    )


def align_daily_series(
    species_frames: dict[str, pd.DataFrame],
    *,
    join: Literal["inner", "outer"] = "inner",
) -> DailyAlignedResult:
    """Align multiple species DataFrames on their ``time_start`` date column.

    Each input DataFrame must contain at least a ``time_start`` column with
    ``datetime.date`` values.  All other columns are preserved and suffixed
    with ``_<species_name>`` in the output.

    Args:
        species_frames: Mapping of species name to its daily DataFrame.
            Must contain at least one entry.
        join: Join strategy — ``'inner'`` keeps only dates shared by all
            species, ``'outer'`` keeps all dates and fills gaps with NaN.

    Returns:
        A ``DailyAlignedResult`` with the merged wide-form DataFrame and
        join diagnostics.

    Raises:
        ValueError: If *species_frames* is empty or *join* is not
            ``'inner'``/``'outer'``.
    """
    if not species_frames:
        raise ValueError("species_frames must contain at least one entry")
    if join not in ("inner", "outer"):
        raise ValueError(f"join must be 'inner' or 'outer', got {join!r}")

    species_names = list(species_frames.keys())

    # Single-species fast path: no merge needed.
    if len(species_names) == 1:
        return _align_single_species(species_names[0], species_frames, join)

    # Multi-species: iterative merge on time_start.
    first_name = species_names[0]
    merged = species_frames[first_name].copy()
    value_cols = [c for c in merged.columns if c != "time_start"]
    merged = merged.rename(columns={c: f"{c}_{first_name}" for c in value_cols})
    original_lengths = {first_name: len(species_frames[first_name])}

    for name in species_names[1:]:
        right = species_frames[name].copy()
        right_value_cols = [c for c in right.columns if c != "time_start"]
        right = right.rename(columns={c: f"{c}_{name}" for c in right_value_cols})
        original_lengths[name] = len(species_frames[name])
        merged = merged.merge(right, on="time_start", how=join)

    merged = merged.sort_values("time_start").reset_index(drop=True)

    # Compute missing counts: rows in the aligned result that were not in the
    # original species data (i.e. dates filled with NaN by an outer join).
    aligned_dates = set(merged["time_start"])
    missing_counts: dict[str, int] = {}
    for name in species_names:
        original_dates = set(species_frames[name]["time_start"])
        missing_counts[name] = len(aligned_dates - original_dates)

    if merged.empty:
        # Disjoint inner join: no dates overlap.
        date_range = (datetime.date.min, datetime.date.min)
    else:
        dates = merged["time_start"]
        date_range = (min(dates), max(dates))

    return DailyAlignedResult(
        aligned_df=merged,
        missing_counts=missing_counts,
        join_type=join,
        date_range=date_range,
    )
