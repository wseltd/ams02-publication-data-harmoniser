"""Join diagnostics for aligned AMS-02 species datasets.

Computes coverage fraction, gap histogram, and overlap count for an
aligned wide-form DataFrame produced by ``DailyAlignedResult.aligned_df``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class JoinDiagnostics:
    """Diagnostics for an aligned multi-species DataFrame.

    Attributes:
        coverage_fraction: Fraction of non-NaN rows per species,
            keyed by species name.
        gap_histogram: List of consecutive-NaN run lengths per species,
            in order of occurrence.  E.g. ``[2, 1, 5]`` means three
            gaps of lengths 2, 1, and 5 from left to right.
        overlap_count: Number of rows where ALL requested species have
            non-NaN values.
    """

    coverage_fraction: dict[str, float]
    gap_histogram: dict[str, list[int]]
    overlap_count: int


def _compute_gap_runs(nan_mask: pd.Series) -> list[int]:
    """Return sorted list of consecutive-True run lengths in *nan_mask*.

    Uses ``itertools.groupby`` on the boolean mask — no pandas rolling
    or window functions.  Chose groupby over pandas run-length encoding
    because the mask is already a simple boolean sequence and groupby
    has zero allocation overhead.
    """
    runs = []
    for is_nan, group in itertools.groupby(nan_mask):
        if is_nan:
            runs.append(sum(1 for _ in group))
    return runs


def compute_join_diagnostics(
    aligned_df: pd.DataFrame,
    species_list: list[str],
) -> JoinDiagnostics:
    """Compute join diagnostics for an aligned species DataFrame.

    Args:
        aligned_df: Wide-form DataFrame with ``y_value_{species}``
            columns, as produced by ``DailyAlignedResult.aligned_df``.
        species_list: Species names to diagnose.  Each must have a
            corresponding ``y_value_{species}`` column in *aligned_df*.

    Returns:
        A ``JoinDiagnostics`` with per-species coverage and gap info,
        plus an overlap count across all requested species.

    Raises:
        ValueError: If *species_list* is empty or a required column
            is missing from *aligned_df*.
    """
    if not species_list:
        raise ValueError("species_list must contain at least one species")

    total_rows = len(aligned_df)

    coverage: dict[str, float] = {}
    gaps: dict[str, list[int]] = {}

    for species in species_list:
        col = f"y_value_{species}"
        if col not in aligned_df.columns:
            raise ValueError(
                f"Column {col!r} not found in aligned_df. "
                f"Available columns: {list(aligned_df.columns)}"
            )

        nan_mask = aligned_df[col].isna()
        non_nan_count = total_rows - int(nan_mask.sum())

        # Coverage: fraction of rows with valid data.
        coverage[species] = non_nan_count / total_rows if total_rows > 0 else 0.0

        # Gap histogram: consecutive NaN run lengths, in occurrence order.
        gaps[species] = _compute_gap_runs(nan_mask)

    # Overlap: rows where ALL species have non-NaN values.
    if total_rows == 0:
        overlap = 0
    else:
        all_valid = pd.Series(True, index=aligned_df.index)
        for species in species_list:
            all_valid = all_valid & aligned_df[f"y_value_{species}"].notna()
        overlap = int(all_valid.sum())

    return JoinDiagnostics(
        coverage_fraction=coverage,
        gap_histogram=gaps,
        overlap_count=overlap,
    )
