"""Bartels-rotation aligner for AMS-02 time-series data.

Bins measurements into 27-day Bartels rotation periods using fixed-epoch
integer arithmetic. The Bartels rotation system is a well-defined numbering
convention — no astronomical ephemeris is needed.

Chose fixed-epoch arithmetic over ephemeris lookup because Bartels rotations
are a pure calendrical convention (27-day periods counted from a fixed date),
not an astronomical observation.
"""

from __future__ import annotations

import datetime

import pandas as pd

# Bartels rotation 1, day 1. All rotation numbers are counted from this date.
BARTELS_EPOCH = datetime.date(1832, 2, 8)

_ROTATION_DAYS = 27


def bartels_rotation_number(date: datetime.date) -> int:
    """Compute the Bartels rotation number for a calendar date.

    Rotation 1 spans days 0–26 after the epoch (27 days total).
    A date exactly on a 27-day boundary belongs to the NEW rotation.

    Args:
        date: Calendar date to classify.

    Returns:
        Positive integer rotation number (1-based).

    Raises:
        ValueError: If *date* is before the Bartels epoch.
    """
    delta_days = (date - BARTELS_EPOCH).days
    if delta_days < 0:
        raise ValueError(
            f"Date {date} is before the Bartels epoch ({BARTELS_EPOCH})"
        )
    return 1 + delta_days // _ROTATION_DAYS


def bartels_rotation_start(rotation: int) -> datetime.date:
    """Return the start date (first day) of the given Bartels rotation.

    Args:
        rotation: Positive integer rotation number (1-based).

    Returns:
        The calendar date on which the rotation begins.

    Raises:
        ValueError: If *rotation* is less than 1.
    """
    if rotation < 1:
        raise ValueError(f"Rotation number must be >= 1, got {rotation}")
    return BARTELS_EPOCH + datetime.timedelta(days=_ROTATION_DAYS * (rotation - 1))


def align_bartels_rotation(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Group measurements by Bartels rotation number.

    The input DataFrame must contain a ``time_start`` column with values
    parseable as dates (strings in ISO format, or datetime-like objects).

    Args:
        df: DataFrame with a ``time_start`` column from the canonical schema.

    Returns:
        Dict mapping Bartels rotation number to the sub-DataFrame of
        measurements falling within that 27-day window.

    Raises:
        ValueError: If *df* does not contain a ``time_start`` column.
    """
    if df.empty:
        return {}

    if "time_start" not in df.columns:
        raise ValueError("DataFrame must contain a 'time_start' column")

    dates = pd.to_datetime(df["time_start"]).dt.date
    rotations = dates.map(bartels_rotation_number)

    result: dict[int, pd.DataFrame] = {}
    for rot_num, group_df in df.groupby(rotations):
        result[int(rot_num)] = group_df.reset_index(drop=True)

    return result
