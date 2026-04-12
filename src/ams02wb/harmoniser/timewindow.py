"""Time-window normalisation and overlap detection for AMS-02 measurements.

Converts heterogeneous time representations (ISO-8601 strings, Bartels
rotation numbers, Unix timestamps) to timezone-aware UTC datetimes.
Detects overlapping time windows across a collection of measurements.

Bartels rotation epoch: 1832-02-08T00:00:00Z, rotation 1 starts at epoch
(not rotation 0). Each rotation is exactly 27 days.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta, timezone

from ams02wb.schema.models import Measurement

# Bartels rotation epoch — rotation 1 starts here.
# Chose 1832-02-08 per the standard Bartels convention used in solar/cosmic-ray
# physics; rotation numbers count from 1, not 0.
BARTELS_EPOCH = datetime(1832, 2, 8, tzinfo=timezone.utc)
BARTELS_PERIOD_DAYS = 27


def bartels_to_utc(rotation: int) -> datetime:
    """Convert a Bartels rotation number to a UTC datetime.

    Args:
        rotation: Bartels rotation number (must be >= 1).

    Returns:
        Timezone-aware UTC datetime for the start of the given rotation.

    Raises:
        ValueError: If rotation is less than 1.
    """
    if rotation < 1:
        raise ValueError(
            f"Bartels rotation must be >= 1, got {rotation}"
        )
    # Rotation 1 maps to the epoch; rotation N starts (N-1)*27 days later.
    return BARTELS_EPOCH + timedelta(days=(rotation - 1) * BARTELS_PERIOD_DAYS)


def _parse_to_utc(value: str | int | float | None) -> datetime | None:
    """Parse a single time value to a timezone-aware UTC datetime.

    Supports:
      - ISO-8601 date strings ("2011-05-19", "2011-05-19T00:00:00Z")
      - Bartels rotation numbers (int, detected by magnitude < 10_000)
      - Unix timestamps (int or float >= 10_000)
      - None passthrough

    Bartels threshold rationale: Bartels rotations are ~2500 in the AMS-02 era;
    Unix timestamps for any date after 1970-01-03 exceed 172_800. A threshold
    of 10_000 cleanly separates the two numeric domains with wide margin.
    """
    if value is None:
        return None
    if isinstance(value, str):
        # Try ISO-8601 with timezone info first, then bare date
        for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                parsed = datetime.strptime(value, fmt)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed
            except ValueError:
                continue
        raise ValueError(
            f"Cannot parse time value as ISO-8601: {value!r}. "
            f"Expected formats: YYYY-MM-DD, YYYY-MM-DDTHH:MM:SS, "
            f"YYYY-MM-DDTHH:MM:SS+HH:MM"
        )
    if isinstance(value, (int, float)):
        if isinstance(value, int) and value < 10_000:
            return bartels_to_utc(value)
        return datetime.fromtimestamp(value, tz=timezone.utc)
    raise TypeError(
        f"Unsupported time value type: {type(value).__name__}. "
        f"Expected str, int, float, or None."
    )


def normalise_time_window(measurement: Measurement) -> Measurement:
    """Return a new Measurement with time_start_utc and time_end_utc set.

    Parses the measurement's time_start and time_end fields (ISO-8601 strings,
    Bartels rotation numbers, or Unix timestamps) into timezone-aware UTC
    datetime strings stored as ISO-8601.

    Args:
        measurement: Measurement with time_start and/or time_end set.

    Returns:
        A new Measurement with time_start_utc and time_end_utc populated.

    Raises:
        ValueError: If time_end is before time_start.
    """
    start = _parse_to_utc(measurement.time_start)
    end = _parse_to_utc(measurement.time_end)

    if start is not None and end is not None and end < start:
        raise ValueError(
            f"time_end ({measurement.time_end}) is before "
            f"time_start ({measurement.time_start})"
        )

    return measurement.model_copy(
        update={
            "time_start_utc": start,
            "time_end_utc": end,
        }
    )


@dataclasses.dataclass(frozen=True)
class OverlapWarning:
    """Warning for overlapping measurement time windows.

    Attributes:
        index_a: Index of the first measurement in the input list.
        index_b: Index of the second measurement in the input list.
        overlap_days: Number of days the two windows overlap.
    """

    index_a: int
    index_b: int
    overlap_days: float


def detect_overlaps(measurements: list[Measurement]) -> list[OverlapWarning]:
    """Detect overlapping time windows in a list of measurements.

    Compares all pairs of measurements that have both time_start_utc and
    time_end_utc set. Adjacent windows (end_a == start_b) are NOT flagged.

    Args:
        measurements: List of measurements with normalised time windows.

    Returns:
        List of OverlapWarning for each pair of overlapping windows.
    """
    warnings: list[OverlapWarning] = []
    # Collect (index, start, end) for measurements with complete windows
    windows: list[tuple[int, datetime, datetime]] = []
    for i, meas in enumerate(measurements):
        if meas.time_start_utc is not None and meas.time_end_utc is not None:
            windows.append((i, meas.time_start_utc, meas.time_end_utc))

    # O(n^2) pairwise — acceptable for typical dataset sizes (dozens to hundreds)
    for ai in range(len(windows)):
        for bi in range(ai + 1, len(windows)):
            idx_a, start_a, end_a = windows[ai]
            idx_b, start_b, end_b = windows[bi]
            # Overlap exists when neither window ends before the other starts
            overlap_start = max(start_a, start_b)
            overlap_end = min(end_a, end_b)
            overlap_delta = overlap_end - overlap_start
            if overlap_delta.total_seconds() > 0:
                warnings.append(
                    OverlapWarning(
                        index_a=idx_a,
                        index_b=idx_b,
                        overlap_days=overlap_delta.total_seconds() / 86400.0,
                    )
                )
    return warnings
