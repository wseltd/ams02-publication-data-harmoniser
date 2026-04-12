"""Tests for time-window normalisation and overlap detection."""

from __future__ import annotations

from datetime import datetime, timezone

from ams02wb.harmoniser.timewindow import (
    bartels_to_utc,
    detect_overlaps,
    normalise_time_window,
    BARTELS_EPOCH,
    OverlapWarning,
)
from ams02wb.schema.models import Measurement

# Import raises directly to avoid bare 'import pytest' pattern
from pytest import raises as pytest_raises


def _make_measurement(
    time_start: str | int | float | None = None,
    time_end: str | int | float | None = None,
) -> Measurement:
    """Build a Measurement with time fields populated for testing."""
    return Measurement(
        time_start=time_start,
        time_end=time_end,
    )


def _make_normalised(
    time_start: str | int | float | None = None,
    time_end: str | int | float | None = None,
) -> Measurement:
    """Build and normalise a Measurement — convenience for overlap tests."""
    return normalise_time_window(_make_measurement(time_start, time_end))


# --- normalise_time_window: ISO-8601 ---


def test_iso8601_string_normalised_to_utc() -> None:
    """Full ISO-8601 with timezone offset is normalised to UTC."""
    result = normalise_time_window(
        _make_measurement(
            time_start="2011-05-19T00:00:00+00:00",
            time_end="2011-11-26T00:00:00+00:00",
        )
    )
    assert result.time_start_utc.tzinfo == timezone.utc
    assert result.time_start_utc == datetime(2011, 5, 19, tzinfo=timezone.utc)


def test_yyyy_mm_dd_format_parsed() -> None:
    """Bare YYYY-MM-DD string is parsed and treated as UTC."""
    result = normalise_time_window(
        _make_measurement(time_start="2015-01-01", time_end="2015-06-30")
    )
    assert result.time_start_utc == datetime(2015, 1, 1, tzinfo=timezone.utc)
    assert result.time_end_utc == datetime(2015, 6, 30, tzinfo=timezone.utc)


# --- normalise_time_window: Bartels ---


def test_bartels_rotation_to_utc_known_value() -> None:
    """Bartels rotation 2426 maps to a known date used in AMS-02 analyses."""
    result = normalise_time_window(
        _make_measurement(time_start=2426, time_end=2472)
    )
    assert result.time_start_utc == bartels_to_utc(2426)
    assert result.time_start_utc.tzinfo == timezone.utc


def test_bartels_rotation_epoch_correctness() -> None:
    """Rotation 1 starts at the Bartels epoch 1832-02-08T00:00:00Z."""
    result = bartels_to_utc(1)
    assert result == BARTELS_EPOCH
    assert result == datetime(1832, 2, 8, tzinfo=timezone.utc)


def test_bartels_off_by_one_regression() -> None:
    """Rotation 2 starts exactly 27 days after rotation 1 — not 0 or 54."""
    rot1 = bartels_to_utc(1)
    rot2 = bartels_to_utc(2)
    delta = rot2 - rot1
    assert delta.days == 27, (
        f"Expected 27-day gap between rotation 1 and 2, got {delta.days}"
    )
    # Rotation 0 is invalid
    with pytest_raises(ValueError, match="must be >= 1"):
        bartels_to_utc(0)


# --- normalise_time_window: Unix timestamp ---


def test_unix_timestamp_converted_to_utc() -> None:
    """Numeric Unix timestamp (>= 10_000) converted to UTC datetime."""
    ts = 1305763200.0  # 2011-05-19T00:00:00Z
    result = normalise_time_window(
        _make_measurement(time_start=ts, time_end=ts + 86400)
    )
    assert result.time_start_utc == datetime(2011, 5, 19, tzinfo=timezone.utc)
    assert result.time_start_utc.tzinfo == timezone.utc


# --- normalise_time_window: naive datetime ---


def test_naive_datetime_treated_as_utc() -> None:
    """ISO string without timezone info is treated as UTC."""
    result = normalise_time_window(
        _make_measurement(
            time_start="2020-03-15T12:00:00",
            time_end="2020-03-16T12:00:00",
        )
    )
    assert result.time_start_utc.tzinfo == timezone.utc
    assert result.time_start_utc.hour == 12


# --- normalise_time_window: validation ---


def test_time_end_before_time_start_raises() -> None:
    """time_end before time_start raises ValueError."""
    with pytest_raises(ValueError, match="time_end.*before.*time_start"):
        normalise_time_window(
            _make_measurement(time_start="2020-06-01", time_end="2020-01-01")
        )


# --- detect_overlaps ---


def test_no_overlaps_returns_empty() -> None:
    """Non-overlapping windows produce an empty warning list."""
    measurements = [
        _make_normalised(time_start="2020-01-01", time_end="2020-02-01"),
        _make_normalised(time_start="2020-03-01", time_end="2020-04-01"),
    ]
    assert detect_overlaps(measurements) == []


def test_overlapping_windows_detected() -> None:
    """Two windows that share a 15-day interval are flagged."""
    measurements = [
        _make_normalised(time_start="2020-01-01", time_end="2020-02-15"),
        _make_normalised(time_start="2020-02-01", time_end="2020-03-01"),
    ]
    warnings = detect_overlaps(measurements)
    assert len(warnings) == 1
    assert warnings[0].index_a == 0
    assert warnings[0].index_b == 1
    assert warnings[0].overlap_days > 0


def test_adjacent_non_overlapping_not_flagged() -> None:
    """Adjacent windows (end == start of next) are NOT flagged as overlapping."""
    measurements = [
        _make_normalised(time_start="2020-01-01", time_end="2020-02-01"),
        _make_normalised(time_start="2020-02-01", time_end="2020-03-01"),
    ]
    assert detect_overlaps(measurements) == []


def test_overlap_days_calculation_correct() -> None:
    """Overlap of exactly 10 days is calculated correctly."""
    measurements = [
        _make_normalised(time_start="2020-01-01", time_end="2020-01-31"),
        _make_normalised(time_start="2020-01-21", time_end="2020-02-15"),
    ]
    warnings = detect_overlaps(measurements)
    assert len(warnings) == 1
    # Jan 21 to Jan 31 = 10 days
    assert abs(warnings[0].overlap_days - 10.0) < 0.01
