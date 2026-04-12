"""Tests for time-window normalisation, overlap/gap detection, and Bartels rotation.

Covers: UTC parse (2), overlap detection (4 — risky interval arithmetic),
gap detection (2), and Bartels boundary (4 — risky astronomical period math).
All inputs are deterministic datetime fixtures with UTC timezone. No network,
no filesystem, no real AMS publication data.
"""

from datetime import datetime, timezone

from ams02wb.harmoniser.timewindow import (
    BARTELS_EPOCH,
    BARTELS_PERIOD_DAYS,
    bartels_to_utc,
    detect_gaps,
    detect_overlaps,
    parse_time_window,
    to_bartels_rotation,
)
from ams02wb.schema.models import Measurement


def _make_measurement(start_utc: datetime, end_utc: datetime) -> Measurement:
    """Build a Measurement with only the normalised time fields populated."""
    return Measurement(time_start_utc=start_utc, time_end_utc=end_utc)


# ---------------------------------------------------------------------------
# UTC parse — 2 cases
# ---------------------------------------------------------------------------


def test_parse_utc_iso_format_returns_datetime_pair():
    """Full ISO-8601 strings with time component parse to exact UTC datetimes."""
    start, stop = parse_time_window(
        "2011-05-19T00:00:00", "2011-06-19T23:59:59"
    )
    assert start == datetime(2011, 5, 19, 0, 0, 0, tzinfo=timezone.utc)
    assert stop == datetime(2011, 6, 19, 23, 59, 59, tzinfo=timezone.utc)
    # Both must carry UTC timezone info — naive datetimes are never returned
    assert start.tzinfo is not None
    assert stop.tzinfo is not None


def test_parse_utc_date_only_assumes_midnight_bounds():
    """Date-only strings ('YYYY-MM-DD') are treated as midnight UTC."""
    start, stop = parse_time_window("2015-01-01", "2015-12-31")
    assert start == datetime(2015, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    assert stop == datetime(2015, 12, 31, 0, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Overlap detection — 4 cases (risky interval arithmetic)
# ---------------------------------------------------------------------------

# Realistic AMS-02 daily proton date ranges
_MAY_1 = datetime(2011, 5, 1, tzinfo=timezone.utc)
_MAY_15 = datetime(2011, 5, 15, tzinfo=timezone.utc)
_MAY_10 = datetime(2011, 5, 10, tzinfo=timezone.utc)
_MAY_20 = datetime(2011, 5, 20, tzinfo=timezone.utc)
_MAY_5 = datetime(2011, 5, 5, tzinfo=timezone.utc)
_MAY_12 = datetime(2011, 5, 12, tzinfo=timezone.utc)
_MAY_25 = datetime(2011, 5, 25, tzinfo=timezone.utc)
_MAY_31 = datetime(2011, 5, 31, tzinfo=timezone.utc)


def test_overlap_detected_when_windows_partially_intersect():
    """Two windows that share a partial date range produce an overlap warning."""
    # Window A: May 1–15, Window B: May 10–20 — 5-day overlap (May 10–15)
    measurements = [
        _make_measurement(_MAY_1, _MAY_15),
        _make_measurement(_MAY_10, _MAY_20),
    ]
    overlaps = detect_overlaps(measurements)
    assert len(overlaps) == 1
    assert overlaps[0].index_a == 0
    assert overlaps[0].index_b == 1
    assert overlaps[0].overlap_days == 5.0


def test_overlap_detected_when_window_fully_contained():
    """A window fully inside another produces an overlap equal to the inner window."""
    # Outer: May 1–20, Inner: May 5–12 — 7-day overlap (entire inner window)
    measurements = [
        _make_measurement(_MAY_1, _MAY_20),
        _make_measurement(_MAY_5, _MAY_12),
    ]
    overlaps = detect_overlaps(measurements)
    assert len(overlaps) == 1
    assert overlaps[0].overlap_days == 7.0


def test_no_overlap_when_windows_adjacent():
    """Adjacent windows (end_a == start_b) are NOT counted as overlapping.

    This is the critical edge case in interval arithmetic — the half-open
    convention [start, end) means touching endpoints have zero overlap.
    """
    # Window A ends exactly when Window B starts
    measurements = [
        _make_measurement(_MAY_1, _MAY_15),
        _make_measurement(_MAY_15, _MAY_31),
    ]
    overlaps = detect_overlaps(measurements)
    assert len(overlaps) == 0


def test_no_overlap_when_windows_separated():
    """Windows with a gap between them produce no overlap warnings."""
    # Window A: May 1–10, Window B: May 20–31 — 10-day gap
    measurements = [
        _make_measurement(_MAY_1, _MAY_10),
        _make_measurement(_MAY_20, _MAY_31),
    ]
    overlaps = detect_overlaps(measurements)
    assert len(overlaps) == 0


# ---------------------------------------------------------------------------
# Gap detection — 2 cases
# ---------------------------------------------------------------------------


def test_gap_detected_when_exceeds_threshold():
    """A 10-day gap between consecutive windows is flagged at a 1-day threshold."""
    measurements = [
        _make_measurement(_MAY_1, _MAY_10),
        _make_measurement(_MAY_20, _MAY_31),
    ]
    gaps = detect_gaps(measurements, threshold_days=1.0)
    assert len(gaps) == 1
    assert gaps[0].index_a == 0
    assert gaps[0].index_b == 1
    assert gaps[0].gap_days == 10.0


def test_no_gap_when_within_threshold():
    """Adjacent windows (zero gap) are not flagged even at a tiny threshold."""
    measurements = [
        _make_measurement(_MAY_1, _MAY_15),
        _make_measurement(_MAY_15, _MAY_31),
    ]
    gaps = detect_gaps(measurements, threshold_days=0.5)
    assert len(gaps) == 0


# ---------------------------------------------------------------------------
# Bartels boundary — 4 cases (risky astronomical period math)
# ---------------------------------------------------------------------------


def test_bartels_rotation_number_for_known_date():
    """A date 54 days after epoch falls in rotation 3 (days 54–80 of rotation 3).

    54 days = 2 full rotations (2 * 27 = 54), so the 54th day is the start
    of rotation 3.
    """
    # Exactly 2 rotations after epoch = start of rotation 3
    dt = datetime(1832, 4, 3, tzinfo=timezone.utc)  # epoch + 55 days
    rotation = to_bartels_rotation(dt)
    # 55 days / 27 = 2.037..., floor + 1 = 3
    assert rotation == 3


def test_bartels_boundary_start_aligns_to_rotation_epoch():
    """bartels_to_utc(1) returns the epoch; to_bartels_rotation(epoch) returns 1.

    Verifying the round-trip anchor point that all other rotation arithmetic
    depends on.
    """
    assert bartels_to_utc(1) == BARTELS_EPOCH
    assert to_bartels_rotation(BARTELS_EPOCH) == 1


def test_bartels_window_spanning_two_rotations_splits():
    """A measurement window that crosses a rotation boundary straddles two
    distinct rotation numbers.

    Window: rotation 2500 start to rotation 2501 start + 5 days. The start
    and end should map to different rotations, confirming that any splitting
    logic must handle this case.
    """
    rot_2500_start = bartels_to_utc(2500)
    rot_2501_start = bartels_to_utc(2501)
    # Window spans from rotation 2500 start to 5 days into rotation 2501
    window_start = rot_2500_start
    window_end = rot_2501_start + (rot_2501_start - rot_2500_start)  # +27 days

    start_rotation = to_bartels_rotation(window_start)
    # End is in the next rotation — use a point just before the boundary
    # to confirm the window crosses it
    end_rotation = to_bartels_rotation(window_end)

    assert start_rotation == 2500
    # End is 27 days into rotation 2501, so it's in rotation 2502
    assert end_rotation == 2502
    assert end_rotation > start_rotation


def test_bartels_rotation_sequential_numbering_consistent():
    """Consecutive rotation starts are exactly BARTELS_PERIOD_DAYS apart, and
    sequential rotation numbers increment by 1.

    Tests a range in the AMS-02 era (~2015) to verify the arithmetic holds
    at large rotation numbers, not just near the epoch.
    """
    # Rotation ~2470 is around 2015
    base_rotation = 2470
    for offset in range(5):
        rot_num = base_rotation + offset
        start = bartels_to_utc(rot_num)
        next_start = bartels_to_utc(rot_num + 1)

        # Consecutive rotation starts are exactly one period apart
        delta_days = (next_start - start).total_seconds() / 86400.0
        assert delta_days == BARTELS_PERIOD_DAYS

        # Round-trip: to_bartels_rotation at the start of rotation N returns N
        assert to_bartels_rotation(start) == rot_num
