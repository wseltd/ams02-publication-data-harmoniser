"""Tests for Bartels-rotation alignment.

Heavy focus on epoch/boundary arithmetic — off-by-one errors at rotation
boundaries are the primary risk surface for this module.
"""

from __future__ import annotations

import datetime

import pandas as pd

from ams02wb.alignment.bartels import (
    BARTELS_EPOCH,
    align_bartels_rotation,
    bartels_rotation_number,
    bartels_rotation_start,
)


class TestBartelsRotationNumber:
    """Boundary arithmetic is the risk — test rotation edges thoroughly."""

    def test_epoch_date_returns_rotation_one(self) -> None:
        """The epoch itself is day 0 of rotation 1."""
        assert bartels_rotation_number(BARTELS_EPOCH) == 1

    def test_last_day_of_rotation_one_is_still_one(self) -> None:
        """Day 26 (the 27th day) is the last day of rotation 1."""
        last_day = BARTELS_EPOCH + datetime.timedelta(days=26)
        assert bartels_rotation_number(last_day) == 1

    def test_first_day_of_rotation_two_boundary(self) -> None:
        """Day 27 is exactly the start of rotation 2 — not still rotation 1."""
        boundary = BARTELS_EPOCH + datetime.timedelta(days=27)
        assert bartels_rotation_number(boundary) == 2

    def test_known_modern_date_matches_published_table(self) -> None:
        """Cross-check against the epoch definition for a modern date.

        Rotation 2544 starts at day 27*2543 = 68661 after epoch, which is
        2020-02-03. This is a direct arithmetic check against the epoch.
        """
        assert bartels_rotation_start(2544) == datetime.date(2020, 2, 3)
        assert bartels_rotation_number(datetime.date(2020, 2, 3)) == 2544
        # The day before belongs to rotation 2543
        assert bartels_rotation_number(datetime.date(2020, 2, 2)) == 2543

    def test_roundtrip_rotation_number_to_start_date(self) -> None:
        """start(number(date)) should return the first day of that rotation."""
        test_date = datetime.date(2023, 6, 15)
        rot = bartels_rotation_number(test_date)
        start = bartels_rotation_start(rot)
        # The start date must be on or before the test date
        assert start <= test_date
        # And the test date must be within 27 days of the start
        assert (test_date - start).days < 27
        # And the rotation number of the start date must match
        assert bartels_rotation_number(start) == rot

    def test_boundary_measurement_belongs_to_new_rotation(self) -> None:
        """A date exactly divisible by 27 days from epoch starts a new rotation."""
        # Rotation 100 starts at day 27*99 = 2673
        rot_100_start = BARTELS_EPOCH + datetime.timedelta(days=27 * 99)
        assert bartels_rotation_number(rot_100_start) == 100
        # The day before is still rotation 99
        day_before = rot_100_start - datetime.timedelta(days=1)
        assert bartels_rotation_number(day_before) == 99


class TestAlignBartelsRotation:
    """Integration tests for the DataFrame grouping function."""

    def test_align_groups_measurements_by_rotation(self) -> None:
        """Measurements spanning two rotations should produce two groups."""
        start = bartels_rotation_start(2542)
        dates = [
            str(start),
            str(start + datetime.timedelta(days=10)),
            str(start + datetime.timedelta(days=27)),  # Next rotation
            str(start + datetime.timedelta(days=30)),
        ]
        df = pd.DataFrame({"time_start": dates, "value": [1.0, 2.0, 3.0, 4.0]})
        result = align_bartels_rotation(df)

        assert len(result) == 2
        assert 2542 in result
        assert 2543 in result
        assert len(result[2542]) == 2
        assert len(result[2543]) == 2

    def test_align_empty_dataframe_returns_empty_dict(self) -> None:
        """An empty DataFrame should yield an empty dict, not raise."""
        df = pd.DataFrame(columns=["time_start", "value"])
        result = align_bartels_rotation(df)
        assert result == {}

    def test_align_single_rotation_returns_one_key(self) -> None:
        """All measurements within one rotation produce exactly one key."""
        start = bartels_rotation_start(2600)
        dates = [
            str(start),
            str(start + datetime.timedelta(days=5)),
            str(start + datetime.timedelta(days=26)),
        ]
        df = pd.DataFrame({"time_start": dates, "value": [10.0, 20.0, 30.0]})
        result = align_bartels_rotation(df)

        assert list(result.keys()) == [2600]
        assert len(result[2600]) == 3
