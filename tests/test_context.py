"""Tests for ams02wb.alignment.context — solar/heliospheric context hooks."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ams02wb.alignment.context import ContextSeries, attach_context, load_context_csv


def _sample_df() -> pd.DataFrame:
    """DataFrame with 5 daily timestamps."""
    return pd.DataFrame({
        "time_start": [
            "2015-06-01T00:00:00+00:00",
            "2015-06-02T00:00:00+00:00",
            "2015-06-03T00:00:00+00:00",
            "2015-06-04T00:00:00+00:00",
            "2015-06-05T00:00:00+00:00",
        ],
        "y_value": [100.0, 200.0, 300.0, 400.0, 500.0],
    })


def _sample_context() -> ContextSeries:
    """Context series with 3 values (not matching all dates)."""
    return ContextSeries(
        name="phi_mod",
        description="Solar modulation potential",
        source="https://example.org/modulation",
        times=[
            datetime(2015, 6, 1, tzinfo=timezone.utc),
            datetime(2015, 6, 3, tzinfo=timezone.utc),
            datetime(2015, 6, 5, tzinfo=timezone.utc),
        ],
        values=[500.0, 520.0, 510.0],
    )


def test_attach_context_nearest() -> None:
    """Nearest-time matching fills all rows with closest context value."""
    df = _sample_df()
    ctx = _sample_context()
    result = attach_context(df, ctx, method="nearest")

    assert "phi_mod" in result.columns
    assert len(result) == 5
    # June 1 should match exactly 500, June 2 nearest to 500 or 520
    assert result["phi_mod"].iloc[0] == pytest.approx(500.0)
    assert result["phi_mod"].iloc[2] == pytest.approx(520.0)
    assert result["phi_mod"].iloc[4] == pytest.approx(510.0)
    # No NaN values with nearest
    assert result["phi_mod"].notna().all()


def test_attach_context_interpolate() -> None:
    """Interpolation linearly fills values between context timestamps."""
    df = _sample_df()
    ctx = _sample_context()
    result = attach_context(df, ctx, method="interpolate")

    assert "phi_mod" in result.columns
    # June 1 = 500, June 3 = 520, so June 2 should be ~510
    assert result["phi_mod"].iloc[0] == pytest.approx(500.0)
    assert result["phi_mod"].iloc[1] == pytest.approx(510.0, abs=1.0)
    assert result["phi_mod"].iloc[2] == pytest.approx(520.0)
    assert result["phi_mod"].iloc[4] == pytest.approx(510.0)


def test_attach_context_exact_with_missing() -> None:
    """Exact matching produces NaN for dates without a context value."""
    df = _sample_df()
    ctx = _sample_context()
    result = attach_context(df, ctx, method="exact")

    assert "phi_mod" in result.columns
    # June 1, 3, 5 should have values
    assert result["phi_mod"].iloc[0] == pytest.approx(500.0)
    assert result["phi_mod"].iloc[2] == pytest.approx(520.0)
    assert result["phi_mod"].iloc[4] == pytest.approx(510.0)
    # June 2, 4 should be NaN
    assert pd.isna(result["phi_mod"].iloc[1])
    assert pd.isna(result["phi_mod"].iloc[3])


def test_attach_context_adds_source_column() -> None:
    """Source attribution column is always added."""
    df = _sample_df()
    ctx = _sample_context()
    result = attach_context(df, ctx, method="nearest")

    assert "phi_mod_source" in result.columns
    assert result["phi_mod_source"].iloc[0] == "https://example.org/modulation"


def test_load_context_csv(tmp_path: Path) -> None:
    """CSV round-trip: write context CSV, load it, verify values."""
    csv_file = tmp_path / "modulation.csv"
    csv_file.write_text(
        "time,value\n"
        "2015-06-01,500.0\n"
        "2015-06-03,520.0\n"
        "2015-06-05,510.0\n"
    )

    ctx = load_context_csv(
        csv_file,
        name="phi",
        description="test modulation",
        source="test",
    )

    assert ctx.name == "phi"
    assert len(ctx.times) == 3
    assert len(ctx.values) == 3
    assert ctx.values[0] == pytest.approx(500.0)
    assert ctx.values[1] == pytest.approx(520.0)
    assert ctx.values[2] == pytest.approx(510.0)


def test_attach_context_missing_time_column() -> None:
    """Raises ValueError when time column doesn't exist."""
    df = pd.DataFrame({"x": [1, 2, 3]})
    ctx = _sample_context()
    with pytest.raises(ValueError, match="not found"):
        attach_context(df, ctx, time_column="time_start")


def test_attach_context_unknown_method() -> None:
    """Raises ValueError for unknown join method."""
    df = _sample_df()
    ctx = _sample_context()
    with pytest.raises(ValueError, match="Unknown method"):
        attach_context(df, ctx, method="magic")
