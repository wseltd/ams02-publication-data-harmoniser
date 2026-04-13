"""Solar/heliospheric context hooks for AMS-02 time-series data.

Pluggable module for attaching external context series (modulation proxies,
solar cycle markers, transient event annotations) to aligned AMS-02 datasets
by time key.

This module does NOT interpret or model heliophysics data — it only provides
join utilities to attach external time-series context for downstream analysis.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ContextSeries:
    """An external time-series to attach to AMS-02 data.

    Attributes:
        name: Column name for the attached values (e.g. "phi_modulation").
        description: Human-readable description of the series.
        source: Attribution URL or reference for the data.
        times: Timestamps for each value.
        values: Corresponding numeric values.
    """

    name: str
    description: str
    source: str
    times: list[datetime] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"ContextSeries(name={self.name!r}, n={len(self.values)}, "
            f"source={self.source!r})"
        )


def attach_context(
    df: pd.DataFrame,
    context: ContextSeries,
    time_column: str = "time_start",
    method: str = "nearest",
) -> pd.DataFrame:
    """Attach an external context series to a DataFrame by time key.

    Parameters
    ----------
    df : pd.DataFrame
        AMS-02 aligned or harmonised DataFrame with a time column.
    context : ContextSeries
        External time-series data to attach.
    time_column : str
        Name of the time column in *df*. Default: "time_start".
    method : str
        Join method: "nearest", "interpolate", or "exact".

    Returns
    -------
    pd.DataFrame
        Copy of *df* with two new columns: ``context.name`` (values)
        and ``{context.name}_source`` (attribution string).
    """
    if time_column not in df.columns:
        raise ValueError(f"Time column {time_column!r} not found in DataFrame")

    result = df.copy()

    # Parse times to datetime
    df_times = pd.to_datetime(result[time_column], utc=True)
    ctx_times = pd.to_datetime(context.times, utc=True)
    ctx_df = pd.DataFrame({
        "_ctx_time": ctx_times,
        context.name: context.values,
    }).sort_values("_ctx_time")

    if method == "nearest":
        # Use merge_asof for nearest-time matching
        result["_merge_time"] = df_times
        result = result.sort_values("_merge_time")
        result = pd.merge_asof(
            result,
            ctx_df.rename(columns={"_ctx_time": "_merge_time"}),
            on="_merge_time",
            direction="nearest",
        )
        result = result.drop(columns=["_merge_time"])

    elif method == "interpolate":
        # Linearly interpolate context values to dataset times
        ctx_timestamps = ctx_times.values.astype(np.int64)
        df_timestamps = df_times.values.astype(np.int64)
        interpolated = np.interp(
            df_timestamps,
            ctx_timestamps,
            context.values,
        )
        result[context.name] = interpolated

    elif method == "exact":
        # Exact date match (left join), NaN for unmatched
        result["_date_key"] = df_times.dt.date
        ctx_df["_date_key"] = pd.Series(ctx_times).dt.date.values
        ctx_lookup = ctx_df.set_index("_date_key")[context.name]
        result[context.name] = result["_date_key"].map(ctx_lookup)
        result = result.drop(columns=["_date_key"])

    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'nearest', 'interpolate', or 'exact'.")

    # Add source attribution column
    result[f"{context.name}_source"] = context.source

    return result


def load_context_csv(
    path: Path | str,
    name: str,
    description: str,
    source: str,
    time_column: str = "time",
    value_column: str = "value",
) -> ContextSeries:
    """Load a CSV file into a ContextSeries.

    Parameters
    ----------
    path : Path | str
        Path to the CSV file.
    name : str
        Name for the context series.
    description : str
        Human-readable description.
    source : str
        Attribution URL or reference.
    time_column : str
        Name of the time column in the CSV. Default: "time".
    value_column : str
        Name of the value column in the CSV. Default: "value".

    Returns
    -------
    ContextSeries
        Loaded context series.
    """
    path = Path(path)
    times: list[datetime] = []
    values: list[float] = []

    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            time_str = row[time_column].strip()
            val_str = row[value_column].strip()
            times.append(pd.Timestamp(time_str).to_pydatetime())
            values.append(float(val_str))

    return ContextSeries(
        name=name,
        description=description,
        source=source,
        times=times,
        values=values,
    )
