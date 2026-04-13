"""CLI command: build-likelihood — construct fit-ready likelihood datasets."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd

from ams02wb.likelihood.diagonal import (
    build_diagonal_covariance,
    MODE as DIAG_MODE,
    UNCERTAINTY_LABEL as DIAG_LABEL,
)
from ams02wb.likelihood.grouped_systematic import (
    build_grouped_systematic_covariance,
    MODE as GROUPED_SYS_MODE,
    UNCERTAINTY_LABEL as GROUPED_SYS_LABEL,
)
from ams02wb.likelihood.kernel_covariance import (
    build_kernel_covariance,
    MODE as KERNEL_MODE,
    UNCERTAINTY_LABEL as KERNEL_LABEL,
)
from ams02wb.likelihood.fitready import build_fit_dataset

logger = logging.getLogger(__name__)

# Valid covariance modes — single source of truth for CLI validation.
VALID_MODES = ("diag", "grouped_sys", "kernel_corr")

# Map mode name to (uncertainty_label, builder_mode_string).
_MODE_METADATA: dict[str, tuple[str, str]] = {
    "diag": (DIAG_LABEL, DIAG_MODE),
    "grouped_sys": (GROUPED_SYS_LABEL, GROUPED_SYS_MODE),
    "kernel_corr": (KERNEL_LABEL, KERNEL_MODE),
}


def _build_covariance(
    mode: str,
    stat_err: np.ndarray,
    sys_err: np.ndarray | None,
    x: np.ndarray | None,
    corr_length: float | None,
) -> np.ndarray:
    """Dispatch to the correct covariance builder based on mode."""
    if mode == "diag":
        return build_diagonal_covariance(stat_err)
    elif mode == "grouped_sys":
        if sys_err is None:
            raise ValueError(
                "grouped_sys mode requires sys_err column in the dataset, "
                "but it was not found or is all NaN."
            )
        return build_grouped_systematic_covariance(stat_err, sys_err)
    elif mode == "kernel_corr":
        if x is None:
            raise ValueError("kernel_corr mode requires x (energy) values.")
        if corr_length is None:
            raise ValueError("kernel_corr mode requires --corr-length.")
        return build_kernel_covariance(stat_err, x, corr_length)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")


def _load_harmonised_parquet(path: Path) -> pd.DataFrame:
    """Load a harmonised Parquet file and validate it has required columns."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    if not path.suffix == ".parquet":
        raise ValueError(f"Expected a .parquet file, got: {path.name}")

    df = pd.read_parquet(path)

    required = {"y_value", "stat_err", "x_centre"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset missing required columns: {sorted(missing)}. "
            f"Available: {sorted(df.columns)}"
        )
    return df


def _build_and_write_single(
    df: pd.DataFrame,
    mode: str,
    corr_length: float | None,
    output_path: Path,
    mode_metadata: dict[str, tuple[str, str]],
) -> None:
    """Build covariance and write fit-ready parquet + JSON sidecar for one subset."""
    y = df["y_value"].to_numpy(dtype=np.float64)
    x = df["x_centre"].to_numpy(dtype=np.float64)
    stat_err = df["stat_err"].to_numpy(dtype=np.float64)

    sys_err: np.ndarray | None = None
    if "sys_err_total" in df.columns and df["sys_err_total"].notna().any():
        sys_err = df["sys_err_total"].fillna(0.0).to_numpy(dtype=np.float64)

    covariance = _build_covariance(mode, stat_err, sys_err, x, corr_length)
    uncertainty_label, mode_string = mode_metadata[mode]

    provenance: dict = {}
    if "provenance_json" in df.columns:
        first_prov = df["provenance_json"].iloc[0]
        if isinstance(first_prov, str):
            try:
                provenance = json.loads(first_prov)
            except json.JSONDecodeError:
                pass
        elif isinstance(first_prov, dict):
            provenance = first_prov

    fit_dataset = build_fit_dataset(
        y=y, x=x, covariance=covariance,
        uncertainty_label=uncertainty_label, mode=mode_string,
        provenance=provenance,
        species=str(df["species_num"].iloc[0]) if "species_num" in df.columns else "",
        x_axis_type=str(df["x_axis_type"].iloc[0]) if "x_axis_type" in df.columns else "",
        y_unit=str(df["y_unit"].iloc[0]) if "y_unit" in df.columns else "",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = df.copy()
    out_df["covariance_matrix"] = [covariance.tolist()] * len(out_df)
    out_df.to_parquet(output_path, index=False)

    sidecar = {
        "uncertainty_label": uncertainty_label,
        "mode": mode_string,
        "n_points": fit_dataset["n_points"],
        "species": fit_dataset["species"],
        "x_axis_type": fit_dataset["x_axis_type"],
        "y_unit": fit_dataset["y_unit"],
        "provenance": provenance,
    }
    if corr_length is not None:
        sidecar["corr_length"] = corr_length

    output_path.with_suffix(".json").write_text(
        json.dumps(sidecar, indent=2, default=str), encoding="utf-8",
    )


@click.command("build-likelihood")
@click.option(
    "--dataset",
    required=True,
    type=click.Path(),
    help="Path to a harmonised Parquet file (output of the harmonise command).",
)
@click.option(
    "--mode",
    required=True,
    type=click.Choice(VALID_MODES, case_sensitive=True),
    help="Covariance construction mode: diag, grouped_sys, or kernel_corr.",
)
@click.option(
    "--corr-length",
    type=float,
    default=None,
    help="Correlation length for kernel_corr mode (in dataset x-axis units). "
    "Required when --mode=kernel_corr, forbidden otherwise.",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(),
    help="Output path for the fit-ready Parquet file. "
    "A JSON sidecar (<name>.json) is written alongside.",
)
@click.option(
    "--time-bin",
    "time_bin",
    type=str,
    default=None,
    help="Filter to a single date (YYYY-MM-DD) or range (YYYY-MM-DD:YYYY-MM-DD) "
    "before building the covariance. Essential for time-series data.",
)
@click.option(
    "--per-day",
    "per_day",
    is_flag=True,
    default=False,
    help="Build one fit-ready dataset per unique date. "
    "--output is treated as a directory path.",
)
def build_likelihood(
    dataset: str,
    mode: str,
    corr_length: float | None,
    output: str,
    time_bin: str | None,
    per_day: bool,
) -> None:
    """Build a fit-ready likelihood dataset from harmonised data.

    Loads a harmonised Parquet dataset, constructs a covariance matrix using
    the selected mode, and writes a fit-ready Parquet file with a JSON sidecar
    containing the uncertainty_label.

    Examples:
        ams02wb build-likelihood --dataset harmonised.parquet --mode diag --output fit.parquet
        ams02wb build-likelihood --dataset harmonised.parquet --mode kernel_corr --corr-length 0.5 --output fit.parquet
    """
    # Validate corr-length / mode mutual exclusivity
    if mode == "kernel_corr" and corr_length is None:
        click.echo(
            "Error: --corr-length is required when --mode=kernel_corr.",
            err=True,
        )
        sys.exit(1)

    if mode != "kernel_corr" and corr_length is not None:
        click.echo(
            f"Error: --corr-length is not allowed when --mode={mode}. "
            f"It is only valid for --mode=kernel_corr.",
            err=True,
        )
        sys.exit(1)

    dataset_path = Path(dataset)
    output_path = Path(output)

    # Load and validate input
    try:
        df = _load_harmonised_parquet(dataset_path)
    except (FileNotFoundError, ValueError) as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    if len(df) == 0:
        click.echo("Error: dataset contains no rows.", err=True)
        sys.exit(1)

    # Find date column for time filtering
    _date_col = None
    for candidate in ("time_start", "date_start", "date"):
        if candidate in df.columns:
            _date_col = candidate
            break

    # Apply --time-bin filter
    if time_bin is not None and _date_col:
        dates = pd.to_datetime(df[_date_col], utc=True)
        if ":" in time_bin:
            start_str, end_str = time_bin.split(":", 1)
            start = pd.Timestamp(start_str, tz="UTC")
            end = pd.Timestamp(end_str, tz="UTC") + pd.Timedelta(days=1)
            df = df[(dates >= start) & (dates < end)]
        else:
            target = pd.Timestamp(time_bin, tz="UTC")
            df = df[dates.dt.date == target.date()]

        df = df.reset_index(drop=True)
        if len(df) == 0:
            click.echo(f"Error: no data matches --time-bin {time_bin!r}.", err=True)
            sys.exit(1)
        click.echo(f"Filtered to {len(df)} rows for time-bin {time_bin}")

    # Handle --per-day: build one likelihood per unique date
    if per_day:
        if not _date_col:
            click.echo("Error: --per-day requires a time column (time_start).", err=True)
            sys.exit(1)

        out_dir = Path(output)
        out_dir.mkdir(parents=True, exist_ok=True)

        dates = pd.to_datetime(df[_date_col], utc=True)
        unique_dates = sorted(dates.dt.date.unique())
        click.echo(f"Building per-day likelihoods for {len(unique_dates)} dates...")

        for day in unique_dates:
            day_df = df[dates.dt.date == day]
            day_str = day.isoformat()
            day_output = out_dir / f"{day_str}.parquet"

            _build_and_write_single(
                day_df, mode, corr_length, day_output, _MODE_METADATA,
            )

        click.echo(f"Wrote {len(unique_dates)} per-day datasets to {out_dir}")
        return

    # Extract arrays for covariance construction
    y = df["y_value"].to_numpy(dtype=np.float64)
    x = df["x_centre"].to_numpy(dtype=np.float64)
    stat_err = df["stat_err"].to_numpy(dtype=np.float64)

    sys_err: np.ndarray | None = None
    if "sys_err_total" in df.columns and df["sys_err_total"].notna().any():
        sys_err = df["sys_err_total"].fillna(0.0).to_numpy(dtype=np.float64)

    # Build covariance matrix
    try:
        covariance = _build_covariance(mode, stat_err, sys_err, x, corr_length)
    except ValueError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    # Get label and mode string from the likelihood module constants
    uncertainty_label, mode_string = _MODE_METADATA[mode]

    # Build provenance from dataset (pass through whatever was in the parquet)
    provenance: dict = {}
    if "provenance_json" in df.columns:
        try:
            first_prov = df["provenance_json"].iloc[0]
            if isinstance(first_prov, str):
                provenance = json.loads(first_prov)
            elif isinstance(first_prov, dict):
                provenance = first_prov
        except (json.JSONDecodeError, IndexError):
            logger.warning("Could not parse provenance_json from dataset.")

    # Assemble fit-ready dataset via the likelihood module
    fit_dataset = build_fit_dataset(
        y=y,
        x=x,
        covariance=covariance,
        uncertainty_label=uncertainty_label,
        mode=mode_string,
        provenance=provenance,
        species=str(df.get("species_num", pd.Series([""]))[0]) if "species_num" in df.columns else "",
        x_axis_type=str(df.get("x_axis_type", pd.Series([""]))[0]) if "x_axis_type" in df.columns else "",
        y_unit=str(df.get("y_unit", pd.Series([""]))[0]) if "y_unit" in df.columns else "",
    )

    # Write output Parquet — include original columns plus covariance_matrix
    output_path.parent.mkdir(parents=True, exist_ok=True)

    out_df = df.copy()
    # Store covariance as a list-of-lists column (each row gets the full matrix)
    # Chose list-of-lists over a separate file because it keeps the covariance
    # co-located with the data — at the cost of redundant storage. For the
    # typical AMS-02 dataset size (< 100 bins) this is negligible.
    cov_list = covariance.tolist()
    out_df["covariance_matrix"] = [cov_list] * len(out_df)

    out_df.to_parquet(output_path, index=False)

    # Write JSON sidecar with uncertainty_label and metadata
    sidecar_path = output_path.with_suffix(".json")
    sidecar = {
        "uncertainty_label": uncertainty_label,
        "mode": mode_string,
        "n_points": fit_dataset["n_points"],
        "species": fit_dataset["species"],
        "x_axis_type": fit_dataset["x_axis_type"],
        "y_unit": fit_dataset["y_unit"],
        "provenance": provenance,
    }
    if corr_length is not None:
        sidecar["corr_length"] = corr_length

    sidecar_path.write_text(
        json.dumps(sidecar, indent=2, default=str),
        encoding="utf-8",
    )

    click.echo(
        f"Wrote fit-ready dataset: {output_path} "
        f"({fit_dataset['n_points']} points, mode={mode_string}, "
        f"label={uncertainty_label})"
    )
    click.echo(f"Wrote sidecar: {sidecar_path}")
