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
def build_likelihood(
    dataset: str,
    mode: str,
    corr_length: float | None,
    output: str,
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
