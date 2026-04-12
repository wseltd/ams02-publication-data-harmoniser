"""Tests for the build-likelihood CLI command."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from click.testing import CliRunner

from ams02wb.cli import cli


def _write_harmonised_parquet(path: Path, n_rows: int = 5, **overrides) -> Path:
    """Write a minimal harmonised Parquet file suitable for likelihood building."""
    data = {
        "species_num": ["PROTON"] * n_rows,
        "species_den": [""] * n_rows,
        "x_axis_type": ["kinetic_energy_per_nucleon"] * n_rows,
        "x_axis_unit": ["GeV"] * n_rows,
        "x_min": np.linspace(1.0, 10.0, n_rows).tolist(),
        "x_max": np.linspace(2.0, 11.0, n_rows).tolist(),
        "x_centre": np.linspace(1.5, 10.5, n_rows).tolist(),
        "y_value": np.linspace(100.0, 200.0, n_rows).tolist(),
        "y_unit": ["m^-2 s^-1 sr^-1 GeV^-1"] * n_rows,
        "stat_err": np.full(n_rows, 5.0).tolist(),
        "sys_err_total": np.full(n_rows, 3.0).tolist(),
        "sys_err_components": [json.dumps({})] * n_rows,
        "time_start": ["2015-01-01T00:00:00+00:00"] * n_rows,
        "time_stop": ["2015-02-01T00:00:00+00:00"] * n_rows,
        "provenance_json": [json.dumps({"source_url": "https://example.com"})] * n_rows,
    }
    data.update(overrides)
    df = pd.DataFrame(data)
    df.to_parquet(path, index=False)
    return path


def test_build_likelihood_help_shows_options() -> None:
    """--help output must list all four required options."""
    runner = CliRunner()
    result = runner.invoke(cli, ["build-likelihood", "--help"])

    assert result.exit_code == 0, result.output
    assert "--dataset" in result.output
    assert "--mode" in result.output
    assert "--corr-length" in result.output
    assert "--output" in result.output


def test_build_likelihood_diag_writes_parquet(tmp_path: Path) -> None:
    """Diagonal mode produces a Parquet file with covariance_matrix column."""
    dataset = _write_harmonised_parquet(tmp_path / "harmonised.parquet")
    output = tmp_path / "fit.parquet"

    runner = CliRunner()
    result = runner.invoke(cli, [
        "build-likelihood",
        "--dataset", str(dataset),
        "--mode", "diag",
        "--output", str(output),
    ])

    assert result.exit_code == 0, result.output
    assert output.exists(), f"Output file not created. Output: {result.output}"

    df = pd.read_parquet(output)
    assert "covariance_matrix" in df.columns, (
        f"Missing covariance_matrix column. Columns: {list(df.columns)}"
    )

    # Covariance should be a 5x5 matrix stored as list-of-lists
    cov_raw = df["covariance_matrix"].iloc[0]
    cov_arr = np.array([list(row) for row in cov_raw], dtype=np.float64)
    assert cov_arr.shape == (5, 5)

    # Diagonal mode: off-diagonal elements should be zero
    off_diag = cov_arr - np.diag(np.diag(cov_arr))
    assert np.allclose(off_diag, 0.0), "Diagonal mode should have zero off-diagonal elements"

    # Diagonal elements should be stat_err^2 = 25.0
    assert np.allclose(np.diag(cov_arr), 25.0), "Diagonal should be stat_err^2"


def test_build_likelihood_kernel_requires_corr_length(tmp_path: Path) -> None:
    """kernel_corr mode must exit non-zero when --corr-length is missing."""
    dataset = _write_harmonised_parquet(tmp_path / "harmonised.parquet")
    output = tmp_path / "fit.parquet"

    runner = CliRunner()
    result = runner.invoke(cli, [
        "build-likelihood",
        "--dataset", str(dataset),
        "--mode", "kernel_corr",
        "--output", str(output),
    ])

    assert result.exit_code != 0, (
        f"Expected non-zero exit for kernel_corr without --corr-length. "
        f"Output: {result.output}"
    )
    # Error message should mention corr-length
    combined = result.output + (result.stderr or "")
    assert "corr-length" in combined.lower() or "corr_length" in combined.lower(), (
        f"Error message should mention corr-length. Got: {combined}"
    )


def test_build_likelihood_kernel_corr_writes_assumed_label(tmp_path: Path) -> None:
    """kernel_corr mode sidecar JSON must contain uncertainty_label='assumed'."""
    dataset = _write_harmonised_parquet(tmp_path / "harmonised.parquet")
    output = tmp_path / "fit.parquet"

    runner = CliRunner()
    result = runner.invoke(cli, [
        "build-likelihood",
        "--dataset", str(dataset),
        "--mode", "kernel_corr",
        "--corr-length", "2.0",
        "--output", str(output),
    ])

    assert result.exit_code == 0, result.output
    assert output.exists()

    sidecar = output.with_suffix(".json")
    assert sidecar.exists(), f"Sidecar JSON not created at {sidecar}"

    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    assert meta["uncertainty_label"] == "assumed", (
        f"kernel_corr label should be 'assumed', got {meta['uncertainty_label']!r}"
    )
    assert meta["mode"] == "kernel_corr"
    assert meta["corr_length"] == 2.0


def test_build_likelihood_diag_rejects_corr_length(tmp_path: Path) -> None:
    """Passing --corr-length with --mode=diag must exit non-zero."""
    dataset = _write_harmonised_parquet(tmp_path / "harmonised.parquet")
    output = tmp_path / "fit.parquet"

    runner = CliRunner()
    result = runner.invoke(cli, [
        "build-likelihood",
        "--dataset", str(dataset),
        "--mode", "diag",
        "--corr-length", "1.0",
        "--output", str(output),
    ])

    assert result.exit_code != 0, (
        f"Expected non-zero exit for diag with --corr-length. Output: {result.output}"
    )
    combined = result.output + (result.stderr or "")
    assert "corr-length" in combined.lower() or "corr_length" in combined.lower(), (
        f"Error should mention corr-length. Got: {combined}"
    )


def test_build_likelihood_missing_dataset_exits_error(tmp_path: Path) -> None:
    """Non-existent --dataset path must exit non-zero with clear error."""
    bogus = tmp_path / "does_not_exist.parquet"
    output = tmp_path / "fit.parquet"

    runner = CliRunner()
    result = runner.invoke(cli, [
        "build-likelihood",
        "--dataset", str(bogus),
        "--mode", "diag",
        "--output", str(output),
    ])

    assert result.exit_code != 0, (
        f"Expected non-zero exit for missing dataset. Output: {result.output}"
    )
    combined = result.output + (result.stderr or "")
    assert "not found" in combined.lower() or "does not exist" in combined.lower(), (
        f"Error should say file not found. Got: {combined}"
    )


def test_build_likelihood_grouped_sys_writes_derived_label(tmp_path: Path) -> None:
    """grouped_sys mode sidecar must contain uncertainty_label='derived'."""
    dataset = _write_harmonised_parquet(tmp_path / "harmonised.parquet")
    output = tmp_path / "fit.parquet"

    runner = CliRunner()
    result = runner.invoke(cli, [
        "build-likelihood",
        "--dataset", str(dataset),
        "--mode", "grouped_sys",
        "--output", str(output),
    ])

    assert result.exit_code == 0, result.output

    sidecar = output.with_suffix(".json")
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    assert meta["uncertainty_label"] == "derived"
    assert meta["mode"] == "grouped_systematic"


def test_build_likelihood_diag_sidecar_has_published_label(tmp_path: Path) -> None:
    """diag mode sidecar must contain uncertainty_label='published'."""
    dataset = _write_harmonised_parquet(tmp_path / "harmonised.parquet")
    output = tmp_path / "fit.parquet"

    runner = CliRunner()
    result = runner.invoke(cli, [
        "build-likelihood",
        "--dataset", str(dataset),
        "--mode", "diag",
        "--output", str(output),
    ])

    assert result.exit_code == 0, result.output

    sidecar = output.with_suffix(".json")
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    assert meta["uncertainty_label"] == "published"
    assert meta["mode"] == "diagonal"


def test_build_likelihood_grouped_sys_rejects_corr_length(tmp_path: Path) -> None:
    """Passing --corr-length with --mode=grouped_sys must exit non-zero."""
    dataset = _write_harmonised_parquet(tmp_path / "harmonised.parquet")
    output = tmp_path / "fit.parquet"

    runner = CliRunner()
    result = runner.invoke(cli, [
        "build-likelihood",
        "--dataset", str(dataset),
        "--mode", "grouped_sys",
        "--corr-length", "1.0",
        "--output", str(output),
    ])

    assert result.exit_code != 0
    combined = result.output + (result.stderr or "")
    assert "corr-length" in combined.lower() or "corr_length" in combined.lower()


def test_build_likelihood_kernel_corr_covariance_has_offdiagonal(tmp_path: Path) -> None:
    """kernel_corr covariance must have non-zero off-diagonal elements."""
    dataset = _write_harmonised_parquet(tmp_path / "harmonised.parquet")
    output = tmp_path / "fit.parquet"

    runner = CliRunner()
    result = runner.invoke(cli, [
        "build-likelihood",
        "--dataset", str(dataset),
        "--mode", "kernel_corr",
        "--corr-length", "2.0",
        "--output", str(output),
    ])

    assert result.exit_code == 0, result.output

    df = pd.read_parquet(output)
    cov_raw = df["covariance_matrix"].iloc[0]
    cov_arr = np.array([list(row) for row in cov_raw], dtype=np.float64)
    off_diag = cov_arr - np.diag(np.diag(cov_arr))
    # With corr_length=2.0 and spread-out x values, there should be correlations
    assert not np.allclose(off_diag, 0.0), (
        "kernel_corr should produce non-zero off-diagonal elements"
    )


def test_build_likelihood_output_preserves_canonical_columns(tmp_path: Path) -> None:
    """Output Parquet must retain all canonical schema columns from input."""
    dataset = _write_harmonised_parquet(tmp_path / "harmonised.parquet")
    output = tmp_path / "fit.parquet"

    runner = CliRunner()
    result = runner.invoke(cli, [
        "build-likelihood",
        "--dataset", str(dataset),
        "--mode", "diag",
        "--output", str(output),
    ])

    assert result.exit_code == 0, result.output

    in_df = pd.read_parquet(dataset)
    out_df = pd.read_parquet(output)

    # All input columns must still be present
    for col in in_df.columns:
        assert col in out_df.columns, f"Input column {col!r} missing from output"

    # Plus the new covariance_matrix column
    assert "covariance_matrix" in out_df.columns
