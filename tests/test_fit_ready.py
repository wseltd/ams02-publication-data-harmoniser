"""Tests for the fit-ready dataset assembler.

Exercises the assembly pipeline: build covariance via the appropriate builder,
then call build_fit_dataset to verify output structure, provenance preservation,
mode selection, and uncertainty_origin labelling.  Does NOT test covariance
matrix numerical correctness (covered by T046-T048).
"""

from __future__ import annotations

import numpy as np

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


# --- Minimal test fixtures as plain data ---

# 3-bin toy dataset: enough to exercise off-diagonal structure in grouped/kernel
# modes without being so large that the test is slow or hard to read.
_N_BINS = 3
_X = np.array([1.0, 2.0, 3.0])
_Y = np.array([10.0, 20.0, 30.0])
_STAT_ERR = np.array([0.1, 0.2, 0.3])
_SYS_ERR = np.array([0.05, 0.10, 0.15])
_PROVENANCE = {
    "paper_doi": "10.1103/PhysRevLett.000.000000",
    "table_id": "Table1",
    "file_url": "https://example.com/data.csv",
    "ingested_at": "2025-01-01T00:00:00Z",
    "source_type": "csv",
}

# Required keys that every fit-ready dataset dict must contain.
_REQUIRED_KEYS = {
    "y", "x", "covariance", "uncertainty_label", "mode",
    "provenance", "species", "x_axis_type", "y_unit", "n_points",
}


def test_fit_ready_contains_required_fields():
    """Assembler output contains all required keys regardless of mode."""
    cov = build_diagonal_covariance(_STAT_ERR)
    result = build_fit_dataset(
        y=_Y, x=_X, covariance=cov,
        uncertainty_label=DIAG_LABEL, mode=DIAG_MODE,
        provenance=_PROVENANCE, species="PROTON",
    )

    missing = _REQUIRED_KEYS - set(result.keys())
    assert not missing, f"Missing required keys: {missing}"
    assert result["n_points"] == _N_BINS
    # y and x arrays must preserve their values, not just shape
    np.testing.assert_array_equal(result["y"], _Y)
    np.testing.assert_array_equal(result["x"], _X)
    assert result["species"] == "PROTON"


def test_fit_ready_provenance_attached():
    """Provenance dict is preserved verbatim in the assembled dataset."""
    cov = build_diagonal_covariance(_STAT_ERR)
    result = build_fit_dataset(
        y=_Y, x=_X, covariance=cov,
        uncertainty_label=DIAG_LABEL, mode=DIAG_MODE,
        provenance=_PROVENANCE,
    )

    # Provenance must be the exact same dict, not a copy or subset
    assert result["provenance"] is _PROVENANCE
    assert result["provenance"]["paper_doi"] == "10.1103/PhysRevLett.000.000000"
    assert result["provenance"]["table_id"] == "Table1"
    assert result["provenance"]["file_url"] == "https://example.com/data.csv"
    # All original keys survive the round trip
    assert set(result["provenance"].keys()) == set(_PROVENANCE.keys())


def test_fit_ready_selects_diag_mode():
    """Diagonal covariance mode sets mode='diagonal' and label='published'."""
    cov = build_diagonal_covariance(_STAT_ERR)
    result = build_fit_dataset(
        y=_Y, x=_X, covariance=cov,
        uncertainty_label=DIAG_LABEL, mode=DIAG_MODE,
        provenance=_PROVENANCE,
    )

    assert result["mode"] == "diagonal"
    assert result["uncertainty_label"] == "published"
    assert result["covariance"].shape == (_N_BINS, _N_BINS)


def test_fit_ready_selects_grouped_sys_mode():
    """Grouped-systematic mode sets mode='grouped_systematic' and label='derived'."""
    cov = build_grouped_systematic_covariance(_STAT_ERR, _SYS_ERR)
    result = build_fit_dataset(
        y=_Y, x=_X, covariance=cov,
        uncertainty_label=GROUPED_SYS_LABEL, mode=GROUPED_SYS_MODE,
        provenance=_PROVENANCE,
    )

    assert result["mode"] == "grouped_systematic"
    assert result["uncertainty_label"] == "derived"
    assert result["covariance"].shape == (_N_BINS, _N_BINS)


def test_fit_ready_selects_kernel_corr_mode():
    """Kernel-correlation mode sets mode='kernel_corr' and label='assumed'."""
    cov = build_kernel_covariance(_STAT_ERR, _X, corr_length=1.0)
    result = build_fit_dataset(
        y=_Y, x=_X, covariance=cov,
        uncertainty_label=KERNEL_LABEL, mode=KERNEL_MODE,
        provenance=_PROVENANCE,
    )

    assert result["mode"] == "kernel_corr"
    assert result["uncertainty_label"] == "assumed"
    assert result["covariance"].shape == (_N_BINS, _N_BINS)


def test_fit_ready_labels_uncertainty_origin():
    """Each covariance mode propagates the correct uncertainty_origin label.

    The label contract: diagonal -> published, grouped_sys -> derived,
    kernel_corr -> assumed.  These labels come from the covariance builder
    module constants, not from the assembler.  This test verifies the full
    chain: builder constant -> assembler input -> assembler output.
    """
    mode_label_pairs = [
        (DIAG_MODE, DIAG_LABEL, build_diagonal_covariance(_STAT_ERR)),
        (GROUPED_SYS_MODE, GROUPED_SYS_LABEL, build_grouped_systematic_covariance(_STAT_ERR, _SYS_ERR)),
        (KERNEL_MODE, KERNEL_LABEL, build_kernel_covariance(_STAT_ERR, _X, corr_length=1.0)),
    ]

    expected_labels = {"published", "derived", "assumed"}
    observed_labels = set()

    for mode, label, cov in mode_label_pairs:
        result = build_fit_dataset(
            y=_Y, x=_X, covariance=cov,
            uncertainty_label=label, mode=mode,
            provenance=_PROVENANCE,
        )
        observed_labels.add(result["uncertainty_label"])
        # Label in the output must match the module constant that was passed in
        assert result["uncertainty_label"] == label, (
            f"Mode {mode!r} should produce label {label!r}, "
            f"got {result['uncertainty_label']!r}"
        )

    # All three canonical labels must appear across the three modes —
    # no two modes should share the same label.
    assert observed_labels == expected_labels, (
        f"Expected all three labels {expected_labels}, got {observed_labels}"
    )
