"""Tests for diagonal likelihood builder — covariance construction and chi2.

Covers: zero/negative error guards (3 cases), shape and symmetry (2 cases),
standard construction (3 cases), chi2 output (1 case).  Uses canonical schema
fields stat_err and sys_err_total.
"""

import numpy as np
import pytest

from ams02wb.likelihood.diagonal import build_diagonal_covariance, diagonal_chi2


# --- Standard construction (3 cases) ---


def test_single_point_returns_1x1_diagonal():
    """Single measurement produces a 1x1 covariance matrix with correct variance."""
    stat_err = np.array([0.5])
    sys_err_total = np.array([0.3])
    cov = build_diagonal_covariance(stat_err, sys_err_total)
    assert cov.shape == (1, 1)
    # C_11 = 0.5^2 + 0.3^2 = 0.25 + 0.09 = 0.34
    assert np.isclose(cov[0, 0], 0.34)


def test_vector_returns_nxn_diagonal():
    """n-element input produces (n, n) diagonal matrix with zero off-diagonals."""
    n = 7
    stat_err = np.linspace(0.1, 0.7, n)
    cov = build_diagonal_covariance(stat_err)
    assert cov.shape == (n, n)
    # All off-diagonal entries must be exactly zero
    off_diag_mask = ~np.eye(n, dtype=bool)
    assert np.all(cov[off_diag_mask] == 0.0)


def test_diagonal_values_equal_variance_sum():
    """Diagonal entries equal stat_err^2 + sys_err_total^2."""
    stat_err = np.array([1.0, 2.0, 3.0])
    sys_err_total = np.array([0.5, 1.0, 1.5])
    cov = build_diagonal_covariance(stat_err, sys_err_total)
    expected = stat_err ** 2 + sys_err_total ** 2
    assert np.allclose(np.diag(cov), expected)
    # Verify exact values: [1.25, 5.0, 11.25]
    assert np.isclose(cov[0, 0], 1.25)
    assert np.isclose(cov[1, 1], 5.0)
    assert np.isclose(cov[2, 2], 11.25)


# --- Zero/negative error guards (3 cases) ---


def test_zero_stat_err_raises_value_error():
    """Zero stat_err must be rejected before matrix construction — singular covariance."""
    with pytest.raises(ValueError, match="strictly positive") as exc_info:
        build_diagonal_covariance(np.array([0.1, 0.0, 0.3]))
    # Verify the error identifies the offending index
    assert "1" in str(exc_info.value)


def test_zero_stat_err_with_nonzero_sys_raises():
    """Zero stat_err is rejected even when sys_err_total is nonzero.

    The guard fires on stat_err alone, before sys_err_total is added.
    A nonzero systematic does not rescue a zero statistical error — the
    stat_err contract requires strictly positive values regardless.
    """
    with pytest.raises(ValueError, match="strictly positive") as exc_info:
        build_diagonal_covariance(
            stat_err=np.array([0.0, 0.0]),
            sys_err_total=np.array([1.0, 2.0]),
        )
    assert "non-positive" in str(exc_info.value)


def test_negative_err_raises_value_error():
    """Negative stat_err is physically meaningless and must be rejected."""
    with pytest.raises(ValueError, match="strictly positive") as exc_info:
        build_diagonal_covariance(np.array([0.1, -0.5, 0.3]))
    # The error should identify the bad index (1) and value (-0.5)
    assert "-0.5" in str(exc_info.value)


# --- Shape and symmetry (2 cases) ---


def test_shape_matches_input_length():
    """Output shape is (n, n) for every n from 1 to a modest size."""
    for n in [1, 2, 5, 20]:
        stat_err = np.full(n, 0.1)
        cov = build_diagonal_covariance(stat_err)
        assert cov.shape == (n, n), f"Expected ({n}, {n}), got {cov.shape}"


def test_shape_square_symmetric():
    """Covariance matrix is square and symmetric for non-trivial input."""
    stat_err = np.array([0.2, 0.7, 1.3, 0.4, 0.9])
    sys_err_total = np.array([0.1, 0.3, 0.5, 0.2, 0.4])
    cov = build_diagonal_covariance(stat_err, sys_err_total)
    rows, cols = cov.shape
    assert rows == cols, "Covariance matrix must be square"
    assert np.array_equal(cov, cov.T), "Covariance matrix must be symmetric"


# --- Chi2 output (1 case) ---


def test_chi2_non_negative_on_toy_residuals():
    """Chi-squared is non-negative and matches hand-computed value for toy data."""
    stat_err = np.array([1.0, 2.0])
    sys_err_total = np.array([0.0, 0.0])
    residuals = np.array([1.0, 2.0])
    chi2 = diagonal_chi2(residuals, stat_err, sys_err_total)
    assert chi2 >= 0.0
    # chi2 = 1^2/1^2 + 2^2/2^2 = 1 + 1 = 2.0
    assert np.isclose(chi2, 2.0)
