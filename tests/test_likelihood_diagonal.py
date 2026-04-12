"""Tests for the diagonal covariance matrix builder."""

import numpy as np
import pytest

from ams02wb.likelihood.diagonal import (
    MODE,
    UNCERTAINTY_LABEL,
    build_diagonal_covariance,
)


class TestBuildDiagonalCovariance:
    """Tests for build_diagonal_covariance."""

    def test_diagonal_shape_matches_input(self):
        """Output matrix shape is (n, n) for n-element input."""
        stat_err = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        cov = build_diagonal_covariance(stat_err)
        assert cov.shape == (5, 5)

    def test_diagonal_values_are_stat_err_squared(self):
        """Diagonal elements are the element-wise squares of input errors."""
        stat_err = np.array([0.1, 0.2, 0.3])
        cov = build_diagonal_covariance(stat_err)
        expected_diag = np.array([0.01, 0.04, 0.09])
        assert np.allclose(np.diag(cov), expected_diag)

    def test_diagonal_off_diagonal_zero(self):
        """All off-diagonal elements are exactly zero — bins are uncorrelated."""
        stat_err = np.array([1.0, 2.0, 3.0])
        cov = build_diagonal_covariance(stat_err)
        off_diagonal = cov - np.diag(np.diag(cov))
        assert np.allclose(off_diagonal, 0.0)

    def test_diagonal_symmetric(self):
        """Covariance matrix is symmetric (trivially true for diagonal, but verifies contract)."""
        stat_err = np.array([0.5, 1.5, 2.5, 3.5])
        cov = build_diagonal_covariance(stat_err)
        assert np.array_equal(cov, cov.T)

    def test_diagonal_single_bin(self):
        """Single-bin input produces a (1, 1) matrix."""
        stat_err = np.array([0.42])
        cov = build_diagonal_covariance(stat_err)
        assert cov.shape == (1, 1)
        assert np.isclose(cov[0, 0], 0.42 ** 2)

    def test_diagonal_empty_input_returns_empty(self):
        """Empty array produces a (0, 0) matrix."""
        stat_err = np.array([])
        cov = build_diagonal_covariance(stat_err)
        assert cov.shape == (0, 0)


class TestEdgeCases:
    """Edge cases and adversarial inputs."""

    def test_non_1d_input_raises(self):
        """2-D input is rejected with a clear error."""
        with pytest.raises(ValueError, match="must be 1-D") as exc_info:
            build_diagonal_covariance(np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert "2-D" in str(exc_info.value)

    def test_zero_errors_raise_value_error(self):
        """Zero stat_err is rejected — would produce a singular covariance matrix."""
        with pytest.raises(ValueError, match="strictly positive") as exc_info:
            build_diagonal_covariance(np.array([0.0, 0.0, 0.0]))
        assert "non-positive" in str(exc_info.value)

    def test_large_values_no_overflow(self):
        """Large but representable errors don't overflow in float64."""
        stat_err = np.array([1e150, 1e150])
        cov = build_diagonal_covariance(stat_err)
        assert np.isfinite(cov).all()
        assert np.isclose(cov[0, 0], 1e300)

    def test_list_input_coerced(self):
        """Plain Python list is accepted and coerced to ndarray."""
        cov = build_diagonal_covariance([0.1, 0.2])
        assert cov.shape == (2, 2)
        assert np.isclose(cov[0, 0], 0.01)


class TestModuleConstants:
    """Module-level constants for provenance tracking."""

    def test_uncertainty_label(self):
        assert UNCERTAINTY_LABEL == "published"

    def test_mode(self):
        assert MODE == "diagonal"
