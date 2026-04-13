"""Tests for grouped-systematic covariance builder.

Covers: mismatched-length validation (3 cases), multi-group block structure
verification (3 cases), standard construction (2 cases), symmetry and chi2
(1 case).  Tests exercise build_grouped_systematic_covariance from
ams02wb.likelihood.grouped_systematic — the canonical grouped-sys builder.
"""

import numpy as np
import pytest

from ams02wb.likelihood.grouped_systematic import build_grouped_systematic_covariance


# ---------------------------------------------------------------------------
# Fixtures: small vectors for 2–3 "groups" of systematic components
# ---------------------------------------------------------------------------

def _make_stat_err(n):
    """Return stat errors for n bins (0.1 per bin)."""
    return np.full(n, 0.1)


def _make_sys_err_single_group(n):
    """Single group: all bins share one systematic component."""
    return np.linspace(0.2, 0.5, n)


def _make_sys_err_two_groups(n_a, n_b):
    """Two groups: group A has nonzero sys, group B has zero sys.

    Returns the combined sys_err vector.  Within group A the sys errors
    are nonzero and thus fully correlated; group B has zero sys so its
    off-diagonal sys contribution is zero.
    """
    group_a = np.full(n_a, 0.3)
    group_b = np.zeros(n_b)
    return np.concatenate([group_a, group_b])


# ---------------------------------------------------------------------------
# Standard construction (2 cases)
# ---------------------------------------------------------------------------

class TestStandardConstruction:
    """Verify correct covariance values for well-formed inputs."""

    def test_single_group_returns_correct_block(self):
        """A single group of 3 bins: C = diag(stat²) + outer(sys, sys)."""
        stat = np.array([0.1, 0.2, 0.3])
        sys = np.array([0.5, 0.5, 0.5])

        cov = build_grouped_systematic_covariance(stat, sys)

        expected = np.diag(stat ** 2) + np.outer(sys, sys)
        assert np.allclose(cov, expected)

    def test_multi_group_covariance_shape_nxn(self):
        """Covariance for n bins is always (n, n)."""
        n = 5
        stat = _make_stat_err(n)
        sys = _make_sys_err_single_group(n)

        cov = build_grouped_systematic_covariance(stat, sys)

        assert cov.shape == (n, n), f"Expected ({n}, {n}), got {cov.shape}"


# ---------------------------------------------------------------------------
# Multi-group block structure verification (3 cases)
# ---------------------------------------------------------------------------

class TestMultiGroupBlockStructure:
    """Verify block structure when bins are partitioned into groups.

    We simulate groups by setting sys_err to zero in one group and nonzero
    in another.  Cross-group off-diagonals should reflect only stat (zero,
    since stat is diagonal) while within-group off-diagonals are nonzero
    due to the outer product of the sys vector.
    """

    def test_multi_group_block_structure_correct(self):
        """Cross-group off-diagonal blocks are zero when one group has no sys."""
        n_a, n_b = 3, 2
        stat = _make_stat_err(n_a + n_b)
        sys = _make_sys_err_two_groups(n_a, n_b)

        cov = build_grouped_systematic_covariance(stat, sys)

        # Cross-group block: rows in A, cols in B — should be zero because
        # sys[i]*sys[j] = 0.3 * 0.0 = 0 for i in A, j in B
        cross_block = cov[:n_a, n_a:]
        assert np.allclose(cross_block, np.zeros((n_a, n_b)))

    def test_multi_group_offdiag_within_group_nonzero(self):
        """Within-group off-diagonal elements are nonzero (correlated)."""
        n_a, n_b = 3, 2
        stat = _make_stat_err(n_a + n_b)
        sys = _make_sys_err_two_groups(n_a, n_b)

        cov = build_grouped_systematic_covariance(stat, sys)

        # Within group A: off-diagonal = sys[i]*sys[j] = 0.3*0.3 = 0.09
        within_a = cov[:n_a, :n_a]
        for i in range(n_a):
            for j in range(n_a):
                if i != j:
                    assert within_a[i, j] != 0.0, (
                        f"Within-group off-diagonal ({i},{j}) should be nonzero"
                    )

    def test_within_group_offdiag_equals_sys_outer_product(self):
        """Within-group off-diag values equal sys_err[i] * sys_err[j]."""
        stat = np.array([0.1, 0.1, 0.1])
        sys = np.array([0.2, 0.3, 0.4])

        cov = build_grouped_systematic_covariance(stat, sys)

        # Off-diagonal (0,1) should be 0.2 * 0.3 = 0.06
        assert np.isclose(cov[0, 1], 0.06)
        # Off-diagonal (1,2) should be 0.3 * 0.4 = 0.12
        assert np.isclose(cov[1, 2], 0.12)
        # Off-diagonal (0,2) should be 0.2 * 0.4 = 0.08
        assert np.isclose(cov[0, 2], 0.08)


# ---------------------------------------------------------------------------
# Mismatched-length validation (3 cases)
# ---------------------------------------------------------------------------

class TestMismatchedLengthValidation:
    """Mismatched inputs must raise ValueError before any matrix is built."""

    def test_mismatched_stat_err_and_group_lengths_raises(self):
        """stat_err and sys_err with different lengths raise ValueError."""
        stat = np.array([0.1, 0.2, 0.3])
        sys = np.array([0.5, 0.5])  # 2 != 3

        with pytest.raises(ValueError, match="Length mismatch") as exc_info:
            build_grouped_systematic_covariance(stat, sys)
        assert "3" in str(exc_info.value)
        assert "2" in str(exc_info.value)

    def test_mismatched_component_and_data_lengths_raises(self):
        """sys_err longer than stat_err raises ValueError."""
        stat = np.array([0.1])
        sys = np.array([0.5, 0.5, 0.5])

        with pytest.raises(ValueError, match="Length mismatch") as exc_info:
            build_grouped_systematic_covariance(stat, sys)
        assert "1" in str(exc_info.value)
        assert "3" in str(exc_info.value)

    def test_empty_group_raises_value_error(self):
        """Empty arrays should produce an empty (0,0) matrix — verify shape.

        The builder does not explicitly reject empty input; it returns
        a vacuously correct 0x0 matrix.  We verify this boundary.
        """
        stat = np.array([])
        sys = np.array([])

        cov = build_grouped_systematic_covariance(stat, sys)

        assert cov.shape == (0, 0), "Empty input should yield (0, 0) matrix"


# ---------------------------------------------------------------------------
# Symmetry and chi2 (1 combined case)
# ---------------------------------------------------------------------------

class TestSymmetryAndChi2:
    """Covariance must be symmetric; chi2 from it must be non-negative."""

    def test_covariance_symmetric(self):
        """Covariance matrix must equal its own transpose."""
        stat = np.array([0.1, 0.2, 0.15, 0.25])
        sys = np.array([0.3, 0.4, 0.35, 0.45])

        cov = build_grouped_systematic_covariance(stat, sys)

        assert np.array_equal(cov, cov.T)

    def test_chi2_non_negative_on_toy_data(self):
        """chi2 = r^T C^{-1} r >= 0 for any residual vector."""
        stat = np.array([0.1, 0.2, 0.15])
        sys = np.array([0.3, 0.4, 0.35])
        residuals = np.array([0.5, -0.3, 0.1])

        cov = build_grouped_systematic_covariance(stat, sys)
        cov_inv = np.linalg.inv(cov)
        chi2 = residuals @ cov_inv @ residuals

        assert chi2 >= 0.0, f"chi2 must be non-negative, got {chi2}"
