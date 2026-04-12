"""Kernel covariance builder tests.

Covers: PSD enforcement (3 cases), jitter regularisation (2 cases),
correlation length parameter behaviour (2 cases), shape and label (2 cases).
"""

import numpy as np

from ams02wb.likelihood.kernel_covariance import (
    UNCERTAINTY_LABEL,
    build_kernel_covariance,
)

# --- Fixtures ---

# Log-uniform energy bins typical of AMS rigidity range 1-1000 GV
LOG_UNIFORM_X = np.logspace(0, 3, 10)  # 10 bins, 1 to 1000
UNIFORM_X = np.linspace(1.0, 100.0, 8)  # 8 bins, uniform spacing
STAT_ERR_UNIFORM = np.full(8, 0.05)
STAT_ERR_LOG = np.linspace(0.01, 0.1, 10)


def _min_eigenvalue(matrix: np.ndarray) -> float:
    """Return the smallest eigenvalue of a symmetric matrix."""
    return float(np.linalg.eigvalsh(matrix)[0])


# --- PSD enforcement (3 cases) ---
# PSD is the critical invariant: all eigenvalues must be >= 0.


def test_psd_on_uniform_spacing():
    """Uniform spacing with moderate correlation length produces a PSD matrix."""
    cov = build_kernel_covariance(STAT_ERR_UNIFORM, UNIFORM_X, corr_length=10.0)
    min_eig = _min_eigenvalue(cov)
    assert min_eig >= -1e-12, f"Not PSD: min eigenvalue = {min_eig}"


def test_psd_on_log_spacing():
    """Log-uniform spacing (typical AMS rigidity bins) produces a PSD matrix."""
    cov = build_kernel_covariance(STAT_ERR_LOG, LOG_UNIFORM_X, corr_length=50.0)
    min_eig = _min_eigenvalue(cov)
    assert min_eig >= -1e-12, f"Not PSD: min eigenvalue = {min_eig}"


def test_psd_on_extreme_correlation_length():
    """Very small and very large correlation lengths must not produce NaN or
    negative eigenvalues.  These are boundary cases where numerical instability
    is most likely."""
    x = np.logspace(0, 2, 20)
    err = np.full(20, 0.03)

    for corr_length in [1e-6, 1e-3, 0.01, 1e4, 1e8]:
        cov = build_kernel_covariance(err, x, corr_length=corr_length)
        assert not np.any(np.isnan(cov)), (
            f"NaN in covariance for corr_length={corr_length}"
        )
        min_eig = _min_eigenvalue(cov)
        assert min_eig >= -1e-12, (
            f"Not PSD for corr_length={corr_length}: min eigenvalue = {min_eig}"
        )


# --- Jitter regularisation (2 cases) ---
# Adding small diagonal jitter improves conditioning.


def test_jitter_increases_diagonal():
    """Adding jitter to the diagonal increases every diagonal entry."""
    cov = build_kernel_covariance(STAT_ERR_LOG, LOG_UNIFORM_X, corr_length=50.0)
    jitter = 1e-8
    cov_jittered = cov + jitter * np.eye(len(LOG_UNIFORM_X))

    for i in range(len(LOG_UNIFORM_X)):
        assert cov_jittered[i, i] > cov[i, i], (
            f"Jitter did not increase diagonal at index {i}"
        )


def test_jitter_restores_psd_for_ill_conditioned():
    """Near-duplicate x values create an ill-conditioned matrix.  Adding
    diagonal jitter must restore numerical PSD (all eigenvalues > 0)."""
    # Near-duplicate bins: spacing ~ 1e-10, making rows nearly identical
    n = 10
    x_tight = np.linspace(1.0, 1.0 + 1e-10, n)
    err = np.full(n, 0.1)

    cov = build_kernel_covariance(err, x_tight, corr_length=1.0)
    # Without jitter, the matrix is rank-deficient or nearly so
    eigenvalues_raw = np.linalg.eigvalsh(cov)

    jitter = 1e-6
    cov_jittered = cov + jitter * np.eye(n)
    eigenvalues_jittered = np.linalg.eigvalsh(cov_jittered)

    # Jitter must push the smallest eigenvalue above zero
    assert eigenvalues_jittered[0] > 0.0, (
        f"Jitter failed to restore PSD: min eigenvalue = {eigenvalues_jittered[0]}"
    )
    # Jittered min eigenvalue must be strictly larger than raw
    assert eigenvalues_jittered[0] > eigenvalues_raw[0], (
        "Jitter did not improve the smallest eigenvalue"
    )


# --- Correlation length parameter behaviour (2 cases) ---


def test_short_correlation_length_near_diagonal():
    """With a very short correlation length, off-diagonal entries decay to
    near zero — the matrix approximates a diagonal."""
    n = 15
    x = np.logspace(0, 2, n)
    err = np.full(n, 0.1)

    cov = build_kernel_covariance(err, x, corr_length=0.01)

    # Off-diagonal entries should be negligible compared to diagonal
    diag = np.diag(cov)
    for i in range(n):
        for j in range(n):
            if i != j:
                assert abs(cov[i, j]) < 1e-6 * diag[i], (
                    f"Off-diagonal [{i},{j}]={cov[i, j]} too large "
                    f"relative to diagonal [{i},{i}]={diag[i]}"
                )


def test_long_correlation_length_nearly_full_corr():
    """With a very long correlation length, the kernel approaches 1 everywhere,
    so the correlation matrix approaches the outer product of errors — all
    entries near sigma_i * sigma_j."""
    n = 12
    x = np.logspace(0, 2, n)
    err = np.linspace(0.02, 0.1, n)

    cov = build_kernel_covariance(err, x, corr_length=1e6)

    # Every entry should be close to sigma_i * sigma_j
    expected_full = np.outer(err, err)
    max_rel_diff = np.max(np.abs(cov - expected_full) / expected_full)
    assert max_rel_diff < 1e-8, (
        f"Long correlation length did not produce near-full correlation: "
        f"max relative difference = {max_rel_diff}"
    )


# --- Shape and label (2 cases) ---


def test_shape_nxn_matches_input():
    """Output shape is (n, n) for various input sizes, and matrix is symmetric."""
    for n in [1, 5, 20]:
        x = np.logspace(0, 2, n)
        err = np.full(n, 0.05)
        cov = build_kernel_covariance(err, x, corr_length=10.0)
        assert cov.shape == (n, n), f"Expected ({n}, {n}), got {cov.shape}"
        assert np.allclose(cov, cov.T), "Covariance matrix must be symmetric"


def test_covariance_label_is_assumed():
    """The module labels this covariance as 'assumed' because the kernel shape
    and correlation length are analyst choices, not published values."""
    assert UNCERTAINTY_LABEL == "assumed"
