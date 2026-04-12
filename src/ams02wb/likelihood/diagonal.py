"""Diagonal covariance matrix builder for published statistical errors.

Constructs a diagonal covariance matrix from per-bin statistical uncertainties
and optional systematic uncertainties, assuming bins are uncorrelated.
C_ii = stat_err_i^2 + sys_err_total_i^2.  No off-diagonal terms, no
systematic grouping, no kernel smoothing.
"""

import numpy as np

# Labels for provenance tracking: this module uses only published stat errors,
# not derived or assumed uncertainties.
UNCERTAINTY_LABEL: str = "published"
MODE: str = "diagonal"


def build_diagonal_covariance(
    stat_err: np.ndarray,
    sys_err_total: np.ndarray | None = None,
) -> np.ndarray:
    """Build a diagonal covariance matrix from statistical and systematic errors.

    Parameters
    ----------
    stat_err : np.ndarray
        1-D array of per-bin statistical uncertainties (standard deviations).
        Must contain only strictly positive values — zero or negative entries
        indicate missing or corrupt data and would produce a singular matrix.
    sys_err_total : np.ndarray or None
        Optional 1-D array of per-bin total systematic uncertainties.
        If None, systematic contribution is zero.

    Returns
    -------
    np.ndarray
        Square (n, n) diagonal matrix where C_ii = stat_err_i^2 + sys_err_total_i^2.

    Raises
    ------
    ValueError
        If stat_err is not 1-D, contains zero or negative values, or if
        sys_err_total has a mismatched length.
    """
    stat_err = np.asarray(stat_err, dtype=np.float64)

    if stat_err.ndim != 1:
        raise ValueError(
            f"stat_err must be 1-D, got {stat_err.ndim}-D array with shape {stat_err.shape}"
        )

    # Reject zero/negative stat_err before matrix construction — a zero diagonal
    # entry yields a singular covariance, which silently breaks downstream chi2.
    if len(stat_err) > 0 and np.any(stat_err <= 0):
        bad_indices = np.where(stat_err <= 0)[0]
        raise ValueError(
            f"stat_err must be strictly positive, but found non-positive values "
            f"at indices {bad_indices.tolist()}: {stat_err[bad_indices].tolist()}"
        )

    variance = stat_err ** 2

    if sys_err_total is not None:
        sys_err_total = np.asarray(sys_err_total, dtype=np.float64)
        if sys_err_total.ndim != 1:
            raise ValueError(
                f"sys_err_total must be 1-D, got {sys_err_total.ndim}-D array"
            )
        if len(sys_err_total) != len(stat_err):
            raise ValueError(
                f"Length mismatch: stat_err has {len(stat_err)} elements, "
                f"sys_err_total has {len(sys_err_total)}"
            )
        variance = variance + sys_err_total ** 2

    return np.diag(variance)


def diagonal_chi2(
    residuals: np.ndarray,
    stat_err: np.ndarray,
    sys_err_total: np.ndarray | None = None,
) -> float:
    """Compute chi-squared using diagonal covariance.

    chi2 = sum_i (r_i^2 / C_ii) where C_ii = stat_err_i^2 + sys_err_total_i^2.
    Avoids full matrix inversion — diagonal inverse is trivial.

    Parameters
    ----------
    residuals : np.ndarray
        1-D array of (data - model) residuals.
    stat_err : np.ndarray
        1-D array of per-bin statistical uncertainties.
    sys_err_total : np.ndarray or None
        Optional 1-D array of per-bin total systematic uncertainties.

    Returns
    -------
    float
        Non-negative chi-squared value.

    Raises
    ------
    ValueError
        If inputs have mismatched lengths or stat_err contains non-positive values.
    """
    residuals = np.asarray(residuals, dtype=np.float64)
    stat_err = np.asarray(stat_err, dtype=np.float64)

    if residuals.ndim != 1:
        raise ValueError(
            f"residuals must be 1-D, got {residuals.ndim}-D array"
        )
    if len(residuals) != len(stat_err):
        raise ValueError(
            f"Length mismatch: residuals has {len(residuals)} elements, "
            f"stat_err has {len(stat_err)}"
        )

    # Build diagonal variances (validation happens inside build_diagonal_covariance)
    cov = build_diagonal_covariance(stat_err, sys_err_total)
    diagonal_variance = np.diag(cov)

    return float(np.sum(residuals ** 2 / diagonal_variance))
