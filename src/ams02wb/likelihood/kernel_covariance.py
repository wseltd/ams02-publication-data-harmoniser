"""Kernel-smoothed covariance builder.

Constructs a covariance matrix using a squared-exponential (Gaussian) kernel
to model bin-to-bin correlations.  The correlation length controls how far
correlations extend in x-space — short lengths give near-diagonal matrices,
long lengths give nearly fully-correlated matrices.

This is labelled 'assumed' because the kernel shape and correlation length
are analyst choices, not published by AMS-02.
"""

import numpy as np

UNCERTAINTY_LABEL: str = "assumed"
MODE: str = "kernel_corr"


def build_kernel_covariance(
    stat_err: np.ndarray,
    x: np.ndarray,
    corr_length: float,
) -> np.ndarray:
    """Build a covariance matrix with squared-exponential kernel correlations.

    Parameters
    ----------
    stat_err : np.ndarray
        1-D array of per-bin uncertainties (standard deviations).
    x : np.ndarray
        1-D array of bin centre positions (same length as stat_err).
    corr_length : float
        Correlation length in the same units as x.  Must be positive.

    Returns
    -------
    np.ndarray
        Square (n, n) covariance matrix where C[i,j] = sigma_i * sigma_j * K(x_i, x_j)
        and K is the squared-exponential kernel.

    Raises
    ------
    ValueError
        If inputs are not 1-D, have mismatched lengths, or corr_length <= 0.
    """
    stat_err = np.asarray(stat_err, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    if stat_err.ndim != 1:
        raise ValueError(
            f"stat_err must be 1-D, got {stat_err.ndim}-D array with shape {stat_err.shape}"
        )
    if x.ndim != 1:
        raise ValueError(
            f"x must be 1-D, got {x.ndim}-D array with shape {x.shape}"
        )
    if len(stat_err) != len(x):
        raise ValueError(
            f"Length mismatch: stat_err has {len(stat_err)} elements, "
            f"x has {len(x)}"
        )
    if corr_length <= 0:
        raise ValueError(
            f"corr_length must be positive, got {corr_length}"
        )

    # Squared-exponential kernel: K(xi, xj) = exp(-0.5 * ((xi - xj) / l)^2)
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    kernel = np.exp(-0.5 * (dx / corr_length) ** 2)

    # Scale by uncertainties: C[i,j] = sigma_i * sigma_j * K[i,j]
    sigma_outer = np.outer(stat_err, stat_err)
    return sigma_outer * kernel
