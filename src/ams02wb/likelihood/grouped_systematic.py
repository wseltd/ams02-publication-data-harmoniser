"""Grouped-systematic covariance builder.

Constructs a covariance matrix that treats statistical errors as
uncorrelated (diagonal) and adds a fully-correlated systematic block.
This is the standard "stat + sys" model where systematic uncertainties
are 100 % correlated across all bins.
"""

import numpy as np

UNCERTAINTY_LABEL: str = "derived"
MODE: str = "grouped_systematic"


def build_grouped_systematic_covariance(
    stat_err: np.ndarray,
    sys_err: np.ndarray,
) -> np.ndarray:
    """Build a covariance matrix with uncorrelated stat and correlated sys.

    Parameters
    ----------
    stat_err : np.ndarray
        1-D array of per-bin statistical uncertainties (standard deviations).
    sys_err : np.ndarray
        1-D array of per-bin systematic uncertainties (standard deviations).
        Treated as fully correlated: off-diagonal element (i,j) = sys_err[i] * sys_err[j].

    Returns
    -------
    np.ndarray
        Square (n, n) covariance matrix.

    Raises
    ------
    ValueError
        If inputs are not 1-D or have mismatched lengths.
    """
    stat_err = np.asarray(stat_err, dtype=np.float64)
    sys_err = np.asarray(sys_err, dtype=np.float64)

    if stat_err.ndim != 1:
        raise ValueError(
            f"stat_err must be 1-D, got {stat_err.ndim}-D array with shape {stat_err.shape}"
        )
    if sys_err.ndim != 1:
        raise ValueError(
            f"sys_err must be 1-D, got {sys_err.ndim}-D array with shape {sys_err.shape}"
        )
    if len(stat_err) != len(sys_err):
        raise ValueError(
            f"Length mismatch: stat_err has {len(stat_err)} elements, "
            f"sys_err has {len(sys_err)}"
        )

    # Diagonal stat component + fully-correlated sys outer product
    return np.diag(stat_err ** 2) + np.outer(sys_err, sys_err)
