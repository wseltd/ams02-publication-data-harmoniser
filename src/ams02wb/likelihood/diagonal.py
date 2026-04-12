"""Diagonal covariance matrix builder for published statistical errors.

Constructs a diagonal covariance matrix from per-bin statistical uncertainties,
assuming bins are uncorrelated. This is the simplest covariance model — no
off-diagonal terms, no systematic grouping, no kernel smoothing.
"""

import numpy as np

# Labels for provenance tracking: this module uses only published stat errors,
# not derived or assumed uncertainties.
UNCERTAINTY_LABEL: str = "published"
MODE: str = "diagonal"


def build_diagonal_covariance(stat_err: np.ndarray) -> np.ndarray:
    """Build a diagonal covariance matrix from statistical errors.

    Parameters
    ----------
    stat_err : np.ndarray
        1-D array of per-bin statistical uncertainties (standard deviations).

    Returns
    -------
    np.ndarray
        Square (n, n) diagonal matrix where diagonal elements are stat_err**2.
        Returns shape (1, 1) for single-bin input, (0, 0) for empty input.

    Raises
    ------
    ValueError
        If stat_err is not 1-D.
    """
    stat_err = np.asarray(stat_err, dtype=np.float64)

    if stat_err.ndim != 1:
        raise ValueError(
            f"stat_err must be 1-D, got {stat_err.ndim}-D array with shape {stat_err.shape}"
        )

    return np.diag(stat_err ** 2)
