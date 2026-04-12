"""Fit-ready dataset builder for likelihood construction.

Assembles validated arrays (y, x, covariance) with metadata into a single
dict that downstream likelihood functions can consume without further
validation.  Shape consistency is enforced here so likelihood code can
assume clean inputs.
"""

from __future__ import annotations

import numpy as np

from ams02wb.schema.models import Measurement, UncertaintyLabel

# Canonical label strings, derived from the single source of truth in
# schema.models.UncertaintyLabel.  Using the enum values avoids duplicating
# the vocabulary and lets governance verify a single definition.
_VALID_LABELS: frozenset[str] = frozenset(label.value for label in UncertaintyLabel)


def build_fit_dataset(
    y: np.ndarray,
    x: np.ndarray,
    covariance: np.ndarray,
    uncertainty_label: str,
    mode: str,
    provenance: dict,
    species: str = "",
    x_axis_type: str = "",
    y_unit: str = "",
) -> dict:
    """Bundle validated arrays and metadata into a fit-ready dataset dict.

    Parameters
    ----------
    y : np.ndarray
        1-D array of measured values (one per bin).
    x : np.ndarray
        1-D array of x-axis values (e.g. energy bin centres), same length as y.
    covariance : np.ndarray
        Square (n, n) covariance matrix where n == len(y).
    uncertainty_label : str
        One of 'published', 'derived', 'assumed' — sourced from
        ``UncertaintyLabel`` in ``ams02wb.schema.models``.
    mode : str
        Covariance construction mode (e.g. 'diagonal', 'grouped_systematic').
    provenance : dict
        Origin metadata (paper DOI, table ID, etc.).
    species : str
        Particle species name (e.g. 'proton').
    x_axis_type : str
        Physical quantity on x-axis (e.g. 'rigidity').
    y_unit : str
        Unit of the measured values.

    Returns
    -------
    dict
        Keys: y, x, covariance, uncertainty_label, mode, provenance,
        species, x_axis_type, y_unit, n_points.

    Raises
    ------
    ValueError
        If array shapes are inconsistent or uncertainty_label is invalid.
    """
    n = len(y)

    if len(x) != n:
        raise ValueError(
            f"Shape mismatch: len(y)={n} but len(x)={len(x)}. "
            f"x and y must have the same number of points."
        )

    if covariance.shape != (n, n):
        raise ValueError(
            f"Shape mismatch: covariance is {covariance.shape} but expected "
            f"({n}, {n}) to match len(y)={n}."
        )

    if uncertainty_label not in _VALID_LABELS:
        raise ValueError(
            f"Invalid uncertainty_label {uncertainty_label!r}. "
            f"Must be one of {sorted(_VALID_LABELS)}."
        )

    return {
        "y": y,
        "x": x,
        "covariance": covariance,
        "uncertainty_label": uncertainty_label,
        "mode": mode,
        "provenance": provenance,
        "species": species,
        "x_axis_type": x_axis_type,
        "y_unit": y_unit,
        "n_points": n,
    }


def build_fit_dataset_from_measurement(
    measurements: list[Measurement],
    covariance: np.ndarray,
    uncertainty_label: str,
    mode: str,
    provenance: dict,
    x_axis_type: str = "",
    y_unit: str = "",
) -> dict:
    """Build a fit-ready dataset directly from Measurement objects.

    Extracts y (value) and x (energy_mid) arrays from the measurement list,
    then delegates to ``build_fit_dataset`` for validation and assembly.

    Parameters
    ----------
    measurements : list[Measurement]
        Ordered list of measurements (one per bin).
    covariance : np.ndarray
        Square covariance matrix matching len(measurements).
    uncertainty_label : str
        One of 'published', 'derived', 'assumed'.
    mode : str
        Covariance construction mode.
    provenance : dict
        Origin metadata.
    x_axis_type : str
        Physical quantity on x-axis.
    y_unit : str
        Unit of the measured values.

    Returns
    -------
    dict
        Same structure as ``build_fit_dataset``.
    """
    y = np.array([m.value for m in measurements], dtype=np.float64)
    x = np.array([m.energy_mid for m in measurements], dtype=np.float64)
    # Use the species from the first measurement; all measurements in a
    # fit dataset should share the same species.
    species = measurements[0].species if measurements else ""

    return build_fit_dataset(
        y=y,
        x=x,
        covariance=covariance,
        uncertainty_label=uncertainty_label,
        mode=mode,
        provenance=provenance,
        species=species,
        x_axis_type=x_axis_type,
        y_unit=y_unit,
    )
