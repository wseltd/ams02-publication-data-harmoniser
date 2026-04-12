"""Axis name normalisation for AMS-02 publication data.

Maps the varied axis labels found across AMS-02 publications (e.g.
"Kinetic Energy (GeV)", "R (GV)", "Ekin/nucleon") to a canonical
AxisName enum.  This is the single source of truth for axis identity;
other modules should import from here rather than maintaining their own
label sets.
"""

from __future__ import annotations

import enum
import re


class AxisName(enum.StrEnum):
    """Canonical axis identifiers used throughout the harmoniser."""

    KINETIC_ENERGY = "kinetic_energy"
    RIGIDITY = "rigidity"
    EKIN_PER_NUC = "ekin_per_nuc"
    KINETIC_ENERGY_PER_NUC = "kinetic_energy_per_nuc"
    MOMENTUM = "momentum"


# ---------------------------------------------------------------------------
# Alias table
# ---------------------------------------------------------------------------
# Each key is a normalised (lowercased, whitespace-collapsed, stripped) alias
# that maps to the canonical AxisName.  New aliases go here — nowhere else.
# Chose a flat dict over regex matching: the alias set is small and finite,
# and exact lookup is O(1) with zero regex edge-case risk.

_ALIAS_TABLE: dict[str, AxisName] = {
    # Kinetic energy
    "kinetic energy": AxisName.KINETIC_ENERGY,
    "kinetic energy (gev)": AxisName.KINETIC_ENERGY,
    "kinetic energy (mev)": AxisName.KINETIC_ENERGY,
    "kinetic energy (tev)": AxisName.KINETIC_ENERGY,
    "kinetic energy gev": AxisName.KINETIC_ENERGY,
    "ek": AxisName.KINETIC_ENERGY,
    "e_k": AxisName.KINETIC_ENERGY,
    # Rigidity
    "rigidity": AxisName.RIGIDITY,
    "rigidity (gv)": AxisName.RIGIDITY,
    "rigidity gv": AxisName.RIGIDITY,
    "r (gv)": AxisName.RIGIDITY,
    "r gv": AxisName.RIGIDITY,
    "r(gv)": AxisName.RIGIDITY,
    # Kinetic energy per nucleon
    "ekin/nucleon": AxisName.EKIN_PER_NUC,
    "ekin/nuc": AxisName.EKIN_PER_NUC,
    "ekin per nucleon": AxisName.EKIN_PER_NUC,
    "ekin_per_nuc": AxisName.EKIN_PER_NUC,
    "ekin_per_nucleon": AxisName.EKIN_PER_NUC,
    "ek/n": AxisName.EKIN_PER_NUC,
    # Kinetic energy per nucleon (full spelling)
    "kinetic energy per nucleon": AxisName.KINETIC_ENERGY_PER_NUC,
    "kinetic energy/nucleon": AxisName.KINETIC_ENERGY_PER_NUC,
    "kinetic energy per nuc": AxisName.KINETIC_ENERGY_PER_NUC,
    # Momentum
    "momentum": AxisName.MOMENTUM,
    "momentum (gev/c)": AxisName.MOMENTUM,
    "momentum (gev)": AxisName.MOMENTUM,
    "p (gev/c)": AxisName.MOMENTUM,
    "p": AxisName.MOMENTUM,
}

# Pre-computed for whitespace collapsing
_MULTI_SPACE = re.compile(r"\s+")


def _normalise_text(raw: str) -> str:
    """Lowercase, collapse whitespace, strip."""
    return _MULTI_SPACE.sub(" ", raw.strip()).lower()


def normalise_axis_label(raw: str) -> AxisName:
    """Map a raw axis label string to the canonical AxisName.

    Parameters
    ----------
    raw : str
        The axis label as found in a publication table header.

    Returns
    -------
    AxisName
        The canonical axis name.

    Raises
    ------
    ValueError
        If the label cannot be mapped to any known axis.  The message
        includes the normalised form of the input to aid debugging.
    """
    normalised = _normalise_text(raw)
    try:
        return _ALIAS_TABLE[normalised]
    except KeyError:
        raise ValueError(
            f"Unknown axis label: {normalised!r}. "
            f"Known labels: {list_known_labels()}"
        ) from None


def list_known_labels() -> list[str]:
    """Return a sorted list of all recognised axis label aliases.

    Returns
    -------
    list[str]
        Every alias string accepted by ``normalise_axis_label``,
        sorted lexicographically.
    """
    return sorted(_ALIAS_TABLE)
