"""Unit conversion utilities for AMS-02 physics quantities.

Provides pure-function converters for energy units, flux area units,
and rigidity <-> kinetic energy transformations using relativistic kinematics.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Energy conversion
# ---------------------------------------------------------------------------
# All (from, to) pairs for MeV, GeV, TeV including identity.
# Chose a flat lookup dict over runtime arithmetic — 9 entries is trivial
# memory, and it avoids floating-point surprises from chained division.

_ENERGY_UNITS: list[str] = ["MeV", "GeV", "TeV"]

_ENERGY_SCALES: dict[str, float] = {
    "MeV": 1.0,
    "GeV": 1e3,
    "TeV": 1e6,
}

ENERGY_CONVERSION_FACTORS: dict[tuple[str, str], float] = {
    (a, b): _ENERGY_SCALES[a] / _ENERGY_SCALES[b]
    for a in _ENERGY_UNITS
    for b in _ENERGY_UNITS
}

_VALID_ENERGY_UNITS = frozenset(_ENERGY_UNITS)


def convert_energy(value: float, from_unit: str, to_unit: str) -> float:
    """Convert an energy value between MeV, GeV, and TeV.

    Parameters
    ----------
    value : float
        The energy value to convert.
    from_unit : str
        Source unit, one of 'MeV', 'GeV', 'TeV'.
    to_unit : str
        Target unit, one of 'MeV', 'GeV', 'TeV'.

    Returns
    -------
    float
        The converted energy value.

    Raises
    ------
    ValueError
        If either unit is not recognised.
    """
    unknown = []
    if from_unit not in _VALID_ENERGY_UNITS:
        unknown.append(from_unit)
    if to_unit not in _VALID_ENERGY_UNITS:
        unknown.append(to_unit)
    if unknown:
        raise ValueError(
            f"Unknown energy unit(s): {unknown}. "
            f"Supported units: {sorted(_VALID_ENERGY_UNITS)}"
        )
    return value * ENERGY_CONVERSION_FACTORS[(from_unit, to_unit)]


# ---------------------------------------------------------------------------
# Flux area-unit conversion
# ---------------------------------------------------------------------------
# Flux is often quoted per m² or per cm².  1 m² = 1e4 cm², so converting
# flux *values* (which are per-area) inverts: flux_in_cm2 = flux_in_m2 / 1e4.
# The scale dict maps each unit to its size in m²; converting flux between
# area units requires dividing by (to_area / from_area) because flux is
# *inversely* proportional to area unit size.

_FLUX_AREA_UNITS: list[str] = ["m2", "cm2"]

# Area of each unit in m²
_FLUX_AREA_SIZE: dict[str, float] = {
    "m2": 1.0,
    "cm2": 1e-4,
}

FLUX_CONVERSION_FACTORS: dict[tuple[str, str], float] = {
    (a, b): _FLUX_AREA_SIZE[b] / _FLUX_AREA_SIZE[a]
    for a in _FLUX_AREA_UNITS
    for b in _FLUX_AREA_UNITS
}

_VALID_FLUX_UNITS = frozenset(_FLUX_AREA_UNITS)


def convert_flux(value: float, from_unit: str, to_unit: str) -> float:
    """Convert a flux value between per-m² and per-cm² area normalisation.

    Parameters
    ----------
    value : float
        The flux value to convert.
    from_unit : str
        Source area unit, one of 'm2', 'cm2'.
    to_unit : str
        Target area unit, one of 'm2', 'cm2'.

    Returns
    -------
    float
        The converted flux value.

    Raises
    ------
    ValueError
        If either unit is not recognised.
    """
    unknown = []
    if from_unit not in _VALID_FLUX_UNITS:
        unknown.append(from_unit)
    if to_unit not in _VALID_FLUX_UNITS:
        unknown.append(to_unit)
    if unknown:
        raise ValueError(
            f"Unknown flux area unit(s): {unknown}. "
            f"Supported units: {sorted(_VALID_FLUX_UNITS)}"
        )
    return value * FLUX_CONVERSION_FACTORS[(from_unit, to_unit)]


# ---------------------------------------------------------------------------
# Rigidity <-> kinetic energy (relativistic)
# ---------------------------------------------------------------------------
# Uses full relativistic kinematics:
#   E_total = sqrt((Z * R)^2 + m^2)    where R = rigidity in GV, m in GeV/c²
#   E_kinetic = E_total - m             (per nucleon: divide by A)
#
# No non-relativistic approximations — AMS-02 covers GeV–TeV where
# relativistic effects are significant for all species.


def rigidity_to_kinetic_energy(
    rigidity_gv: float,
    charge_z: int,
    mass_gev: float,
    mass_number_a: int = 1,
) -> float:
    """Convert magnetic rigidity to kinetic energy per nucleon.

    Parameters
    ----------
    rigidity_gv : float
        Magnetic rigidity in GV (gigavolts).
    charge_z : int
        Particle charge number (e.g. 1 for proton, 2 for helium-4).
    mass_gev : float
        Total particle rest mass in GeV/c² (e.g. 0.93827 for proton,
        3.7274 for helium-4).
    mass_number_a : int, optional
        Mass number (nucleon count). Default 1 (proton). For helium-4
        use 4 to get kinetic energy *per nucleon*.

    Returns
    -------
    float
        Kinetic energy per nucleon in GeV.
    """
    # total momentum p = Z * R (in GeV/c when R in GV)
    p = charge_z * rigidity_gv
    e_total = math.sqrt(p * p + mass_gev * mass_gev)
    e_kinetic = e_total - mass_gev
    return e_kinetic / mass_number_a


def kinetic_energy_to_rigidity(
    ek_per_nucleon_gev: float,
    charge_z: int,
    mass_gev: float,
    mass_number_a: int = 1,
) -> float:
    """Convert kinetic energy per nucleon to magnetic rigidity.

    Parameters
    ----------
    ek_per_nucleon_gev : float
        Kinetic energy per nucleon in GeV.
    charge_z : int
        Particle charge number.
    mass_gev : float
        Total particle rest mass in GeV/c².
    mass_number_a : int, optional
        Mass number. Default 1.

    Returns
    -------
    float
        Magnetic rigidity in GV.
    """
    e_kinetic = ek_per_nucleon_gev * mass_number_a
    e_total = e_kinetic + mass_gev
    p = math.sqrt(e_total * e_total - mass_gev * mass_gev)
    return p / charge_z
