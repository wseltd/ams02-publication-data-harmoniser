"""Species normalisation and property lookup for AMS-02 data.

Maps varied species name strings (aliases, abbreviations, mixed case) to
canonical uppercase names and provides physical property lookup for axis
conversions.  This is the single source of truth for species identity —
other modules import from here rather than maintaining their own tables.

Rejected alternative: per-module species dicts.  Duplicated dicts drift
and cause silent physics bugs when mass or charge values diverge.
"""

from __future__ import annotations

from ams02wb.schema.models import Species

# Canonical species definitions.  Mass values are PDG 2022 recommendations.
# Built once at import time — these never change at runtime.
KNOWN_SPECIES: dict[str, Species] = {
    "PROTON": Species(
        name="PROTON", charge_z=1, mass_gev=0.93827, mass_number_a=1,
    ),
    "HELIUM": Species(
        name="HELIUM", charge_z=2, mass_gev=3.72738, mass_number_a=4,
    ),
    "ELECTRON": Species(
        name="ELECTRON", charge_z=1, mass_gev=0.000511, mass_number_a=1,
    ),
    "POSITRON": Species(
        name="POSITRON", charge_z=1, mass_gev=0.000511, mass_number_a=1,
    ),
}

# Alias table: maps common name variants to canonical names.
# Lowercase keys for case-insensitive lookup; uppercase keys
# handled separately via .upper() in normalise_species.
_ALIASES: dict[str, str] = {
    "p": "PROTON",
    "proton": "PROTON",
    "he": "HELIUM",
    "helium": "HELIUM",
    "e-": "ELECTRON",
    "electron": "ELECTRON",
    "e+": "POSITRON",
    "positron": "POSITRON",
}


def normalise_species(name: str) -> str:
    """Map a species name or alias to its canonical uppercase form.

    Parameters
    ----------
    name : str
        Species name, alias, or abbreviation (case-insensitive for
        canonical names, case-sensitive for symbol aliases like 'e-').

    Returns
    -------
    str
        Canonical species name (e.g. 'PROTON', 'HELIUM').

    Raises
    ------
    ValueError
        If the name cannot be mapped to any known species.
    """
    upper = name.upper()
    if upper in KNOWN_SPECIES:
        return upper
    if name in _ALIASES:
        return _ALIASES[name]
    raise ValueError(
        f"Unknown species: {name!r}. "
        f"Known species: {sorted(KNOWN_SPECIES)}"
    )


def get_species(name: str) -> Species:
    """Look up species properties by name or alias.

    Parameters
    ----------
    name : str
        Species name or alias accepted by ``normalise_species``.

    Returns
    -------
    Species
        The species properties (charge, mass, mass number).

    Raises
    ------
    ValueError
        If the species is not recognised.
    """
    canonical = normalise_species(name)
    return KNOWN_SPECIES[canonical]
