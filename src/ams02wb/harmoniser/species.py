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
    "ANTIPROTON": Species(
        name="ANTIPROTON", charge_z=1, mass_gev=0.93827, mass_number_a=1,
    ),
    "CARBON": Species(
        name="CARBON", charge_z=6, mass_gev=11.1749, mass_number_a=12,
    ),
    "BORON": Species(
        name="BORON", charge_z=5, mass_gev=10.2553, mass_number_a=11,
    ),
    "OXYGEN": Species(
        name="OXYGEN", charge_z=8, mass_gev=14.8951, mass_number_a=16,
    ),
    "NITROGEN": Species(
        name="NITROGEN", charge_z=7, mass_gev=13.0437, mass_number_a=14,
    ),
    "LITHIUM": Species(
        name="LITHIUM", charge_z=3, mass_gev=6.5339, mass_number_a=7,
    ),
    "BERYLLIUM": Species(
        name="BERYLLIUM", charge_z=4, mass_gev=8.3935, mass_number_a=9,
    ),
    "DEUTERON": Species(
        name="DEUTERON", charge_z=1, mass_gev=1.87561, mass_number_a=2,
    ),
    "NEON": Species(
        name="NEON", charge_z=10, mass_gev=18.6227, mass_number_a=20,
    ),
    "MAGNESIUM": Species(
        name="MAGNESIUM", charge_z=12, mass_gev=22.3410, mass_number_a=24,
    ),
    "SILICON": Species(
        name="SILICON", charge_z=14, mass_gev=26.0598, mass_number_a=28,
    ),
    "IRON": Species(
        name="IRON", charge_z=26, mass_gev=52.1032, mass_number_a=56,
    ),
    "SULFUR": Species(
        name="SULFUR", charge_z=16, mass_gev=29.7819, mass_number_a=32,
    ),
    "SODIUM": Species(
        name="SODIUM", charge_z=11, mass_gev=21.4094, mass_number_a=23,
    ),
    "ALUMINUM": Species(
        name="ALUMINUM", charge_z=13, mass_gev=25.1265, mass_number_a=27,
    ),
    "FLUORINE": Species(
        name="FLUORINE", charge_z=9, mass_gev=17.6926, mass_number_a=19,
    ),
}

# Alias table: maps common name variants to canonical names.
_ALIASES: dict[str, str] = {
    "p": "PROTON",
    "proton": "PROTON",
    "he": "HELIUM",
    "helium": "HELIUM",
    "e-": "ELECTRON",
    "electron": "ELECTRON",
    "e+": "POSITRON",
    "positron": "POSITRON",
    "pbar": "ANTIPROTON",
    "antiproton": "ANTIPROTON",
    "anti-proton": "ANTIPROTON",
    "c": "CARBON",
    "carbon": "CARBON",
    "b": "BORON",
    "boron": "BORON",
    "o": "OXYGEN",
    "oxygen": "OXYGEN",
    "n": "NITROGEN",
    "nitrogen": "NITROGEN",
    "li": "LITHIUM",
    "lithium": "LITHIUM",
    "be": "BERYLLIUM",
    "beryllium": "BERYLLIUM",
    "d": "DEUTERON",
    "deuteron": "DEUTERON",
    "deuterium": "DEUTERON",
    "ne": "NEON",
    "neon": "NEON",
    "mg": "MAGNESIUM",
    "magnesium": "MAGNESIUM",
    "si": "SILICON",
    "silicon": "SILICON",
    "fe": "IRON",
    "iron": "IRON",
    "s": "SULFUR",
    "sulfur": "SULFUR",
    "na": "SODIUM",
    "sodium": "SODIUM",
    "al": "ALUMINUM",
    "aluminum": "ALUMINUM",
    "f": "FLUORINE",
    "fluorine": "FLUORINE",
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
