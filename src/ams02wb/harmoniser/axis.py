"""Axis harmoniser — converts measurements to a common energy axis and unit.

Composes the unit converter (T006) and axis normaliser (T007) to transform
a list of Measurements to a target energy unit and axis type.  No unit
conversion logic lives here — this module is pure orchestration of existing
primitives.

Design: map-style pass — each Measurement is independently transformed.
Returns new Measurement instances; originals are never mutated.
"""

from __future__ import annotations

from ams02wb.harmoniser.species import get_species
from ams02wb.schema.models import Measurement
from ams02wb.schema.units import (
    ENERGY_CONVERSION_FACTORS,
    convert_energy,
    rigidity_to_kinetic_energy,
)


def harmonise_axes(
    measurements: list[Measurement],
    target_energy_unit: str = "GeV",
    target_axis: str = "kinetic_energy_per_nucleon",
) -> list[Measurement]:
    """Convert measurements to a common energy axis and unit.

    Applies axis conversion (e.g. rigidity -> kinetic energy per nucleon)
    then unit conversion (e.g. GeV -> MeV), adjusting the flux value
    denominator when the energy unit changes.

    Parameters
    ----------
    measurements : list[Measurement]
        Input measurements.  Not modified.
    target_energy_unit : str
        Target energy unit ('MeV', 'GeV', 'TeV').  Default 'GeV'.
    target_axis : str
        Target axis type.  Default 'kinetic_energy_per_nucleon'.

    Returns
    -------
    list[Measurement]
        New Measurement instances with converted fields.
    """
    return [
        _harmonise_one(m, target_energy_unit, target_axis) for m in measurements
    ]


def _harmonise_one(
    m: Measurement,
    target_energy_unit: str,
    target_axis: str,
) -> Measurement:
    """Convert a single measurement to the target axis and unit."""
    energy_low = m.energy_low
    energy_high = m.energy_high
    energy_mid = m.energy_mid
    value = m.value
    current_unit = m.unit
    current_axis = m.axis_type

    # Step 1: axis conversion (e.g. rigidity -> kinetic energy per nucleon).
    # After this step, energy values are in GeV (the natural output unit
    # of the rigidity_to_kinetic_energy converter from T006).
    if current_axis == "rigidity" and target_axis != "rigidity":
        species = get_species(m.species)
        energy_low = rigidity_to_kinetic_energy(
            energy_low, species.charge_z, species.mass_gev, species.mass_number_a,
        )
        energy_high = rigidity_to_kinetic_energy(
            energy_high, species.charge_z, species.mass_gev, species.mass_number_a,
        )
        energy_mid = rigidity_to_kinetic_energy(
            energy_mid, species.charge_z, species.mass_gev, species.mass_number_a,
        )
        current_unit = "GeV"
        current_axis = target_axis

    # Step 2: energy unit conversion (e.g. GeV -> MeV).
    # The flux value has an energy unit in its denominator (particles/m²/s/GeV),
    # so changing the energy unit requires dividing by the conversion factor.
    # Example: 100 particles/m²/s/GeV = 0.1 particles/m²/s/MeV because
    # 1 GeV = 1000 MeV, so each MeV bin is 1/1000 of a GeV bin.
    if current_unit != target_energy_unit:
        factor = ENERGY_CONVERSION_FACTORS[(current_unit, target_energy_unit)]
        energy_low = convert_energy(energy_low, current_unit, target_energy_unit)
        energy_high = convert_energy(energy_high, current_unit, target_energy_unit)
        energy_mid = convert_energy(energy_mid, current_unit, target_energy_unit)
        # Flux denominator scales inversely with energy unit size
        value = value / factor

    return Measurement(
        energy_low=energy_low,
        energy_high=energy_high,
        energy_mid=energy_mid,
        value=value,
        unit=target_energy_unit,
        axis_type=current_axis,
        species=m.species,
        stat_error_low=m.stat_error_low,
        stat_error_high=m.stat_error_high,
        sys_error_low=m.sys_error_low,
        sys_error_high=m.sys_error_high,
    )
