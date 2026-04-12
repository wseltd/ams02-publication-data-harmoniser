"""Tests for the axis harmoniser.

Focused on the highest-risk surfaces: flux denominator correction during
unit conversion, and correct propagation of rigidity-to-kinetic-energy
through energy fields.
"""

from __future__ import annotations

import math

from ams02wb.harmoniser.axis import harmonise_axes
from ams02wb.schema.models import Measurement


def _make_measurement(
    energy_low: float = 1.0,
    energy_high: float = 2.0,
    energy_mid: float = 1.5,
    value: float = 100.0,
    unit: str = "GeV",
    axis_type: str = "kinetic_energy_per_nucleon",
    species: str = "PROTON",
) -> Measurement:
    """Build a Measurement with sensible defaults for testing."""
    return Measurement(
        energy_low=energy_low,
        energy_high=energy_high,
        energy_mid=energy_mid,
        value=value,
        unit=unit,
        axis_type=axis_type,
        species=species,
    )


class TestEnergyUnitConversion:
    def test_gev_to_mev_converts_energy_fields(self) -> None:
        """GeV->MeV should multiply all energy fields by 1000."""
        m = _make_measurement(
            energy_low=1.0, energy_high=2.0, energy_mid=1.5,
            unit="GeV",
        )
        result = harmonise_axes([m], target_energy_unit="MeV")
        r = result[0]
        assert math.isclose(r.energy_low, 1000.0, rel_tol=1e-9)
        assert math.isclose(r.energy_high, 2000.0, rel_tol=1e-9)
        assert math.isclose(r.energy_mid, 1500.0, rel_tol=1e-9)
        assert r.unit == "MeV"

    def test_flux_denominator_adjusts_with_unit_change(self) -> None:
        """Flux per GeV != flux per MeV.  Converting GeV->MeV means each
        MeV bin is 1/1000 of a GeV bin, so flux value divides by 1000."""
        m = _make_measurement(value=100.0, unit="GeV")
        result = harmonise_axes([m], target_energy_unit="MeV")
        # 100 particles/m²/s/GeV = 0.1 particles/m²/s/MeV
        assert math.isclose(result[0].value, 0.1, rel_tol=1e-9)

    def test_mev_to_gev_flux_scales_up(self) -> None:
        """Inverse of GeV->MeV: flux per MeV -> flux per GeV multiplies
        by 1000 (each GeV bin contains 1000 MeV bins)."""
        m = _make_measurement(value=0.5, unit="MeV")
        result = harmonise_axes([m], target_energy_unit="GeV")
        assert math.isclose(result[0].value, 500.0, rel_tol=1e-9)


class TestAxisConversion:
    def test_rigidity_to_kinetic_energy_conversion(self) -> None:
        """Rigidity in GV should be converted to kinetic energy per nucleon
        using the proton mass from the species lookup."""
        m = _make_measurement(
            energy_low=1.0, energy_high=10.0, energy_mid=5.0,
            value=50.0, unit="GV", axis_type="rigidity", species="PROTON",
        )
        result = harmonise_axes([m])
        r = result[0]
        assert r.axis_type == "kinetic_energy_per_nucleon"
        assert r.unit == "GeV"
        # Proton at 1 GV: E_k = sqrt(1² + 0.93827²) - 0.93827 ≈ 0.433 GeV
        expected_low = math.sqrt(1.0**2 + 0.93827**2) - 0.93827
        assert math.isclose(r.energy_low, expected_low, rel_tol=1e-4)
        # energy_high (10 GV) should be much larger than energy_low (1 GV)
        assert r.energy_high > r.energy_low

    def test_rigidity_to_ke_then_unit_conversion(self) -> None:
        """Rigidity -> KE/n in GeV, then GeV -> MeV: both conversions
        should compose correctly."""
        m = _make_measurement(
            energy_low=1.0, energy_high=2.0, energy_mid=1.5,
            value=100.0, unit="GV", axis_type="rigidity", species="PROTON",
        )
        result = harmonise_axes([m], target_energy_unit="MeV")
        r = result[0]
        assert r.unit == "MeV"
        assert r.axis_type == "kinetic_energy_per_nucleon"
        # After rigidity->KE, energy is in GeV; then GeV->MeV multiplies by 1000
        expected_low_gev = math.sqrt(1.0**2 + 0.93827**2) - 0.93827
        assert math.isclose(r.energy_low, expected_low_gev * 1000.0, rel_tol=1e-4)


class TestEdgeCases:
    def test_already_normalised_returns_unchanged(self) -> None:
        """Measurement already in target axis and unit should come back
        with identical field values (but as a new instance)."""
        m = _make_measurement(
            energy_low=5.0, energy_high=10.0, energy_mid=7.5,
            value=42.0, unit="GeV", axis_type="kinetic_energy_per_nucleon",
        )
        result = harmonise_axes([m])
        r = result[0]
        assert r.energy_low == 5.0
        assert r.energy_high == 10.0
        assert r.energy_mid == 7.5
        assert r.value == 42.0
        assert r.unit == "GeV"
        assert r.axis_type == "kinetic_energy_per_nucleon"

    def test_empty_list_returns_empty(self) -> None:
        result = harmonise_axes([])
        assert result == []

    def test_original_measurements_not_mutated(self) -> None:
        """harmonise_axes must return new instances, not modify originals."""
        m = _make_measurement(
            energy_low=1.0, energy_high=2.0, energy_mid=1.5,
            value=100.0, unit="GeV",
        )
        original_low = m.energy_low
        original_value = m.value
        result = harmonise_axes([m], target_energy_unit="MeV")
        # Original should be untouched
        assert m.energy_low == original_low
        assert m.value == original_value
        assert m.unit == "GeV"
        # Result should be a different instance
        assert result[0] is not m

    def test_multiple_measurements_all_converted(self) -> None:
        """All measurements in the list should be converted, not just the first."""
        measurements = [
            _make_measurement(energy_low=1.0, value=100.0, unit="GeV"),
            _make_measurement(energy_low=2.0, value=200.0, unit="GeV"),
        ]
        results = harmonise_axes(measurements, target_energy_unit="MeV")
        assert len(results) == 2
        assert math.isclose(results[0].energy_low, 1000.0, rel_tol=1e-9)
        assert math.isclose(results[1].energy_low, 2000.0, rel_tol=1e-9)
