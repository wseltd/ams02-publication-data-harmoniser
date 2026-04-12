"""Tests for ams02wb.schema.units conversion functions."""

from __future__ import annotations

import math

from ams02wb.schema.units import (
    ENERGY_CONVERSION_FACTORS,
    convert_energy,
    convert_flux,
    kinetic_energy_to_rigidity,
    rigidity_to_kinetic_energy,
)

# ---------------------------------------------------------------------------
# Physical constants used across tests
# ---------------------------------------------------------------------------
PROTON_MASS_GEV = 0.93827  # GeV/c²
HELIUM4_MASS_GEV = 3.7274  # GeV/c², 4He nucleus
HELIUM4_Z = 2
HELIUM4_A = 4


# ---------------------------------------------------------------------------
# Energy conversion tests
# ---------------------------------------------------------------------------


class TestConvertEnergy:
    def test_convert_energy_mev_to_gev_exact(self) -> None:
        result = convert_energy(1000.0, "MeV", "GeV")
        assert abs(result - 1.0) < 1e-12, f"expected 1.0, got {result}"

    def test_convert_energy_gev_to_tev_exact(self) -> None:
        result = convert_energy(1000.0, "GeV", "TeV")
        assert abs(result - 1.0) < 1e-12, f"expected 1.0, got {result}"

    def test_convert_energy_roundtrip_is_identity(self) -> None:
        """Converting MeV->TeV->MeV must recover the original value."""
        original = 42.5
        via_tev = convert_energy(original, "MeV", "TeV")
        back = convert_energy(via_tev, "TeV", "MeV")
        assert math.isclose(back, original, rel_tol=1e-12), (
            f"roundtrip failed: {back} != {original}"
        )

    def test_convert_energy_unknown_unit_raises_value_error(self) -> None:
        try:
            convert_energy(1.0, "eV", "GeV")
        except ValueError as exc:
            assert "Unknown energy unit" in str(exc)
        else:
            raise AssertionError("expected ValueError for unknown source unit")

    def test_convert_energy_unknown_target_raises(self) -> None:
        try:
            convert_energy(1.0, "GeV", "keV")
        except ValueError as exc:
            assert "Unknown energy unit" in str(exc)
        else:
            raise AssertionError("expected ValueError for unknown target unit")

    def test_convert_energy_both_unknown_reports_both(self) -> None:
        try:
            convert_energy(1.0, "eV", "keV")
        except ValueError as exc:
            msg = str(exc)
            assert "eV" in msg and "keV" in msg, (
                f"expected both bad units in message, got: {msg}"
            )
        else:
            raise AssertionError("expected ValueError for two unknown units")

    def test_convert_energy_identity(self) -> None:
        result = convert_energy(3.14, "GeV", "GeV")
        assert math.isclose(result, 3.14), f"identity failed: {result}"

    def test_energy_conversion_factors_completeness(self) -> None:
        """All 9 (from, to) pairs must be present."""
        units = ["MeV", "GeV", "TeV"]
        for a in units:
            for b in units:
                assert (a, b) in ENERGY_CONVERSION_FACTORS, (
                    f"missing factor for ({a}, {b})"
                )


# ---------------------------------------------------------------------------
# Flux conversion tests
# ---------------------------------------------------------------------------


class TestConvertFlux:
    def test_convert_flux_m2_to_cm2_factor_is_1e4(self) -> None:
        """Flux per m² -> per cm² divides by 1e4 (same rate spread over smaller unit)."""
        result = convert_flux(1e4, "m2", "cm2")
        assert math.isclose(result, 1.0, rel_tol=1e-12), (
            f"expected 1.0, got {result}"
        )

    def test_convert_flux_roundtrip_is_identity(self) -> None:
        original = 7.77e-3
        via_cm2 = convert_flux(original, "m2", "cm2")
        back = convert_flux(via_cm2, "cm2", "m2")
        assert math.isclose(back, original, rel_tol=1e-12), (
            f"roundtrip failed: {back} != {original}"
        )

    def test_convert_flux_unknown_unit_raises_value_error(self) -> None:
        try:
            convert_flux(1.0, "m2", "km2")
        except ValueError as exc:
            assert "Unknown flux area unit" in str(exc)
        else:
            raise AssertionError("expected ValueError for unknown flux unit")

    def test_convert_flux_identity(self) -> None:
        result = convert_flux(2.5, "cm2", "cm2")
        assert math.isclose(result, 2.5), f"identity failed: {result}"


# ---------------------------------------------------------------------------
# Rigidity <-> kinetic energy tests
# ---------------------------------------------------------------------------


class TestRigidityKineticEnergy:
    def test_rigidity_to_kinetic_energy_proton_at_1gv(self) -> None:
        """Proton at R=1 GV: Ek = sqrt(1 + m^2) - m."""
        m = PROTON_MASS_GEV
        expected = math.sqrt(1.0 + m * m) - m
        result = rigidity_to_kinetic_energy(1.0, 1, m)
        assert math.isclose(result, expected, rel_tol=1e-9), (
            f"proton 1 GV: {result} != {expected}"
        )

    def test_rigidity_to_kinetic_energy_helium_differs_from_proton(self) -> None:
        """He-4 at same rigidity gives different Ek/n than proton."""
        ek_proton = rigidity_to_kinetic_energy(10.0, 1, PROTON_MASS_GEV)
        ek_helium = rigidity_to_kinetic_energy(
            10.0, HELIUM4_Z, HELIUM4_MASS_GEV, HELIUM4_A
        )
        # They must differ — same rigidity, different Z/A/m
        assert not math.isclose(ek_proton, ek_helium, rel_tol=1e-3), (
            f"proton and helium Ek/n should differ at same rigidity: "
            f"proton={ek_proton}, helium={ek_helium}"
        )

    def test_kinetic_energy_to_rigidity_proton_at_1gev(self) -> None:
        """Proton at Ek=1 GeV: R = sqrt((1+m)^2 - m^2)."""
        m = PROTON_MASS_GEV
        e_total = 1.0 + m
        expected_r = math.sqrt(e_total * e_total - m * m)
        result = kinetic_energy_to_rigidity(1.0, 1, m)
        assert math.isclose(result, expected_r, rel_tol=1e-9), (
            f"proton 1 GeV: {result} != {expected_r}"
        )

    def test_rigidity_kinetic_energy_roundtrip_proton(self) -> None:
        """R -> Ek -> R must recover the original rigidity for proton."""
        r_original = 5.0
        m = PROTON_MASS_GEV
        ek = rigidity_to_kinetic_energy(r_original, 1, m)
        r_back = kinetic_energy_to_rigidity(ek, 1, m)
        assert math.isclose(r_back, r_original, rel_tol=1e-12), (
            f"roundtrip failed: {r_back} != {r_original}"
        )

    def test_rigidity_kinetic_energy_roundtrip_helium(self) -> None:
        """R -> Ek/n -> R must recover the original rigidity for He-4."""
        r_original = 20.0
        ek = rigidity_to_kinetic_energy(
            r_original, HELIUM4_Z, HELIUM4_MASS_GEV, HELIUM4_A
        )
        r_back = kinetic_energy_to_rigidity(
            ek, HELIUM4_Z, HELIUM4_MASS_GEV, HELIUM4_A
        )
        assert math.isclose(r_back, r_original, rel_tol=1e-12), (
            f"roundtrip failed: {r_back} != {r_original}"
        )

    def test_high_rigidity_ultrarelativistic_limit(self) -> None:
        """At very high rigidity, Ek ~ p = Z*R (mass negligible)."""
        r = 1000.0  # 1000 GV
        m = PROTON_MASS_GEV
        ek = rigidity_to_kinetic_energy(r, 1, m)
        # Ek should be very close to R for proton at 1 TV (within ~0.1%)
        assert math.isclose(ek, r, rel_tol=1e-3), (
            f"ultrarelativistic limit: Ek={ek} should be close to R={r}"
        )
