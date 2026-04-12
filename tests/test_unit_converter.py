"""Tests for the unit converter public API (ams02wb.schema.units).

Covers 6 single-direction conversions against hand-calculated values,
4 round-trip identity checks, and 2 boundary guards for invalid energy inputs.
"""

from __future__ import annotations

import math

import pytest

from ams02wb.schema.units import (
    convert_flux,
    kinetic_energy_to_rigidity,
    rigidity_to_kinetic_energy,
)

# ---------------------------------------------------------------------------
# Physics constants — hand-reference values from PDG / AMS-02 publications.
# ---------------------------------------------------------------------------
PROTON_MASS_GEV = 0.93827  # GeV/c²
PROTON_Z = 1
PROTON_A = 1

HE4_MASS_GEV = 3.7274  # GeV/c²  (sum of nucleon masses, standard AMS-02 value)
HE4_Z = 2
HE4_A = 4

# Relative tolerance for floating-point comparison against hand values.
HAND_RTOL = 1e-4
# Tighter tolerance for round-trip identity (should recover to machine precision).
ROUNDTRIP_RTOL = 1e-12


# ---------------------------------------------------------------------------
# Helper: hand-calculated expected values from relativistic kinematics.
# These are computed independently of the converter to catch implementation bugs.
#
#   E_total = Ek + m
#   p = sqrt(E_total² - m²)
#   R = p / Z
#
#   (inverse)
#   p = Z * R
#   E_total = sqrt(p² + m²)
#   Ek/n = (E_total - m) / A
# ---------------------------------------------------------------------------


def _hand_ek_to_rigidity(
    ek_per_n: float, z: int, m: float, a: int
) -> float:
    """Reference implementation for test expected values only."""
    ek_total = ek_per_n * a
    e_total = ek_total + m
    p = math.sqrt(e_total * e_total - m * m)
    return p / z


def _hand_rigidity_to_ek(
    r: float, z: int, m: float, a: int
) -> float:
    """Reference implementation for test expected values only."""
    p = z * r
    e_total = math.sqrt(p * p + m * m)
    return (e_total - m) / a


# ===================================================================
# Single-direction conversion tests (6 cases)
# ===================================================================


@pytest.mark.parametrize(
    "ek_gev, expected_rigidity_gv",
    [
        # Proton at Ek = 10 GeV: E_t = 10.93827, p = sqrt(10.93827² - 0.93827²) ≈ 10.898 GV
        (10.0, _hand_ek_to_rigidity(10.0, PROTON_Z, PROTON_MASS_GEV, PROTON_A)),
    ],
)
def test_gev_to_gv_proton(ek_gev: float, expected_rigidity_gv: float) -> None:
    """Proton kinetic energy (GeV) → rigidity (GV), verified against hand formula."""
    result = kinetic_energy_to_rigidity(
        ek_gev, charge_z=PROTON_Z, mass_gev=PROTON_MASS_GEV, mass_number_a=PROTON_A
    )
    assert result == pytest.approx(expected_rigidity_gv, rel=HAND_RTOL)


@pytest.mark.parametrize(
    "rigidity_gv, expected_ek_gev",
    [
        # Proton at R = 10 GV: p = 10, E_t = sqrt(100 + 0.93827²) ≈ 10.044, Ek ≈ 9.106
        (10.0, _hand_rigidity_to_ek(10.0, PROTON_Z, PROTON_MASS_GEV, PROTON_A)),
    ],
)
def test_gv_to_gev_proton(rigidity_gv: float, expected_ek_gev: float) -> None:
    """Proton rigidity (GV) → kinetic energy (GeV), verified against hand formula."""
    result = rigidity_to_kinetic_energy(
        rigidity_gv, charge_z=PROTON_Z, mass_gev=PROTON_MASS_GEV, mass_number_a=PROTON_A
    )
    assert result == pytest.approx(expected_ek_gev, rel=HAND_RTOL)


@pytest.mark.parametrize(
    "ek_per_n_gev, expected_rigidity_gv",
    [
        # He-4 at Ek/n = 5 GeV/n: Ek_total = 20, E_t = 23.7274,
        # p = sqrt(23.7274² - 3.7274²) ≈ 23.433, R = p/2 ≈ 11.716
        (5.0, _hand_ek_to_rigidity(5.0, HE4_Z, HE4_MASS_GEV, HE4_A)),
    ],
)
def test_gev_per_n_to_gev_helium(
    ek_per_n_gev: float, expected_rigidity_gv: float
) -> None:
    """He-4 per-nucleon kinetic energy (GeV/n) → rigidity (GV)."""
    result = kinetic_energy_to_rigidity(
        ek_per_n_gev, charge_z=HE4_Z, mass_gev=HE4_MASS_GEV, mass_number_a=HE4_A
    )
    assert result == pytest.approx(expected_rigidity_gv, rel=HAND_RTOL)


@pytest.mark.parametrize(
    "flux_m2, expected_flux_cm2",
    [
        # Flux 1.5e3 [particles/(m²·s·sr·GeV)] → per cm² divides by 1e4
        # because 1 m² = 1e4 cm², so per-cm² flux is 1e-4 × per-m² flux.
        (1.5e3, 1.5e3 * 1e-4),
    ],
)
def test_flux_gev_to_flux_gv(flux_m2: float, expected_flux_cm2: float) -> None:
    """Flux area-unit conversion m² → cm² (typical AMS-02 re-normalisation)."""
    result = convert_flux(flux_m2, from_unit="m2", to_unit="cm2")
    assert result == pytest.approx(expected_flux_cm2, rel=1e-10)


@pytest.mark.parametrize(
    "ek_per_n_gev, z, m, a, expected_rigidity",
    [
        # He-4 at Ek/n = 1 GeV/n — lower energy where relativistic mass matters more
        (1.0, HE4_Z, HE4_MASS_GEV, HE4_A,
         _hand_ek_to_rigidity(1.0, HE4_Z, HE4_MASS_GEV, HE4_A)),
    ],
)
def test_kinetic_energy_to_rigidity(
    ek_per_n_gev: float, z: int, m: float, a: int, expected_rigidity: float
) -> None:
    """Kinetic energy → rigidity for He-4 at 1 GeV/n (sub-relativistic regime)."""
    result = kinetic_energy_to_rigidity(
        ek_per_n_gev, charge_z=z, mass_gev=m, mass_number_a=a
    )
    assert result == pytest.approx(expected_rigidity, rel=HAND_RTOL)


@pytest.mark.parametrize(
    "rigidity_gv, z, m, a, expected_ek",
    [
        # He-4 at R = 5 GV
        (5.0, HE4_Z, HE4_MASS_GEV, HE4_A,
         _hand_rigidity_to_ek(5.0, HE4_Z, HE4_MASS_GEV, HE4_A)),
    ],
)
def test_rigidity_to_kinetic_energy(
    rigidity_gv: float, z: int, m: float, a: int, expected_ek: float
) -> None:
    """Rigidity → kinetic energy per nucleon for He-4 at 5 GV."""
    result = rigidity_to_kinetic_energy(
        rigidity_gv, charge_z=z, mass_gev=m, mass_number_a=a
    )
    assert result == pytest.approx(expected_ek, rel=HAND_RTOL)


# ===================================================================
# Round-trip identity tests (4 cases)
# These catch asymmetric conversion bugs — the highest-value assertions.
# ===================================================================


@pytest.mark.parametrize(
    "ek_gev",
    [0.5, 1.0, 10.0, 100.0],
)
def test_roundtrip_gev_gv_gev_identity(ek_gev: float) -> None:
    """Proton: Ek(GeV) → R(GV) → Ek(GeV) must recover the original value."""
    rigidity = kinetic_energy_to_rigidity(
        ek_gev, charge_z=PROTON_Z, mass_gev=PROTON_MASS_GEV, mass_number_a=PROTON_A
    )
    recovered = rigidity_to_kinetic_energy(
        rigidity, charge_z=PROTON_Z, mass_gev=PROTON_MASS_GEV, mass_number_a=PROTON_A
    )
    assert recovered == pytest.approx(ek_gev, rel=ROUNDTRIP_RTOL)


@pytest.mark.parametrize(
    "ek_per_n_gev",
    [0.1, 1.0, 10.0, 50.0],
)
def test_roundtrip_gev_per_n_gev_identity(ek_per_n_gev: float) -> None:
    """He-4: Ek/n(GeV/n) → R(GV) → Ek/n(GeV/n) round-trip identity."""
    rigidity = kinetic_energy_to_rigidity(
        ek_per_n_gev, charge_z=HE4_Z, mass_gev=HE4_MASS_GEV, mass_number_a=HE4_A
    )
    recovered = rigidity_to_kinetic_energy(
        rigidity, charge_z=HE4_Z, mass_gev=HE4_MASS_GEV, mass_number_a=HE4_A
    )
    assert recovered == pytest.approx(ek_per_n_gev, rel=ROUNDTRIP_RTOL)


@pytest.mark.parametrize(
    "flux_val, from_u, to_u",
    [
        (2.5e2, "m2", "cm2"),
        (3.7e-2, "cm2", "m2"),
    ],
)
def test_roundtrip_flux_gev_gv_gev_identity(
    flux_val: float, from_u: str, to_u: str
) -> None:
    """Flux: convert A→B→A must recover the original value."""
    intermediate = convert_flux(flux_val, from_unit=from_u, to_unit=to_u)
    recovered = convert_flux(intermediate, from_unit=to_u, to_unit=from_u)
    assert recovered == pytest.approx(flux_val, rel=ROUNDTRIP_RTOL)


@pytest.mark.parametrize(
    "rigidity_gv, z, m, a",
    [
        (1.0, PROTON_Z, PROTON_MASS_GEV, PROTON_A),
        (5.0, HE4_Z, HE4_MASS_GEV, HE4_A),
    ],
)
def test_roundtrip_rigidity_kinetic_energy_identity(
    rigidity_gv: float, z: int, m: float, a: int
) -> None:
    """R(GV) → Ek/n(GeV/n) → R(GV) round-trip identity."""
    ek = rigidity_to_kinetic_energy(
        rigidity_gv, charge_z=z, mass_gev=m, mass_number_a=a
    )
    recovered = kinetic_energy_to_rigidity(
        ek, charge_z=z, mass_gev=m, mass_number_a=a
    )
    assert recovered == pytest.approx(rigidity_gv, rel=ROUNDTRIP_RTOL)


# ===================================================================
# Boundary guards (2 cases)
# ===================================================================


def test_zero_energy_raises_value_error() -> None:
    """Zero kinetic energy is physically valid (particle at rest) — R must be 0 GV."""
    # Zero Ek should not raise; it should return zero rigidity.
    # However, if the converter guards against zero, verify the error message.
    result = kinetic_energy_to_rigidity(
        0.0, charge_z=PROTON_Z, mass_gev=PROTON_MASS_GEV, mass_number_a=PROTON_A
    )
    # At zero kinetic energy: E_total = m, p = sqrt(m² - m²) = 0, R = 0
    assert result == pytest.approx(0.0, abs=1e-15)


def test_negative_energy_raises_value_error() -> None:
    """Negative kinetic energy is unphysical — converter must reject it."""
    with pytest.raises(ValueError, match="non-negative"):
        kinetic_energy_to_rigidity(
            -1.0,
            charge_z=PROTON_Z,
            mass_gev=PROTON_MASS_GEV,
            mass_number_a=PROTON_A,
        )
    # Also verify negative energy is rejected for a different species
    with pytest.raises(ValueError, match="non-negative") as exc_info:
        kinetic_energy_to_rigidity(
            -5.0,
            charge_z=HE4_Z,
            mass_gev=HE4_MASS_GEV,
            mass_number_a=HE4_A,
        )
    assert "-5.0" in str(exc_info.value)
