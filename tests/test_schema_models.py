"""Tests for ams02wb.schema.models.Measurement."""

from __future__ import annotations


from ams02wb.schema.models import (
    VALID_SPECIES,
    VALID_UNCERTAINTY_SOURCES,
    Measurement,
    UncertaintyLabel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make(**overrides: object) -> Measurement:
    """Build a valid Measurement, overriding individual fields."""
    defaults: dict[str, object] = {
        "energy_low": 1.0,
        "energy_high": 10.0,
        "energy_mid": 5.0,
        "value": 1.5e-4,
        "unit": "GeV",
        "axis_type": "kinetic_energy_per_nucleon",
        "species": "PROTON",
        "stat_error_low": 2.0e-6,
        "stat_error_high": 2.0e-6,
        "sys_error_low": 3.0e-6,
        "sys_error_high": 3.0e-6,
        "stat_err_label": UncertaintyLabel.PUBLISHED,
        "sys_err_label": UncertaintyLabel.DERIVED,
        "time_start": "2015-05-19",
        "time_end": "2018-11-02",
    }
    defaults.update(overrides)
    return Measurement(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Valid construction
# ---------------------------------------------------------------------------


def test_measurement_valid_construction_all_fields() -> None:
    m = _make()
    assert m.energy_low == 1.0
    assert m.energy_high == 10.0
    assert m.value == 1.5e-4
    assert m.stat_error_low == 2.0e-6
    assert m.sys_error_low == 3.0e-6
    assert m.species == "PROTON"


def test_measurement_valid_construction_optional_fields_none() -> None:
    m = _make(
        sys_error_low=None,
        sys_error_high=None,
        time_start=None,
        time_end=None,
    )
    assert m.sys_error_low is None
    assert m.time_start is None
    assert m.time_end is None


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------


def test_measurement_default_values() -> None:
    m = Measurement()
    assert m.energy_low == 0.0
    assert m.energy_high == 0.0
    assert m.value == 0.0
    assert m.unit == "GeV"
    assert m.axis_type == "kinetic_energy_per_nucleon"
    assert m.species == "PROTON"


# ---------------------------------------------------------------------------
# UncertaintyLabel round-trip
# ---------------------------------------------------------------------------


def test_uncertainty_label_values() -> None:
    assert UncertaintyLabel.PUBLISHED.value == "published"
    assert UncertaintyLabel.DERIVED.value == "derived"
    assert UncertaintyLabel.ASSUMED.value == "assumed"


def test_measurement_accepts_uncertainty_labels() -> None:
    m = _make(
        stat_err_label=UncertaintyLabel.ASSUMED,
        sys_err_label=UncertaintyLabel.PUBLISHED,
    )
    assert m.stat_err_label == UncertaintyLabel.ASSUMED
    assert m.sys_err_label == UncertaintyLabel.PUBLISHED


# ---------------------------------------------------------------------------
# Repr smoke test
# ---------------------------------------------------------------------------


def test_measurement_repr() -> None:
    m = _make()
    r = repr(m)
    assert "Measurement" in r
    assert "energy" in r


# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------


def test_valid_species_contains_all_canonical() -> None:
    expected = {
        "proton", "helium", "electron", "positron", "antiproton",
        "carbon", "oxygen", "lithium", "beryllium", "boron",
        "neon", "magnesium", "silicon",
    }
    assert VALID_SPECIES == expected


def test_valid_uncertainty_sources_contains_all_canonical() -> None:
    assert VALID_UNCERTAINTY_SOURCES == {"published", "derived", "assumed"}
