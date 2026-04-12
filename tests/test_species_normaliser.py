"""Tests for species normalisation and property lookup."""

from __future__ import annotations

from ams02wb.harmoniser.species import (
    KNOWN_SPECIES,
    get_species,
    normalise_species,
)


class TestNormaliseSpecies:
    def test_canonical_name_returns_itself(self) -> None:
        assert normalise_species("PROTON") == "PROTON"
        assert normalise_species("HELIUM") == "HELIUM"

    def test_lowercase_normalises_to_uppercase(self) -> None:
        assert normalise_species("proton") == "PROTON"
        assert normalise_species("helium") == "HELIUM"

    def test_alias_maps_correctly(self) -> None:
        assert normalise_species("p") == "PROTON"
        assert normalise_species("e-") == "ELECTRON"
        assert normalise_species("e+") == "POSITRON"

    def test_unknown_species_raises_value_error(self) -> None:
        raised = False
        try:
            normalise_species("MUON")
        except ValueError:
            raised = True
        assert raised, "Expected ValueError for unknown species"


class TestGetSpecies:
    def test_returns_correct_properties(self) -> None:
        proton = get_species("PROTON")
        assert proton.charge_z == 1
        assert proton.mass_number_a == 1
        assert proton.mass_gev > 0.93

    def test_helium_has_charge_2(self) -> None:
        he = get_species("HELIUM")
        assert he.charge_z == 2
        assert he.mass_number_a == 4

    def test_alias_lookup_returns_species(self) -> None:
        proton = get_species("p")
        assert proton.name == "PROTON"

    def test_all_known_species_have_positive_mass(self) -> None:
        for name, sp in KNOWN_SPECIES.items():
            assert sp.mass_gev > 0, f"{name} has non-positive mass"
