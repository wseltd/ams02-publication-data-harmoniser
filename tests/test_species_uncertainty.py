"""Tests for species normaliser and uncertainty labeller.

Species normaliser tests (6): exercise normalise_species with every known
AMS-02 variant, verifying case-insensitivity and that unmapped names raise
rather than silently passing through.

Uncertainty labeller tests (4): verify label_uncertainties returns exact
UncertaintyLabel values for each provenance path.
"""

from __future__ import annotations

import pytest

from ams02wb.harmoniser.species import normalise_species
from ams02wb.harmoniser.uncertainty import label_uncertainties
from ams02wb.parsers.context import ParseContext
from ams02wb.schema.models import Measurement, UncertaintyLabel


# ---------------------------------------------------------------------------
# Species normaliser — 6 tests
# ---------------------------------------------------------------------------


class TestNormaliseSpeciesVariants:
    """Each test covers one canonical species with all its known aliases,
    including case variants and whitespace to verify trimming / rejection."""

    def test_normalise_species_proton_variants(self) -> None:
        # Canonical and alias forms, various casings
        assert normalise_species("PROTON") == "PROTON"
        assert normalise_species("proton") == "PROTON"
        assert normalise_species("Proton") == "PROTON"
        assert normalise_species("p") == "PROTON"
        # Case-insensitive via .upper() path
        assert normalise_species("PrOtOn") == "PROTON"
        # Whitespace must NOT silently map — padded name is unmapped
        with pytest.raises(ValueError) as exc_info:
            normalise_species(" proton")
        assert "Unknown species" in str(exc_info.value)

    def test_normalise_species_helium_variants(self) -> None:
        assert normalise_species("HELIUM") == "HELIUM"
        assert normalise_species("helium") == "HELIUM"
        assert normalise_species("Helium") == "HELIUM"
        assert normalise_species("he") == "HELIUM"
        assert normalise_species("HeLiUm") == "HELIUM"
        # Trailing whitespace is not stripped — must raise
        with pytest.raises(ValueError) as exc_info:
            normalise_species("helium ")
        assert "Unknown species" in str(exc_info.value)

    def test_normalise_species_electron_variants(self) -> None:
        assert normalise_species("ELECTRON") == "ELECTRON"
        assert normalise_species("electron") == "ELECTRON"
        assert normalise_species("Electron") == "ELECTRON"
        assert normalise_species("e-") == "ELECTRON"
        assert normalise_species("eLECTRON") == "ELECTRON"
        # Symbol alias is case-sensitive: 'E-' goes through .upper() path
        # which yields 'E-' — not in KNOWN_SPECIES, and 'E-' not in _ALIASES
        with pytest.raises(ValueError) as exc_info:
            normalise_species("E-")
        assert "Unknown species" in str(exc_info.value)

    def test_normalise_species_positron_variants(self) -> None:
        assert normalise_species("POSITRON") == "POSITRON"
        assert normalise_species("positron") == "POSITRON"
        assert normalise_species("Positron") == "POSITRON"
        assert normalise_species("e+") == "POSITRON"
        assert normalise_species("poSiTrOn") == "POSITRON"
        # Whitespace around symbol alias must not match
        with pytest.raises(ValueError) as exc_info:
            normalise_species(" e+")
        assert "Unknown species" in str(exc_info.value)

    def test_normalise_species_antiproton_variants(self) -> None:
        # Antiproton is NOT in KNOWN_SPECIES or _ALIASES — every attempt
        # must raise.  This documents that the current species table does
        # not include antiproton, which is the correct behaviour until
        # antiproton data is added to the harmoniser.
        for alias in ("pbar", "antiproton", "ANTIPROTON", "Antiproton"):
            with pytest.raises(ValueError) as exc_info:
                normalise_species(alias)
            assert "Unknown species" in str(exc_info.value)
            assert alias in str(exc_info.value)

    def test_normalise_species_unknown_raises(self) -> None:
        """Completely unknown name must raise with actionable message."""
        with pytest.raises(ValueError) as exc_info:
            normalise_species("MUON")
        err_msg = str(exc_info.value)
        assert "Unknown species" in err_msg
        assert "MUON" in err_msg
        # Error message should list known species to help the caller
        assert "PROTON" in err_msg


# ---------------------------------------------------------------------------
# Uncertainty labeller — 4 tests
# ---------------------------------------------------------------------------


class TestLabelUncertainty:
    """Each test exercises one provenance path through label_uncertainties."""

    def test_label_uncertainty_published(self) -> None:
        """Both errors from table, no transformations → both PUBLISHED."""
        m = Measurement(value=1.0, stat_err_pos=0.1, sys_err_pos=0.2)
        ctx = ParseContext(stat_err_from_table=True, sys_err_from_table=True)
        result = label_uncertainties(m, ctx)
        assert result.stat_err_label == UncertaintyLabel.PUBLISHED
        assert result.sys_err_label == UncertaintyLabel.PUBLISHED
        # Verify exact string values from the enum
        assert result.stat_err_label.value == "published"
        assert result.sys_err_label.value == "published"

    def test_label_uncertainty_derived(self) -> None:
        """Heuristic split → both stat and sys labelled DERIVED,
        even if table flags are also set (derived takes precedence)."""
        m = Measurement(value=1.0, stat_err_pos=0.05, sys_err_pos=0.05)
        ctx = ParseContext(
            stat_err_from_table=True,
            sys_err_from_table=True,
            err_split_heuristic=True,
        )
        result = label_uncertainties(m, ctx)
        assert result.stat_err_label == UncertaintyLabel.DERIVED
        assert result.sys_err_label == UncertaintyLabel.DERIVED
        assert result.stat_err_label.value == "derived"
        assert result.sys_err_label.value == "derived"

    def test_label_uncertainty_assumed(self) -> None:
        """No context flags set → both labels ASSUMED (default path)."""
        m = Measurement(value=1.0)
        ctx = ParseContext()  # all flags False
        result = label_uncertainties(m, ctx)
        assert result.stat_err_label == UncertaintyLabel.ASSUMED
        assert result.sys_err_label == UncertaintyLabel.ASSUMED
        assert result.stat_err_label.value == "assumed"
        assert result.sys_err_label.value == "assumed"

    def test_label_uncertainty_unknown_tag_raises(self) -> None:
        """UncertaintyLabel only accepts the three known values.
        Constructing with an invalid string must raise."""
        with pytest.raises(ValueError) as exc_info:
            UncertaintyLabel("invented")
        err_msg = str(exc_info.value)
        assert "invented" in err_msg.lower() or "not a valid" in err_msg.lower()
