"""Tests for axis name normalisation (ams02wb.schema.axes)."""

from __future__ import annotations

import pytest

from ams02wb.schema.axes import AxisName, list_known_labels, normalise_axis_label


# -------------------------------------------------------------------
# normalise_axis_label — happy paths
# -------------------------------------------------------------------


class TestNormaliseHappyPaths:
    """Verify that common axis label variants resolve correctly."""

    def test_normalise_kinetic_energy_canonical_string(self) -> None:
        assert normalise_axis_label("Kinetic Energy") == AxisName.KINETIC_ENERGY

    def test_normalise_kinetic_energy_with_unit_suffix(self) -> None:
        assert normalise_axis_label("Kinetic Energy (GeV)") == AxisName.KINETIC_ENERGY

    def test_normalise_rigidity_r_gv_shorthand(self) -> None:
        assert normalise_axis_label("R (GV)") == AxisName.RIGIDITY

    def test_normalise_rigidity_full_string(self) -> None:
        assert normalise_axis_label("Rigidity (GV)") == AxisName.RIGIDITY

    def test_normalise_ekin_per_nuc_slash_notation(self) -> None:
        assert normalise_axis_label("Ekin/nucleon") == AxisName.EKIN_PER_NUC

    def test_normalise_ekin_per_nuc_underscore_notation(self) -> None:
        assert normalise_axis_label("ekin_per_nuc") == AxisName.EKIN_PER_NUC


# -------------------------------------------------------------------
# normalise_axis_label — edge cases
# -------------------------------------------------------------------


class TestNormaliseEdgeCases:
    """Verify whitespace and case handling."""

    def test_normalise_is_case_insensitive(self) -> None:
        assert normalise_axis_label("RIGIDITY") == AxisName.RIGIDITY
        assert normalise_axis_label("rigidity") == AxisName.RIGIDITY
        assert normalise_axis_label("Rigidity") == AxisName.RIGIDITY

    def test_normalise_collapses_extra_whitespace(self) -> None:
        assert normalise_axis_label("  Kinetic   Energy  ") == AxisName.KINETIC_ENERGY


# -------------------------------------------------------------------
# normalise_axis_label — error paths
# -------------------------------------------------------------------


class TestNormaliseErrors:
    """Verify that unknown labels raise informative errors."""

    def test_unknown_label_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown axis label") as exc_info:
            normalise_axis_label("totally bogus axis")
        assert type(exc_info.value) is ValueError
        assert "totally bogus axis" in str(exc_info.value)
        assert "Known labels" in str(exc_info.value)

    def test_value_error_message_contains_normalised_label(self) -> None:
        with pytest.raises(ValueError, match="totally bogus axis") as exc_info:
            normalise_axis_label("  Totally  Bogus  Axis  ")
        assert type(exc_info.value) is ValueError
        assert "totally bogus axis" in str(exc_info.value)
        assert "  " not in str(exc_info.value)


# -------------------------------------------------------------------
# list_known_labels
# -------------------------------------------------------------------


class TestListKnownLabels:
    """Verify the label listing helper."""

    def test_list_known_labels_returns_nonempty_sorted_list(self) -> None:
        labels = list_known_labels()
        assert len(labels) > 0
        assert labels == sorted(labels)
