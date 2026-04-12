"""Tests for axis normalisation — canonical mappings, case-insensitivity, unknown fallback.

Covers the public API of ams02wb.schema.axes (normalise_axis_label).
Does not test internal regex or lookup tables directly.
"""

from __future__ import annotations

import pytest

from ams02wb.schema.axes import AxisName, normalise_axis_label


# -------------------------------------------------------------------
# Canonical header mappings (3 cases)
# -------------------------------------------------------------------


def test_rigidity_gv_header_normalises() -> None:
    """'Rigidity (GV)' resolves to the canonical RIGIDITY axis."""
    assert normalise_axis_label("Rigidity (GV)") == AxisName.RIGIDITY


def test_kinetic_energy_gev_header_normalises() -> None:
    """'Kinetic Energy (GeV)' resolves to the canonical KINETIC_ENERGY axis."""
    assert normalise_axis_label("Kinetic Energy (GeV)") == AxisName.KINETIC_ENERGY


def test_kinetic_energy_per_nucleon_header_normalises() -> None:
    """'Kinetic Energy per nucleon' resolves to KINETIC_ENERGY_PER_NUC."""
    assert normalise_axis_label("Kinetic Energy per nucleon") == AxisName.KINETIC_ENERGY_PER_NUC


# -------------------------------------------------------------------
# Case-insensitive matching (3 cases)
# Highest-risk area: AMS tables use inconsistent capitalisation across
# publications (e.g. 'RIGIDITY (GV)', 'rigidity (gv)', 'Rigidity (Gv)').
# -------------------------------------------------------------------


def test_case_insensitive_rigidity() -> None:
    """All-caps and all-lower rigidity headers resolve identically."""
    assert normalise_axis_label("RIGIDITY (GV)") == AxisName.RIGIDITY
    assert normalise_axis_label("rigidity (gv)") == AxisName.RIGIDITY


def test_case_insensitive_kinetic_energy() -> None:
    """All-caps and all-lower kinetic energy headers resolve identically."""
    assert normalise_axis_label("KINETIC ENERGY (GEV)") == AxisName.KINETIC_ENERGY
    assert normalise_axis_label("kinetic energy (gev)") == AxisName.KINETIC_ENERGY


def test_case_insensitive_mixed_case() -> None:
    """Mixed-case headers (common in AMS publications) still normalise."""
    assert normalise_axis_label("Rigidity (Gv)") == AxisName.RIGIDITY
    assert normalise_axis_label("kinetic ENERGY (Gev)") == AxisName.KINETIC_ENERGY


# -------------------------------------------------------------------
# Unknown-header fallback (2 cases)
# Must raise ValueError with actionable message — never silently map
# to a wrong axis, which would corrupt downstream species joins.
# -------------------------------------------------------------------


def test_unknown_header_returns_sentinel() -> None:
    """An unrecognised non-empty header raises ValueError with the normalised label."""
    with pytest.raises(ValueError, match="Unknown axis label") as exc_info:
        normalise_axis_label("Totally Unknown Axis")
    # Verify the error message includes the normalised form for debugging
    assert "totally unknown axis" in str(exc_info.value)


def test_unknown_header_empty_string() -> None:
    """An empty string is not a valid axis label and must raise ValueError."""
    with pytest.raises(ValueError, match="Unknown axis label") as exc_info:
        normalise_axis_label("")
    # Verify the error references what was actually passed (empty after normalisation)
    assert "Known labels" in str(exc_info.value)
