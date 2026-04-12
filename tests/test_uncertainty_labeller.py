"""Tests for the uncertainty labeller.

Focus is on precedence rules and label correctness — the highest-risk
area is DERIVED-beats-PUBLISHED precedence when multiple flags are set.
"""

from __future__ import annotations

from ams02wb.harmoniser.uncertainty import label_uncertainties
from ams02wb.parsers.context import ParseContext
from ams02wb.schema.models import Measurement, UncertaintyLabel


def _make_measurement(
    stat_err_pos: float = 0.1,
    stat_err_neg: float = 0.1,
    sys_err_pos: float = 0.2,
    sys_err_neg: float = 0.2,
) -> Measurement:
    """Helper: build a Measurement with the given uncertainty values."""
    return Measurement(
        energy_low=1.0,
        energy_high=2.0,
        energy_mid=1.5,
        value=10.0,
        stat_err_pos=stat_err_pos,
        stat_err_neg=stat_err_neg,
        sys_err_pos=sys_err_pos,
        sys_err_neg=sys_err_neg,
    )


def test_stat_err_from_table_labelled_published() -> None:
    ctx = ParseContext(stat_err_from_table=True)
    result = label_uncertainties(_make_measurement(), ctx)
    assert result.stat_err_label == UncertaintyLabel.PUBLISHED


def test_sys_err_from_table_labelled_published() -> None:
    ctx = ParseContext(sys_err_from_table=True)
    result = label_uncertainties(_make_measurement(), ctx)
    assert result.sys_err_label == UncertaintyLabel.PUBLISHED


def test_symmetrised_sys_err_labelled_derived() -> None:
    ctx = ParseContext(sys_err_symmetrised=True)
    result = label_uncertainties(_make_measurement(), ctx)
    assert result.sys_err_label == UncertaintyLabel.DERIVED


def test_heuristic_split_labelled_derived() -> None:
    """err_split_heuristic marks both stat and sys as DERIVED."""
    ctx = ParseContext(err_split_heuristic=True)
    result = label_uncertainties(_make_measurement(), ctx)
    assert result.stat_err_label == UncertaintyLabel.DERIVED
    assert result.sys_err_label == UncertaintyLabel.DERIVED


def test_missing_err_labelled_assumed() -> None:
    """No flags set and no values → ASSUMED."""
    ctx = ParseContext()
    result = label_uncertainties(
        _make_measurement(stat_err_pos=0.0, stat_err_neg=0.0,
                          sys_err_pos=0.0, sys_err_neg=0.0),
        ctx,
    )
    assert result.stat_err_label == UncertaintyLabel.ASSUMED
    assert result.sys_err_label == UncertaintyLabel.ASSUMED


def test_derived_precedence_over_published() -> None:
    """A field that is both from_table AND symmetrised must be DERIVED.

    This is the critical precedence rule: transformation takes priority
    over source provenance.
    """
    ctx = ParseContext(
        sys_err_from_table=True,
        sys_err_symmetrised=True,
    )
    result = label_uncertainties(_make_measurement(), ctx)
    assert result.sys_err_label == UncertaintyLabel.DERIVED


def test_all_fields_from_table_all_published() -> None:
    ctx = ParseContext(stat_err_from_table=True, sys_err_from_table=True)
    result = label_uncertainties(_make_measurement(), ctx)
    assert result.stat_err_label == UncertaintyLabel.PUBLISHED
    assert result.sys_err_label == UncertaintyLabel.PUBLISHED


def test_no_context_flags_all_assumed() -> None:
    ctx = ParseContext()
    result = label_uncertainties(_make_measurement(), ctx)
    assert result.stat_err_label == UncertaintyLabel.ASSUMED
    assert result.sys_err_label == UncertaintyLabel.ASSUMED


def test_label_does_not_mutate_uncertainty_values() -> None:
    """Labelling must not change the uncertainty magnitudes."""
    original = _make_measurement(
        stat_err_pos=0.5, stat_err_neg=0.3,
        sys_err_pos=0.8, sys_err_neg=0.7,
    )
    ctx = ParseContext(stat_err_from_table=True, sys_err_symmetrised=True)
    result = label_uncertainties(original, ctx)
    assert result.stat_err_pos == 0.5
    assert result.stat_err_neg == 0.3
    assert result.sys_err_pos == 0.8
    assert result.sys_err_neg == 0.7


def test_returns_new_measurement_instance() -> None:
    """label_uncertainties must not mutate the input — returns a new object."""
    original = _make_measurement()
    ctx = ParseContext(stat_err_from_table=True)
    result = label_uncertainties(original, ctx)
    assert result is not original
    assert original.stat_err_label is None  # original unchanged


def test_mixed_stat_published_sys_derived() -> None:
    """stat from table (PUBLISHED), sys symmetrised (DERIVED)."""
    ctx = ParseContext(stat_err_from_table=True, sys_err_symmetrised=True)
    result = label_uncertainties(_make_measurement(), ctx)
    assert result.stat_err_label == UncertaintyLabel.PUBLISHED
    assert result.sys_err_label == UncertaintyLabel.DERIVED


def test_zero_uncertainty_still_labelled() -> None:
    """Even zero-valued uncertainties get a label — zero is a valid value."""
    m = _make_measurement(
        stat_err_pos=0.0, stat_err_neg=0.0,
        sys_err_pos=0.0, sys_err_neg=0.0,
    )
    ctx = ParseContext(stat_err_from_table=True, sys_err_from_table=True)
    result = label_uncertainties(m, ctx)
    assert result.stat_err_label == UncertaintyLabel.PUBLISHED
    assert result.sys_err_label == UncertaintyLabel.PUBLISHED


def test_heuristic_split_stat_from_table_stat_is_derived() -> None:
    """err_split_heuristic overrides stat_err_from_table → DERIVED.

    If the total error was split heuristically, the stat component is
    derived even if a stat column existed in the table.
    """
    ctx = ParseContext(stat_err_from_table=True, err_split_heuristic=True)
    result = label_uncertainties(_make_measurement(), ctx)
    assert result.stat_err_label == UncertaintyLabel.DERIVED
