"""Uncertainty labeller: tags each uncertainty field with its provenance.

Rule-based tagger driven by ParseContext flags — deterministic, no ML.
Chose a flat if/elif/else over a lookup table because there are only two
axes (stat/sys) and three precedence levels; a table would obscure the
precedence logic without reducing code.
"""

from __future__ import annotations

from ams02wb.parsers.context import ParseContext
from ams02wb.schema.models import Measurement, UncertaintyLabel


def _stat_label(ctx: ParseContext) -> UncertaintyLabel:
    """Determine label for statistical uncertainty fields.

    Precedence: DERIVED (heuristic split produced the stat component)
    beats PUBLISHED (value read from table).  If neither flag is set,
    the value was filled from a default → ASSUMED.
    """
    # err_split_heuristic means total error was heuristically split into
    # stat/sys components — the stat part is a derived quantity even if
    # a table value existed.
    if ctx.err_split_heuristic:
        return UncertaintyLabel.DERIVED
    if ctx.stat_err_from_table:
        return UncertaintyLabel.PUBLISHED
    return UncertaintyLabel.ASSUMED


def _sys_label(ctx: ParseContext) -> UncertaintyLabel:
    """Determine label for systematic uncertainty fields.

    Precedence: DERIVED (symmetrised or heuristic-split) beats PUBLISHED
    (value read from table).  If neither flag is set → ASSUMED.
    """
    # Transformation flags take priority over source flags: a value that
    # was read from a table but then symmetrised is DERIVED, not PUBLISHED,
    # because the number the user sees is no longer the published one.
    if ctx.sys_err_symmetrised or ctx.err_split_heuristic:
        return UncertaintyLabel.DERIVED
    if ctx.sys_err_from_table:
        return UncertaintyLabel.PUBLISHED
    return UncertaintyLabel.ASSUMED


def label_uncertainties(
    measurement: Measurement, parse_context: ParseContext
) -> Measurement:
    """Return a new Measurement with stat_err_label and sys_err_label set.

    Does not modify uncertainty values — only attaches labels indicating
    whether each uncertainty is PUBLISHED, DERIVED, or ASSUMED.

    Args:
        measurement: The measurement to label.
        parse_context: Flags from the parser describing how uncertainties
            were obtained.

    Returns:
        A new Measurement instance with label fields populated.
    """
    return measurement.model_copy(
        update={
            "stat_err_label": _stat_label(parse_context),
            "sys_err_label": _sys_label(parse_context),
        }
    )
