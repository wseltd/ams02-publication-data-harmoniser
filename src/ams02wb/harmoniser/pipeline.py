"""Harmonisation pipeline — applies all four stages in fixed order.

Stages run sequentially: species normalisation, axis harmonisation,
uncertainty labelling, time-window normalisation.  Each stage produces
new Measurement instances; originals are never mutated.

Returns canonical-schema dicts with merged provenance metadata so that
downstream consumers (exporters, likelihood builders) get a flat,
self-describing record.
"""

from __future__ import annotations

from datetime import datetime, timezone

from ams02wb.harmoniser.species import normalise_species
from ams02wb.harmoniser.axis import harmonise_axes
from ams02wb.harmoniser.uncertainty import label_uncertainties
from ams02wb.harmoniser.timewindow import normalise_time_window
from ams02wb.parsers.context import ParseContext
from ams02wb.schema.models import Measurement


# Canonical output field names — defined once, used for mapping.
# Species ratio encoding: "PROTON" → num="PROTON", den=None;
# future ratio species like "positron/electron" would split on "/".
_SPECIES_RATIO_SEP = "/"


def _split_species(species: str) -> tuple[str, str | None]:
    """Split species into numerator/denominator for ratio fluxes."""
    if _SPECIES_RATIO_SEP in species:
        num, den = species.split(_SPECIES_RATIO_SEP, 1)
        return num.strip(), den.strip()
    return species, None


def _symmetrise(pos: float | None, neg: float | None) -> float | None:
    """Symmetrise asymmetric errors as (|pos| + |neg|) / 2.

    Returns None if both inputs are None.  If only one side is present,
    uses that value as the symmetric error — best-effort, not invented.
    """
    if pos is not None and neg is not None:
        return (abs(pos) + abs(neg)) / 2.0
    if pos is not None:
        return abs(pos)
    if neg is not None:
        return abs(neg)
    return None


def _measurement_to_canonical(
    m: Measurement,
    original_axis_type: str,
    original_axis_unit: str,
    steps_applied: list[str],
    provenance_json: dict,
) -> dict:
    """Map a harmonised Measurement to a canonical-schema dict."""
    species_num, species_den = _split_species(m.species)

    harmonisation_metadata = {
        "steps_applied": steps_applied,
        "original_x_axis_type": original_axis_type,
        "original_x_axis_unit": original_axis_unit,
        "harmonised_at": datetime.now(timezone.utc).isoformat(),
    }

    # Merge original provenance with harmonisation metadata
    merged_provenance = {**provenance_json, "harmonisation_metadata": harmonisation_metadata}

    # Symmetrise stat errors from low/high pairs
    stat_err = _symmetrise(m.stat_error_low, m.stat_error_high)

    sys_err_total = _symmetrise(m.sys_error_low, m.sys_error_high)

    return {
        "species_num": species_num,
        "species_den": species_den,
        "x_axis_type": m.axis_type,
        "x_axis_unit": m.unit,
        "x_min": m.energy_low,
        "x_max": m.energy_high,
        "x_centre": m.energy_mid,
        "y_value": m.value,
        "y_unit": m.unit,
        "stat_err": stat_err,
        "sys_err_total": sys_err_total,
        "sys_err_components": [],
        "time_start": m.time_start_utc.isoformat() if m.time_start_utc else None,
        "time_stop": m.time_end_utc.isoformat() if m.time_end_utc else None,
        "provenance_json": merged_provenance,
    }


def run_harmonisation_pipeline(
    measurements: list[Measurement],
    parse_context: ParseContext,
    provenance_json: dict,
) -> list[dict]:
    """Apply all four harmonisation stages and return canonical-schema dicts.

    Stages run in fixed order:
      1. normalise_species — canonical species names
      2. harmonise_axes — common energy axis and unit
      3. label_uncertainties — tag each error as published/derived/assumed
      4. normalise_time_window — UTC datetimes for time fields

    Before axis harmonisation, the original axis type and unit are
    snapshotted for provenance tracking.

    Args:
        measurements: Raw Measurement instances from ingestion.
        parse_context: Parser flags for uncertainty labelling.
        provenance_json: Original provenance dict (paper, table, URL).

    Returns:
        List of dicts using canonical schema field names, one per
        measurement, with merged provenance metadata.
    """
    results: list[dict] = []

    for m in measurements:
        steps_applied: list[str] = []

        # Snapshot original axis before any transformation
        original_axis_type = m.axis_type
        original_axis_unit = m.unit

        # Stage 1: species normalisation
        canonical_species = normalise_species(m.species)
        if canonical_species != m.species:
            steps_applied.append("normalise_species")
        m = m.model_copy(update={"species": canonical_species})

        # Stage 2: axis harmonisation
        # harmonise_axes builds new Measurement instances that only carry
        # energy/value/unit/species/stat_error/sys_error fields — it drops
        # time, asymmetric error, and label fields.  Merge the converted
        # energy fields back onto the full measurement to preserve everything.
        harmonised = harmonise_axes([m])[0]
        if harmonised.axis_type != original_axis_type or harmonised.unit != original_axis_unit:
            steps_applied.append("harmonise_axes")
        m = m.model_copy(update={
            "energy_low": harmonised.energy_low,
            "energy_high": harmonised.energy_high,
            "energy_mid": harmonised.energy_mid,
            "value": harmonised.value,
            "unit": harmonised.unit,
            "axis_type": harmonised.axis_type,
        })

        # Stage 3: uncertainty labelling
        labelled = label_uncertainties(m, parse_context)
        if labelled.stat_err_label != m.stat_err_label or labelled.sys_err_label != m.sys_err_label:
            steps_applied.append("label_uncertainties")
        m = labelled

        # Stage 4: time-window normalisation
        time_normalised = normalise_time_window(m)
        if time_normalised.time_start_utc != m.time_start_utc or time_normalised.time_end_utc != m.time_end_utc:
            steps_applied.append("normalise_time_window")
        m = time_normalised

        results.append(
            _measurement_to_canonical(
                m, original_axis_type, original_axis_unit, steps_applied, provenance_json,
            )
        )

    return results
