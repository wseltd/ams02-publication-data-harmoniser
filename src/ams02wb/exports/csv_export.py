"""CSV exporter for AMS-02 measurement data in canonical schema order.

Writes a list of Measurement objects to a CSV file using the canonical
field ordering defined by the AMS-02 harmonisation schema.  Nested dict
fields (sys_err_components, metadata_json, provenance_json) are serialised
via json.dumps so that downstream readers can recover them with json.loads.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from ams02wb.schema.models import Measurement

# Canonical column order — every CSV written by this module uses exactly
# these columns in exactly this order.  The list is the single source of
# truth; do not duplicate it elsewhere.
CANONICAL_FIELDS: list[str] = [
    "dataset_id",
    "publication_id",
    "publication_title",
    "publication_url",
    "table_id",
    "species_num",
    "species_den",
    "measurement_type",
    "x_axis_type",
    "x_axis_unit",
    "x_min",
    "x_max",
    "x_centre",
    "y_value",
    "y_unit",
    "time_start",
    "time_stop",
    "time_label",
    "stat_err",
    "sys_err_total",
    "sys_err_components",
    "scale_err",
    "upper_limit_flag",
    "metadata_json",
    "provenance_json",
]


def _measurement_to_row(m: Measurement) -> dict[str, str]:
    """Map a Measurement to a flat dict keyed by canonical field names.

    Fields not present in the Measurement model are left as empty strings.
    Dict-valued fields are serialised as JSON strings.
    """
    sys_err_components: dict[str, float | None] = {
        "stat_err_pos": m.stat_error_high,
        "stat_err_neg": m.stat_error_low,
        "sys_err_pos": m.sys_error_high,
        "sys_err_neg": m.sys_error_low,
    }

    provenance: dict[str, str | None] = {
        "stat_err_label": m.stat_err_label.value if m.stat_err_label else None,
        "sys_err_label": m.sys_err_label.value if m.sys_err_label else None,
        "time_start_utc": m.time_start_utc.isoformat() if m.time_start_utc else None,
        "time_end_utc": m.time_end_utc.isoformat() if m.time_end_utc else None,
    }

    def _opt(v: object) -> str:
        return "" if v is None else str(v)

    return {
        "dataset_id": "",
        "publication_id": "",
        "publication_title": "",
        "publication_url": "",
        "table_id": "",
        "species_num": m.species,
        "species_den": "",
        "measurement_type": "",
        "x_axis_type": m.axis_type,
        "x_axis_unit": m.unit,
        "x_min": str(m.energy_low),
        "x_max": str(m.energy_high),
        "x_centre": str(m.energy_mid),
        "y_value": str(m.value),
        "y_unit": m.unit,
        "time_start": _opt(m.time_start),
        "time_stop": _opt(m.time_end),
        "time_label": "",
        "stat_err": _opt(m.stat_error_high),
        "sys_err_total": _opt(m.sys_error_high),
        "sys_err_components": json.dumps(sys_err_components),
        "scale_err": "",
        "upper_limit_flag": "",
        "metadata_json": json.dumps({}),
        "provenance_json": json.dumps(provenance),
    }


def export_csv_from_dicts(
    records: list[dict], path: Path, fieldnames: list[str] | None = None,
) -> Path:
    """Write a list of dicts (e.g. harmonised records) to CSV.

    Uses CANONICAL_FIELDS as the column order by default. Missing keys
    produce empty cells. Dict-valued fields are JSON-serialised.
    """
    resolved = Path(path).resolve()
    cols = fieldnames or CANONICAL_FIELDS

    with open(resolved, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            row: dict[str, str] = {}
            for col in cols:
                val = rec.get(col, "")
                if isinstance(val, (dict, list)):
                    row[col] = json.dumps(val)
                elif val is None:
                    row[col] = ""
                else:
                    row[col] = str(val)
            writer.writerow(row)

    return resolved


def export_csv(measurements: list[Measurement], path: Path) -> Path:
    """Write measurements to a CSV file in canonical schema order.

    Args:
        measurements: List of Measurement objects to serialise.
        path: Destination file path.  Parent directory must exist.

    Returns:
        Resolved Path to the written CSV file.
    """
    resolved = Path(path).resolve()

    with open(resolved, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CANONICAL_FIELDS)
        writer.writeheader()
        for m in measurements:
            writer.writerow(_measurement_to_row(m))

    return resolved
