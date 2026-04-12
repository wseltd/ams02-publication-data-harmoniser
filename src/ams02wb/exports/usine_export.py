"""USINE-format exporter for AMS-02 measurement data.

Produces plain-text files compatible with the USINE cosmic-ray database format.
Each file has a comment-prefixed header block followed by fixed-width data columns.

The mapping constants (AXIS_TO_USINE, MEASUREMENT_TYPE_TO_USINE) and the
quantity-string builder live here rather than in a separate constants module
because they are only consumed by this exporter — no other module needs them.
"""

from __future__ import annotations

from ams02wb.schema.models import Measurement

# --- Mapping constants ------------------------------------------------

# Canonical x_axis_type values → USINE energy type codes.
AXIS_TO_USINE: dict[str, str] = {
    "rigidity": "R",
    "kinetic_energy": "Ek",
    "kinetic_energy_per_nucleon": "Ekn",
}

# Canonical measurement_type → USINE data type strings.
MEASUREMENT_TYPE_TO_USINE: dict[str, str] = {
    "flux": "flux",
    "ratio": "ratio",
}


# --- Pure helpers -----------------------------------------------------

def build_usine_quantity_string(
    species_num: str,
    species_den: str | None,
) -> str:
    """Build a USINE quantity string from numerator and optional denominator.

    Args:
        species_num: Primary (numerator) species, e.g. 'B', 'H'.
        species_den: Denominator species for ratios, or None for single-species.

    Returns:
        USINE quantity string — 'B/C' for ratios, 'H' for single-species flux.
    """
    if species_den:
        return f"{species_num}/{species_den}"
    return species_num


# --- Formatter --------------------------------------------------------

# Fixed column widths for the data section.  Chosen to align with USINE
# conventions and keep columns readable at a glance.
_COL_WIDTH = 14


def _format_header(
    quantity: str,
    energy_type: str,
    experiment: str,
    bibcode: str | None,
) -> str:
    """Build the comment-prefixed header block."""
    lines = [
        f"# experiment {experiment}",
        f"# quantity {quantity}",
        f"# energy_type {energy_type}",
    ]
    if bibcode:
        lines.append(f"# bibcode {bibcode}")
    return "\n".join(lines)


def _format_data_rows(
    measurements: list[Measurement],
    columns: list[str],
) -> str:
    """Format measurement data as fixed-width text lines."""
    header = "".join(col.ljust(_COL_WIDTH) for col in columns)
    rows = [f"# {header}"]

    for m in measurements:
        vals = [
            m.energy_low,
            m.energy_high,
            m.energy_mid,
            m.value,
            m.stat_err_pos,
            m.sys_err_pos,
        ]
        row = "".join(
            str(v if v is not None else 0.0).ljust(_COL_WIDTH) for v in vals
        )
        rows.append(row)

    return "\n".join(rows)


# --- Public entry point -----------------------------------------------

_DATA_COLUMNS = ["e_low", "e_high", "e_mid", "value", "stat_err", "sys_err"]


def export_usine(
    measurements: list[Measurement],
    *,
    species_num: str,
    species_den: str | None = None,
    x_axis_type: str = "kinetic_energy_per_nucleon",
    experiment: str = "AMS-02",
    bibcode: str | None = None,
) -> str:
    """Export measurements to a USINE-format string.

    Args:
        measurements: Data points to export.
        species_num: Numerator species name (e.g. 'H', 'B').
        species_den: Denominator species for ratios, or None for flux.
        x_axis_type: Canonical axis type — must be a key in AXIS_TO_USINE.
        experiment: Experiment name for the header (default 'AMS-02').
        bibcode: Optional ADS bibcode to include in the header.

    Returns:
        Complete USINE-format text as a string.

    Raises:
        ValueError: If x_axis_type is not a recognised USINE energy type.
    """
    if x_axis_type not in AXIS_TO_USINE:
        raise ValueError(
            f"Unknown x_axis_type {x_axis_type!r}; "
            f"expected one of {sorted(AXIS_TO_USINE)}"
        )

    quantity = build_usine_quantity_string(species_num, species_den)
    energy_type = AXIS_TO_USINE[x_axis_type]

    header = _format_header(quantity, energy_type, experiment, bibcode)
    data = _format_data_rows(measurements, _DATA_COLUMNS)

    return f"{header}\n{data}\n"
