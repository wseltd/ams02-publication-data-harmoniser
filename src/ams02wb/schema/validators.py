"""Schema validators for AMS-02 harmonised records.

Provides physical-bound constants and field-level validation for
canonical AMS-02 data records. Each validator returns a list of
ValidationFinding named-tuples describing any problems found.
"""

from typing import Any, Dict, List, NamedTuple


class ValidationFinding(NamedTuple):
    """A single validation problem found in a record."""

    field: str
    value: Any
    reason: str

    def __repr__(self) -> str:
        return (
            f"ValidationFinding(field={self.field!r}, "
            f"value={self.value!r}, reason={self.reason!r})"
        )

# Physical bounds for AMS-02 energy measurements (GeV).
# Range covers the full AMS-02 rigidity/energy range across species.
ENERGY_MIN_GEV = 0.1
ENERGY_MAX_GEV = 5000.0

# Flux must be strictly positive for a real measurement.
FLUX_MIN = 0.0

REQUIRED_ENERGY_FIELDS = ["energy_centre_gev", "energy_low_gev", "energy_high_gev"]
REQUIRED_FLUX_FIELDS = ["flux"]
REQUIRED_UNCERTAINTY_FIELDS = ["flux_err_stat_lo", "flux_err_stat_hi"]

_ALL_REQUIRED_FIELDS = (
    REQUIRED_ENERGY_FIELDS + REQUIRED_FLUX_FIELDS + REQUIRED_UNCERTAINTY_FIELDS
)


def validate_energy_fields(record: Dict[str, Any]) -> List[ValidationFinding]:
    """Validate energy fields for presence, type, bounds, and ordering.

    Checks:
    - Each REQUIRED_ENERGY_FIELDS key is present and not None
    - Present values are numeric and within [ENERGY_MIN_GEV, ENERGY_MAX_GEV]
    - energy_low_gev < energy_high_gev (bin edges must not be inverted)
    """
    findings: List[ValidationFinding] = []

    # Track which fields have usable numeric values for the ordering check.
    numeric_values: Dict[str, float] = {}

    for field_name in REQUIRED_ENERGY_FIELDS:
        if field_name not in record or record[field_name] is None:
            findings.append(ValidationFinding(
                field=field_name,
                value=record.get(field_name),
                reason="required field is missing or None",
            ))
            continue

        value = record[field_name]

        if not isinstance(value, (int, float)):
            findings.append(ValidationFinding(
                field=field_name,
                value=value,
                reason=f"expected numeric value, got {type(value).__name__}",
            ))
            continue

        numeric_values[field_name] = value

        if value < ENERGY_MIN_GEV:
            findings.append(ValidationFinding(
                field=field_name,
                value=value,
                reason=f"below minimum {ENERGY_MIN_GEV} GeV",
            ))
        elif value > ENERGY_MAX_GEV:
            findings.append(ValidationFinding(
                field=field_name,
                value=value,
                reason=f"above maximum {ENERGY_MAX_GEV} GeV",
            ))

    # Check bin-edge ordering when both low and high are present and numeric.
    low = numeric_values.get("energy_low_gev")
    high = numeric_values.get("energy_high_gev")
    if low is not None and high is not None and low >= high:
        findings.append(ValidationFinding(
            field="energy_low_gev",
            value=low,
            reason=f"energy_low_gev ({low}) >= energy_high_gev ({high})",
        ))

    return findings


def validate_flux_fields(record: Dict[str, Any]) -> List[ValidationFinding]:
    """Validate flux field for presence, type, and physical bounds.

    Checks:
    - 'flux' key is present and not None
    - Present values are numeric
    - Flux is strictly positive (zero is flagged because a measured bin
      with exactly zero flux is almost certainly a placeholder, not a
      real measurement)
    """
    findings: List[ValidationFinding] = []

    if "flux" not in record or record["flux"] is None:
        findings.append(ValidationFinding(
            field="flux",
            value=record.get("flux"),
            reason="required field is missing or None",
        ))
        return findings

    value = record["flux"]

    if not isinstance(value, (int, float)):
        findings.append(ValidationFinding(
            field="flux",
            value=value,
            reason=f"expected numeric value, got {type(value).__name__}",
        ))
        return findings

    if value <= FLUX_MIN:
        findings.append(ValidationFinding(
            field="flux",
            value=value,
            reason="flux must be strictly positive",
        ))

    return findings


def validate_uncertainty_fields(record: Dict[str, Any]) -> List[ValidationFinding]:
    """Validate statistical uncertainty fields for presence, type, and sign.

    Checks:
    - Each REQUIRED_UNCERTAINTY_FIELDS key is present and not None
    - Present values are numeric
    - Values are >= 0 (negative uncertainty indicates data corruption
      or parsing errors)
    """
    findings: List[ValidationFinding] = []

    for field_name in REQUIRED_UNCERTAINTY_FIELDS:
        if field_name not in record or record[field_name] is None:
            findings.append(ValidationFinding(
                field=field_name,
                value=record.get(field_name),
                reason="required field is missing or None",
            ))
            continue

        value = record[field_name]

        if not isinstance(value, (int, float)):
            findings.append(ValidationFinding(
                field=field_name,
                value=value,
                reason=f"expected numeric value, got {type(value).__name__}",
            ))
            continue

        if value < 0:
            findings.append(ValidationFinding(
                field=field_name,
                value=value,
                reason="statistical uncertainty must be non-negative",
            ))

    return findings


def validate_record(record: Dict[str, Any]) -> List[ValidationFinding]:
    """Run all validators on a single record, collecting all findings.

    Does not short-circuit: every validator runs regardless of earlier
    findings, so callers get a complete picture of all problems.

    Args:
        record: A dict mapping field names to values.

    Returns:
        A list of ValidationFinding for every problem detected.
        Empty list means the record is valid.
    """
    findings: List[ValidationFinding] = []

    # Check for missing required fields first.
    for field_name in _ALL_REQUIRED_FIELDS:
        if field_name not in record:
            findings.append(ValidationFinding(
                field=field_name,
                value=None,
                reason="required field is missing",
            ))

    # Run domain validators regardless of missing fields — they handle
    # None values via .get() and skip gracefully.
    findings.extend(validate_energy_fields(record))
    findings.extend(validate_flux_fields(record))
    findings.extend(validate_uncertainty_fields(record))

    return findings
