"""Schema validators for AMS-02 harmonised records.

Provides physical-bound constants and field-level validation for
canonical AMS-02 data records. Each validator returns a list of
ValidationFinding named-tuples describing any problems found.
"""

from typing import Any, Dict, List, NamedTuple

ValidationFinding = NamedTuple(
    "ValidationFinding",
    [("field", str), ("value", Any), ("reason", str)],
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
    """Validate energy fields against physical bounds and consistency.

    Checks:
    - energy_centre_gev within [ENERGY_MIN_GEV, ENERGY_MAX_GEV]
    - energy_low_gev < energy_high_gev (bin edges must not be inverted)
    """
    findings: List[ValidationFinding] = []

    centre = record.get("energy_centre_gev")
    if centre is not None:
        if centre < ENERGY_MIN_GEV:
            findings.append(ValidationFinding(
                field="energy_centre_gev",
                value=centre,
                reason=f"below minimum {ENERGY_MIN_GEV} GeV",
            ))
        elif centre > ENERGY_MAX_GEV:
            findings.append(ValidationFinding(
                field="energy_centre_gev",
                value=centre,
                reason=f"above maximum {ENERGY_MAX_GEV} GeV",
            ))

    low = record.get("energy_low_gev")
    high = record.get("energy_high_gev")
    if low is not None and high is not None and low >= high:
        findings.append(ValidationFinding(
            field="energy_low_gev",
            value=low,
            reason=f"energy_low_gev ({low}) >= energy_high_gev ({high})",
        ))

    return findings


def validate_flux_fields(record: Dict[str, Any]) -> List[ValidationFinding]:
    """Validate that flux is strictly positive.

    Zero flux is flagged because a measured bin with exactly zero flux
    is almost certainly a placeholder or missing value, not a real
    measurement.
    """
    findings: List[ValidationFinding] = []

    flux = record.get("flux")
    if flux is not None and flux <= FLUX_MIN:
        findings.append(ValidationFinding(
            field="flux",
            value=flux,
            reason="flux must be strictly positive",
        ))

    return findings


def validate_uncertainty_fields(record: Dict[str, Any]) -> List[ValidationFinding]:
    """Validate that statistical uncertainties are non-negative.

    Negative uncertainty values indicate data corruption or parsing errors.
    """
    findings: List[ValidationFinding] = []

    for field_name in REQUIRED_UNCERTAINTY_FIELDS:
        value = record.get(field_name)
        if value is not None and value < 0:
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
