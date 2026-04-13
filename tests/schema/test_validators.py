"""Tests for ams02wb.schema.validators."""

from ams02wb.schema.validators import (
    ENERGY_MAX_GEV,
    ENERGY_MIN_GEV,
    ValidationFinding,
    validate_energy_fields,
    validate_flux_fields,
    validate_record,
    validate_uncertainty_fields,
)


def _valid_record():
    """A fully valid record with all required fields."""
    return {
        "x_centre": 10.0,
        "x_min": 9.0,
        "x_max": 11.0,
        "y_value": 1.5e-4,
        "stat_err": 0.1e-4,
    }


# --- validate_record integration ---


def test_validate_record_returns_empty_list_for_valid_row():
    findings = validate_record(_valid_record())
    assert findings == [], f"expected no findings, got {findings}"


def test_validate_record_collects_all_findings_without_short_circuit():
    """A record with multiple problems should report ALL of them."""
    record = {
        "x_centre": -1.0,
        "x_min": 5.0,
        "x_max": 3.0,
        "y_value": -0.5,
        "stat_err": -0.01,
    }
    findings = validate_record(record)
    fields_with_findings = {f.field for f in findings}
    assert "x_centre" in fields_with_findings
    assert "x_min" in fields_with_findings
    assert "y_value" in fields_with_findings
    assert "stat_err" in fields_with_findings
    assert len(findings) >= 4, f"expected at least 4 findings, got {len(findings)}"


# --- Energy validation ---


def test_energy_centre_below_minimum_produces_finding():
    record = _valid_record()
    record["x_centre"] = 0.01
    findings = validate_energy_fields(record)
    assert len(findings) == 1
    assert findings[0].field == "x_centre"
    assert "below minimum" in findings[0].reason


def test_energy_centre_above_maximum_produces_finding():
    record = _valid_record()
    record["x_centre"] = 10_000.0
    findings = validate_energy_fields(record)
    assert len(findings) == 1
    assert findings[0].field == "x_centre"
    assert "above maximum" in findings[0].reason


def test_energy_low_above_energy_high_produces_finding():
    record = _valid_record()
    record["x_min"] = 15.0
    record["x_max"] = 10.0
    findings = validate_energy_fields(record)
    low_findings = [f for f in findings if f.field == "x_min"]
    assert len(low_findings) == 1
    assert "x_min" in low_findings[0].reason


def test_energy_at_exact_minimum_is_valid():
    record = _valid_record()
    record["x_centre"] = ENERGY_MIN_GEV
    findings = validate_energy_fields(record)
    assert findings == [], f"exact minimum should be valid, got {findings}"


def test_energy_at_exact_maximum_is_valid():
    record = _valid_record()
    record["x_centre"] = ENERGY_MAX_GEV
    findings = validate_energy_fields(record)
    assert findings == [], f"exact maximum should be valid, got {findings}"


# --- Flux validation ---


def test_negative_flux_produces_finding():
    record = _valid_record()
    record["y_value"] = -1.0e-5
    findings = validate_flux_fields(record)
    assert len(findings) == 1
    assert findings[0].field == "y_value"
    assert "strictly positive" in findings[0].reason


def test_zero_flux_produces_finding():
    record = _valid_record()
    record["y_value"] = 0.0
    findings = validate_flux_fields(record)
    assert len(findings) == 1
    assert findings[0].field == "y_value"
    assert "strictly positive" in findings[0].reason


# --- Uncertainty validation ---


def test_negative_stat_uncertainty_produces_finding():
    record = _valid_record()
    record["stat_err"] = -0.01
    findings = validate_uncertainty_fields(record)
    assert len(findings) == 1
    assert findings[0].field == "stat_err"
    assert "non-negative" in findings[0].reason


def test_zero_uncertainty_is_valid():
    record = _valid_record()
    record["stat_err"] = 0.0
    findings = validate_uncertainty_fields(record)
    assert findings == [], f"zero uncertainty should be valid, got {findings}"


# --- Missing fields ---


def test_missing_required_field_produces_finding():
    record = _valid_record()
    del record["y_value"]
    del record["x_max"]
    findings = validate_record(record)
    missing_fields = {f.field for f in findings if "missing" in f.reason}
    assert "y_value" in missing_fields
    assert "x_max" in missing_fields


# --- Finding attributes ---


def test_finding_field_attribute_matches_offending_key():
    record = _valid_record()
    record["x_centre"] = -5.0
    findings = validate_record(record)
    energy_findings = [f for f in findings if f.field == "x_centre"]
    assert len(energy_findings) == 1
    assert energy_findings[0].value == -5.0
    assert isinstance(energy_findings[0], ValidationFinding)


def test_finding_repr_contains_all_fields():
    finding = ValidationFinding(field="y_value", value=-1.0, reason="bad")
    text = repr(finding)
    assert "y_value" in text
    assert "-1.0" in text
    assert "bad" in text
