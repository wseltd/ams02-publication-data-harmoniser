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
        "energy_centre_gev": 10.0,
        "energy_low_gev": 9.0,
        "energy_high_gev": 11.0,
        "flux": 1.5e-4,
        "flux_err_stat_lo": 0.1e-4,
        "flux_err_stat_hi": 0.1e-4,
    }


# --- validate_record integration ---


def test_validate_record_returns_empty_list_for_valid_row():
    findings = validate_record(_valid_record())
    assert findings == [], f"expected no findings, got {findings}"


def test_validate_record_collects_all_findings_without_short_circuit():
    """A record with multiple problems should report ALL of them."""
    record = {
        "energy_centre_gev": -1.0,
        "energy_low_gev": 5.0,
        "energy_high_gev": 3.0,
        "flux": -0.5,
        "flux_err_stat_lo": -0.01,
        "flux_err_stat_hi": 0.01,
    }
    findings = validate_record(record)
    fields_with_findings = {f.field for f in findings}
    # energy_centre below min, low >= high, negative flux, negative uncertainty
    assert "energy_centre_gev" in fields_with_findings
    assert "energy_low_gev" in fields_with_findings
    assert "flux" in fields_with_findings
    assert "flux_err_stat_lo" in fields_with_findings
    assert len(findings) >= 4, f"expected at least 4 findings, got {len(findings)}"


# --- Energy validation ---


def test_energy_centre_below_minimum_produces_finding():
    record = _valid_record()
    record["energy_centre_gev"] = 0.01
    findings = validate_energy_fields(record)
    assert len(findings) == 1
    assert findings[0].field == "energy_centre_gev"
    assert "below minimum" in findings[0].reason


def test_energy_centre_above_maximum_produces_finding():
    record = _valid_record()
    record["energy_centre_gev"] = 10_000.0
    findings = validate_energy_fields(record)
    assert len(findings) == 1
    assert findings[0].field == "energy_centre_gev"
    assert "above maximum" in findings[0].reason


def test_energy_low_above_energy_high_produces_finding():
    record = _valid_record()
    record["energy_low_gev"] = 15.0
    record["energy_high_gev"] = 10.0
    findings = validate_energy_fields(record)
    low_findings = [f for f in findings if f.field == "energy_low_gev"]
    assert len(low_findings) == 1
    assert "energy_low_gev" in low_findings[0].reason


def test_energy_at_exact_minimum_is_valid():
    record = _valid_record()
    record["energy_centre_gev"] = ENERGY_MIN_GEV
    findings = validate_energy_fields(record)
    assert findings == [], f"exact minimum should be valid, got {findings}"


def test_energy_at_exact_maximum_is_valid():
    record = _valid_record()
    record["energy_centre_gev"] = ENERGY_MAX_GEV
    findings = validate_energy_fields(record)
    assert findings == [], f"exact maximum should be valid, got {findings}"


# --- Flux validation ---


def test_negative_flux_produces_finding():
    record = _valid_record()
    record["flux"] = -1.0e-5
    findings = validate_flux_fields(record)
    assert len(findings) == 1
    assert findings[0].field == "flux"
    assert "strictly positive" in findings[0].reason


def test_zero_flux_produces_finding():
    record = _valid_record()
    record["flux"] = 0.0
    findings = validate_flux_fields(record)
    assert len(findings) == 1
    assert findings[0].field == "flux"
    assert "strictly positive" in findings[0].reason


# --- Uncertainty validation ---


def test_negative_stat_uncertainty_produces_finding():
    record = _valid_record()
    record["flux_err_stat_lo"] = -0.01
    findings = validate_uncertainty_fields(record)
    assert len(findings) == 1
    assert findings[0].field == "flux_err_stat_lo"
    assert "non-negative" in findings[0].reason


def test_zero_uncertainty_is_valid():
    record = _valid_record()
    record["flux_err_stat_lo"] = 0.0
    record["flux_err_stat_hi"] = 0.0
    findings = validate_uncertainty_fields(record)
    assert findings == [], f"zero uncertainty should be valid, got {findings}"


# --- Missing fields ---


def test_missing_required_field_produces_finding():
    record = _valid_record()
    del record["flux"]
    del record["energy_high_gev"]
    findings = validate_record(record)
    missing_fields = {f.field for f in findings if "missing" in f.reason}
    assert "flux" in missing_fields
    assert "energy_high_gev" in missing_fields


# --- Finding attributes ---


def test_finding_field_attribute_matches_offending_key():
    record = _valid_record()
    record["energy_centre_gev"] = -5.0
    findings = validate_record(record)
    energy_findings = [f for f in findings if f.field == "energy_centre_gev"]
    assert len(energy_findings) == 1
    assert energy_findings[0].value == -5.0
    assert isinstance(energy_findings[0], ValidationFinding)


def test_finding_repr_contains_all_fields():
    finding = ValidationFinding(field="flux", value=-1.0, reason="bad")
    text = repr(finding)
    assert "flux" in text
    assert "-1.0" in text
    assert "bad" in text
