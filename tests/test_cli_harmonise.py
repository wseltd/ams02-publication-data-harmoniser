"""Tests for the harmonise CLI command and pipeline integration."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from ams02wb.cli import cli


# Canonical schema fields that every output record must contain.
CANONICAL_FIELDS = {
    "species_num", "species_den", "x_axis_type", "x_axis_unit",
    "x_min", "x_max", "x_centre", "y_value", "y_unit",
    "stat_err", "sys_err_total", "sys_err_components",
    "time_start", "time_stop", "provenance_json",
}


def _write_dataset(directory: Path, filename: str, measurements: list[dict],
                   provenance: dict | None = None) -> Path:
    """Write a minimal ingested-format JSON file into directory."""
    dataset = {
        "publication_id": "test-001",
        "title": "Test Publication",
        "source_url": "https://example.com/test",
        "measurements": measurements,
        "provenance": provenance or {
            "source_url": "https://example.com/test",
            "content_hash": "sha256:abc123",
            "parse_method": "csv_table_extraction",
            "ingested_at": "2025-01-01T00:00:00+00:00",
        },
    }
    path = directory / filename
    path.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    return path


def _sample_measurement(**overrides: object) -> dict:
    """Build a minimal measurement dict with sensible defaults."""
    base = {
        "energy_low": 1.0,
        "energy_high": 2.0,
        "energy_mid": 1.5,
        "value": 100.0,
        "unit": "GeV",
        "axis_type": "kinetic_energy_per_nucleon",
        "species": "PROTON",
        "stat_err_pos": 5.0,
        "stat_err_neg": 5.0,
        "sys_err_pos": 3.0,
        "sys_err_neg": 3.0,
        "time_start": "2015-01-01",
        "time_end": "2015-02-01",
    }
    base.update(overrides)
    return base


def test_harmonise_produces_normalised_output_files(tmp_path: Path) -> None:
    """Harmonise command writes JSON output files matching input filenames."""
    in_dir = tmp_path / "input"
    in_dir.mkdir()
    out_dir = tmp_path / "output"

    _write_dataset(in_dir, "pub_001.json", [_sample_measurement()])

    runner = CliRunner()
    result = runner.invoke(cli, [
        "harmonise", "--input-dir", str(in_dir), "--output-dir", str(out_dir),
    ])

    assert result.exit_code == 0, result.output
    output_file = out_dir / "pub_001.json"
    assert output_file.exists(), f"Expected output file not found. Output: {result.output}"

    records = json.loads(output_file.read_text(encoding="utf-8"))
    assert len(records) == 1
    assert records[0]["species_num"] == "PROTON"


def test_harmonise_output_matches_canonical_schema_fields(tmp_path: Path) -> None:
    """Every output record must contain exactly the canonical field names."""
    in_dir = tmp_path / "input"
    in_dir.mkdir()
    out_dir = tmp_path / "output"

    _write_dataset(in_dir, "pub_002.json", [
        _sample_measurement(),
        _sample_measurement(species="HELIUM", energy_mid=3.0),
    ])

    runner = CliRunner()
    result = runner.invoke(cli, [
        "harmonise", "--input-dir", str(in_dir), "--output-dir", str(out_dir),
    ])

    assert result.exit_code == 0, result.output
    records = json.loads((out_dir / "pub_002.json").read_text(encoding="utf-8"))

    for record in records:
        assert set(record.keys()) == CANONICAL_FIELDS, (
            f"Field mismatch: extra={set(record.keys()) - CANONICAL_FIELDS}, "
            f"missing={CANONICAL_FIELDS - set(record.keys())}"
        )


def test_harmonise_preserves_original_provenance(tmp_path: Path) -> None:
    """Original provenance fields must survive into output provenance_json."""
    in_dir = tmp_path / "input"
    in_dir.mkdir()
    out_dir = tmp_path / "output"

    original_provenance = {
        "source_url": "https://ams02.space/paper/42",
        "content_hash": "sha256:deadbeef",
        "parse_method": "csv_table_extraction",
        "ingested_at": "2025-06-15T12:00:00+00:00",
    }
    _write_dataset(in_dir, "pub_003.json", [_sample_measurement()],
                   provenance=original_provenance)

    runner = CliRunner()
    result = runner.invoke(cli, [
        "harmonise", "--input-dir", str(in_dir), "--output-dir", str(out_dir),
    ])

    assert result.exit_code == 0, result.output
    records = json.loads((out_dir / "pub_003.json").read_text(encoding="utf-8"))
    prov = records[0]["provenance_json"]

    # All original provenance keys must be present and unchanged
    for key, value in original_provenance.items():
        assert prov[key] == value, f"Provenance key {key!r} changed: {prov.get(key)!r} != {value!r}"


def test_harmonise_provenance_includes_harmonisation_steps(tmp_path: Path) -> None:
    """Provenance metadata must include steps_applied listing stage names."""
    in_dir = tmp_path / "input"
    in_dir.mkdir()
    out_dir = tmp_path / "output"

    # Use lowercase species alias to trigger normalise_species step,
    # and time fields to trigger normalise_time_window step.
    _write_dataset(in_dir, "pub_004.json", [
        _sample_measurement(species="proton", time_start="2015-01-01"),
    ])

    runner = CliRunner()
    result = runner.invoke(cli, [
        "harmonise", "--input-dir", str(in_dir), "--output-dir", str(out_dir),
    ])

    assert result.exit_code == 0, result.output
    records = json.loads((out_dir / "pub_004.json").read_text(encoding="utf-8"))
    metadata = records[0]["provenance_json"]["harmonisation_metadata"]

    assert "steps_applied" in metadata
    assert isinstance(metadata["steps_applied"], list)
    # Species was lowercase → normalise_species should appear
    assert "normalise_species" in metadata["steps_applied"]
    # Time fields were set → normalise_time_window should appear
    assert "normalise_time_window" in metadata["steps_applied"]


def test_harmonise_provenance_includes_original_axis_values(tmp_path: Path) -> None:
    """Harmonisation metadata must record original axis type and unit."""
    in_dir = tmp_path / "input"
    in_dir.mkdir()
    out_dir = tmp_path / "output"

    _write_dataset(in_dir, "pub_005.json", [
        _sample_measurement(axis_type="kinetic_energy_per_nucleon", unit="GeV"),
    ])

    runner = CliRunner()
    result = runner.invoke(cli, [
        "harmonise", "--input-dir", str(in_dir), "--output-dir", str(out_dir),
    ])

    assert result.exit_code == 0, result.output
    records = json.loads((out_dir / "pub_005.json").read_text(encoding="utf-8"))
    metadata = records[0]["provenance_json"]["harmonisation_metadata"]

    assert metadata["original_x_axis_type"] == "kinetic_energy_per_nucleon"
    assert metadata["original_x_axis_unit"] == "GeV"
    assert "harmonised_at" in metadata


def test_harmonise_species_filter_limits_output(tmp_path: Path) -> None:
    """--species flag limits output to only the requested species."""
    in_dir = tmp_path / "input"
    in_dir.mkdir()
    out_dir = tmp_path / "output"

    _write_dataset(in_dir, "pub_006.json", [
        _sample_measurement(species="PROTON"),
        _sample_measurement(species="HELIUM", energy_mid=3.0),
        _sample_measurement(species="ELECTRON", energy_mid=0.1),
    ])

    runner = CliRunner()
    result = runner.invoke(cli, [
        "harmonise", "--input-dir", str(in_dir), "--output-dir", str(out_dir),
        "--species", "PROTON,ELECTRON",
    ])

    assert result.exit_code == 0, result.output
    records = json.loads((out_dir / "pub_006.json").read_text(encoding="utf-8"))

    species_in_output = {r["species_num"] for r in records}
    assert species_in_output == {"PROTON", "ELECTRON"}, (
        f"Expected PROTON and ELECTRON only, got {species_in_output}"
    )
    assert len(records) == 2


def test_harmonise_empty_input_dir_exits_nonzero(tmp_path: Path) -> None:
    """Command exits non-zero when input directory has no JSON files."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    runner = CliRunner()
    result = runner.invoke(cli, [
        "harmonise", "--input-dir", str(empty_dir),
    ])

    assert result.exit_code != 0
    assert "no JSON files" in result.output.lower() or "no json files" in (result.output + (result.stderr or "")).lower()


def test_harmonise_pipeline_applies_stages_in_order(tmp_path: Path) -> None:
    """Pipeline applies stages in the correct fixed order.

    Uses a measurement with lowercase species alias and time fields to
    verify that species normalisation happens before axis harmonisation
    (axis harmonisation needs canonical species names for rigidity
    conversion lookups), and that time normalisation runs last.
    """
    in_dir = tmp_path / "input"
    in_dir.mkdir()
    out_dir = tmp_path / "output"

    # lowercase alias triggers species normalisation; GV rigidity triggers
    # axis harmonisation; time fields trigger time-window normalisation;
    # uncertainties always get labelled.
    _write_dataset(in_dir, "pub_007.json", [
        _sample_measurement(
            species="proton",
            axis_type="rigidity",
            unit="GV",
            time_start="2015-01-01",
            time_end="2015-02-01",
        ),
    ])

    runner = CliRunner()
    result = runner.invoke(cli, [
        "harmonise", "--input-dir", str(in_dir), "--output-dir", str(out_dir),
    ])

    assert result.exit_code == 0, result.output
    records = json.loads((out_dir / "pub_007.json").read_text(encoding="utf-8"))
    record = records[0]
    metadata = record["provenance_json"]["harmonisation_metadata"]

    # All four stages should have fired
    steps = metadata["steps_applied"]
    assert "normalise_species" in steps
    assert "harmonise_axes" in steps
    assert "label_uncertainties" in steps
    assert "normalise_time_window" in steps

    # Verify ordering: species before axes, axes before uncertainties,
    # uncertainties before time-window
    idx_species = steps.index("normalise_species")
    idx_axes = steps.index("harmonise_axes")
    idx_unc = steps.index("label_uncertainties")
    idx_time = steps.index("normalise_time_window")
    assert idx_species < idx_axes < idx_unc < idx_time, (
        f"Stage order wrong: {steps}"
    )

    # Species was normalised to canonical form
    assert record["species_num"] == "PROTON"
    # Axis was converted from rigidity/GV to kinetic_energy_per_nucleon/GeV
    assert record["x_axis_type"] == "kinetic_energy_per_nucleon"
    assert record["x_axis_unit"] == "GeV"
    # Original axis preserved in metadata
    assert metadata["original_x_axis_type"] == "rigidity"
    assert metadata["original_x_axis_unit"] == "GV"
    # Time was normalised to UTC ISO strings
    assert record["time_start"] is not None
    assert record["time_stop"] is not None
