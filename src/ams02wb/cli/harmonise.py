"""CLI command: harmonise — run the harmonisation pipeline on ingested datasets."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

from ams02wb.harmoniser.pipeline import run_harmonisation_pipeline
from ams02wb.parsers.context import ParseContext
from ams02wb.schema.models import Measurement

logger = logging.getLogger(__name__)


def _load_measurements_from_file(path: Path) -> tuple[list[Measurement], dict]:
    """Load measurements and provenance from an ingested dataset JSON file.

    Returns (measurements, provenance_dict).  Measurements are built from
    the 'measurements' array if present; otherwise returns an empty list.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    provenance = raw.get("provenance", {})

    measurement_dicts = raw.get("measurements", [])
    measurements = [Measurement(**md) for md in measurement_dicts]

    return measurements, provenance


@click.command("harmonise")
@click.option(
    "--input-dir",
    required=True,
    type=click.Path(),
    help="Directory containing ingested JSON dataset files.",
)
@click.option(
    "--output-dir",
    default="./ams02wb-harmonised/",
    type=click.Path(),
    help="Directory to write harmonised output files into.",
)
@click.option(
    "--species",
    default=None,
    type=str,
    help="Comma-separated list of species to include (e.g. 'PROTON,HELIUM'). "
    "If omitted, all species are included.",
)
def harmonise_datasets(input_dir: str, output_dir: str, species: str | None) -> None:
    """Run harmonisation pipeline on all ingested JSON datasets in INPUT_DIR.

    Applies species normalisation, axis harmonisation, uncertainty labelling,
    and time-window normalisation.  Writes one JSON file per input file into
    OUTPUT_DIR with canonical schema field names.
    """
    in_path = Path(input_dir)
    out_path = Path(output_dir)

    if not in_path.is_dir():
        click.echo(f"Error: input directory does not exist: {in_path}", err=True)
        sys.exit(1)

    json_files = sorted(in_path.glob("*.json"))
    if not json_files:
        click.echo("Error: no JSON files found in input directory.", err=True)
        sys.exit(1)

    # Parse species filter if provided
    species_filter: set[str] | None = None
    if species is not None:
        species_filter = {s.strip().upper() for s in species.split(",") if s.strip()}

    out_path.mkdir(parents=True, exist_ok=True)

    # Default parse context — flags default to False (ASSUMED labels).
    # Per-file parse contexts could be stored alongside ingested data in
    # future; for now a single default context is correct because the
    # ingestion pipeline does not yet persist parse flags.
    parse_context = ParseContext()

    total_records = 0

    for json_file in json_files:
        try:
            measurements, provenance = _load_measurements_from_file(json_file)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Skipping %s: %s", json_file.name, exc)
            continue

        if not measurements:
            logger.info("No measurements in %s, skipping.", json_file.name)
            continue

        canonical_records = run_harmonisation_pipeline(
            measurements, parse_context, provenance,
        )

        # Apply species filter after harmonisation (species names are now canonical)
        if species_filter is not None:
            canonical_records = [
                r for r in canonical_records if r["species_num"] in species_filter
            ]

        if not canonical_records:
            continue

        dest = out_path / json_file.name
        dest.write_text(
            json.dumps(canonical_records, indent=2, default=str),
            encoding="utf-8",
        )
        total_records += len(canonical_records)
        logger.info("Harmonised %s -> %s (%d records)", json_file.name, dest, len(canonical_records))

    click.echo(f"Harmonised {total_records} records from {len(json_files)} files.")
