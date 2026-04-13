"""CLI entry point for the AMS-02 Publication Data Harmonizer."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Literal, cast

import click
import pandas as pd
from importlib.metadata import version, PackageNotFoundError

from ams02wb.alignment.daily import align_daily_series
from ams02wb.alignment.bartels import align_bartels_rotation

logger = logging.getLogger(__name__)


def _resolve_version() -> str:
    """Read version from installed package metadata.

    Falls back to 'unknown' if the package is not installed
    (e.g. during development without pip install -e).
    """
    try:
        return version("ams02wb")
    except PackageNotFoundError:
        return "unknown"


@click.group(invoke_without_command=True)
@click.version_option(version=_resolve_version(), prog_name="ams02wb")
@click.option("--verbose", is_flag=True, default=False, help="Enable debug logging.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """AMS-02 Publication Data Harmonizer and Likelihood Workbench."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


def _load_harmonised_dataframe(species: str, input_dir: Path) -> pd.DataFrame:
    """Load harmonised dataset for a species from all JSON files in input_dir.

    Scans all ``*.json`` files in *input_dir*, loads records, and filters
    to rows matching *species* (case-insensitive on ``species_num``).
    Returns an empty DataFrame if no matching records are found.
    """
    if not input_dir.is_dir():
        logger.warning("Harmonised data directory does not exist: %s", input_dir)
        return pd.DataFrame()

    all_records: list[dict] = []
    for json_file in sorted(input_dir.glob("*.json")):
        try:
            raw = json.loads(json_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(raw, list):
            all_records.extend(raw)
        elif isinstance(raw, dict) and "measurements" in raw:
            all_records.extend(raw["measurements"])

    if not all_records:
        logger.warning("No records found in %s", input_dir)
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    if "species_num" not in df.columns:
        return pd.DataFrame()

    filtered = df[df["species_num"].str.upper() == species.upper()]
    if filtered.empty:
        logger.warning("No records for species %s in %s", species, input_dir)

    return filtered


@click.command("align-time-series")
@click.option(
    "--input-dir",
    default="./ams02wb-data/harmonised/",
    type=click.Path(),
    help="Directory containing harmonised JSON files.",
)
@click.option(
    "--species",
    multiple=True,
    required=True,
    type=str,
    help="Species to align (repeatable, e.g. --species proton --species helium).",
)
@click.option(
    "--join",
    "join_type",
    type=click.Choice(["intersection", "union"], case_sensitive=False),
    default="intersection",
    help="Join strategy: intersection (inner) or union (outer). Default: intersection.",
)
@click.option(
    "--cadence",
    type=click.Choice(["daily", "bartels"], case_sensitive=False),
    default="daily",
    help="Time cadence for alignment. Default: daily.",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(),
    help="Path to write the aligned output (parquet).",
)
def align_time_series(
    input_dir: str,
    species: tuple[str, ...],
    join_type: str,
    cadence: str,
    output: str,
) -> None:
    """Align harmonised species time-series on a shared time grid.

    Loads harmonised datasets for each --species, aligns them by the
    chosen --cadence, and writes the result to --output.
    """
    # Set up stderr logging for join diagnostics.
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.INFO)
    stderr_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(stderr_handler)
    logger.setLevel(logging.INFO)

    harmonised_dir = Path(input_dir)
    species_frames: dict[str, pd.DataFrame] = {}
    for sp in species:
        df = _load_harmonised_dataframe(sp, harmonised_dir)
        if df.empty:
            logger.warning("Skipping species %s: no data loaded.", sp)
            continue
        species_frames[sp] = df

    if not species_frames:
        click.echo("Error: no data loaded for any requested species.", err=True)
        sys.exit(1)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Map CLI join names to pandas join types.
    join_map = {"intersection": "inner", "union": "outer"}
    pandas_join = cast(Literal["inner", "outer"], join_map[join_type])

    if cadence == "daily":
        result = align_daily_series(species_frames, join=pandas_join)
        aligned_df = result.aligned_df

        # Log join diagnostics to stderr.
        for sp, count in result.missing_counts.items():
            if count > 0:
                logger.info("Dropped/imputed %d dates for species %s", count, sp)
        logger.info(
            "Aligned %d rows, date range %s to %s (join=%s)",
            len(aligned_df),
            result.date_range[0],
            result.date_range[1],
            result.join_type,
        )
    else:
        # Bartels cadence: concatenate all species then group by rotation.
        all_frames = pd.concat(species_frames.values(), ignore_index=True)
        rotation_groups = align_bartels_rotation(all_frames)

        # Build a wide-form result: one row per rotation with counts per species.
        rows = []
        for rot_num, group_df in sorted(rotation_groups.items()):
            row: dict[str, object] = {"bartels_rotation": rot_num, "n_measurements": len(group_df)}
            rows.append(row)

        aligned_df = pd.DataFrame(rows)
        logger.info("Aligned into %d Bartels rotations", len(aligned_df))

    aligned_df.to_parquet(out_path, index=False)
    click.echo(f"Wrote aligned data to {out_path}")


@click.command("export-dataset")
@click.option(
    "--dataset",
    required=True,
    type=click.Path(exists=True),
    help="Path to the dataset file to export.",
)
@click.option(
    "--format",
    "fmt",
    required=True,
    type=click.Choice(["parquet", "csv", "json", "usine"], case_sensitive=False),
    help="Output format: parquet, csv, json, or usine.",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(),
    help="Path to write the exported file.",
)
def export_dataset(dataset: str, fmt: str, output: str) -> None:
    """Export a dataset to the specified format.

    Reads the dataset from --dataset and writes to --output in the
    chosen --format.
    """
    dataset_path = Path(dataset)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw = json.loads(dataset_path.read_text(encoding="utf-8"))

    if fmt == "parquet":
        from ams02wb.exports.parquet import export_parquet

        records = raw if isinstance(raw, list) else raw.get("measurements", [])
        provenance = raw.get("provenance", {}) if isinstance(raw, dict) else {}
        df = pd.DataFrame(records)
        export_parquet({"data": df, "provenance": provenance, "covariance_label": "unknown"}, out_path)
    elif fmt == "csv":
        from ams02wb.exports.csv_export import CANONICAL_FIELDS, export_csv_from_dicts

        records = raw if isinstance(raw, list) else raw.get("measurements", [])
        export_csv_from_dicts(records, out_path, CANONICAL_FIELDS)
    elif fmt == "json":
        from ams02wb.exports.json_export import export_json

        export_json(raw, out_path)
    elif fmt == "usine":
        from ams02wb.exports.usine_export import export_usine
        from ams02wb.schema.models import Measurement

        records = raw if isinstance(raw, list) else raw.get("measurements", [])
        # Try to build Measurement objects; fall back to empty list
        measurements = []
        for rec in records:
            try:
                measurements.append(Measurement(**rec))
            except (TypeError, ValueError, KeyError):
                logger.debug("Skipping record that doesn't fit Measurement schema")
                continue
        species = measurements[0].species if measurements else "UNKNOWN"
        usine_text = export_usine(measurements, species_num=species)
        out_path.write_text(usine_text, encoding="utf-8")
    else:
        click.echo(f"Error: unknown format {fmt!r}. Supported: parquet, csv, json, usine.", err=True)
        sys.exit(1)

    click.echo(f"Exported dataset to {out_path} ({fmt})")


# Register subcommands — imported here to avoid circular imports.
from ams02wb.cli.index_publications import index_publications  # noqa: E402
from ams02wb.cli.ingest_publication import ingest_publication  # noqa: E402
from ams02wb.cli.ingest_all import ingest_all  # noqa: E402
from ams02wb.cli.validate import validate_datasets  # noqa: E402
from ams02wb.cli.harmonise import harmonise_datasets  # noqa: E402
from ams02wb.cli.build_likelihood import build_likelihood  # noqa: E402

cli.add_command(index_publications)
cli.add_command(ingest_publication)
cli.add_command(ingest_all)
cli.add_command(validate_datasets)
cli.add_command(harmonise_datasets)
cli.add_command(build_likelihood)
cli.add_command(align_time_series)
cli.add_command(export_dataset)
