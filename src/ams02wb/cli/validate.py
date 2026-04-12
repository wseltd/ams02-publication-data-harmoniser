"""CLI command: validate — run schema validation on dataset JSON files."""

from __future__ import annotations

import sys

import click

from ams02wb.schema.dataset_validator import validate_dataset_dir


@click.command("validate")
@click.option(
    "--input-dir",
    required=True,
    type=click.Path(),
    help="Directory containing JSON dataset files to validate.",
)
def validate_datasets(input_dir: str) -> None:
    """Validate all JSON dataset files in a directory.

    Runs schema validation on every *.json file in INPUT_DIR and reports
    per-file PASS/FAIL status. Exits 0 if all files pass, 1 if any fail,
    2 if no JSON files are found.
    """
    from pathlib import Path

    dir_path = Path(input_dir)

    try:
        results = validate_dataset_dir(dir_path)
    except FileNotFoundError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    if not results:
        click.echo("No JSON files found.", err=True)
        sys.exit(2)

    has_failures = False

    for result in results:
        status = result["status"]
        fname = result["file_path"]
        click.echo(f"{fname}: {status}")

        if status == "FAIL":
            has_failures = True
            for finding in result["findings"]:
                click.echo(
                    f"  {finding.field}: {finding.reason} "
                    f"(value={finding.value!r})"
                )

    sys.exit(1 if has_failures else 0)
