"""CLI entry point for the AMS-02 Publication Data Harmonizer."""

import logging

import click
from importlib.metadata import version, PackageNotFoundError


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


# Register subcommands — imported here to avoid circular imports.
from ams02wb.cli.index_publications import index_publications  # noqa: E402
from ams02wb.cli.ingest_publication import ingest_publication  # noqa: E402
from ams02wb.cli.ingest_all import ingest_all  # noqa: E402

cli.add_command(index_publications)
cli.add_command(ingest_publication)
cli.add_command(ingest_all)
