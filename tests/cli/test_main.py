"""Tests that verify CLI subcommand wiring in main.py."""

from click.testing import CliRunner

from ams02wb.cli import cli, index_publications, ingest_publication


def test_index_publications_registered_as_subcommand():
    """index-publications must appear as a subcommand of the cli group."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "index-publications" in result.output


def test_ingest_publication_registered_as_subcommand():
    """ingest-publication must appear as a subcommand of the cli group."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "ingest-publication" in result.output


def test_index_publications_help_shows_output_dir_option():
    """index-publications --help must expose the --output-dir option."""
    runner = CliRunner()
    result = runner.invoke(cli, ["index-publications", "--help"])
    assert result.exit_code == 0
    assert "--output-dir" in result.output


def test_ingest_publication_help_shows_publication_id_option():
    """ingest-publication --help must expose the --publication-id option."""
    runner = CliRunner()
    result = runner.invoke(cli, ["ingest-publication", "--help"])
    assert result.exit_code == 0
    assert "--publication-id" in result.output


def test_index_publications_importable_from_cli_package():
    """index_publications must be importable from ams02wb.cli.__init__."""
    assert callable(index_publications)
    assert index_publications.name == "index-publications"


def test_ingest_publication_importable_from_cli_package():
    """ingest_publication must be importable from ams02wb.cli.__init__."""
    assert callable(ingest_publication)
    assert ingest_publication.name == "ingest-publication"
