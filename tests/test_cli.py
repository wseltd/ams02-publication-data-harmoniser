"""Tests for the CLI entry point."""

import logging

from click.testing import CliRunner

from ams02wb.cli.main import cli


def test_cli_group_invocable_without_subcommand():
    """Bare invocation (no subcommand) should succeed and print help."""
    runner = CliRunner()
    result = runner.invoke(cli, [])
    assert result.exit_code == 0


def test_cli_help_text_contains_ams02():
    """Help output must mention AMS-02 so users know what tool they are running."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "AMS-02" in result.output


def test_cli_verbose_flag_sets_debug_logging():
    """--verbose must configure the root logger to DEBUG level."""
    runner = CliRunner()
    # Reset root logger to a known state before invoking
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.WARNING)

    result = runner.invoke(cli, ["--verbose"])
    assert result.exit_code == 0
    assert root.level == logging.DEBUG


def test_cli_version_flag_outputs_version_string():
    """--version must print a version string (not crash on missing metadata)."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    # Output should contain the prog name and some version identifier
    assert "ams02wb" in result.output


def test_cli_no_args_shows_help():
    """Running with no arguments should display help text, not an error."""
    runner = CliRunner()
    result = runner.invoke(cli, [])
    assert result.exit_code == 0
    assert "--verbose" in result.output
    assert "--version" in result.output
    assert "--help" in result.output


def test_cli_verbose_without_flag_sets_info_logging():
    """Without --verbose, root logger should be at INFO level."""
    runner = CliRunner()
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.WARNING)

    result = runner.invoke(cli, [])
    assert result.exit_code == 0
    assert root.level == logging.INFO


def test_cli_help_describes_tool_purpose():
    """Help text must describe the tool as a harmonizer and workbench."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert "Harmonizer" in result.output
    assert "Likelihood Workbench" in result.output
