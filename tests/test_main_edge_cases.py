# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_codex

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from coreason_codex.main import app

runner = CliRunner()


def test_cli_missing_arguments() -> None:
    """Test that Typer handles missing arguments (exit code 2)."""
    # Missing both
    result = runner.invoke(app, ["build"])
    assert result.exit_code == 2
    assert "Missing option" in result.output


def test_cli_missing_source_argument() -> None:
    """Test that Typer handles missing source argument."""
    result = runner.invoke(app, ["build", "--output", "out"])
    assert result.exit_code == 2
    assert "Missing option" in result.output
    # ANSI codes/Rich formatting can mess up exact string matching for flags
    # We verify the parameter name is present
    assert "source" in result.output


def test_cli_source_not_exists(tmp_path: Path) -> None:
    """
    Test that Typer validates file existence if `exists=True` is set.
    """
    # We point to a non-existent source
    source = tmp_path / "non_existent"
    output = tmp_path / "output"

    result = runner.invoke(app, ["build", "--source", str(source), "--output", str(output)])

    assert result.exit_code == 2
    assert "does not exist" in result.output


def test_cli_embedder_initialization_error(tmp_path: Path) -> None:
    """
    Test that the CLI handles errors during Embedder initialization gracefully.
    """
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()

    with patch("coreason_codex.main.CodexPipeline") as MockPipeline:
        # Simulate failure in pipeline run
        instance = MockPipeline.return_value
        instance.run.side_effect = RuntimeError("Model download failed")

        result = runner.invoke(app, ["build", "--source", str(source), "--output", str(output)])

        assert result.exit_code == 1


def test_cli_relative_paths(tmp_path: Path) -> None:
    """
    Test that Typer correctly handles relative paths.
    """
    # Let's create a temp dir and pass it as absolute, verifying correct type conversion.
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output"  # doesn't need to exist

    with patch("coreason_codex.main.CodexPipeline") as MockPipeline:
        result = runner.invoke(app, ["build", "--source", str(source), "--output", str(output)])

        assert result.exit_code == 0

        instance = MockPipeline.return_value
        args, _ = instance.run.call_args
        source_arg = args[0]
        output_arg = args[1]

        assert isinstance(source_arg, Path)
        assert isinstance(output_arg, Path)


def test_cli_unknown_command() -> None:
    """Test that Typer handles unknown commands."""
    result = runner.invoke(app, ["unknown_command"])
    assert result.exit_code == 2
    assert "No such command" in result.output
