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

from coreason_codex.main import app, main
from coreason_codex.schemas import CodexMatch, Concept

runner = CliRunner()


def test_main_build_success(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()

    with patch("coreason_codex.main.CodexPipeline") as MockPipeline:
        mock_pipeline_instance = MockPipeline.return_value

        result = runner.invoke(app, ["build", "--source", str(source), "--output", str(output)])

        assert result.exit_code == 0

        # Verify Pipeline was initialized and methods called
        MockPipeline.assert_called_once()
        mock_pipeline_instance.run.assert_called_once()
        args, _ = mock_pipeline_instance.run.call_args
        assert args[0] == source
        assert args[1] == output
        assert args[2] == "cpu"


def test_main_build_with_device(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()

    with patch("coreason_codex.main.CodexPipeline") as MockPipeline:
        mock_pipeline_instance = MockPipeline.return_value
        result = runner.invoke(app, ["build", "--source", str(source), "--output", str(output), "--device", "cuda"])
        assert result.exit_code == 0
        mock_pipeline_instance.run.assert_called_once()
        args, _ = mock_pipeline_instance.run.call_args
        assert args[2] == "cuda"


def test_main_build_failure(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()

    with patch("coreason_codex.main.CodexPipeline") as MockPipeline:
        mock_pipeline_instance = MockPipeline.return_value
        # Simulate failure
        mock_pipeline_instance.run.side_effect = RuntimeError("Build failed")

        result = runner.invoke(app, ["build", "--source", str(source), "--output", str(output)])

        assert result.exit_code == 1


def test_main_normalize_success(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    pack.mkdir()

    match = CodexMatch(
        input_text="test",
        match_concept=Concept(
            concept_id=1, concept_name="C", domain_id="D", vocabulary_id="V", concept_class_id="C", concept_code="1"
        ),
        similarity_score=1.0,
        is_standard=True,
    )

    with patch("coreason_codex.main.initialize") as mock_init:
        with patch("coreason_codex.main.CodexPipeline") as MockPipeline:
            mock_pipeline_instance = MockPipeline.return_value
            mock_pipeline_instance.search.return_value = [match]

            result = runner.invoke(app, ["normalize", "test input", "--pack", str(pack)])

            assert result.exit_code == 0
            mock_init.assert_called_once_with(str(pack))
            # Verify search called (we can't easily check context without custom matcher)
            mock_pipeline_instance.search.assert_called_once()
            args, kwargs = mock_pipeline_instance.search.call_args
            assert args[0] == "test input"
            assert kwargs["domain_filter"] is None
            assert '"concept_id": 1' in result.output


def test_main_normalize_failure(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    pack.mkdir()

    with patch("coreason_codex.main.initialize") as mock_init:
        mock_init.side_effect = RuntimeError("Init failed")

        result = runner.invoke(app, ["normalize", "test", "--pack", str(pack)])

        assert result.exit_code == 1


def test_version_command() -> None:
    """Test the version command."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "coreason-codex v" in result.output


def test_entry_point_function() -> None:
    """
    Test the main() function directly to ensure the app() is called.
    This covers the `def main(): app()` line.
    """
    with patch("coreason_codex.main.app") as mock_app:
        main()
        mock_app.assert_called_once()
