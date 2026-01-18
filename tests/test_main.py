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

    with patch("coreason_codex.main.CodexBuilder") as MockBuilder:
        with patch("coreason_codex.main.SapBertEmbedder") as MockEmbedder:
            mock_builder_instance = MockBuilder.return_value
            mock_embedder_instance = MockEmbedder.return_value

            result = runner.invoke(app, ["build", "--source", str(source), "--output", str(output)])

            assert result.exit_code == 0

            # Verify Builder was initialized and methods called
            MockBuilder.assert_called_once_with(source, output)
            mock_builder_instance.build_vocab.assert_called_once()
            mock_builder_instance.build_vectors.assert_called_once_with(mock_embedder_instance)
            mock_builder_instance.generate_manifest.assert_called_once()

            # Verify Embedder was initialized
            MockEmbedder.assert_called_once_with(device="cpu")


def test_main_build_with_device(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()

    with patch("coreason_codex.main.CodexBuilder"):
        with patch("coreason_codex.main.SapBertEmbedder") as MockEmbedder:
            result = runner.invoke(app, ["build", "--source", str(source), "--output", str(output), "--device", "cuda"])
            assert result.exit_code == 0
            MockEmbedder.assert_called_once_with(device="cuda")


def test_main_build_failure(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()

    with patch("coreason_codex.main.CodexBuilder") as MockBuilder:
        with patch("coreason_codex.main.SapBertEmbedder"):
            mock_builder_instance = MockBuilder.return_value
            # Simulate failure
            mock_builder_instance.build_vocab.side_effect = RuntimeError("Build failed")

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
        with patch("coreason_codex.main.codex_normalize") as mock_norm:
            mock_norm.return_value = [match]

            result = runner.invoke(app, ["normalize", "test input", "--pack", str(pack)])

            assert result.exit_code == 0
            mock_init.assert_called_once_with(str(pack))
            mock_norm.assert_called_once_with("test input", domain_filter=None)
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
