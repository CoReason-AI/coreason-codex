# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_codex

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from coreason_codex.main import main


def test_cli_missing_arguments() -> None:
    """Test that the CLI exits with status 2 (argparse error) when required arguments are missing."""
    # Missing --source and --output
    test_args = ["coreason-codex", "build"]
    with patch.object(sys, "argv", test_args):
        # argparse prints to stderr and exits with 2
        with pytest.raises(SystemExit) as e:
            main()
        assert e.value.code == 2


def test_cli_missing_source_argument() -> None:
    """Test that the CLI exits with status 2 when --source is missing."""
    test_args = ["coreason-codex", "build", "--output", "out"]
    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit) as e:
            main()
        assert e.value.code == 2


def test_cli_embedder_initialization_error(tmp_path: Path) -> None:
    """
    Test that the CLI handles errors during Embedder initialization gracefully.
    Should catch the exception, log it, and exit with status 1.
    """
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()

    test_args = ["coreason-codex", "build", "--source", str(source), "--output", str(output)]

    with patch.object(sys, "argv", test_args):
        with patch("coreason_codex.main.CodexBuilder") as MockBuilder:
            with patch("coreason_codex.main.SapBertEmbedder") as MockEmbedder:
                # Mock Embedder to raise an exception on init
                MockEmbedder.side_effect = RuntimeError("Model download failed")

                # Mock Builder to ensure build_vocab is called before embedder init
                mock_builder = MockBuilder.return_value

                with pytest.raises(SystemExit) as e:
                    main()

                assert e.value.code == 1

                # Verify build_vocab was called (as it happens before embedder init in main.py)
                mock_builder.build_vocab.assert_called_once()
                # Verify build_vectors was NOT called because embedder init failed
                mock_builder.build_vectors.assert_not_called()


def test_cli_relative_paths(tmp_path: Path) -> None:
    """
    Test that the CLI correctly handles relative paths and converts them to Path objects.
    """
    # Create a condition where we are 'in' tmp_path (conceptually)
    # We pass relative strings "src_rel" and "out_rel"

    test_args = ["coreason-codex", "build", "--source", "src_rel", "--output", "out_rel"]

    with patch.object(sys, "argv", test_args):
        with patch("coreason_codex.main.CodexBuilder") as MockBuilder:
            with patch("coreason_codex.main.SapBertEmbedder"):
                main()

                # Verify CodexBuilder was called with Path objects
                # Note: Path("src_rel") creates a relative path object.
                # CodexBuilder logic doesn't strictly require absolute paths during init,
                # but we verify they are passed as Paths.
                args, _ = MockBuilder.call_args
                source_arg, output_arg = args

                assert isinstance(source_arg, Path)
                assert isinstance(output_arg, Path)
                assert str(source_arg) == "src_rel"
                assert str(output_arg) == "out_rel"


def test_cli_unknown_command() -> None:
    """Test that the CLI handles unknown commands."""
    test_args = ["coreason-codex", "unknown_command"]
    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit) as e:
            main()
        # argparse usually exits with 2 for invalid choices/commands
        assert e.value.code == 2
