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
from unittest.mock import MagicMock, patch

import pytest

from coreason_codex.main import main


def test_main_build_success(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()

    # Mock arguments
    test_args = ["coreason-codex", "build", "--source", str(source), "--output", str(output)]

    with patch.object(sys, "argv", test_args):
        with patch("coreason_codex.main.CodexBuilder") as MockBuilder:
            with patch("coreason_codex.main.SapBertEmbedder") as MockEmbedder:
                mock_builder_instance = MockBuilder.return_value
                mock_embedder_instance = MockEmbedder.return_value

                main()

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

    test_args = ["coreason-codex", "build", "--source", str(source), "--output", str(output), "--device", "cuda"]

    with patch.object(sys, "argv", test_args):
        with patch("coreason_codex.main.CodexBuilder") as MockBuilder:
            with patch("coreason_codex.main.SapBertEmbedder") as MockEmbedder:
                main()
                MockEmbedder.assert_called_once_with(device="cuda")


def test_main_build_failure(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"

    test_args = ["coreason-codex", "build", "--source", str(source), "--output", str(output)]

    with patch.object(sys, "argv", test_args):
        with patch("coreason_codex.main.CodexBuilder") as MockBuilder:
            with patch("coreason_codex.main.SapBertEmbedder"):
                mock_builder_instance = MockBuilder.return_value
                # Simulate failure
                mock_builder_instance.build_vocab.side_effect = RuntimeError("Build failed")

                # Expect exit code 1
                with pytest.raises(SystemExit) as e:
                    main()
                assert e.value.code == 1
