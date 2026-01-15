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
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from coreason_codex.build import CodexBuilder
from coreason_codex.pipeline import CodexContext


def test_builder_source_dir_not_found(tmp_path: Path) -> None:
    non_existent = tmp_path / "non_existent"
    builder = CodexBuilder(non_existent, tmp_path / "out")
    with pytest.raises(FileNotFoundError, match="Source directory not found"):
        builder.build_vocab()


def test_build_vocab_connect_failure(source_csvs: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "out_connect_fail"
    builder = CodexBuilder(source_csvs, output_dir)

    # Case 1: Connect fails immediately
    with patch("duckdb.connect", side_effect=Exception("Connect Fail")):
        with pytest.raises(RuntimeError, match="Build failed"):
            builder.build_vocab()


def test_build_vocab_process_failure_cleanup(source_csvs: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "out_process_fail"
    output_dir.mkdir(parents=True, exist_ok=True)
    builder = CodexBuilder(source_csvs, output_dir)

    # Pre-create artifact to test cleanup
    db_path = output_dir / "vocab.duckdb"
    db_path.touch()

    # Mock _load_table to fail AFTER connection is made
    # We need real connection or mock that returns an object
    with patch("coreason_codex.build.duckdb.connect") as mock_connect:
        mock_con = MagicMock()
        mock_connect.return_value = mock_con

        # We patch the instance method _load_table on the class or instance?
        # Easier to patch the method on the class
        with patch.object(CodexBuilder, "_load_table", side_effect=Exception("Load Fail")):
            with pytest.raises(RuntimeError, match="Build failed"):
                builder.build_vocab()

        # Verify close called
        mock_con.close.assert_called()

    # Verify cleanup (file should be gone)
    assert not db_path.exists()


def test_build_vectors_missing_vocab(tmp_path: Path, mock_embedder: Any) -> None:
    builder = CodexBuilder(tmp_path, tmp_path / "out")
    # Vocab not built
    with pytest.raises(FileNotFoundError, match="Vocab artifact not found"):
        builder.build_vectors(mock_embedder)


def test_build_vectors_embed_failure(source_csvs: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    builder = CodexBuilder(source_csvs, output_dir)
    builder.build_vocab()

    # Mock embedder to fail
    failing_embedder = MagicMock()
    failing_embedder.embed_batch.side_effect = Exception("Embed fail")

    with pytest.raises(RuntimeError, match="Vector build failed"):
        builder.build_vectors(failing_embedder)


def test_pipeline_get_instance_error() -> None:
    CodexContext._instance = None
    with pytest.raises(RuntimeError, match="CodexContext not initialized"):
        CodexContext.get_instance()
