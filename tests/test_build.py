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

import pytest

from coreason_codex.build import CodexBuilder
from coreason_codex.loader import CodexLoader

# Note: source_csvs fixture is now in conftest.py


def test_builder_full_workflow(source_csvs: Path, tmp_path: Path, mock_embedder: Any) -> None:
    output_dir = tmp_path / "build_output"

    builder = CodexBuilder(source_csvs, output_dir)

    # 1. Build Vocab
    vocab_path = builder.build_vocab()
    assert vocab_path.exists()
    assert (output_dir / "vocab.duckdb").exists()

    # 2. Build Vectors
    builder.build_vectors(mock_embedder)
    # Check if LanceDB artifact exists (table 'vectors')
    # It usually creates a directory structure.
    # We can check by listing.
    assert (output_dir / "vectors.lance").exists()

    # 3. Generate Manifest
    manifest_path = builder.generate_manifest()
    assert manifest_path.exists()

    # 4. Verify with Loader
    loader = CodexLoader(output_dir)
    assert loader.verify_integrity()
    con, lancedb_con = loader.load_codex()
    assert con is not None

    # Verify Data
    result = con.execute("SELECT COUNT(*) FROM concept").fetchone()
    assert result is not None
    count = result[0]
    assert count == 2


def test_builder_missing_files(tmp_path: Path) -> None:
    empty_src = tmp_path / "empty"
    empty_src.mkdir()
    builder = CodexBuilder(empty_src, tmp_path / "out")

    with pytest.raises(FileNotFoundError, match="Required file not found"):
        builder.build_vocab()
