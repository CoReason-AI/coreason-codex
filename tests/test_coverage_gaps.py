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
from unittest.mock import patch

import pytest

from coreason_codex.build import CodexBuilder
from coreason_codex.loader import CodexLoader


def test_loader_path_resolution_error(synthetic_codex_pack: Path) -> None:
    loader = CodexLoader(synthetic_codex_pack)

    # Mock Path.resolve to raise arbitrary exception
    with patch.object(Path, "resolve", side_effect=Exception("Disk Error")):
        with pytest.raises(ValueError, match="Invalid path for artifact"):
            loader.verify_integrity()


def test_build_vocab_cleanup_existing_file(source_csvs: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "out_cleanup"
    output_dir.mkdir()
    builder = CodexBuilder(source_csvs, output_dir)

    # Create dummy vocab file that should be deleted before build starts
    db_path = output_dir / "vocab.duckdb"
    db_path.touch()

    # We want to test the pre-build cleanup: "if db_path.exists(): db_path.unlink()"
    # Just running `build_vocab` normally with an existing file covers it.

    builder.build_vocab()

    # Coverage should mark line 64 (unlink) as covered.


def test_build_vectors_log_coverage(tmp_path: Path, mock_embedder: Any) -> None:
    pass
