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

from coreason_codex.loader import CodexLoader


def test_synthetic_pack_structure(synthetic_codex_pack: Path) -> None:
    """
    Verifies that the synthetic fixture creates the expected files.
    """
    assert (synthetic_codex_pack / "vocab.duckdb").exists()
    assert (synthetic_codex_pack / "vectors.lance").exists()
    assert (synthetic_codex_pack / "manifest.json").exists()


def test_codex_loader_integrity(synthetic_codex_pack: Path) -> None:
    """
    Verifies that the CodexLoader accepts the synthetic pack.
    """
    loader = CodexLoader(synthetic_codex_pack)
    # verify_integrity returns True on success
    assert loader.verify_integrity() is True


def test_codex_loader_connections(synthetic_codex_pack: Path) -> None:
    """
    Verifies that we can actually query the loaded artifacts.
    """
    loader = CodexLoader(synthetic_codex_pack)
    con, lancedb_con = loader.load_codex()

    # 1. Test DuckDB
    result = con.execute("SELECT COUNT(*) FROM concept").fetchone()
    assert result is not None
    count = result[0]
    assert count >= 5

    # 2. Test LanceDB
    # lancedb_con is the DB object. We need to open the table.
    # The fixture creates table "vectors"
    table = lancedb_con.open_table("vectors")
    # LanceDB length check might vary by version, check count via len or query
    assert len(table) >= 5


def test_mock_embedder(mock_embedder: Any) -> None:
    """
    Verifies the mock embedder is deterministic and correct shape.
    """
    t1 = "Heart Attack"
    t2 = "Myocardial Infarction"

    v1 = mock_embedder.embed(t1)
    v2 = mock_embedder.embed(t2)
    v1_again = mock_embedder.embed(t1)

    # Deterministic
    assert v1 == v1_again
    # Different inputs, different outputs (highly likely)
    assert v1 != v2
    # Correct dimension
    assert len(v1) == 128

    # Normalized
    norm = sum(x**2 for x in v1) ** 0.5
    assert abs(norm - 1.0) < 1e-6
