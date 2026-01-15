# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_codex

import csv
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock

import pytest

from coreason_codex.build import CodexBuilder
from coreason_codex.interfaces import Embedder
from coreason_codex.loader import CodexLoader
from coreason_codex.normalizer import CodexNormalizer

# --- Mocks ---


class MismatchedEmbedder(Embedder):
    """Embedder that returns fewer vectors than inputs."""

    def embed(self, text: str) -> List[float]:
        return [0.0] * 128

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        # Return only 1 vector regardless of input size (if > 0)
        if not texts:
            return []
        return [[0.0] * 128]  # Always return 1 vector even if texts has 10 items


class MockEmbedder(Embedder):
    """Deterministic mock embedder."""

    def embed(self, text: str) -> List[float]:
        return [0.1] * 128

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [[0.1] * 128 for _ in texts]


# --- Tests ---


def test_normalizer_resilience_orphan_ids(synthetic_codex_pack: Path, mock_embedder: Any) -> None:
    """
    Test that the Normalizer handles cases where Vector DB returns IDs
    that are missing from DuckDB (integrity drift).
    """
    loader = CodexLoader(synthetic_codex_pack)
    con, lancedb_con = loader.load_codex()

    norm = CodexNormalizer(mock_embedder, con, lancedb_con)

    # Mock LanceDB table search to return a mix of valid and invalid IDs
    # Valid ID from synthetic pack: 312327
    # Invalid ID: 999999999

    mock_table = MagicMock()
    mock_query = MagicMock()
    mock_limit = MagicMock()

    mock_table.search.return_value = mock_query
    mock_query.limit.return_value = mock_limit

    # Return matches
    mock_limit.to_list.return_value = [
        {"concept_id": 312327, "_distance": 0.1},  # Valid
        {"concept_id": 999999999, "_distance": 0.2},  # Orphan
    ]

    norm.table = mock_table

    matches = norm.normalize("input")

    # Should contain only the valid one
    assert len(matches) == 1
    assert matches[0].match_concept.concept_id == 312327


def test_unicode_handling(tmp_path: Path) -> None:
    """Test full cycle with Unicode characters."""
    source_dir = tmp_path / "unicode_src"
    source_dir.mkdir()

    # Create files with Unicode
    with open(source_dir / "CONCEPT.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "concept_id",
                "concept_name",
                "domain_id",
                "vocabulary_id",
                "concept_class_id",
                "standard_concept",
                "concept_code",
                "invalid_reason",
            ]
        )
        writer.writerow([100, "Ménière's disease", "Condition", "SNOMED", "Clinical Finding", "S", "CODE_100", ""])
        writer.writerow([101, "β-blocker", "Drug", "RxNorm", "Ingredient", "S", "CODE_101", ""])

    # Required files
    for fname in ["CONCEPT_ANCESTOR.csv", "CONCEPT_RELATIONSHIP.csv"]:
        with open(source_dir / fname, "w", newline="") as f:
            # write headers only
            if fname == "CONCEPT_ANCESTOR.csv":
                f.write("ancestor_concept_id,descendant_concept_id,min_levels_of_separation,max_levels_of_separation\n")
            else:
                f.write("concept_id_1,concept_id_2,relationship_id,valid_start_date,valid_end_date,invalid_reason\n")

    # Build
    output_dir = tmp_path / "unicode_out"
    builder = CodexBuilder(source_dir, output_dir)
    builder.build_vocab()

    embedder = MockEmbedder()
    builder.build_vectors(embedder)
    builder.generate_manifest()

    # Load and Verify
    loader = CodexLoader(output_dir)
    con, lancedb_con = loader.load_codex()

    # 1. Check DuckDB hydration of unicode
    res = con.execute("SELECT concept_name FROM concept WHERE concept_id = 100").fetchone()
    assert res is not None
    assert res[0] == "Ménière's disease"

    res2 = con.execute("SELECT concept_name FROM concept WHERE concept_id = 101").fetchone()
    assert res2 is not None
    assert res2[0] == "β-blocker"

    # 2. Check Normalizer usage
    norm = CodexNormalizer(embedder, con, lancedb_con)
    matches = norm.normalize("Ménière")
    assert len(matches) > 0
    names = {m.match_concept.concept_name for m in matches}
    assert "Ménière's disease" in names or "β-blocker" in names


def test_builder_embedder_mismatch(source_csvs: Path, tmp_path: Path) -> None:
    """Test that builder fails if embedder returns mismatched list length."""
    output_dir = tmp_path / "mismatch_out"
    builder = CodexBuilder(source_csvs, output_dir)
    builder.build_vocab()

    bad_embedder = MismatchedEmbedder()

    with pytest.raises(RuntimeError, match="Vector build failed"):
        builder.build_vectors(bad_embedder, batch_size=2)
