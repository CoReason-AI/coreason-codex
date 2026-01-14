# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_codex

from typing import Any, Tuple
from unittest.mock import MagicMock

import duckdb
import pytest

from coreason_codex.interfaces import Embedder
from coreason_codex.loader import CodexLoader
from coreason_codex.normalizer import CodexNormalizer


@pytest.fixture
def loaded_components(synthetic_codex_pack: Any, mock_embedder: Any) -> Tuple[duckdb.DuckDBPyConnection, Any, Embedder]:
    """
    Returns (duckdb_conn, lancedb_conn, embedder)
    """
    loader = CodexLoader(synthetic_codex_pack)
    con, lancedb_con = loader.load_codex()
    return con, lancedb_con, mock_embedder


def test_normalizer_init(loaded_components: Any) -> None:
    con, lancedb_con, embedder = loaded_components
    norm = CodexNormalizer(embedder, con, lancedb_con)
    assert norm is not None


def test_normalizer_init_bad_table(loaded_components: Any) -> None:
    con, lancedb_con, embedder = loaded_components
    # Try to init with non-existent table
    with pytest.raises(ValueError, match="not found"):
        CodexNormalizer(embedder, con, lancedb_con, vector_table_name="non_existent_table")


def test_normalize_exact_match(loaded_components: Any) -> None:
    con, lancedb_con, embedder = loaded_components
    norm = CodexNormalizer(embedder, con, lancedb_con)

    # Search for "Acute myocardial infarction" (Concept 312327)
    # Since we use a deterministic MockEmbedder, the vector for the exact string will match perfectly.
    query = "Acute myocardial infarction"
    matches = norm.normalize(query, k=5)

    assert len(matches) > 0
    top_match = matches[0]

    assert top_match.match_concept.concept_id == 312327
    assert top_match.match_concept.concept_name == "Acute myocardial infarction"
    # Similarity should be very close to 1.0 (since dist is 0)
    assert top_match.similarity_score > 0.99
    assert top_match.is_standard is True


def test_normalize_domain_filtering(loaded_components: Any) -> None:
    con, lancedb_con, embedder = loaded_components
    norm = CodexNormalizer(embedder, con, lancedb_con)

    # "Metformin" is a Drug (1503297).
    query = "Metformin"

    # 1. Search without filter
    matches_all = norm.normalize(query, k=5)
    # Should find it
    found = any(m.match_concept.concept_id == 1503297 for m in matches_all)
    assert found

    # 2. Search WITH filter "Condition"
    # Metformin is a Drug, so it should be filtered out.
    matches_filtered = norm.normalize(query, k=5, domain_filter="Condition")

    # Should NOT find it
    found_filtered = any(m.match_concept.concept_id == 1503297 for m in matches_filtered)
    assert not found_filtered


def test_normalize_empty_input(loaded_components: Any) -> None:
    con, lancedb_con, embedder = loaded_components
    norm = CodexNormalizer(embedder, con, lancedb_con)
    assert norm.normalize("") == []
    assert norm.normalize("   ") == []


def test_normalize_no_matches(loaded_components: Any) -> None:
    con, lancedb_con, embedder = loaded_components
    norm = CodexNormalizer(embedder, con, lancedb_con)

    # Search for something that shouldn't exist or is far away.
    # With random vectors, everything is somewhat close, but we can verify it doesn't crash.
    matches = norm.normalize("Alien space virus from Mars", k=2)
    assert isinstance(matches, list)
    # It might return results because approximate search always returns closest.
    # We just check structure.
    if matches:
        assert isinstance(matches[0].similarity_score, float)


def test_normalize_synonym_score_preservation(loaded_components: Any) -> None:
    con, lancedb_con, embedder = loaded_components
    norm = CodexNormalizer(embedder, con, lancedb_con)

    # Mock the LanceDB table to return explicit results simulating synonyms
    # Synonym A (Close) -> ID 123, Dist 0.1 (Score 0.9)
    # Synonym B (Far)   -> ID 123, Dist 0.5 (Score 0.5)
    # We want to ensure ID 123 gets Score 0.9

    mock_table = MagicMock()
    mock_query = MagicMock()
    mock_limit = MagicMock()

    # Chain: search() -> limit() -> to_list()
    mock_table.search.return_value = mock_query
    mock_query.limit.return_value = mock_limit
    mock_limit.to_list.return_value = [
        {"concept_id": 312327, "_distance": 0.1},  # First match (Best)
        {"concept_id": 312327, "_distance": 0.5},  # Second match (Worst)
    ]

    norm.table = mock_table

    matches = norm.normalize("Some query")

    assert len(matches) == 1
    assert matches[0].match_concept.concept_id == 312327
    # Should be 1.0 - 0.1 = 0.9
    # If bug exists, it would be 1.0 - 0.5 = 0.5
    assert abs(matches[0].similarity_score - 0.9) < 0.0001
