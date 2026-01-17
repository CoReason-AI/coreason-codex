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
    matches = norm.normalize("Alien space virus from Mars", k=2)
    assert isinstance(matches, list)


def test_normalize_zero_results(loaded_components: Any) -> None:
    con, lancedb_con, embedder = loaded_components
    norm = CodexNormalizer(embedder, con, lancedb_con)

    # Mock table to return empty list
    mock_table = MagicMock()
    mock_query = MagicMock()
    mock_limit = MagicMock()

    mock_table.search.return_value = mock_query
    mock_query.limit.return_value = mock_limit
    mock_limit.to_list.return_value = []

    norm.table = mock_table
    assert norm.normalize("test") == []


def test_normalize_synonym_score_preservation(loaded_components: Any) -> None:
    con, lancedb_con, embedder = loaded_components
    norm = CodexNormalizer(embedder, con, lancedb_con)

    mock_table = MagicMock()
    mock_query = MagicMock()
    mock_limit = MagicMock()

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
    assert abs(matches[0].similarity_score - 0.9) < 0.0001


def test_normalize_mapped_standard(loaded_components: Any) -> None:
    """
    Test that searching for a non-standard concept (ICD10 I21.9)
    returns the mapped standard concept ID (SNOMED 312327).
    """
    con, lancedb_con, embedder = loaded_components
    norm = CodexNormalizer(embedder, con, lancedb_con)

    # 999999 is "Acute myocardial infarction, unspecified" (ICD10 I21.9)
    # It maps to 312327 (SNOMED Acute myocardial infarction)
    query = "Acute myocardial infarction, unspecified"
    matches = norm.normalize(query, k=5)

    assert len(matches) > 0

    # Find the match for our non-standard concept
    match = next((m for m in matches if m.match_concept.concept_id == 999999), None)
    assert match is not None
    assert match.is_standard is False  # standard_concept is NULL
    assert match.mapped_standard_id == 312327


def test_normalize_mapped_standard_error(loaded_components: Any) -> None:
    """
    Test that error handling works when fetching mapped standard concept.
    """
    con, lancedb_con, embedder = loaded_components
    norm = CodexNormalizer(embedder, con, lancedb_con)

    # 1. We want to simulate an error ONLY when querying concept_relationship.
    # The normalizer makes 2 types of queries:
    # A. To 'concept' table (to hydrate).
    # B. To 'concept_relationship' table (to map).

    # We can wrap the duckdb connection to intercept execute.
    # But it's easier to patch 'execute' on the connection object if it were a python object.
    # DuckDBPyConnection is a C extension type.
    # Instead, we can inject a mock connection that delegates to the real one, but fails on specific query.

    real_execute = con.execute

    def side_effect(query: str, params: list = None):
        if "concept_relationship" in query:
            raise RuntimeError("Database exploded")
        return real_execute(query, params)

    # We need to mock the connection object provided to CodexNormalizer
    mock_conn = MagicMock(wraps=con)
    mock_conn.execute.side_effect = side_effect

    norm = CodexNormalizer(embedder, mock_conn, lancedb_con)

    # Trigger the logic: search for non-standard concept
    query = "Acute myocardial infarction, unspecified"
    matches = norm.normalize(query, k=5)

    # Should still return matches, but mapped_standard_id should be None (and log warning)
    match = next((m for m in matches if m.match_concept.concept_id == 999999), None)
    assert match is not None
    assert match.is_standard is False
    assert match.mapped_standard_id is None
