# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_codex

from typing import Any
from unittest.mock import MagicMock

import duckdb
import pytest

from coreason_codex.normalizer import CodexNormalizer
from coreason_codex.schemas import CodexMatch, Concept


@pytest.fixture
def memory_db() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")

    # Create minimal schema
    con.execute("""
        CREATE TABLE concept_relationship (
            concept_id_1 INTEGER,
            concept_id_2 INTEGER,
            relationship_id VARCHAR,
            invalid_reason VARCHAR
        )
    """)
    return con


@pytest.fixture
def mock_components() -> Any:
    """Returns (mock_embedder, mock_lancedb)"""
    return MagicMock(), MagicMock()


def create_match(concept_id: int, is_standard: bool) -> CodexMatch:
    """Helper to create a dummy CodexMatch"""
    c = Concept(
        concept_id=concept_id,
        concept_name=f"Concept {concept_id}",
        domain_id="Condition",
        vocabulary_id="Test",
        concept_class_id="Test",
        standard_concept="S" if is_standard else None,
        concept_code="CODE",
    )
    return CodexMatch(
        input_text="test", match_concept=c, similarity_score=1.0, is_standard=is_standard, mapped_standard_id=None
    )


def test_mapping_missing_relationship(memory_db: duckdb.DuckDBPyConnection, mock_components: Any) -> None:
    embedder, lancedb = mock_components
    # Mock table existence check
    lancedb.open_table.return_value = MagicMock()

    norm = CodexNormalizer(embedder, memory_db, lancedb)

    # Input: Non-standard concept 100
    # DB State: Empty relationship table
    matches = [create_match(100, is_standard=False)]

    norm._hydrate_mapped_standard_ids(matches)

    assert matches[0].mapped_standard_id is None


def test_mapping_invalid_relationship(memory_db: duckdb.DuckDBPyConnection, mock_components: Any) -> None:
    embedder, lancedb = mock_components
    lancedb.open_table.return_value = MagicMock()
    norm = CodexNormalizer(embedder, memory_db, lancedb)

    # Input: Non-standard concept 100
    matches = [create_match(100, is_standard=False)]

    # DB State: Relationship exists but is invalid
    memory_db.execute("INSERT INTO concept_relationship VALUES (100, 200, 'Maps to', 'D')")

    norm._hydrate_mapped_standard_ids(matches)

    assert matches[0].mapped_standard_id is None


def test_mapping_multiple_targets(memory_db: duckdb.DuckDBPyConnection, mock_components: Any) -> None:
    embedder, lancedb = mock_components
    lancedb.open_table.return_value = MagicMock()
    norm = CodexNormalizer(embedder, memory_db, lancedb)

    # Input: Non-standard concept 100
    matches = [create_match(100, is_standard=False)]

    # DB State: Maps to 200 AND 300
    # Note: DuckDB query without ORDER BY is non-deterministic, but the dict comprehension
    # {row[0]: row[1] for row in rows} will overwrite.
    # We just want to ensure it picks ONE valid ID.
    memory_db.execute("INSERT INTO concept_relationship VALUES (100, 200, 'Maps to', NULL)")
    memory_db.execute("INSERT INTO concept_relationship VALUES (100, 300, 'Maps to', NULL)")

    norm._hydrate_mapped_standard_ids(matches)

    assert matches[0].mapped_standard_id in [200, 300]


def test_mixed_batch_hydration(memory_db: duckdb.DuckDBPyConnection, mock_components: Any) -> None:
    embedder, lancedb = mock_components
    lancedb.open_table.return_value = MagicMock()
    norm = CodexNormalizer(embedder, memory_db, lancedb)

    # Input batch:
    # 1. Standard (10) - Should be ignored
    # 2. Non-Standard Mapped (20 -> 200)
    # 3. Non-Standard Unmapped (30)

    matches = [
        create_match(10, is_standard=True),
        create_match(20, is_standard=False),
        create_match(30, is_standard=False),
    ]

    # DB State
    memory_db.execute("INSERT INTO concept_relationship VALUES (20, 200, 'Maps to', NULL)")

    norm._hydrate_mapped_standard_ids(matches)

    assert matches[0].mapped_standard_id is None  # Standard, untouched
    assert matches[1].mapped_standard_id == 200  # Mapped
    assert matches[2].mapped_standard_id is None  # Unmapped
