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
from typing import Any, Generator, List
from unittest.mock import Mock

import lancedb
import pytest

from coreason_codex.interfaces import Embedder
from coreason_codex.normalizer import CodexNormalizer


class MockEmbedder(Embedder):
    """
    Mock embedder that returns deterministic vectors based on text content.
    """

    def __init__(self, dim: int = 4):
        self.dim = dim

    def embed(self, text: str) -> List[float]:
        # Simple deterministic vector generation
        # e.g., "A" -> [1.0, 0.0, ...], "B" -> [0.0, 1.0, ...]
        # For simplicity, we just use hash based values or predefined ones for test cases
        if text == "Condition A":
            return [1.0, 0.0, 0.0, 0.0]
        elif text == "Condition B":
            return [0.0, 1.0, 0.0, 0.0]
        elif text == "Drug A":
            return [0.0, 0.0, 1.0, 0.0]
        else:
            # Random-ish vector
            val = float(len(text)) / 100.0
            return [val] * self.dim


@pytest.fixture
def lance_test_db(tmp_path: Path) -> Generator[Any, None, None]:
    """Creates a temporary LanceDB with a populated 'vectors' table."""
    db_path = tmp_path / "lancedb_test"
    db = lancedb.connect(str(db_path))

    # Data corresponding to MockEmbedder vectors
    # Vectors are size 4
    data = [
        {
            "vector": [1.0, 0.0, 0.0, 0.0],
            "concept_id": 1001,
            "concept_name": "Condition A (Standard)",
            "domain_id": "Condition",
            "vocabulary_id": "SNOMED",
            "concept_class_id": "Clinical Finding",
            "standard_concept": "S",
            "concept_code": "C1001",
        },
        {
            "vector": [0.0, 1.0, 0.0, 0.0],
            "concept_id": 1002,
            "concept_name": "Condition B (Non-Standard)",
            "domain_id": "Condition",
            "vocabulary_id": "ICD10",
            "concept_class_id": "Diagnosis",
            "standard_concept": None,
            "concept_code": "C1002",
        },
        {
            "vector": [0.0, 0.0, 1.0, 0.0],
            "concept_id": 2001,
            "concept_name": "Drug A (Standard)",
            "domain_id": "Drug",
            "vocabulary_id": "RxNorm",
            "concept_class_id": "Ingredient",
            "standard_concept": "S",
            "concept_code": "D2001",
        },
    ]

    table = db.create_table("vectors", data=data)
    yield table
    # cleanup handled by tmp_path


def test_normalize_exact_match(lance_test_db: Any) -> None:
    """Test finding an exact match for a concept."""
    embedder = MockEmbedder()
    normalizer = CodexNormalizer(lance_test_db, embedder)

    # Search for "Condition A" -> Vector [1, 0, 0, 0] -> Matches id 1001
    match = normalizer.normalize("Condition A")

    assert match is not None
    assert match.input_text == "Condition A"
    assert match.match_concept.concept_id == 1001
    assert match.match_concept.concept_name == "Condition A (Standard)"
    assert match.is_standard is True
    assert match.mapped_standard_id == 1001

    # Distance should be 0 (exact match), so similarity 1.0
    # Floating point comparison
    assert abs(match.similarity_score - 1.0) < 1e-5


def test_normalize_non_standard(lance_test_db: Any) -> None:
    """Test finding a non-standard concept."""
    embedder = MockEmbedder()
    normalizer = CodexNormalizer(lance_test_db, embedder)

    # Search for "Condition B"
    match = normalizer.normalize("Condition B")

    assert match is not None
    assert match.match_concept.concept_id == 1002
    assert match.is_standard is False
    assert match.mapped_standard_id is None


def test_normalize_domain_filter_success(lance_test_db: Any) -> None:
    """Test filtering by correct domain."""
    embedder = MockEmbedder()
    normalizer = CodexNormalizer(lance_test_db, embedder)

    # Search for "Condition A" with filter "Condition"
    match = normalizer.normalize("Condition A", domain_filter="Condition")

    assert match is not None
    assert match.match_concept.concept_id == 1001


def test_normalize_domain_filter_exclusion(lance_test_db: Any) -> None:
    """Test filtering excludes matches from other domains."""
    embedder = MockEmbedder()
    normalizer = CodexNormalizer(lance_test_db, embedder)

    # Search for "Condition A" (which is mostly Condition) but filter by "Drug"
    # Even if vector matches Condition A perfectly, if we filter by Drug,
    # it should either return nothing or the Drug A if close enough.
    # In our small DB, Drug A is orthogonal ([0,0,1,0] vs [1,0,0,0]).
    # So it might return Drug A with low similarity or nothing if query limit/radius.
    # LanceDB search returns k-NN. If we filter, it finds nearest in that subset.

    match = normalizer.normalize("Condition A", domain_filter="Drug")

    # It should match Drug A (2001) because that's the only Drug.
    assert match is not None
    assert match.match_concept.domain_id == "Drug"
    assert match.match_concept.concept_id == 2001
    # Similarity should be low (0 vs 1) -> Distance usually calculated.
    # Cosine distance between [1,0,0,0] and [0,0,1,0] is 1.0. Similarity 0.0.
    assert match.similarity_score < 0.1


def test_normalize_no_results_empty_db(tmp_path: Path) -> None:
    """Test handling when DB is empty."""
    db_path = tmp_path / "empty_db"
    db = lancedb.connect(str(db_path))

    # Create empty table with explicit schema
    import pyarrow as pa

    schema = pa.schema(
        [
            pa.field("vector", pa.list_(pa.float32(), 4)),
            pa.field("concept_id", pa.int64()),
            pa.field("concept_name", pa.string()),
            pa.field("domain_id", pa.string()),
            pa.field("vocabulary_id", pa.string()),
            pa.field("concept_class_id", pa.string()),
            pa.field("standard_concept", pa.string()),
            pa.field("concept_code", pa.string()),
        ]
    )
    table = db.create_table("vectors", schema=schema)

    embedder = MockEmbedder()
    normalizer = CodexNormalizer(table, embedder)

    match = normalizer.normalize("Anything")
    assert match is None


def test_embedder_failure(lance_test_db: Any) -> None:
    """Test graceful handling of embedder failure."""

    class BrokenEmbedder(Embedder):
        def embed(self, text: str) -> List[float]:
            raise ValueError("Embedding crashed")

    normalizer = CodexNormalizer(lance_test_db, BrokenEmbedder())
    match = normalizer.normalize("Test")
    assert match is None


def test_lancedb_search_failure(lance_test_db: Any) -> None:
    """Test graceful handling of LanceDB search failure."""
    # Mock the table object to raise an exception
    mock_table = Mock()
    mock_table.search.side_effect = Exception("DB Connection Lost")

    embedder = MockEmbedder()
    normalizer = CodexNormalizer(mock_table, embedder)

    match = normalizer.normalize("Test")
    assert match is None


def test_pydantic_validation_error(tmp_path: Path) -> None:
    """Test handling of malformed data in LanceDB."""
    db_path = tmp_path / "malformed_db"
    db = lancedb.connect(str(db_path))

    # Create table with missing required field (e.g., concept_id is missing or null?)
    # LanceDB/Arrow usually enforces schema, but we can have missing columns if schema is loose
    # or if we map to Pydantic and data is wrong (e.g. string for int).

    # Let's create a table that has valid vector but invalid other fields for Concept model.
    # Concept requires concept_id (int). Let's provide a string.
    # Note: LanceDB might enforce types if we define schema, but if we just insert dicts, it infers.

    data = [
        {
            "vector": [1.0, 0.0, 0.0, 0.0],
            "concept_id": "not_an_int",  # Invalid type
            "concept_name": "Bad Concept",
            "domain_id": "Condition",
            "vocabulary_id": "S",
            "concept_class_id": "C",
            "standard_concept": "S",
            "concept_code": "123",
        }
    ]

    table = db.create_table("vectors", data=data)

    embedder = MockEmbedder()
    normalizer = CodexNormalizer(table, embedder)

    # Should find the record, but fail to parse into Concept, returning None
    match = normalizer.normalize("Condition A")
    assert match is None


def test_invalid_domain_filter_injection(lance_test_db: Any) -> None:
    """Test that invalid domain filter strings are rejected."""
    embedder = MockEmbedder()
    normalizer = CodexNormalizer(lance_test_db, embedder)

    # Attempt Injection
    injection_attempt = "Condition' OR '1'='1"
    match = normalizer.normalize("Condition A", domain_filter=injection_attempt)

    assert match is None
