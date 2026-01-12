# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_codex

import shutil
from pathlib import Path
from typing import Generator, List, Tuple

import lancedb
import numpy as np
import pytest
from coreason_codex.interfaces import Embedder
from coreason_codex.normalizer import CodexNormalizer


class MockEmbedder(Embedder):
    """Deterministic mock embedder for testing."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim

    def embed(self, texts: List[str]) -> np.ndarray:
        # Deterministic embedding based on text length/chars to ensure stability
        embeddings = []
        for text in texts:
            # Create a simple deterministic vector
            val = sum(ord(c) for c in text) % 100 / 100.0
            vec = np.array([val] * self.dim, dtype=np.float32)
            # Normalize to unit length for cosine similarity
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec)
        return np.array(embeddings)


@pytest.fixture
def temp_lancedb(tmp_path: Path) -> Generator[lancedb.table.Table, None, None]:
    """Create a temporary LanceDB table with dummy data."""
    db_path = tmp_path / "lancedb"
    db = lancedb.connect(db_path)

    # Define dummy data
    # We need a vector column. MockEmbedder(dim=4)
    # Let's say "Headache" -> val roughly 0.5 -> [0.5, 0.5, 0.5, 0.5] normalized

    data = [
        {
            "vector": [0.5, 0.5, 0.5, 0.5],
            "concept_id": 1001,
            "concept_name": "Headache",
            "domain_id": "Condition",
            "vocabulary_id": "SNOMED",
            "concept_class_id": "Clinical Finding",
            "standard_concept": "S",
            "concept_code": "12345",
        },
        {
            "vector": [0.6, 0.6, 0.6, 0.6],  # Close to Headache
            "concept_id": 1002,
            "concept_name": "Cephalalgia",
            "domain_id": "Condition",
            "vocabulary_id": "ICD10",
            "concept_class_id": "Diagnosis",
            "standard_concept": None,  # Non-standard
            "concept_code": "R51",
        },
        {
            "vector": [0.1, 0.1, 0.1, 0.1],  # Different
            "concept_id": 2001,
            "concept_name": "Aspirin",
            "domain_id": "Drug",
            "vocabulary_id": "RxNorm",
            "concept_class_id": "Ingredient",
            "standard_concept": "S",
            "concept_code": "1191",
        },
    ]

    tbl = db.create_table("concepts", data=data)
    yield tbl
    shutil.rmtree(db_path, ignore_errors=True)


def test_normalizer_finds_match(temp_lancedb: lancedb.table.Table) -> None:
    embedder = MockEmbedder(dim=4)
    normalizer = CodexNormalizer(temp_lancedb, embedder)
    # This test was empty in previous iteration, completing it or relying on others
    matches = normalizer.normalize("Headache")
    assert len(matches) > 0


@pytest.fixture
def temp_lancedb_with_embedder(tmp_path: Path) -> Generator[Tuple[lancedb.table.Table, Embedder], None, None]:
    """Create a temporary LanceDB table with data embedded by MockEmbedder."""
    db_path = tmp_path / "lancedb_embedded"
    db = lancedb.connect(db_path)
    embedder = MockEmbedder(dim=4)

    # Define concepts
    concepts = [
        ("Headache", "Condition", "S", 1001),
        ("Cephalalgia", "Condition", None, 1002),
        ("Aspirin", "Drug", "S", 2001),
        ("Tylenol", "Drug", None, 2002),
    ]

    data = []
    for name, domain, std, cid in concepts:
        vec = embedder.embed([name])[0]
        data.append(
            {
                "vector": vec,
                "concept_id": cid,
                "concept_name": name,
                "domain_id": domain,
                "vocabulary_id": "Mixed",
                "concept_class_id": "Test",
                "standard_concept": std,
                "concept_code": f"C{cid}",
            }
        )

    tbl = db.create_table("concepts", data=data)
    yield tbl, embedder
    shutil.rmtree(db_path, ignore_errors=True)


def test_normalizer_exact_match(temp_lancedb_with_embedder: Tuple[lancedb.table.Table, Embedder]) -> None:
    table, embedder = temp_lancedb_with_embedder
    normalizer = CodexNormalizer(table, embedder)

    matches = normalizer.normalize("Headache")
    assert len(matches) > 0
    top_match = matches[0]

    assert top_match.match_concept.concept_name == "Headache"
    assert top_match.match_concept.concept_id == 1001
    assert top_match.is_standard is True
    assert top_match.similarity_score > 0.99  # Should be practically 1.0


def test_normalizer_domain_filter(temp_lancedb_with_embedder: Tuple[lancedb.table.Table, Embedder]) -> None:
    table, embedder = temp_lancedb_with_embedder
    normalizer = CodexNormalizer(table, embedder)

    # "Aspirin" is a Drug. If we filter by "Condition", we shouldn't find it,
    # or we should find "Headache" if it's somewhat close?
    # MockEmbedder is character based. "Aspirin" vs "Headache".
    # Let's search for "Aspirin" but filter "Condition".

    matches = normalizer.normalize("Aspirin", domain_filter="Condition")

    # Should NOT return Aspirin (Drug)
    for match in matches:
        assert match.match_concept.domain_id == "Condition"
        assert match.match_concept.concept_name != "Aspirin"


def test_normalizer_strict_typing(temp_lancedb_with_embedder: Tuple[lancedb.table.Table, Embedder]) -> None:
    table, embedder = temp_lancedb_with_embedder
    normalizer = CodexNormalizer(table, embedder)

    matches = normalizer.normalize("Headache")
    concept = matches[0].match_concept

    assert isinstance(concept.concept_id, int)
    assert isinstance(concept.concept_name, str)


def test_empty_embeddings() -> None:
    """Test handling of empty embeddings."""

    class EmptyEmbedder(Embedder):
        def embed(self, texts: List[str]) -> np.ndarray:
            return np.array([])

    embedder = EmptyEmbedder()
    # Mock table (not used)
    normalizer = CodexNormalizer(None, embedder)

    matches = normalizer.normalize("test")
    assert matches == []


def test_normalizer_malformed_data(tmp_path: Path) -> None:
    """Test handling of malformed data in LanceDB."""
    db_path = tmp_path / "lancedb_malformed"
    db = lancedb.connect(db_path)
    embedder = MockEmbedder(dim=4)

    # Create data with missing required field (e.g. concept_id is a string "invalid" or missing)
    # LanceDB is schema-less on insert usually unless defined, but Pydantic will complain.
    # Note: LanceDB might enforce types if we define schema, but here we just pass dicts.

    vec = embedder.embed(["Malformed"])[0]
    data = [
        {
            "vector": vec,
            # concept_id is missing/wrong type
            "concept_id": "NOT_AN_INT",
            "concept_name": "Malformed Concept",
            "domain_id": "Error",
            "vocabulary_id": "ERR",
            "concept_class_id": "Err",
            "standard_concept": None,
            "concept_code": "E1",
        }
    ]

    tbl = db.create_table("concepts_malformed", data=data)

    normalizer = CodexNormalizer(tbl, embedder)

    # This should trigger the exception block but continue
    matches = normalizer.normalize("Malformed")

    # Should handle the error gracefully and return empty list (since the only result failed)
    assert len(matches) == 0

    shutil.rmtree(db_path, ignore_errors=True)
