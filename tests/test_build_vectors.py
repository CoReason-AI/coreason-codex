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
from typing import Any, Iterator, List
from unittest.mock import MagicMock, patch

import lancedb
import numpy as np
import pytest

from coreason_codex.build import CodexBuilder
from coreason_codex.interfaces import Embedder


class MockEmbedder(Embedder):
    """Deterministic mock embedder."""

    def embed(self, texts: List[str]) -> np.ndarray:
        # Return deterministic vectors based on string length or simple hash
        # Shape: (len(texts), 2)
        vectors = []
        for t in texts:
            # Simple deterministic float generation
            val = float(len(t))
            vectors.append([val, val + 0.5])
        return np.array(vectors, dtype=np.float32)


class FailingEmbedder(Embedder):
    """Embedder that raises an error."""

    def embed(self, texts: List[str]) -> np.ndarray:
        raise ValueError("Embedding failed")


@pytest.fixture
def mock_athena_data(tmp_path: Path) -> Path:
    """Creates dummy Athena CSV files."""
    source_dir = tmp_path / "athena_source"
    source_dir.mkdir()

    # Create CONCEPT.csv
    with open(source_dir / "CONCEPT.csv", "w", newline="") as f:
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
                "valid_start_date",
                "valid_end_date",
                "invalid_reason",
            ]
        )
        writer.writerow(
            [1, "Test Concept 1", "Condition", "SNOMED", "Clinical Finding", "S", "1001", "19700101", "20991231", ""]
        )
        writer.writerow([2, "Test Concept 2", "Drug", "RxNorm", "Ingredient", "S", "2002", "19700101", "20991231", ""])
        # Add a concept with None name (should be skipped by query)
        writer.writerow([3, "", "Metadata", "None", "None", "", "3003", "19700101", "20991231", ""])

    # Create other required files
    (source_dir / "CONCEPT_RELATIONSHIP.csv").touch()
    (source_dir / "CONCEPT_ANCESTOR.csv").touch()

    # Fill them with headers to avoid errors
    with open(source_dir / "CONCEPT_RELATIONSHIP.csv", "w", newline="") as f:
        csv.writer(f).writerow(
            ["concept_id_1", "concept_id_2", "relationship_id", "valid_start_date", "valid_end_date", "invalid_reason"]
        )

    with open(source_dir / "CONCEPT_ANCESTOR.csv", "w", newline="") as f:
        csv.writer(f).writerow(
            ["ancestor_concept_id", "descendant_concept_id", "min_levels_of_separation", "max_levels_of_separation"]
        )

    return source_dir


def test_build_vectors_success(mock_athena_data: Path, tmp_path: Path) -> None:
    """Test successful generation of LanceDB vectors."""
    output_dir = tmp_path / "output"
    builder = CodexBuilder(source_dir=mock_athena_data, output_dir=output_dir)

    # 1. Build Vocab first (prerequisite)
    builder.build_vocab()

    # 2. Build Vectors
    embedder = MockEmbedder()
    builder.build_vectors(embedder=embedder, table_name="vectors", batch_size=1)

    # 3. Verify
    db = lancedb.connect(str(output_dir))
    # list_tables() returns a response object in new lancedb versions, table_names() returns list
    assert "vectors" in db.table_names()

    table = db.open_table("vectors")

    # Use to_pylist instead of to_pandas to avoid pandas dependency
    data = table.to_arrow().to_pylist()

    # Check length
    # ID 1 and 2 should be present. ID 3 has empty name in CSV (,,) which DuckDB treats as NULL.
    # Query filters out NULLs.
    assert len(data) == 2

    # Check content
    row1 = next(r for r in data if r["concept_id"] == 1)
    assert row1["concept_name"] == "Test Concept 1"
    # Arrow list/array is returned as list in pylist
    assert isinstance(row1["vector"], list)
    assert len(row1["vector"]) == 2
    assert row1["vector"][0] == 14.0


def test_build_vectors_missing_vocab(tmp_path: Path) -> None:
    """Test error when vocab.duckdb is missing."""
    builder = CodexBuilder(source_dir=tmp_path, output_dir=tmp_path)
    embedder = MockEmbedder()

    with pytest.raises(FileNotFoundError, match="Vocab artifact not found"):
        builder.build_vectors(embedder)


def test_build_vectors_overwrites(mock_athena_data: Path, tmp_path: Path) -> None:
    """Test that build_vectors overwrites existing table."""
    output_dir = tmp_path / "output"
    builder = CodexBuilder(source_dir=mock_athena_data, output_dir=output_dir)
    builder.build_vocab()
    embedder = MockEmbedder()

    # Build once
    builder.build_vectors(embedder)

    # Build again
    builder.build_vectors(embedder)

    db = lancedb.connect(str(output_dir))
    table = db.open_table("vectors")
    data = table.to_arrow().to_pylist()
    # Should still have correct number of rows, not doubled
    assert len(data) == 2


def test_build_vectors_embedding_error(mock_athena_data: Path, tmp_path: Path) -> None:
    """Test handling of embedding errors."""
    output_dir = tmp_path / "output"
    builder = CodexBuilder(source_dir=mock_athena_data, output_dir=output_dir)
    builder.build_vocab()
    embedder = FailingEmbedder()

    with pytest.raises(RuntimeError, match="Vector build failed"):
        builder.build_vectors(embedder)


def test_build_vectors_db_error(mock_athena_data: Path, tmp_path: Path) -> None:
    """Test handling of DB errors."""
    output_dir = tmp_path / "output"
    builder = CodexBuilder(source_dir=mock_athena_data, output_dir=output_dir)
    builder.build_vocab()
    embedder = MockEmbedder()

    # Mock lancedb.connect to fail
    with patch("coreason_codex.build.lancedb.connect", side_effect=RuntimeError("DB Error")):
        with pytest.raises(RuntimeError, match="Vector build failed"):
            builder.build_vectors(embedder)


def test_batch_generator_coverage(mock_athena_data: Path, tmp_path: Path) -> None:
    """Test to ensure batch generator is fully exhausted to cover the break statement."""
    output_dir = tmp_path / "output"
    builder = CodexBuilder(source_dir=mock_athena_data, output_dir=output_dir)
    builder.build_vocab()
    embedder = MockEmbedder()

    with patch("coreason_codex.build.lancedb.connect") as mock_connect:
        mock_db = mock_connect.return_value

        def consume_generator(name: str, data: Iterator[Any], mode: str) -> Any:
            # Consume all items
            for _ in data:
                pass
            return MagicMock()

        mock_db.create_table.side_effect = consume_generator

        # Run build_vectors with batch_size=1 to force multiple iterations
        builder.build_vectors(embedder, batch_size=1)

        # Verify create_table was called
        assert mock_db.create_table.called
