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
from unittest.mock import patch

import lancedb
import numpy as np
import pytest

from coreason_codex.build import CodexBuilder
from coreason_codex.interfaces import Embedder


class MismatchedEmbedder(Embedder):
    """Embedder that returns fewer vectors than inputs."""

    def embed(self, texts: List[str]) -> np.ndarray:
        # Return only 1 vector regardless of input size (if > 0)
        if not texts:
            return np.array([])
        return np.array([[0.0, 0.0]], dtype=np.float32)


class MockEmbedder(Embedder):
    """Deterministic mock embedder."""

    def embed(self, texts: List[str]) -> np.ndarray:
        vectors = []
        for t in texts:
            val = float(len(t))
            vectors.append([val, val + 0.5])
        return np.array(vectors, dtype=np.float32)


@pytest.fixture
def unicode_athena_data(tmp_path: Path) -> Path:
    """Creates Athena CSV files with Unicode characters."""
    source_dir = tmp_path / "unicode_source"
    source_dir.mkdir()

    # Create CONCEPT.csv with Unicode
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
                "valid_start_date",
                "valid_end_date",
                "invalid_reason",
            ]
        )
        writer.writerow(
            [1, "Ménière's disease", "Condition", "SNOMED", "Clinical Finding", "S", "1001", "19700101", "20991231", ""]
        )
        writer.writerow([2, "β-blocker", "Drug", "RxNorm", "Ingredient", "S", "2002", "19700101", "20991231", ""])

    # Required files
    (source_dir / "CONCEPT_RELATIONSHIP.csv").touch()
    (source_dir / "CONCEPT_ANCESTOR.csv").touch()

    with open(source_dir / "CONCEPT_RELATIONSHIP.csv", "w", newline="") as f:
        csv.writer(f).writerow(
            ["concept_id_1", "concept_id_2", "relationship_id", "valid_start_date", "valid_end_date", "invalid_reason"]
        )

    with open(source_dir / "CONCEPT_ANCESTOR.csv", "w", newline="") as f:
        csv.writer(f).writerow(
            ["ancestor_concept_id", "descendant_concept_id", "min_levels_of_separation", "max_levels_of_separation"]
        )

    return source_dir


def test_vectors_unicode_handling(unicode_athena_data: Path, tmp_path: Path) -> None:
    """Test that Unicode characters are handled correctly in build and storage."""
    output_dir = tmp_path / "output"
    builder = CodexBuilder(source_dir=unicode_athena_data, output_dir=output_dir)
    builder.build_vocab()

    embedder = MockEmbedder()
    builder.build_vectors(embedder)

    # Verify
    db = lancedb.connect(str(output_dir))
    table = db.open_table("vectors")
    data = table.to_arrow().to_pylist()

    assert len(data) == 2

    # Check 1: Ménière's disease
    row1 = next(r for r in data if r["concept_id"] == 1)
    assert row1["concept_name"] == "Ménière's disease"

    # Check 2: β-blocker
    row2 = next(r for r in data if r["concept_id"] == 2)
    assert row2["concept_name"] == "β-blocker"


def test_vectors_mismatched_embeddings(unicode_athena_data: Path, tmp_path: Path) -> None:
    """Test that an error is raised if embedder returns mismatched vector counts."""
    output_dir = tmp_path / "output"
    builder = CodexBuilder(source_dir=unicode_athena_data, output_dir=output_dir)
    builder.build_vocab()

    embedder = MismatchedEmbedder()

    # We expect an IndexError because the loop will try to access index 1 when only index 0 exists
    # Or RuntimeError if wrapped
    # The code wraps exceptions in RuntimeError("Vector build failed: ...")
    with pytest.raises(RuntimeError) as excinfo:
        builder.build_vectors(embedder, batch_size=2)

    assert "Vector build failed" in str(excinfo.value)
    # The underlying cause should be IndexError
    assert isinstance(excinfo.value.__cause__, IndexError)


def test_vectors_generator_interruption(unicode_athena_data: Path, tmp_path: Path) -> None:
    """Test handling of an exception raised during batch generation (cursor fetch)."""
    output_dir = tmp_path / "output"
    builder = CodexBuilder(source_dir=unicode_athena_data, output_dir=output_dir)
    builder.build_vocab()
    embedder = MockEmbedder()

    # We want to mock duckdb.connect -> con -> cursor -> fetchmany
    # But doing this deeply is hard. Easier to mock the cursor execute result
    # However, `build_vectors` calls `con.execute(query)` then `cursor.fetchmany`.

    with patch("coreason_codex.build.duckdb.connect") as mock_connect:
        mock_con = mock_connect.return_value
        mock_cursor = mock_con.execute.return_value

        # Setup fetchmany to return data first time, then raise error
        # Data format: list of tuples
        batch1 = [(1, "Test", "D", "V", "C", "S", "1")]

        class StatefulSideEffect:
            def __init__(self) -> None:
                self.counter = 0

            def __call__(self, size: int) -> List[Any]:
                if self.counter == 0:
                    self.counter += 1
                    return batch1
                raise IOError("Cursor failure")

        mock_cursor.fetchmany.side_effect = StatefulSideEffect()

        # We also need to mock lancedb because we mocked duckdb (so vocab check might fail if we don't handle it)
        # Wait, `build_vectors` connects to `vocab.duckdb`. If we mock duckdb.connect, it won't read the file.
        # That's fine, we are providing the data via mock.

        # We need to make sure `db_path.exists()` passes. The fixture builds vocab, so it exists.

        with patch("coreason_codex.build.lancedb.connect") as mock_lance:
            # We need to consume the generator for it to run.
            # lancedb.create_table does that.
            mock_db = mock_lance.return_value

            def consume(name: str, data: Any, mode: str) -> None:
                for _ in data:
                    pass

            mock_db.create_table.side_effect = consume

            with pytest.raises(RuntimeError) as excinfo:
                builder.build_vectors(embedder, batch_size=1)

            assert "Cursor failure" in str(excinfo.value)
