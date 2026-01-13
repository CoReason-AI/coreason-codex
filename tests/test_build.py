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
from typing import Any

import duckdb
import pytest

from coreason_codex.build import CodexBuilder


@pytest.fixture
def mock_athena_data(tmp_path: Path) -> Path:
    """Creates dummy Athena CSV files in a temporary directory."""
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

    # Create CONCEPT_RELATIONSHIP.csv
    with open(source_dir / "CONCEPT_RELATIONSHIP.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["concept_id_1", "concept_id_2", "relationship_id", "valid_start_date", "valid_end_date", "invalid_reason"]
        )
        writer.writerow([1, 2, "Maps to", "19700101", "20991231", ""])

    # Create CONCEPT_ANCESTOR.csv
    with open(source_dir / "CONCEPT_ANCESTOR.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["ancestor_concept_id", "descendant_concept_id", "min_levels_of_separation", "max_levels_of_separation"]
        )
        writer.writerow([1, 1, 0, 0])
        writer.writerow([1, 2, 1, 1])

    return source_dir


def test_builder_initialization(tmp_path: Path) -> None:
    """Test that CodexBuilder initializes correctly."""
    builder = CodexBuilder(source_dir=tmp_path, output_dir=tmp_path)
    assert builder.source_dir == tmp_path
    assert builder.output_dir == tmp_path


def test_build_vocab_success(mock_athena_data: Path, tmp_path: Path) -> None:
    """Test successful build of DuckDB artifact."""
    output_dir = tmp_path / "output"
    builder = CodexBuilder(source_dir=mock_athena_data, output_dir=output_dir)

    db_path = builder.build_vocab()

    assert db_path.exists()
    assert db_path.name == "vocab.duckdb"

    # Verify contents
    con = duckdb.connect(str(db_path), read_only=True)

    # Check CONCEPT table
    concepts = con.execute("SELECT * FROM CONCEPT ORDER BY concept_id").fetchall()
    assert len(concepts) == 2
    assert concepts[0][0] == 1
    assert concepts[1][0] == 2

    # Check CONCEPT_RELATIONSHIP table
    rels = con.execute("SELECT * FROM CONCEPT_RELATIONSHIP").fetchall()
    assert len(rels) == 1
    assert rels[0][0] == 1
    assert rels[0][1] == 2

    # Check CONCEPT_ANCESTOR table
    ancestors = con.execute(
        "SELECT * FROM CONCEPT_ANCESTOR ORDER BY ancestor_concept_id, descendant_concept_id"
    ).fetchall()
    assert len(ancestors) == 2

    con.close()


def test_build_missing_source_dir(tmp_path: Path) -> None:
    """Test error when source directory does not exist."""
    builder = CodexBuilder(source_dir=tmp_path / "nonexistent", output_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="Source directory not found"):
        builder.build_vocab()


def test_build_missing_required_file(tmp_path: Path) -> None:
    """Test error when a required CSV file is missing."""
    # Create empty dir
    source_dir = tmp_path / "partial_source"
    source_dir.mkdir()
    # Create only one file
    (source_dir / "CONCEPT.csv").touch()

    builder = CodexBuilder(source_dir=source_dir, output_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="Required file not found"):
        builder.build_vocab()


def test_build_overwrites_existing(mock_athena_data: Path, tmp_path: Path) -> None:
    """Test that the builder overwrites an existing artifact."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Create a dummy file where the DB should be
    existing_db = output_dir / "vocab.duckdb"
    existing_db.write_text("dummy content")

    builder = CodexBuilder(source_dir=mock_athena_data, output_dir=output_dir)
    db_path = builder.build_vocab()

    # Verify it's a valid DB now, not dummy text
    con = duckdb.connect(str(db_path), read_only=True)
    result = con.execute("SELECT count(*) FROM CONCEPT").fetchone()
    assert result is not None
    assert result[0] == 2
    con.close()


def test_build_failure_cleanup(mock_athena_data: Path, tmp_path: Path) -> None:
    """Test that partial artifacts are cleaned up on failure."""
    output_dir = tmp_path / "output"
    builder = CodexBuilder(source_dir=mock_athena_data, output_dir=output_dir)

    # Mock _load_table to fail
    # We need to monkeypatch the instance method or use mock
    # Since we are testing the class logic, let's subclass or patch

    with pytest.raises(RuntimeError, match="Build failed"):
        # We inject a failure by providing a corrupted CSV effectively by messing with the file
        # *after* check but before load?
        # Easier to mock duckdb.connect or internal method.
        # Let's mock _load_table on the instance.

        # We can't easily mock instance method on an already created instance without some tricks,
        # but we can patch the class method.
        original_load = CodexBuilder._load_table

        def failing_load(self: Any, con: duckdb.DuckDBPyConnection, table_name: str) -> None:
            if table_name == "CONCEPT_RELATIONSHIP":
                raise ValueError("Simulated Load Error")
            original_load(self, con, table_name)

        # Temporarily replace
        CodexBuilder._load_table = failing_load  # type: ignore
        try:
            builder.build_vocab()
        finally:
            CodexBuilder._load_table = original_load  # type: ignore

    # Verify cleanup
    assert not (output_dir / "vocab.duckdb").exists()
