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

import duckdb
import pytest

from coreason_codex.build import CodexBuilder


@pytest.fixture
def complex_athena_data(tmp_path: Path) -> Path:
    """Creates complex Athena CSV files in a temporary directory."""
    source_dir = tmp_path / "complex_source"
    source_dir.mkdir()

    # Create empty placeholders for required files not being tested primarily
    # but needed for verification check
    for name in ["CONCEPT_RELATIONSHIP.csv", "CONCEPT_ANCESTOR.csv"]:
        with open(source_dir / name, "w", newline="") as f:
            writer = csv.writer(f)
            if name == "CONCEPT_RELATIONSHIP.csv":
                writer.writerow(
                    [
                        "concept_id_1",
                        "concept_id_2",
                        "relationship_id",
                        "valid_start_date",
                        "valid_end_date",
                        "invalid_reason",
                    ]
                )
            else:
                writer.writerow(
                    [
                        "ancestor_concept_id",
                        "descendant_concept_id",
                        "min_levels_of_separation",
                        "max_levels_of_separation",
                    ]
                )

    return source_dir


def test_build_large_ids(complex_athena_data: Path, tmp_path: Path) -> None:
    """Test that the builder handles large integers (BIGINT) correctly."""
    # Write CONCEPT.csv with large IDs
    large_id = 5_000_000_000  # Larger than 2^32 (approx 4.29e9)
    with open(complex_athena_data / "CONCEPT.csv", "w", newline="") as f:
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
            [
                large_id,
                "Large Concept",
                "Condition",
                "SNOMED",
                "Clinical Finding",
                "S",
                "1001",
                "19700101",
                "20991231",
                "",
            ]
        )

    builder = CodexBuilder(source_dir=complex_athena_data, output_dir=tmp_path)
    db_path = builder.build_vocab()

    con = duckdb.connect(str(db_path), read_only=True)
    # Verify type is BIGINT or HUGEINT, not INTEGER
    # And verify value is correct
    result = con.execute("SELECT concept_id FROM CONCEPT").fetchone()
    assert result[0] == large_id
    con.close()


def test_build_complex_text(complex_athena_data: Path, tmp_path: Path) -> None:
    """Test handling of special characters in CSV."""
    complex_name = 'Concept with "Quotes", \nNewlines, and Emoji 💊'
    with open(complex_athena_data / "CONCEPT.csv", "w", newline="", encoding="utf-8") as f:
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
        writer.writerow([100, complex_name, "Drug", "RxNorm", "Brand Name", "S", "C100", "19700101", "20991231", ""])

    builder = CodexBuilder(source_dir=complex_athena_data, output_dir=tmp_path)
    db_path = builder.build_vocab()

    con = duckdb.connect(str(db_path), read_only=True)
    result = con.execute("SELECT concept_name FROM CONCEPT WHERE concept_id=100").fetchone()
    assert result[0] == complex_name
    con.close()


def test_build_missing_columns_failure(complex_athena_data: Path, tmp_path: Path) -> None:
    """Test that build fails if required columns for indexing are missing."""
    # CONCEPT.csv missing 'concept_id'
    with open(complex_athena_data / "CONCEPT.csv", "w", newline="") as f:
        writer = csv.writer(f)
        # Note: missing concept_id
        writer.writerow(["concept_name", "domain_id", "vocabulary_id"])
        writer.writerow(["Test", "Condition", "SNOMED"])

    builder = CodexBuilder(source_dir=complex_athena_data, output_dir=tmp_path)

    # It should fail when trying to CREATE INDEX ON CONCEPT(concept_id)
    # or earlier if we add schema validation.
    with pytest.raises(RuntimeError, match="Build failed"):
        builder.build_vocab()


def test_build_empty_tables(complex_athena_data: Path, tmp_path: Path) -> None:
    """Test building with headers only (no rows)."""
    with open(complex_athena_data / "CONCEPT.csv", "w", newline="") as f:
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
        # No rows

    builder = CodexBuilder(source_dir=complex_athena_data, output_dir=tmp_path)
    db_path = builder.build_vocab()

    con = duckdb.connect(str(db_path), read_only=True)
    count = con.execute("SELECT count(*) FROM CONCEPT").fetchone()[0]
    assert count == 0
    con.close()
