# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_codex

import duckdb
import pytest

from coreason_codex.crosswalker import CodexCrossWalker
from coreason_codex.schemas import Concept


@pytest.fixture
def duckdb_con() -> duckdb.DuckDBPyConnection:
    """Creates a temporary in-memory DuckDB connection with test data."""
    con = duckdb.connect(":memory:")

    # Create tables
    con.execute("""
        CREATE TABLE CONCEPT (
            concept_id INTEGER,
            concept_name VARCHAR,
            domain_id VARCHAR,
            vocabulary_id VARCHAR,
            concept_class_id VARCHAR,
            standard_concept VARCHAR,
            concept_code VARCHAR,
            invalid_reason VARCHAR
        )
    """)

    con.execute("""
        CREATE TABLE CONCEPT_RELATIONSHIP (
            concept_id_1 INTEGER,
            concept_id_2 INTEGER,
            relationship_id VARCHAR,
            invalid_reason VARCHAR
        )
    """)

    # Insert test data
    # Source Concept: 1 (SNOMED 'A')
    # Target Concept: 2 (ICD10 'B')
    # Target Concept: 3 (ICD10 'C') - Invalid
    # Target Concept: 4 (OtherVocab 'D')

    # Concepts
    con.execute("INSERT INTO CONCEPT VALUES (1, 'A', 'Condition', 'SNOMED', 'Clinc', 'S', 'A01', NULL)")
    con.execute("INSERT INTO CONCEPT VALUES (2, 'B', 'Condition', 'ICD10CM', 'Code', NULL, 'B02', NULL)")
    con.execute("INSERT INTO CONCEPT VALUES (3, 'C', 'Condition', 'ICD10CM', 'Code', NULL, 'C03', 'D')")  # Invalid
    con.execute("INSERT INTO CONCEPT VALUES (4, 'D', 'Condition', 'Other', 'Code', NULL, 'D04', NULL)")

    # Relationships
    con.execute("INSERT INTO CONCEPT_RELATIONSHIP VALUES (1, 2, 'Maps to', NULL)")
    con.execute("INSERT INTO CONCEPT_RELATIONSHIP VALUES (1, 3, 'Maps to', NULL)")
    con.execute("INSERT INTO CONCEPT_RELATIONSHIP VALUES (1, 4, 'Maps to', NULL)")
    con.execute(
        "INSERT INTO CONCEPT_RELATIONSHIP VALUES (1, 2, 'Maps to', 'D')"
    )  # Invalid relationship duplicate (mocking scenario)

    return con


def test_translate_code_success(duckdb_con: duckdb.DuckDBPyConnection) -> None:
    """Test successful translation."""
    walker = CodexCrossWalker(duckdb_con)
    translations = walker.translate_code(1, "ICD10CM")

    assert len(translations) == 1
    c = translations[0]
    assert isinstance(c, Concept)
    assert c.concept_id == 2
    assert c.vocabulary_id == "ICD10CM"
    assert c.concept_code == "B02"


def test_translate_code_no_match(duckdb_con: duckdb.DuckDBPyConnection) -> None:
    """Test translation with no matches in target vocabulary."""
    walker = CodexCrossWalker(duckdb_con)
    translations = walker.translate_code(1, "RxNorm")
    assert translations == []


def test_translate_code_invalid_target(duckdb_con: duckdb.DuckDBPyConnection) -> None:
    """Test that invalid concepts are filtered out."""
    # We set up concept 3 as invalid in fixture.
    # It has a valid relationship from 1, but the concept itself is invalid.
    walker = CodexCrossWalker(duckdb_con)
    # If we query for ICD10CM, we should only get concept 2, not 3.
    translations = walker.translate_code(1, "ICD10CM")
    ids = [c.concept_id for c in translations]
    assert 2 in ids
    assert 3 not in ids


def test_translate_code_error(duckdb_con: duckdb.DuckDBPyConnection) -> None:
    """Test handling of DB errors."""
    duckdb_con.close()
    walker = CodexCrossWalker(duckdb_con)
    translations = walker.translate_code(1, "ICD10CM")
    assert translations == []
