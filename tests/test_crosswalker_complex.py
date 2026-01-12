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


@pytest.fixture
def complex_crosswalker_con() -> duckdb.DuckDBPyConnection:
    """
    Creates a DuckDB connection with complex mapping scenarios.
    """
    con = duckdb.connect(":memory:")

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

    # Scenario 1: Deprecated Source -> Valid Target
    # Concept 100 (Old) -> Maps to -> Concept 200 (New, Valid)
    con.execute("INSERT INTO CONCEPT VALUES (100, 'Old', 'Cond', 'VocabA', 'Cls', NULL, 'A100', 'D')")
    con.execute("INSERT INTO CONCEPT VALUES (200, 'New', 'Cond', 'VocabB', 'Cls', 'S', 'B200', NULL)")
    con.execute("INSERT INTO CONCEPT_RELATIONSHIP VALUES (100, 200, 'Maps to', NULL)")

    # Scenario 2: Valid Source -> Deprecated Target (Should be filtered out)
    # Concept 300 -> Maps to -> Concept 400 (Deprecated)
    con.execute("INSERT INTO CONCEPT VALUES (300, 'Src', 'Cond', 'VocabA', 'Cls', NULL, 'A300', NULL)")
    con.execute("INSERT INTO CONCEPT VALUES (400, 'TgtDeprecated', 'Cond', 'VocabB', 'Cls', NULL, 'B400', 'D')")
    con.execute("INSERT INTO CONCEPT_RELATIONSHIP VALUES (300, 400, 'Maps to', NULL)")

    # Scenario 3: Multiple Relationship Types
    # Concept 500 -> Maps to -> Concept 600 (Valid)
    # Concept 500 -> Is a -> Concept 700 (Valid)
    # The current implementation DOES NOT filter by relationship_id, only by target vocab and validity.
    # So if we query for VocabB, we get both if they are in VocabB.
    con.execute("INSERT INTO CONCEPT VALUES (500, 'Multi', 'Cond', 'VocabA', 'Cls', NULL, 'A500', NULL)")
    con.execute("INSERT INTO CONCEPT VALUES (600, 'RelMap', 'Cond', 'VocabB', 'Cls', 'S', 'B600', NULL)")
    con.execute("INSERT INTO CONCEPT VALUES (700, 'RelIsA', 'Cond', 'VocabB', 'Cls', 'S', 'B700', NULL)")
    con.execute("INSERT INTO CONCEPT_RELATIONSHIP VALUES (500, 600, 'Maps to', NULL)")
    con.execute("INSERT INTO CONCEPT_RELATIONSHIP VALUES (500, 700, 'Is a', NULL)")

    return con


def test_deprecated_source_to_valid_target(complex_crosswalker_con: duckdb.DuckDBPyConnection) -> None:
    """
    Verifies that we can translate FROM a deprecated code.
    This is critical for legacy data support.
    """
    walker = CodexCrossWalker(complex_crosswalker_con)
    results = walker.translate_code(100, "VocabB")

    assert len(results) == 1
    assert results[0].concept_id == 200


def test_target_must_be_valid(complex_crosswalker_con: duckdb.DuckDBPyConnection) -> None:
    """
    Verifies that if the target concept is deprecated, it is NOT returned.
    We only want to cross-walk to active standards.
    """
    walker = CodexCrossWalker(complex_crosswalker_con)
    results = walker.translate_code(300, "VocabB")

    # Target 400 is deprecated ('D')
    assert len(results) == 0


def test_sql_injection_resilience(complex_crosswalker_con: duckdb.DuckDBPyConnection) -> None:
    """
    Test resilience against SQL injection in vocabulary_id.
    """
    walker = CodexCrossWalker(complex_crosswalker_con)
    # Try to inject OR 1=1
    # If injection works, it might match 'VocabA' rows or throw syntax error
    results = walker.translate_code(100, "VocabB' OR '1'='1")

    # Should be empty because no vocabulary matches that literal string
    assert len(results) == 0


def test_broad_relationship_mapping(complex_crosswalker_con: duckdb.DuckDBPyConnection) -> None:
    """
    Verifies that currently ALL valid relationships to target vocab are returned.
    If 500 -> 600 (Maps to) and 500 -> 700 (Is a), both in VocabB, both returned.
    """
    walker = CodexCrossWalker(complex_crosswalker_con)
    results = walker.translate_code(500, "VocabB")

    ids = sorted([c.concept_id for c in results])
    assert ids == [600, 700]
