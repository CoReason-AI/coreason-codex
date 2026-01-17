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
from coreason_codex.normalizer import CodexNormalizer


@pytest.fixture
def mock_components() -> Tuple[duckdb.DuckDBPyConnection, Any, Embedder]:
    """
    Sets up an in-memory DuckDB with specific edge case data
    and a mock LanceDB/Embedder.
    """
    # 1. Setup in-memory DuckDB
    con = duckdb.connect(":memory:")

    # Schema
    con.execute("""
        CREATE TABLE concept (
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
        CREATE TABLE concept_relationship (
            concept_id_1 INTEGER,
            concept_id_2 INTEGER,
            relationship_id VARCHAR,
            valid_start_date DATE,
            valid_end_date DATE,
            invalid_reason VARCHAR
        )
    """)

    # Data
    # Concept 100: Non-standard, No mapping
    # Concept 200: Non-standard, Invalid mapping
    # Concept 300: Non-standard, Multiple mappings
    # Concept 1000: Standard target A
    # Concept 1001: Standard target B

    concepts = [
        (100, "No Mapping Code", "Condition", "Test", "Test Class", None, "C100", None),
        (200, "Invalid Mapping Code", "Condition", "Test", "Test Class", None, "C200", None),
        (300, "Multi Mapping Code", "Condition", "Test", "Test Class", None, "C300", None),
        (1000, "Standard A", "Condition", "Test", "Test Class", "S", "S1000", None),
        (1001, "Standard B", "Condition", "Test", "Test Class", "S", "S1001", None),
    ]
    con.executemany("INSERT INTO concept VALUES (?, ?, ?, ?, ?, ?, ?, ?)", concepts)

    relationships = [
        # 200 -> 1000 (Invalid)
        (200, 1000, "Maps to", "2020-01-01", "2022-01-01", "D"),
        # 300 -> 1000 (Valid)
        (300, 1000, "Maps to", "2020-01-01", "2099-12-31", None),
        # 300 -> 1001 (Valid)
        (300, 1001, "Maps to", "2020-01-01", "2099-12-31", None),
    ]
    con.executemany("INSERT INTO concept_relationship VALUES (?, ?, ?, CAST(? AS DATE), CAST(? AS DATE), ?)", relationships)

    # 2. Mock LanceDB
    mock_table = MagicMock()
    mock_lancedb = MagicMock()
    mock_lancedb.open_table.return_value = mock_table

    # 3. Mock Embedder
    mock_embedder = MagicMock(spec=Embedder)
    mock_embedder.embed.return_value = [0.1] * 128

    return con, mock_lancedb, mock_embedder


def setup_normalizer_with_result(components, concept_id: int):
    con, lancedb, embedder = components

    # Configure LanceDB to return the specific concept
    mock_query = MagicMock()
    mock_limit = MagicMock()

    # Mock search result
    mock_result = [{"concept_id": concept_id, "_distance": 0.0}]

    lancedb.open_table.return_value.search.return_value = mock_query
    mock_query.limit.return_value = mock_limit
    mock_limit.to_list.return_value = mock_result

    norm = CodexNormalizer(embedder, con, lancedb)
    # Inject the mock table since __init__ calls open_table
    norm.table = lancedb.open_table.return_value

    return norm


def test_mapping_no_relationship(mock_components):
    """
    Test a non-standard concept that has NO entry in concept_relationship table.
    """
    norm = setup_normalizer_with_result(mock_components, 100)

    matches = norm.normalize("test")

    assert len(matches) == 1
    m = matches[0]
    assert m.match_concept.concept_id == 100
    assert m.is_standard is False
    # Should be None because no relationship exists
    assert m.mapped_standard_id is None


def test_mapping_invalid_relationship(mock_components):
    """
    Test a non-standard concept that has a 'Maps to' entry, but it is invalid (invalid_reason='D').
    """
    norm = setup_normalizer_with_result(mock_components, 200)

    matches = norm.normalize("test")

    assert len(matches) == 1
    m = matches[0]
    assert m.match_concept.concept_id == 200
    assert m.is_standard is False
    # Should be None because relationship is invalid
    assert m.mapped_standard_id is None


def test_mapping_multiple_relationships(mock_components):
    """
    Test a non-standard concept that maps to MULTIPLE standard concepts.
    The normalizer should pick one (via LIMIT 1) and not crash.
    """
    norm = setup_normalizer_with_result(mock_components, 300)

    matches = norm.normalize("test")

    assert len(matches) == 1
    m = matches[0]
    assert m.match_concept.concept_id == 300
    assert m.is_standard is False
    # Should be one of the valid targets (1000 or 1001)
    assert m.mapped_standard_id in [1000, 1001]
