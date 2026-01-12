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

from coreason_codex.hierarchy import CodexHierarchy


@pytest.fixture
def duckdb_con() -> duckdb.DuckDBPyConnection:
    """Creates a temporary in-memory DuckDB connection with test data."""
    con = duckdb.connect(":memory:")

    # Create CONCEPT_ANCESTOR table
    con.execute("""
        CREATE TABLE CONCEPT_ANCESTOR (
            ancestor_concept_id INTEGER,
            descendant_concept_id INTEGER,
            min_levels_of_separation INTEGER,
            max_levels_of_separation INTEGER
        )
    """)

    # Insert test data
    # 1 is ancestor of 2 and 3
    # 2 is ancestor of 4
    # 5 has no descendants
    con.execute("INSERT INTO CONCEPT_ANCESTOR VALUES (1, 1, 0, 0)")  # Self
    con.execute("INSERT INTO CONCEPT_ANCESTOR VALUES (1, 2, 1, 1)")
    con.execute("INSERT INTO CONCEPT_ANCESTOR VALUES (1, 3, 1, 1)")
    con.execute("INSERT INTO CONCEPT_ANCESTOR VALUES (1, 4, 2, 2)")
    con.execute("INSERT INTO CONCEPT_ANCESTOR VALUES (2, 2, 0, 0)")  # Self
    con.execute("INSERT INTO CONCEPT_ANCESTOR VALUES (2, 4, 1, 1)")
    con.execute("INSERT INTO CONCEPT_ANCESTOR VALUES (3, 3, 0, 0)")  # Self
    con.execute("INSERT INTO CONCEPT_ANCESTOR VALUES (5, 5, 0, 0)")  # Self

    return con


def test_get_descendants_found(duckdb_con: duckdb.DuckDBPyConnection) -> None:
    """Test finding descendants for a concept with children."""
    hierarchy = CodexHierarchy(duckdb_con)
    descendants = hierarchy.get_descendants(1)

    # Should include 1 (self), 2, 3, 4
    assert len(descendants) == 4
    assert set(descendants) == {1, 2, 3, 4}


def test_get_descendants_leaf(duckdb_con: duckdb.DuckDBPyConnection) -> None:
    """Test finding descendants for a leaf node (only self)."""
    hierarchy = CodexHierarchy(duckdb_con)
    descendants = hierarchy.get_descendants(3)

    assert len(descendants) == 1
    assert descendants == [3]


def test_get_descendants_not_found(duckdb_con: duckdb.DuckDBPyConnection) -> None:
    """Test finding descendants for a non-existent concept."""
    hierarchy = CodexHierarchy(duckdb_con)
    descendants = hierarchy.get_descendants(999)

    assert descendants == []


def test_get_descendants_error(duckdb_con: duckdb.DuckDBPyConnection) -> None:
    """Test handling of DB errors."""
    # Close connection to trigger error
    duckdb_con.close()

    hierarchy = CodexHierarchy(duckdb_con)
    descendants = hierarchy.get_descendants(1)

    # Should return empty list and log error (checked via logs if needed, but here just return value)
    assert descendants == []
