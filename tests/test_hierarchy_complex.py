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
def complex_hierarchy_con() -> duckdb.DuckDBPyConnection:
    """
    Creates a DuckDB connection with a complex hierarchy scenario.
    """
    con = duckdb.connect(":memory:")

    con.execute("""
        CREATE TABLE CONCEPT_ANCESTOR (
            ancestor_concept_id INTEGER,
            descendant_concept_id INTEGER,
            min_levels_of_separation INTEGER,
            max_levels_of_separation INTEGER
        )
    """)

    # 1. Diamond Hierarchy
    # A(10) -> B(11) -> D(13)
    # A(10) -> C(12) -> D(13)
    #
    # In CONCEPT_ANCESTOR (pre-computed), this looks like:
    # (10, 10), (10, 11), (10, 12), (10, 13)
    # (11, 11), (11, 13)
    # (12, 12), (12, 13)
    # (13, 13)

    inserts = [
        (10, 10, 0, 0),
        (10, 11, 1, 1),
        (10, 12, 1, 1),
        (10, 13, 2, 2),  # path via B or C, distance might vary but existence doesn't
        (11, 11, 0, 0),
        (11, 13, 1, 1),
        (12, 12, 0, 0),
        (12, 13, 1, 1),
        (13, 13, 0, 0),
    ]

    con.executemany("INSERT INTO CONCEPT_ANCESTOR VALUES (?, ?, ?, ?)", inserts)
    return con


def test_diamond_hierarchy(complex_hierarchy_con: duckdb.DuckDBPyConnection) -> None:
    """
    Test that a diamond hierarchy correctly returns all unique descendants.
    Crucial to ensure no duplicates if the underlying table had issues (though here we control it).
    """
    hierarchy = CodexHierarchy(complex_hierarchy_con)
    descendants = hierarchy.get_descendants(10)

    # Expect: 10, 11, 12, 13
    assert len(descendants) == 4
    assert sorted(descendants) == [10, 11, 12, 13]


def test_invalid_ids(complex_hierarchy_con: duckdb.DuckDBPyConnection) -> None:
    """Test response to zero or negative IDs."""
    hierarchy = CodexHierarchy(complex_hierarchy_con)

    assert hierarchy.get_descendants(0) == []
    assert hierarchy.get_descendants(-1) == []


def test_large_descendant_set() -> None:
    """
    Test performance/handling of a larger set of descendants.
    """
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE CONCEPT_ANCESTOR (ancestor_concept_id INTEGER, descendant_concept_id INTEGER)")

    # Root 100 has 10,000 descendants
    # Generate data in bulk
    root_id = 100
    descendants = [(root_id, i) for i in range(1000, 11000)]

    con.executemany("INSERT INTO CONCEPT_ANCESTOR VALUES (?, ?)", descendants)

    hierarchy = CodexHierarchy(con)
    results = hierarchy.get_descendants(root_id)

    assert len(results) == 10000
    assert 1000 in results
    assert 10999 in results
