# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_codex

from typing import Any

import duckdb
import pytest

from coreason_codex.hierarchy import CodexHierarchy
from coreason_codex.loader import CodexLoader


@pytest.fixture
def hierarchy_engine(synthetic_codex_pack: Any) -> CodexHierarchy:
    loader = CodexLoader(synthetic_codex_pack)
    con, _ = loader.load_codex()
    return CodexHierarchy(con)


def test_hierarchy_init_success(synthetic_codex_pack: Any) -> None:
    loader = CodexLoader(synthetic_codex_pack)
    con, _ = loader.load_codex()
    h = CodexHierarchy(con)
    assert h is not None


def test_hierarchy_init_failure(tmp_path: Any) -> None:
    # Create an empty DB without the table
    db_path = tmp_path / "empty.duckdb"
    con = duckdb.connect(str(db_path))

    with pytest.raises(ValueError, match="concept_ancestor' is missing"):
        CodexHierarchy(con)


def test_get_descendants(hierarchy_engine: CodexHierarchy) -> None:
    # 441840 (Clinical Finding) is ancestor of 312327, 201820, 31967
    descendants = hierarchy_engine.get_descendants(441840)

    assert 312327 in descendants
    assert 201820 in descendants
    assert 31967 in descendants
    assert 441840 in descendants  # Self reference


def test_get_descendants_leaf(hierarchy_engine: CodexHierarchy) -> None:
    # 312327 (Acute MI) is a leaf in our sample graph (besides self)
    descendants = hierarchy_engine.get_descendants(312327)
    assert descendants == [312327]


def test_get_descendants_invalid_id(hierarchy_engine: CodexHierarchy) -> None:
    # ID not in DB
    descendants = hierarchy_engine.get_descendants(99999999)
    assert descendants == []
