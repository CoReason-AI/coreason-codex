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
from unittest.mock import MagicMock

import duckdb
import pytest

from coreason_codex.crosswalker import CodexCrossWalker
from coreason_codex.loader import CodexLoader


@pytest.fixture
def crosswalker_engine(synthetic_codex_pack: Any) -> CodexCrossWalker:
    loader = CodexLoader(synthetic_codex_pack)
    con, _ = loader.load_codex()
    return CodexCrossWalker(con)


def test_crosswalker_init_success(crosswalker_engine: CodexCrossWalker) -> None:
    assert crosswalker_engine is not None


def test_crosswalker_init_failure(tmp_path: Any) -> None:
    # Create an empty DB without the table
    db_path = tmp_path / "empty.duckdb"
    con = duckdb.connect(str(db_path))

    with pytest.raises(ValueError, match="concept_relationship' is missing"):
        CodexCrossWalker(con)


def test_translate_icd_to_snomed(crosswalker_engine: CodexCrossWalker) -> None:
    # 999999 (ICD10 I21.9) -> "Maps to" -> 312327 (SNOMED Acute MI)
    results = crosswalker_engine.translate_code(999999, relationship_id="Maps to")

    assert len(results) == 1
    target = results[0]
    assert target.concept_id == 312327
    assert target.vocabulary_id == "SNOMED"
    assert target.standard_concept == "S"


def test_translate_with_vocab_filter(crosswalker_engine: CodexCrossWalker) -> None:
    # 999999 -> 312327 (SNOMED)

    # 1. Correct Filter
    results = crosswalker_engine.translate_code(999999, target_vocabulary_id="SNOMED")
    assert len(results) == 1

    # 2. Incorrect Filter
    results_empty = crosswalker_engine.translate_code(999999, target_vocabulary_id="RxNorm")
    assert len(results_empty) == 0


def test_translate_no_mapping(crosswalker_engine: CodexCrossWalker) -> None:
    # 312327 (Standard Concept) usually maps to itself in 'Maps to' only if explicit.
    # In our sample data, 312327 has NO outgoing 'Maps to' relationship defined.
    results = crosswalker_engine.translate_code(312327)
    assert len(results) == 0


def test_translate_invalid_id(crosswalker_engine: CodexCrossWalker) -> None:
    results = crosswalker_engine.translate_code(-1)
    assert len(results) == 0


def test_check_relationship_success(crosswalker_engine: CodexCrossWalker) -> None:
    # 999999 -> 312327 (Maps to)
    assert crosswalker_engine.check_relationship(999999, 312327, "Maps to") is True


def test_check_relationship_failure(crosswalker_engine: CodexCrossWalker) -> None:
    # Wrong target
    assert crosswalker_engine.check_relationship(999999, 201820, "Maps to") is False
    # Wrong relationship
    assert crosswalker_engine.check_relationship(999999, 312327, "Is a") is False
    # Reverse (relationships are directed)
    assert crosswalker_engine.check_relationship(312327, 999999, "Maps to") is False


def test_check_relationship_invalid(tmp_path: Any) -> None:
    """Test with a relationship that has an invalid_reason set."""
    # Create a fresh DB
    db_path = tmp_path / "test_invalid.duckdb"
    con = duckdb.connect(str(db_path))

    # Create schema (minimal for check_relationship)
    con.execute(
        "CREATE TABLE concept_relationship (concept_id_1 INTEGER, concept_id_2 INTEGER, relationship_id VARCHAR, invalid_reason VARCHAR)"
    )

    # Insert invalid row (invalid_reason = 'D')
    con.execute("INSERT INTO concept_relationship VALUES (1, 2, 'Bad', 'D')")

    # Initialize CrossWalker
    cw = CodexCrossWalker(con)

    # It should return False because invalid_reason is 'D'
    assert cw.check_relationship(1, 2, "Bad") is False

    con.close()


def test_check_relationship_exception(crosswalker_engine: CodexCrossWalker) -> None:
    """Test exception handling in check_relationship."""
    # Replace connection with mock
    mock_conn = MagicMock()
    mock_conn.execute.side_effect = Exception("DB Error")
    crosswalker_engine.duckdb_conn = mock_conn

    res = crosswalker_engine.check_relationship(1, 2, "Rel")
    assert res is False
