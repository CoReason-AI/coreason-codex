# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_codex

from unittest.mock import MagicMock

from coreason_codex.crosswalker import CodexCrossWalker
from coreason_codex.hierarchy import CodexHierarchy


def test_hierarchy_query_error() -> None:
    mock_con = MagicMock()
    mock_con.execute.side_effect = [
        MagicMock(),  # Init check
        Exception("Query Fail"),  # get_descendants
    ]

    h = CodexHierarchy(mock_con)
    res = h.get_descendants(1)
    assert res == []


def test_crosswalker_query_error() -> None:
    mock_con = MagicMock()
    mock_con.execute.side_effect = [
        MagicMock(),  # Init check
        Exception("Query Fail"),  # translate
    ]

    cw = CodexCrossWalker(mock_con)
    res = cw.translate_code(1)
    assert res == []
