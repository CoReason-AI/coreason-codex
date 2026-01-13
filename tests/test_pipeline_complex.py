# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_codex

from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

import coreason_codex.pipeline as pipeline
from coreason_codex.pipeline import get_context, initialize


@pytest.fixture(autouse=True)
def reset_context() -> Generator[None, None, None]:
    """Reset the global context before and after each test."""
    pipeline._CONTEXT = None
    yield
    pipeline._CONTEXT = None


@patch("coreason_codex.pipeline.CodexLoader")
@patch("coreason_codex.pipeline.SapBertEmbedder")
@patch("coreason_codex.pipeline.CodexNormalizer")
@patch("coreason_codex.pipeline.CodexHierarchy")
@patch("coreason_codex.pipeline.CodexCrossWalker")
def test_reinitialization_failure_preserves_context(
    mock_cw: MagicMock,
    mock_hier: MagicMock,
    mock_norm: MagicMock,
    mock_emb: MagicMock,
    mock_loader: MagicMock,
) -> None:
    """Test that a failed re-initialization does not destroy the existing context."""
    # 1. Setup successful first init
    loader_instance = mock_loader.return_value
    lancedb_con = MagicMock()
    loader_instance.load_codex.return_value = (MagicMock(), lancedb_con)
    lancedb_con.open_table.return_value = MagicMock()

    initialize("/path/success")
    ctx_initial = get_context()
    assert ctx_initial is not None

    # 2. Setup failing second init
    # Simulate loader failure on second call
    # CodexLoader("/path/fail") -> raises Error
    # mock_loader is the class. side_effect on the class constructor?
    # We need side_effect to depend on input args.

    def side_effect(path: str) -> MagicMock:
        if path == "/path/fail":
            raise ValueError("Corrupt pack")
        return loader_instance  # type: ignore[no-any-return]

    mock_loader.side_effect = side_effect

    # 3. Trigger failure
    with pytest.raises(RuntimeError, match="Codex initialization failed: Corrupt pack"):
        initialize("/path/fail")

    # 4. Assert context is preserved
    ctx_after = get_context()
    assert ctx_after is ctx_initial
    assert ctx_after.normalizer is ctx_initial.normalizer


@patch("coreason_codex.pipeline.CodexLoader")
@patch("coreason_codex.pipeline.SapBertEmbedder")
@patch("coreason_codex.pipeline.CodexNormalizer")
@patch("coreason_codex.pipeline.CodexHierarchy")
@patch("coreason_codex.pipeline.CodexCrossWalker")
def test_component_initialization_failure(
    mock_cw: MagicMock,
    mock_hier: MagicMock,
    mock_norm: MagicMock,
    mock_emb: MagicMock,
    mock_loader: MagicMock,
) -> None:
    """Test that failure in a downstream component aborts initialization."""
    # Setup successful loader
    loader_instance = mock_loader.return_value
    lancedb_con = MagicMock()
    loader_instance.load_codex.return_value = (MagicMock(), lancedb_con)
    lancedb_con.open_table.return_value = MagicMock()

    # Simulate Hierarchy initialization failure
    mock_hier.side_effect = RuntimeError("DuckDB connection lost")

    # Trigger init
    with pytest.raises(RuntimeError, match="Codex initialization failed: DuckDB connection lost"):
        initialize("/path/valid")

    # Assert context is still None (was not set)
    with pytest.raises(RuntimeError, match="Codex not initialized"):
        get_context()
