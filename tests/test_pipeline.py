# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_codex

from unittest.mock import MagicMock, patch

import pytest

import coreason_codex.pipeline as pipeline
from coreason_codex.pipeline import get_context, initialize


from typing import Generator


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
def test_initialize_success(
    mock_crosswalker: MagicMock,
    mock_hierarchy: MagicMock,
    mock_normalizer: MagicMock,
    mock_embedder: MagicMock,
    mock_loader: MagicMock,
) -> None:
    """Test successful initialization of the Codex system."""
    # Setup mocks
    loader_instance = mock_loader.return_value
    duckdb_con = MagicMock()
    lancedb_con = MagicMock()
    loader_instance.load_codex.return_value = (duckdb_con, lancedb_con)

    # Setup LanceDB table mock
    vector_table = MagicMock()
    lancedb_con.open_table.return_value = vector_table

    # Call initialize
    pack_path = "/tmp/codex_pack"
    initialize(pack_path)

    # Verify CodexLoader usage
    mock_loader.assert_called_once_with(pack_path)
    loader_instance.load_codex.assert_called_once()

    # Verify SapBertEmbedder usage
    mock_embedder.assert_called_once()

    # Verify Engines initialization
    lancedb_con.open_table.assert_called_once_with("vectors")
    mock_normalizer.assert_called_once_with(table=vector_table, embedder=mock_embedder.return_value)
    mock_hierarchy.assert_called_once_with(con=duckdb_con)
    mock_crosswalker.assert_called_once_with(con=duckdb_con)

    # Verify Context is set
    context = get_context()
    assert context.normalizer == mock_normalizer.return_value
    assert context.hierarchy == mock_hierarchy.return_value
    assert context.crosswalker == mock_crosswalker.return_value


@patch("coreason_codex.pipeline.CodexLoader")
def test_initialize_failure_loader(mock_loader: MagicMock) -> None:
    """Test initialization failure when loader fails."""
    mock_loader.side_effect = Exception("Loader error")

    with pytest.raises(RuntimeError, match="Codex initialization failed: Loader error"):
        initialize("/tmp/path")

    # Verify context is None
    with pytest.raises(RuntimeError, match="Codex not initialized"):
        get_context()


@patch("coreason_codex.pipeline.CodexLoader")
@patch("coreason_codex.pipeline.SapBertEmbedder")
def test_initialize_failure_table_missing(mock_embedder: MagicMock, mock_loader: MagicMock) -> None:
    """Test initialization failure when vector table is missing."""
    loader_instance = mock_loader.return_value
    lancedb_con = MagicMock()
    loader_instance.load_codex.return_value = (MagicMock(), lancedb_con)

    # Make open_table fail
    lancedb_con.open_table.side_effect = Exception("Table not found")

    with pytest.raises(RuntimeError, match="Codex initialization failed: Vector table 'vectors' not found"):
        initialize("/tmp/path")


def test_get_context_uninitialized() -> None:
    """Test get_context raises error when not initialized."""
    with pytest.raises(RuntimeError, match="Codex not initialized"):
        get_context()


@patch("coreason_codex.pipeline.CodexLoader")
@patch("coreason_codex.pipeline.SapBertEmbedder")
@patch("coreason_codex.pipeline.CodexNormalizer")
@patch("coreason_codex.pipeline.CodexHierarchy")
@patch("coreason_codex.pipeline.CodexCrossWalker")
def test_reinitialization(
    mock_cw: MagicMock,
    mock_hier: MagicMock,
    mock_norm: MagicMock,
    mock_emb: MagicMock,
    mock_loader: MagicMock,
) -> None:
    """Test that calling initialize twice updates the context."""
    # Setup mocks
    loader_instance = mock_loader.return_value
    loader_instance.load_codex.return_value = (MagicMock(), MagicMock())
    lancedb_con = loader_instance.load_codex.return_value[1]
    lancedb_con.open_table.return_value = MagicMock()

    # First initialization
    initialize("/path/1")
    ctx1 = get_context()

    # Second initialization
    initialize("/path/2")
    ctx2 = get_context()

    assert ctx1 is not ctx2
    assert mock_loader.call_count == 2
