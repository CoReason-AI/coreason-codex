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

import numpy as np
import pytest

from coreason_codex.embedders import SapBertEmbedder


@patch("coreason_codex.embedders.SentenceTransformer")
def test_sapbert_initialization(mock_st: MagicMock) -> None:
    """Test that the model is initialized with the correct default name."""
    embedder = SapBertEmbedder()
    mock_st.assert_called_once_with("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    assert embedder.model == mock_st.return_value


@patch("coreason_codex.embedders.SentenceTransformer")
def test_sapbert_initialization_custom(mock_st: MagicMock) -> None:
    """Test initialization with a custom model name."""
    _ = SapBertEmbedder(model_name="custom/model")
    mock_st.assert_called_once_with("custom/model")


@patch("coreason_codex.embedders.SentenceTransformer")
def test_sapbert_initialization_failure(mock_st: MagicMock) -> None:
    """Test that initialization raises error if model load fails."""
    mock_st.side_effect = RuntimeError("Model not found")

    with pytest.raises(RuntimeError, match="Model not found"):
        SapBertEmbedder()


@patch("coreason_codex.embedders.SentenceTransformer")
def test_embed_success(mock_st: MagicMock) -> None:
    """Test the embed method returns a numpy array."""
    # Setup mock return value
    mock_model = mock_st.return_value
    expected_array = np.array([[0.1, 0.2], [0.3, 0.4]])
    mock_model.encode.return_value = expected_array

    embedder = SapBertEmbedder()
    texts = ["one", "two"]
    result = embedder.embed(texts)

    mock_model.encode.assert_called_once_with(texts)
    assert np.array_equal(result, expected_array)
    assert isinstance(result, np.ndarray)


@patch("coreason_codex.embedders.SentenceTransformer")
def test_embed_converts_to_numpy(mock_st: MagicMock) -> None:
    """Test that if encode returns a list, it is converted to numpy array."""
    mock_model = mock_st.return_value
    # Return list of lists
    mock_model.encode.return_value = [[0.1, 0.2], [0.3, 0.4]]

    embedder = SapBertEmbedder()
    result = embedder.embed(["test"])

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)


@patch("coreason_codex.embedders.SentenceTransformer")
def test_embed_empty_input(mock_st: MagicMock) -> None:
    """Test embedding an empty list of texts."""
    mock_model = mock_st.return_value
    # SentenceTransformer usually returns empty array or list for empty input
    mock_model.encode.return_value = np.array([])

    embedder = SapBertEmbedder()
    result = embedder.embed([])

    mock_model.encode.assert_called_once_with([])
    assert isinstance(result, np.ndarray)
    assert result.size == 0


@patch("coreason_codex.embedders.SentenceTransformer")
def test_embed_whitespace_strings(mock_st: MagicMock) -> None:
    """Test embedding strings that are empty or whitespace."""
    mock_model = mock_st.return_value
    # Mock return for 2 items
    mock_model.encode.return_value = np.array([[0.1, 0.1], [0.2, 0.2]])

    embedder = SapBertEmbedder()
    texts = ["", "   "]
    result = embedder.embed(texts)

    mock_model.encode.assert_called_once_with(texts)
    assert result.shape == (2, 2)
