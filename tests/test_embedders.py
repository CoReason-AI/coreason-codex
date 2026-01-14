# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_codex

from typing import Any, Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from coreason_codex.embedders import SapBertEmbedder


@pytest.fixture
def mock_sentence_transformer() -> Generator[MagicMock, None, None]:
    with patch("coreason_codex.embedders.SentenceTransformer") as MockClass:
        mock_instance = MockClass.return_value

        # Configure encode mock to return valid shapes
        def side_effect(sentences: Any, **kwargs: Any) -> np.ndarray:
            # Check if input is list or string
            if isinstance(sentences, list):
                # Return batch: (N, 768)
                return np.random.rand(len(sentences), 768).astype(np.float32)
            else:
                # Return single: (768,)
                return np.random.rand(768).astype(np.float32)

        mock_instance.encode.side_effect = side_effect
        yield MockClass


def test_sapbert_init(mock_sentence_transformer: Any) -> None:
    embedder = SapBertEmbedder()
    mock_sentence_transformer.assert_called_once()
    assert embedder.model is not None


def test_sapbert_embed_single(mock_sentence_transformer: Any) -> None:
    embedder = SapBertEmbedder()
    text = "Heart Attack"
    vector = embedder.embed(text)

    assert isinstance(vector, list)
    assert len(vector) == 768
    assert isinstance(vector[0], float)

    # Verify mock call
    # Use explicit cast or ignore if mypy complains about assert_called_with on Any/MagicMock
    embedder.model.encode.assert_called_with(text, convert_to_numpy=True)  # type: ignore


def test_sapbert_embed_batch(mock_sentence_transformer: Any) -> None:
    embedder = SapBertEmbedder()
    texts = ["Heart Attack", "Diabetes"]
    vectors = embedder.embed_batch(texts)

    assert isinstance(vectors, list)
    assert len(vectors) == 2
    assert len(vectors[0]) == 768

    # Verify mock call
    embedder.model.encode.assert_called_with(texts, convert_to_numpy=True, show_progress_bar=False)  # type: ignore


def test_sapbert_init_failure() -> None:
    with patch("coreason_codex.embedders.SentenceTransformer", side_effect=Exception("Download failed")):
        with pytest.raises(RuntimeError, match="Could not load embedding model"):
            SapBertEmbedder()
