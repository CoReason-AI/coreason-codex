# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_codex

from typing import List

from sentence_transformers import SentenceTransformer

from coreason_codex.interfaces import Embedder
from coreason_codex.utils.logger import logger


class SapBertEmbedder(Embedder):
    """
    Embedder implementation using SapBERT model via SentenceTransformer.
    Reference: cambridgeltl/SapBERT-from-PubMedBERT-fulltext
    """

    def __init__(self, model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext", device: str = "cpu") -> None:
        """
        Initialize the SapBERT embedder.

        Args:
            model_name: The name of the HuggingFace model to load.
            device: 'cpu', 'cuda', etc.
        """
        logger.info(f"Loading embedding model: {model_name} on {device}")
        try:
            self.model = SentenceTransformer(model_name, device=device)
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise RuntimeError(f"Could not load embedding model: {e}") from e

    def embed(self, text: str) -> List[float]:
        """
        Embed a single string.
        """
        if not text:
            # Handle empty string specifically if model fails or return empty/zero logic upstream
            pass

        # encode returns ndarray or tensor
        vector = self.model.encode(text, convert_to_numpy=True)
        # mypy thinks this is Any, so we cast list(float)
        return [float(x) for x in vector.tolist()]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of strings.
        """
        if not texts:
            return []

        vectors = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        # mypy thinks this is Any, so we cast
        return [[float(x) for x in vec.tolist()] for vec in vectors]
