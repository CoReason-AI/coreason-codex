# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_codex

from typing import List, cast

import numpy as np
from sentence_transformers import SentenceTransformer

from coreason_codex.utils.logger import logger


class SapBertEmbedder:
    """
    Embedder implementation using SapBERT model via SentenceTransformer.
    """

    def __init__(self, model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext") -> None:
        """
        Initialize the SapBERT embedder.

        Args:
            model_name: The name of the HuggingFace model to load.
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts into a numpy array of vectors.

        Args:
            texts: A list of strings to embed.

        Returns:
            A numpy array of shape (len(texts), embedding_dim).
        """
        logger.info(f"Embedding {len(texts)} texts")
        embeddings = self.model.encode(texts)
        # Ensure it returns numpy array (encode usually does, but let's be safe)
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)

        # Cast to Any first if necessary, but just cast to ndarray to satisfy mypy
        return cast(np.ndarray, embeddings)
