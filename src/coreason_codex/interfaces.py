# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_codex

from typing import List, Protocol


class Embedder(Protocol):
    """
    Protocol for text embedding models.
    """

    def embed(self, text: str) -> List[float]:
        """
        Embeds a single string into a vector.
        """
        ...

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of strings into a list of vectors.
        """
        ...
