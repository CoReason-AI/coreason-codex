# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_codex

import re
from typing import Any, Optional

from pydantic import ValidationError

from coreason_codex.interfaces import Embedder
from coreason_codex.models import CodexMatch, Concept
from coreason_codex.utils.logger import logger


class CodexNormalizer:
    """
    Semantic Normalizer for mapping text to OMOP Concepts using vector search.
    """

    def __init__(self, vector_table: Any, embedder: Embedder):
        """
        Initialize the normalizer.

        Args:
            vector_table: A lancedb.table.Table object ready for search.
            embedder: Component to generate embeddings for input text.
        """
        self.vector_table = vector_table
        self.embedder = embedder

    def _validate_domain_filter(self, domain_filter: str) -> bool:
        """
        Validates the domain filter to prevent injection.
        OMOP Domain IDs are typically alphanumeric (e.g., 'Condition', 'Drug').
        Allows alphanumeric and underscores.
        """
        return bool(re.match(r"^[a-zA-Z0-9_]+$", domain_filter))

    def normalize(self, text: str, domain_filter: Optional[str] = None) -> Optional[CodexMatch]:
        """
        Find the best matching standard concept for the input text.

        Args:
            text: The clinical text to normalize (e.g., "tummy ache").
            domain_filter: Optional OMOP Domain ID to filter by (e.g., "Condition").

        Returns:
            CodexMatch object if a match is found, None otherwise.
        """
        try:
            vector = self.embedder.embed(text)
        except Exception as e:
            logger.error(f"Embedding failed for text '{text}': {e}")
            return None

        try:
            query = self.vector_table.search(vector)

            if domain_filter:
                if not self._validate_domain_filter(domain_filter):
                    logger.warning(f"Invalid domain_filter provided: {domain_filter}")
                    return None

                # Use prefilter=True to filter before vector search for efficiency
                # safe because we validated domain_filter is alphanumeric
                query = query.where(f"domain_id = '{domain_filter}'", prefilter=True)

            # Limit to top 1 match
            results = query.limit(1).to_list()
        except Exception as e:
            logger.error(f"LanceDB search failed: {e}")
            return None

        if not results:
            return None

        top_result = results[0]

        # Handle similarity score
        # LanceDB returns _distance. Assuming Cosine distance: Similarity = 1 - Distance
        distance = top_result.pop("_distance", 0.0)
        similarity = 1.0 - distance

        # Remove vector column if present to avoid Pydantic validation error
        if "vector" in top_result:
            del top_result["vector"]

        try:
            concept = Concept(**top_result)
        except ValidationError as e:
            logger.error(f"Data validation error for concept: {e}")
            return None

        is_standard = concept.standard_concept == "S"

        # If it's a standard concept, it maps to itself.
        # If non-standard, we return None for mapped_id (caller must use CrossWalker).
        mapped_id = concept.concept_id if is_standard else None

        return CodexMatch(
            input_text=text,
            match_concept=concept,
            similarity_score=similarity,
            is_standard=is_standard,
            mapped_standard_id=mapped_id,
        )
