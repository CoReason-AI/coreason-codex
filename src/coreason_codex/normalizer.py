# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_codex

from typing import List, Optional

import lancedb

from coreason_codex.interfaces import Embedder
from coreason_codex.schemas import CodexMatch, Concept
from coreason_codex.utils.logger import logger


class CodexNormalizer:
    """
    Semantic Normalizer that maps free text to Standard Concept IDs.
    Uses Vector Search (LanceDB) and Domain Filtering.
    """

    def __init__(self, table: lancedb.table.Table, embedder: Embedder) -> None:
        """
        Initialize the CodexNormalizer.

        Args:
            table: An initialized LanceDB table containing the vocabulary vectors.
            embedder: An instance of an Embedder (e.g., SapBERT or Mock).
        """
        self.table = table
        self.embedder = embedder

    def normalize(self, text: str, domain_filter: Optional[str] = None) -> List[CodexMatch]:
        """
        Normalize input text to a list of matching concepts.

        Args:
            text: The input text to normalize (e.g., "Patient felt queasy").
            domain_filter: Optional domain ID to filter results (e.g., "Condition").

        Returns:
            A list of CodexMatch objects sorted by similarity.
        """
        logger.info(f"Normalizing text: '{text}' with domain_filter='{domain_filter}'")

        # 1. Embed the input text
        # The embedder is expected to return a numpy array (or list of list)
        # We handle single string input by wrapping it in a list
        embeddings = self.embedder.embed([text])
        if len(embeddings) == 0:
            logger.warning("Embedder returned empty embeddings.")
            return []

        vector = embeddings[0]

        # 2. Build Query
        query = self.table.search(vector)

        # 3. Apply Domain Filter if provided
        if domain_filter:
            query = query.where(f"domain_id = '{domain_filter}'")

        # 4. Execute Search
        # We default to top 10 matches for now, could be configurable
        results = query.limit(10).to_list()

        matches: List[CodexMatch] = []
        for res in results:
            # Map LanceDB result to Concept schema
            # Assuming LanceDB columns match Concept schema fields + 'vector' + '_distance'

            # Note: LanceDB search returns distance. Similarity is usually 1 - distance (for cosine)
            # or depending on metric. Assuming standard cosine distance here or similar.
            # If metric is "cosine", smaller is better. If "inner_product", larger is better.
            # For this implementation, we will use the '_distance' field provided by LanceDB.
            # However, CodexMatch expects 'similarity_score'.
            # If the vector search uses cosine distance, score = 1 - distance.

            distance = res.get("_distance", 0.0)
            similarity = 1.0 - distance

            try:
                concept = Concept(
                    concept_id=res["concept_id"],
                    concept_name=res["concept_name"],
                    domain_id=res["domain_id"],
                    vocabulary_id=res["vocabulary_id"],
                    concept_class_id=res["concept_class_id"],
                    standard_concept=res.get("standard_concept"),
                    concept_code=res["concept_code"],
                )

                # Determine is_standard logic
                # PRD: "Standard Concepts" usually SNOMED/RxNorm.
                # PRD Schema: standard_concept: Optional[str] # "S" (Standard) or NULL
                is_standard = concept.standard_concept == "S"

                # Mapped standard ID logic would ideally come from the Concept Relationship/Cross-Walker
                # For this atomic unit, we might not have it unless it's in the vector store.
                # PRD says "matches 'Nausea' (ConceptID: 31967)".
                # If the match itself is non-standard, we need a way to find the standard map.
                # For now, we will leave mapped_standard_id as None or self if standard.
                mapped_id = concept.concept_id if is_standard else None

                match = CodexMatch(
                    input_text=text,
                    match_concept=concept,
                    similarity_score=similarity,
                    is_standard=is_standard,
                    mapped_standard_id=mapped_id,
                )
                matches.append(match)
            except Exception as e:
                logger.error(f"Failed to map result to Concept: {e}")
                continue

        return matches
