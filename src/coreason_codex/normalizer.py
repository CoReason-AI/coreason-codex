# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_codex

from typing import Any, List, Optional

import duckdb
from loguru import logger

from coreason_codex.interfaces import Embedder
from coreason_codex.schemas import CodexMatch, Concept


class CodexNormalizer:
    """
    The Semantic Normalizer. Maps free text to Standard OMOP Concepts.

    Mechanism:
    1. Embeds input text using the provided Embedder.
    2. Searches LanceDB for nearest neighbor vectors to find Concept IDs.
    3. Hydrates Concept details from DuckDB.
    4. Applies domain filtering and formatting.
    """

    def __init__(
        self,
        embedder: Embedder,
        duckdb_conn: duckdb.DuckDBPyConnection,
        lancedb_conn: Any,
        vector_table_name: str = "vectors",
    ):
        self.embedder = embedder
        self.duckdb_conn = duckdb_conn
        self.lancedb_conn = lancedb_conn
        self.vector_table_name = vector_table_name

        # Verify table exists
        try:
            self.table = self.lancedb_conn.open_table(self.vector_table_name)
        except Exception as e:
            logger.error(f"Failed to open LanceDB table '{self.vector_table_name}': {e}")
            raise ValueError(f"LanceDB table '{self.vector_table_name}' not found.") from e

    def normalize(self, text: str, k: int = 10, domain_filter: Optional[str] = None) -> List[CodexMatch]:
        """
        Normalizes text to standard concepts.

        Args:
            text: The input text (e.g. "heart attack").
            k: Number of candidates to retrieve.
            domain_filter: Optional OMOP Domain ID to filter by (e.g. "Condition", "Drug").
        """
        if not text.strip():
            return []

        # 1. Embed
        vector = self.embedder.embed(text)

        # 2. Vector Search (LanceDB)
        # Returns a PyArrow table or similar. We convert to list of dicts.
        # We assume the schema has 'concept_id' and 'concept_name'.
        results = self.table.search(vector).limit(k).to_list()

        if not results:
            return []

        # Extract IDs and scores
        # We need to map concept_id -> score to attach later
        # NOTE: LanceDB results are usually sorted by distance (ASC).
        # We want to keep the BEST (first) score for each unique concept_id.
        concept_scores = {}
        for r in results:
            c_id = r["concept_id"]
            if c_id not in concept_scores:
                concept_scores[c_id] = 1.0 - r["_distance"]

        concept_ids = list(concept_scores.keys())

        # 3. Hydrate from DuckDB
        # We query the concept details.
        # Note: We enforce standard concept status if desired, but usually normalizer returns what matches.
        # However, PRD says: "Matches 'Nausea' (ConceptID: 31967)".
        # And CodexMatch has `is_standard`.
        query = f"""
            SELECT
                concept_id,
                concept_name,
                domain_id,
                vocabulary_id,
                concept_class_id,
                standard_concept,
                concept_code
            FROM concept
            WHERE concept_id IN ({",".join(["?"] * len(concept_ids))})
        """

        # Add domain filter if present
        params = list(concept_ids)
        if domain_filter:
            query += " AND domain_id = ?"
            params.append(domain_filter)

        cursor = self.duckdb_conn.execute(query, params)
        rows = cursor.fetchall()

        # 4. Construct Matches
        matches: List[CodexMatch] = []
        for row in rows:
            # Row order: id, name, domain, vocab, class, standard, code
            c_id = row[0]

            concept = Concept(
                concept_id=c_id,
                concept_name=row[1],
                domain_id=row[2],
                vocabulary_id=row[3],
                concept_class_id=row[4],
                standard_concept=row[5],
                concept_code=row[6],
            )

            # Re-attach score
            score = concept_scores.get(c_id, 0.0)

            # Determine match status
            is_std = concept.standard_concept == "S"

            # If it's not standard, we might want to find the mapping, but that's the CrossWalker's job usually.
            # However, CodexMatch has `mapped_standard_id`.
            # For this iteration, we leave mapped_standard_id as None unless we do another lookup.
            # The PRD says: "is_standard: False... mapped_standard_id: 1125315".
            # This implies the Normalizer *should* try to find the standard mapping if possible.
            # But the "One Step Rule" suggests keeping logic atomic.
            # The "CrossWalker" is a separate component.
            # I will leave mapped_standard_id as None for now to keep this unit focused on "Vector Search + Hydration".

            matches.append(
                CodexMatch(
                    input_text=text,
                    match_concept=concept,
                    similarity_score=score,
                    is_standard=is_std,
                    mapped_standard_id=None,
                )
            )

        # Sort by score descending
        matches.sort(key=lambda x: x.similarity_score, reverse=True)

        return matches
