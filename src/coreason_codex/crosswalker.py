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

import duckdb
from loguru import logger

from coreason_codex.schemas import Concept


class CodexCrossWalker:
    """
    Handles translation between vocabularies using the OMOP CONCEPT_RELATIONSHIP table.
    """

    def __init__(self, duckdb_conn: duckdb.DuckDBPyConnection):
        self.duckdb_conn = duckdb_conn

        # Verify table exists
        try:
            self.duckdb_conn.execute("SELECT 1 FROM concept_relationship LIMIT 1")
        except Exception as e:
            logger.error(f"Table 'concept_relationship' not found or invalid: {e}")
            raise ValueError("Table 'concept_relationship' is missing in the vocabulary.") from e

    def translate_code(
        self, source_id: int, relationship_id: str = "Maps to", target_vocabulary_id: Optional[str] = None
    ) -> List[Concept]:
        """
        Translates a source concept ID to target concept(s) via a specific relationship.
        Useful for mapping ICD-10 to SNOMED (Maps to) or ATC to RxNorm.

        Args:
            source_id: The Concept ID to translate from.
            relationship_id: The relationship type (default: "Maps to").
            target_vocabulary_id: Optional filter for target vocabulary (e.g. "SNOMED").

        Returns:
            List[Concept]: The translated target concepts.
        """
        query = """
            SELECT
                c.concept_id,
                c.concept_name,
                c.domain_id,
                c.vocabulary_id,
                c.concept_class_id,
                c.standard_concept,
                c.concept_code
            FROM concept_relationship cr
            JOIN concept c ON cr.concept_id_2 = c.concept_id
            WHERE cr.concept_id_1 = ?
              AND cr.relationship_id = ?
              AND cr.invalid_reason IS NULL
              AND c.invalid_reason IS NULL
        """

        params = [source_id, relationship_id]

        if target_vocabulary_id:
            query += " AND c.vocabulary_id = ?"
            params.append(target_vocabulary_id)

        try:
            cursor = self.duckdb_conn.execute(query, params)
            rows = cursor.fetchall()

            concepts = []
            for row in rows:
                concepts.append(
                    Concept(
                        concept_id=row[0],
                        concept_name=row[1],
                        domain_id=row[2],
                        vocabulary_id=row[3],
                        concept_class_id=row[4],
                        standard_concept=row[5],
                        concept_code=row[6],
                    )
                )
            return concepts

        except Exception as e:
            logger.error(f"Error translating code {source_id}: {e}")
            return []

    def check_relationship(self, concept_id_1: int, concept_id_2: int, relationship_id: str) -> bool:
        """
        Checks if a specific relationship exists between two concepts.
        Useful for verification (e.g. "Does Drug X treat Condition Y?").

        Args:
            concept_id_1: Source Concept ID.
            concept_id_2: Target Concept ID.
            relationship_id: The relationship type (e.g. "Indication - Drug").

        Returns:
            bool: True if the relationship exists and is valid, False otherwise.
        """
        query = """
            SELECT 1
            FROM concept_relationship
            WHERE concept_id_1 = ?
              AND concept_id_2 = ?
              AND relationship_id = ?
              AND invalid_reason IS NULL
            LIMIT 1
        """

        try:
            result = self.duckdb_conn.execute(query, [concept_id_1, concept_id_2, relationship_id]).fetchone()
            return result is not None
        except Exception as e:
            logger.error(f"Error checking relationship between {concept_id_1} and {concept_id_2}: {e}")
            return False
