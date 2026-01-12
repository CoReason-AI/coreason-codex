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

import duckdb

from coreason_codex.schemas import Concept
from coreason_codex.utils.logger import logger


class CodexCrossWalker:
    """
    Cross-Walker that maps concepts between vocabularies.
    Uses the CONCEPT_RELATIONSHIP and CONCEPT tables.
    """

    def __init__(self, con: duckdb.DuckDBPyConnection) -> None:
        """
        Initialize the CodexCrossWalker.

        Args:
            con: An initialized DuckDB connection.
        """
        self.con = con

    def translate_code(self, source_id: int, target_vocabulary: str) -> List[Concept]:
        """
        Maps a source concept to concepts in the target vocabulary.
        Typically traverses 'Maps to' or 'Maps to value' relationships.

        Args:
            source_id: The source concept ID.
            target_vocabulary: The target vocabulary ID (e.g., 'ICD10CM', 'SNOMED').

        Returns:
            A list of matching Concept objects in the target vocabulary.
        """
        logger.info(f"Translating concept {source_id} to vocabulary {target_vocabulary}")

        try:
            # Query Logic:
            # 1. Join CONCEPT_RELATIONSHIP (cr) on concept_id_1 = source_id
            # 2. Join CONCEPT (c) on c.concept_id = cr.concept_id_2
            # 3. Filter where c.vocabulary_id = target_vocabulary
            # 4. Filter for valid relationships (usually 'Maps to', 'Maps to value', or standard mapping)
            #    For broad translation, we might not strictly filter relationship_id unless specified.
            #    PRD says: SNOMED ID -> Maps to (inverse) -> ICD-10 Code.
            #    So we are looking for valid relationships.
            #    Let's select relevant columns to build Concept objects.

            query = """
                SELECT
                    c.concept_id,
                    c.concept_name,
                    c.domain_id,
                    c.vocabulary_id,
                    c.concept_class_id,
                    c.standard_concept,
                    c.concept_code
                FROM CONCEPT_RELATIONSHIP cr
                JOIN CONCEPT c ON cr.concept_id_2 = c.concept_id
                WHERE cr.concept_id_1 = ?
                  AND c.vocabulary_id = ?
                  AND cr.invalid_reason IS NULL
                  AND c.invalid_reason IS NULL
            """

            # Note: We assume standard OMOP schemas where invalid_reason IS NULL means active.

            results = self.con.execute(query, [source_id, target_vocabulary]).fetchall()

            concepts: List[Concept] = []
            for row in results:
                concept = Concept(
                    concept_id=row[0],
                    concept_name=row[1],
                    domain_id=row[2],
                    vocabulary_id=row[3],
                    concept_class_id=row[4],
                    standard_concept=row[5],
                    concept_code=row[6],
                )
                concepts.append(concept)

            logger.info(f"Found {len(concepts)} translations for {source_id} in {target_vocabulary}")
            return concepts

        except Exception as e:
            logger.error(f"Failed to translate concept {source_id}: {e}")
            return []
