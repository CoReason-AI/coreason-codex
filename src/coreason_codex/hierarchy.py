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

from coreason_codex.utils.logger import logger


class CodexHierarchy:
    """
    Hierarchy Engine that exploits the pre-computed OMOP hierarchy for reasoning.
    Uses the CONCEPT_ANCESTOR table to find descendants.
    """

    def __init__(self, con: duckdb.DuckDBPyConnection) -> None:
        """
        Initialize the CodexHierarchy.

        Args:
            con: An initialized DuckDB connection.
        """
        self.con = con

    def get_descendants(self, concept_id: int) -> List[int]:
        """
        Get all descendant concept IDs for a given concept ID.
        Uses the CONCEPT_ANCESTOR table where ancestor_concept_id matches the input.

        Args:
            concept_id: The ancestor concept ID.

        Returns:
            A list of descendant concept IDs.
        """
        logger.info(f"Fetching descendants for concept_id: {concept_id}")

        try:
            # Query the CONCEPT_ANCESTOR table
            # We select descendant_concept_id where ancestor_concept_id is the input
            query = """
                SELECT descendant_concept_id
                FROM CONCEPT_ANCESTOR
                WHERE ancestor_concept_id = ?
            """
            results = self.con.execute(query, [concept_id]).fetchall()

            # Extract IDs from the result tuples
            descendants = [row[0] for row in results]

            logger.info(f"Found {len(descendants)} descendants for {concept_id}")
            return descendants

        except Exception as e:
            logger.error(f"Failed to fetch descendants for {concept_id}: {e}")
            # Depending on desired behavior, we could raise or return empty list.
            # Returning empty list is safer for the agent workflow, but logging the error is crucial.
            return []
