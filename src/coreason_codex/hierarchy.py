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
from loguru import logger


class CodexHierarchy:
    """
    Handles hierarchical reasoning using the OMOP CONCEPT_ANCESTOR table.
    """

    def __init__(self, duckdb_conn: duckdb.DuckDBPyConnection):
        self.duckdb_conn = duckdb_conn

        # Verify table exists
        try:
            self.duckdb_conn.execute("SELECT 1 FROM concept_ancestor LIMIT 1")
        except Exception as e:
            logger.error(f"Table 'concept_ancestor' not found or invalid: {e}")
            raise ValueError("Table 'concept_ancestor' is missing in the vocabulary.") from e

    def get_descendants(self, concept_id: int) -> List[int]:
        """
        Returns a list of descendant concept IDs for a given concept ID.
        Uses the transitive closure table (concept_ancestor).
        Includes the concept itself (distance=0).
        """
        query = """
            SELECT descendant_concept_id
            FROM concept_ancestor
            WHERE ancestor_concept_id = ?
        """
        try:
            results = self.duckdb_conn.execute(query, [concept_id]).fetchall()
            # Results is list of tuples [(id,), (id,)]
            return [r[0] for r in results]
        except Exception as e:
            logger.error(f"Error querying descendants for {concept_id}: {e}")
            return []
