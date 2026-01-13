# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_codex

from pathlib import Path
from typing import Union

import duckdb

from coreason_codex.utils.logger import logger


class CodexBuilder:
    """
    Offline Builder utility to compile raw Athena CSVs into DuckDB artifacts.
    """

    REQUIRED_FILES = ["CONCEPT.csv", "CONCEPT_RELATIONSHIP.csv", "CONCEPT_ANCESTOR.csv"]

    def __init__(self, source_dir: Union[str, Path], output_dir: Union[str, Path]):
        """
        Initialize the CodexBuilder.

        Args:
            source_dir: Directory containing the raw Athena CSV files.
            output_dir: Directory where the artifacts will be saved.
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)

    def _verify_source_files(self) -> None:
        """Verify that all required CSV files exist in the source directory."""
        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {self.source_dir}")

        for filename in self.REQUIRED_FILES:
            file_path = self.source_dir / filename
            if not file_path.exists():
                raise FileNotFoundError(f"Required file not found: {filename}")

    def build_vocab(self) -> Path:
        """
        Builds the vocab.duckdb artifact from the source CSVs.

        Returns:
            The path to the generated vocab.duckdb file.
        """
        self._verify_source_files()

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)

        db_path = self.output_dir / "vocab.duckdb"

        # Remove existing file if it exists to ensure a clean build
        if db_path.exists():
            db_path.unlink()

        logger.info(f"Building vocab artifact at {db_path}")

        con = None
        try:
            con = duckdb.connect(str(db_path))

            # Load CONCEPT table
            self._load_table(con, "CONCEPT")

            # Load CONCEPT_RELATIONSHIP table
            self._load_table(con, "CONCEPT_RELATIONSHIP")

            # Load CONCEPT_ANCESTOR table
            self._load_table(con, "CONCEPT_ANCESTOR")

            # Create Indexes
            self._create_indexes(con)

            con.close()
            con = None
            logger.info("Vocab build complete.")
            return db_path

        except Exception as e:
            logger.error(f"Failed to build vocab artifact: {e}")
            if con:
                con.close()
            if db_path.exists():
                db_path.unlink()  # Cleanup partial build
            raise RuntimeError(f"Build failed: {e}") from e

    def _load_table(self, con: duckdb.DuckDBPyConnection, table_name: str) -> None:
        """Loads a single CSV file into a DuckDB table."""
        filename = f"{table_name}.csv"
        file_path = self.source_dir / filename
        logger.info(f"Loading {table_name} from {file_path}")

        # specific for Athena: typically tab-separated, but read_csv_auto is smart.
        # We explicitly assume headers exist.
        query = f"""
            CREATE TABLE {table_name} AS
            SELECT * FROM read_csv_auto('{file_path}', header=True)
        """
        con.execute(query)
        logger.info(f"Loaded {table_name}")

    def _create_indexes(self, con: duckdb.DuckDBPyConnection) -> None:
        """Creates indexes for performance optimization."""
        logger.info("Creating indexes...")

        # CONCEPT: Primary lookup
        con.execute("CREATE INDEX idx_concept_id ON CONCEPT(concept_id)")

        # CONCEPT_ANCESTOR: Hierarchy lookups
        con.execute("CREATE INDEX idx_ancestor_id ON CONCEPT_ANCESTOR(ancestor_concept_id)")
        con.execute("CREATE INDEX idx_descendant_id ON CONCEPT_ANCESTOR(descendant_concept_id)")

        # CONCEPT_RELATIONSHIP: Cross-walking
        con.execute("CREATE INDEX idx_cr_concept_1 ON CONCEPT_RELATIONSHIP(concept_id_1)")
        # We might also want concept_id_2 for reverse lookups if needed, but PRD emphasizes forward mapping.
        # Adding it helps with complete graph traversal.
        con.execute("CREATE INDEX idx_cr_concept_2 ON CONCEPT_RELATIONSHIP(concept_id_2)")

        logger.info("Indexes created.")
