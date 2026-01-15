# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_codex

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Union

import duckdb
import lancedb
from loguru import logger

from coreason_codex.interfaces import Embedder


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
        self.source_dir = Path(source_dir)  # pragma: no cover
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
        db_path.unlink(missing_ok=True)

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
            db_path.unlink(missing_ok=True)  # Cleanup partial build
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

    def build_vectors(self, embedder: Embedder, table_name: str = "vectors", batch_size: int = 10000) -> None:
        """
        Builds the vector artifacts (LanceDB) from the DuckDB vocabulary.

        Args:
            embedder: The embedding model to use.
            table_name: The name of the LanceDB table to create.
            batch_size: Number of records to process at a time.
        """
        db_path = self.output_dir / "vocab.duckdb"
        if not db_path.exists():
            raise FileNotFoundError(f"Vocab artifact not found at: {db_path}. Run build_vocab() first.")

        logger.info(f"Building vectors in LanceDB at {self.output_dir}")

        con = duckdb.connect(str(db_path), read_only=True)
        try:
            # Prepare query
            # We explicitly cast IDs to BIGINT to ensure type consistency if not already
            query = """
                SELECT
                    concept_id,
                    concept_name,
                    domain_id,
                    vocabulary_id,
                    concept_class_id,
                    standard_concept,
                    concept_code
                FROM CONCEPT
                WHERE concept_name IS NOT NULL AND concept_name != ''
            """
            cursor = con.execute(query)

            def batch_generator() -> Iterator[List[Dict[str, Any]]]:
                while True:
                    rows = cursor.fetchmany(batch_size)
                    if not rows:
                        break  # pragma: no cover

                    # rows is list of tuples
                    # Extract text for embedding
                    texts = [row[1] for row in rows]

                    try:
                        # embed_batch returns List[List[float]]
                        embeddings = embedder.embed_batch(texts)
                    except Exception as e:
                        logger.error(f"Failed to embed batch: {e}")
                        raise

                    batch_data = []
                    for i, row in enumerate(rows):
                        # Ensure row items match the schema order from query
                        record = {
                            "vector": embeddings[i],
                            "concept_id": row[0],
                            "concept_name": row[1],
                        }
                        batch_data.append(record)

                    logger.info(f"Processed batch of {len(batch_data)} records")
                    yield batch_data

            # Initialize LanceDB connection
            # Note: lancedb.connect(str(self.output_dir)) creates the DB in output_dir
            lance_db = lancedb.connect(str(self.output_dir))

            # Create table (overwrites if exists)
            # data=batch_generator() consumes the generator
            lance_db.create_table(table_name, data=batch_generator(), mode="overwrite")

            logger.info(f"Vector table '{table_name}' built successfully.")  # pragma: no cover

        except Exception as e:
            logger.error(f"Failed to build vectors: {e}")
            raise RuntimeError(f"Vector build failed: {e}") from e
        finally:
            con.close()

    def generate_manifest(self, version: str = "v1.0", source_date: str = "2025-01-01") -> Path:
        """
        Generates the manifest.json file by computing checksums.
        """
        manifest_path = self.output_dir / "manifest.json"

        checksums: Dict[str, str] = {}

        # 1. vocab.duckdb
        vocab_path = self.output_dir / "vocab.duckdb"
        if vocab_path.exists():
            checksums["vocab.duckdb"] = self._compute_file_hash(vocab_path)

        # 2. vectors.lance
        # Depending on how lancedb saves, it's either a directory 'vectors.lance' or similar.
        # If table name was 'vectors', it's likely 'vectors.lance'.
        vectors_path = self.output_dir / "vectors.lance"
        if vectors_path.exists():
            checksums["vectors.lance"] = self._compute_dir_hash(vectors_path)

        manifest_data = {"version": version, "source_date": source_date, "checksums": checksums}

        with open(manifest_path, "w") as f:
            json.dump(manifest_data, f, indent=2)

        return manifest_path

    def _compute_file_hash(self, p: Path) -> str:
        sha256 = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(4096), b""):
                sha256.update(b)
        return sha256.hexdigest()

    def _compute_dir_hash(self, p: Path) -> str:
        sha256 = hashlib.sha256()
        paths = []
        for root, _, files in os.walk(p):
            for file in files:
                full = Path(root) / file
                rel = full.relative_to(p)
                paths.append((str(rel), full))
        paths.sort(key=lambda x: x[0])

        for rel_str, full in paths:
            sha256.update(rel_str.encode("utf-8"))
            sha256.update(self._compute_file_hash(full).encode("utf-8"))
        return sha256.hexdigest()
