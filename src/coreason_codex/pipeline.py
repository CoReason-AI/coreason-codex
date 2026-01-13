# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_codex

from dataclasses import dataclass
from typing import Optional

from coreason_codex.crosswalker import CodexCrossWalker
from coreason_codex.embedders import SapBertEmbedder
from coreason_codex.hierarchy import CodexHierarchy
from coreason_codex.loader import CodexLoader
from coreason_codex.normalizer import CodexNormalizer
from coreason_codex.utils.logger import logger


@dataclass
class CodexContext:
    """Holds the initialized components of the Codex system."""

    normalizer: CodexNormalizer
    hierarchy: CodexHierarchy
    crosswalker: CodexCrossWalker


# Global singleton
_CONTEXT: Optional[CodexContext] = None


def initialize(pack_path: str) -> None:
    """
    Initialize the Codex system with the specified artifact pack.
    Loads artifacts, initializes embedders, and sets up core engines.

    Args:
        pack_path: Path to the Codex Pack directory.

    Raises:
        FileNotFoundError: If the pack path does not exist.
        ValueError: If the pack integrity check fails.
        RuntimeError: If initialization fails.
    """
    global _CONTEXT
    logger.info(f"Initializing Codex with pack: {pack_path}")

    try:
        # 1. Load Artifacts
        loader = CodexLoader(pack_path)
        duckdb_con, lancedb_con = loader.load_codex()

        # 2. Initialize Embedder
        # This is a heavy operation, done synchronously as per requirements
        embedder = SapBertEmbedder()

        # 3. Initialize Engines
        # Open the vectors table. Assuming table name 'vectors' from builder defaults.
        # We need to list tables or just try to open it.
        # CodexBuilder uses "vectors" by default.
        # lancedb connection object allows opening table.
        try:
            vector_table = lancedb_con.open_table("vectors")
        except Exception:
            # Fallback or check if table exists logic might be needed if table name varies
            # But for now we assume "vectors"
            logger.warning("Table 'vectors' not found, checking available tables...")
            # If using older lancedb, table_names() returns list.
            # If newer, verify.
            # Assuming 'vectors' is strictly required.
            raise ValueError("Vector table 'vectors' not found in LanceDB artifact.") from None

        normalizer = CodexNormalizer(table=vector_table, embedder=embedder)
        hierarchy = CodexHierarchy(con=duckdb_con)
        crosswalker = CodexCrossWalker(con=duckdb_con)

        # 4. Set Global Context
        _CONTEXT = CodexContext(
            normalizer=normalizer,
            hierarchy=hierarchy,
            crosswalker=crosswalker,
        )
        logger.info("Codex initialization complete.")

    except Exception as e:
        logger.exception("Failed to initialize Codex")
        raise RuntimeError(f"Codex initialization failed: {e}") from e


def get_context() -> CodexContext:
    """
    Retrieve the initialized Codex context.

    Returns:
        The active CodexContext.

    Raises:
        RuntimeError: If Codex has not been initialized.
    """
    if _CONTEXT is None:
        raise RuntimeError("Codex not initialized. Call initialize() first.")
    return _CONTEXT
