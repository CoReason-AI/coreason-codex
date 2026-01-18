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

from loguru import logger

from coreason_codex.crosswalker import CodexCrossWalker
from coreason_codex.embedders import SapBertEmbedder
from coreason_codex.hierarchy import CodexHierarchy
from coreason_codex.loader import CodexLoader
from coreason_codex.normalizer import CodexNormalizer
from coreason_codex.schemas import CodexMatch, Concept


class CodexContext:
    """
    Global context/singleton for accessing Codex services.
    """

    _instance: Optional["CodexContext"] = None

    def __init__(self, pack_path: str):
        logger.info(f"Initializing Codex Context with pack: {pack_path}")
        self.loader = CodexLoader(pack_path)
        self.duckdb_conn, self.lancedb_conn = self.loader.load_codex()

        # Initialize components
        # Note: We use SapBertEmbedder by default. In a real app, this might be injected.
        self.embedder = SapBertEmbedder()

        self.normalizer = CodexNormalizer(
            embedder=self.embedder, duckdb_conn=self.duckdb_conn, lancedb_conn=self.lancedb_conn
        )

        self.hierarchy = CodexHierarchy(duckdb_conn=self.duckdb_conn)

        self.crosswalker = CodexCrossWalker(duckdb_conn=self.duckdb_conn)

    @classmethod
    def initialize(cls, pack_path: str) -> None:
        cls._instance = cls(pack_path)

    @classmethod
    def get_instance(cls) -> "CodexContext":
        if cls._instance is None:
            raise RuntimeError("CodexContext not initialized. Call initialize() first.")
        return cls._instance


# --- Public API Functions (MCP Tools) ---


def initialize(pack_path: str) -> None:
    """Initializes the Codex system."""
    CodexContext.initialize(pack_path)


def codex_normalize(text: str, domain_filter: Optional[str] = None) -> List[CodexMatch]:
    """
    Normalizes text to Standard Concepts.
    """
    ctx = CodexContext.get_instance()
    return ctx.normalizer.normalize(text, domain_filter=domain_filter)


def codex_get_descendants(concept_id: int) -> List[int]:
    """
    Returns a list of descendant concept IDs.
    """
    ctx = CodexContext.get_instance()
    return ctx.hierarchy.get_descendants(concept_id)


def codex_translate_code(source_id: int, target_vocabulary: Optional[str] = None) -> List[Concept]:
    """
    Translates a concept ID to another vocabulary (e.g. SNOMED).
    """
    ctx = CodexContext.get_instance()
    return ctx.crosswalker.translate_code(source_id, target_vocabulary_id=target_vocabulary)


def codex_check_relationship(concept_id_1: int, concept_id_2: int, relationship_id: str) -> bool:
    """
    Checks if a specific relationship exists between two concepts.
    """
    ctx = CodexContext.get_instance()
    return ctx.crosswalker.check_relationship(concept_id_1, concept_id_2, relationship_id)
