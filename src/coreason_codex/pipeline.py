# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_codex

import os
import shutil
from pathlib import Path
from typing import List, Optional

from loguru import logger

from coreason_codex.crosswalker import CodexCrossWalker
from coreason_codex.embedders import SapBertEmbedder
from coreason_codex.hierarchy import CodexHierarchy
from coreason_codex.loader import CodexLoader, load_repository
from coreason_codex.normalizer import CodexNormalizer
from coreason_codex.schemas import CodexMatch, Concept
from coreason_identity.models import UserContext


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

    def search_code(self, query: str, limit: int = 20) -> List[dict]:
        """
        Search for code snippets in the vector index.
        """
        vectors = self.embedder.embed_batch([query])
        query_vector = vectors[0]

        try:
            tbl = self.lancedb_conn.open_table("code_vectors")
            results = tbl.search(query_vector).limit(limit).to_list()
            return results
        except Exception as e:
            logger.warning(f"Code search failed (table might not exist): {e}")
            return []

    @classmethod
    def initialize(cls, pack_path: str) -> None:
        cls._instance = cls(pack_path)

    @classmethod
    def get_instance(cls) -> "CodexContext":
        if cls._instance is None:
            raise RuntimeError("CodexContext not initialized. Call initialize() first.")
        return cls._instance


class IndexingPipeline:
    """
    Pipeline for indexing repositories.
    """

    def __init__(self):
        self.context = CodexContext.get_instance()

    def run(self, url: str, user_context: UserContext):
        """
        Runs the indexing pipeline: fetch -> embed -> store.
        """
        logger.info(f"Starting indexing for {url}")
        repo_path = load_repository(url, user_context)

        try:
            self._embed_and_store(repo_path, user_context)
        finally:
            shutil.rmtree(repo_path, ignore_errors=True)

    def _embed_and_store(self, repo_path: Path, user_context: UserContext):
        documents = []
        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith((".py", ".md", ".txt", ".java", ".js", ".ts", ".go")):
                    full_path = Path(root) / file
                    try:
                        content = full_path.read_text(errors="ignore")
                        if not content.strip():
                            continue

                        # Embed - Using the context embedder (SapBert) as placeholder for CodeBert/OpenAI
                        # Truncate to avoid context limit issues
                        truncated_content = content[:512]
                        vectors = self.context.embedder.embed_batch([truncated_content])
                        vector = vectors[0]

                        doc = {
                            "vector": vector,
                            "text": content[:2000],  # Store truncated text for display
                            "source": str(full_path),
                            "owner_id": user_context.user_id,
                            "allowed_groups": user_context.groups,
                        }
                        documents.append(doc)
                    except Exception as e:
                        logger.warning(f"Failed to embed {file}: {e}")

        if documents:
            logger.info(f"Storing {len(documents)} documents to LanceDB")
            db = self.context.lancedb_conn
            table_name = "code_vectors"
            try:
                if table_name in db.table_names():
                    tbl = db.open_table(table_name)
                    tbl.add(documents)
                else:
                    db.create_table(table_name, data=documents)
            except Exception as e:
                logger.error(f"Failed to store vectors: {e}")
                # Fallback: try creating if open failed weirdly
                try:
                    db.create_table(table_name, data=documents, mode="overwrite")
                except Exception as ex:
                    logger.error(f"Retry failed: {ex}")


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
