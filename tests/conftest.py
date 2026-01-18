# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_codex

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Generator, List, cast

import duckdb
import lancedb
import numpy as np
import pytest

from coreason_codex.interfaces import Embedder

# --- Mocks ---


class MockEmbedder:
    """
    Deterministic embedder for testing.
    Generates vectors based on string hash.
    Dimension: 128
    """

    def __init__(self, dim: int = 128):
        self.dim = dim

    def embed(self, text: str) -> List[float]:
        # Create a deterministic seed from the text
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        # Generate normalized vector
        vec = rng.random(self.dim)
        return cast(List[float], (vec / np.linalg.norm(vec)).tolist())

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]


# --- Fixtures ---


@pytest.fixture
def mock_embedder() -> Embedder:
    return MockEmbedder()


@pytest.fixture
def synthetic_codex_pack(tmp_path: Path, mock_embedder: Embedder) -> Generator[Path, None, None]:
    """
    Creates a temporary Codex Pack with:
    - vocab.duckdb (Concepts, Ancestors, Relationships)
    - vectors.lance (Embeddings)
    - manifest.json
    """
    pack_dir = tmp_path / "codex_vtest"
    pack_dir.mkdir()

    # 1. Create DuckDB Artifact
    db_path = pack_dir / "vocab.duckdb"
    con = duckdb.connect(str(db_path))

    # Create schema matching schemas.Concept
    con.execute("""
        CREATE TABLE concept (
            concept_id INTEGER,
            concept_name VARCHAR,
            domain_id VARCHAR,
            vocabulary_id VARCHAR,
            concept_class_id VARCHAR,
            standard_concept VARCHAR,
            concept_code VARCHAR,
            invalid_reason VARCHAR
        )
    """)

    # Create ancestor table
    con.execute("""
        CREATE TABLE concept_ancestor (
            ancestor_concept_id INTEGER,
            descendant_concept_id INTEGER,
            min_levels_of_separation INTEGER,
            max_levels_of_separation INTEGER
        )
    """)

    # Create relationship table
    con.execute("""
        CREATE TABLE concept_relationship (
            concept_id_1 INTEGER,
            concept_id_2 INTEGER,
            relationship_id VARCHAR,
            valid_start_date DATE,
            valid_end_date DATE,
            invalid_reason VARCHAR
        )
    """)

    # Sample Data (OMOP Standard Concepts)
    # 312327: Acute myocardial infarction (Condition)
    # 201820: Diabetes mellitus (Condition)
    # 1125315: Acetaminophen (Drug)
    # 1503297: Metformin (Drug)
    # 31967: Nausea (Condition)
    # 441840: Clinical Finding (High level)
    # 43530807: Heart Attack (Non-standard SNOMED)
    # 999999: ICD-10 Code "I21.9" (Non-standard)

    concepts = [
        (312327, "Acute myocardial infarction", "Condition", "SNOMED", "Clinical Finding", "S", "22298006", None),
        (201820, "Diabetes mellitus", "Condition", "SNOMED", "Clinical Finding", "S", "73211009", None),
        (1125315, "Acetaminophen", "Drug", "RxNorm", "Ingredient", "S", "161", None),
        (1503297, "Metformin", "Drug", "RxNorm", "Ingredient", "S", "6809", None),
        (31967, "Nausea", "Condition", "SNOMED", "Clinical Finding", "S", "422587005", None),
        (441840, "Clinical Finding", "Condition", "SNOMED", "Clinical Finding", "S", "CF", None),
        (43530807, "Heart Attack", "Condition", "SNOMED", "Clinical Finding", None, "12345_fake", None),
        (999999, "Acute myocardial infarction, unspecified", "Condition", "ICD10", "ICD10 Code", None, "I21.9", None),
    ]

    con.executemany("INSERT INTO concept VALUES (?, ?, ?, ?, ?, ?, ?, ?)", concepts)

    # Sample Ancestry
    ancestors = [
        (441840, 312327, 1, 1),
        (441840, 201820, 1, 1),
        (441840, 31967, 1, 1),
        (312327, 312327, 0, 0),
        (201820, 201820, 0, 0),
        (441840, 441840, 0, 0),
    ]
    con.executemany("INSERT INTO concept_ancestor VALUES (?, ?, ?, ?)", ancestors)

    # Sample Relationships
    # ICD10 "I21.9" (999999) "Maps to" SNOMED "Acute MI" (312327)
    relationships = [(999999, 312327, "Maps to", "1970-01-01", "2099-12-31", None)]
    con.executemany(
        "INSERT INTO concept_relationship VALUES (?, ?, ?, CAST(? AS DATE), CAST(? AS DATE), ?)", relationships
    )

    con.close()

    # 2. Create LanceDB Artifact
    # We will embed the 'concept_name' for search
    lance_db = lancedb.connect(str(pack_dir))

    vectors_data = []
    for cid, cname, _, _, _, _, _, _ in concepts:
        vec = mock_embedder.embed(cname)
        vectors_data.append({"concept_id": cid, "concept_name": cname, "vector": vec})

    # Create table 'vectors' to match expected directory 'vectors.lance'
    lance_db.create_table("vectors", data=vectors_data, mode="overwrite")

    # 3. Create Manifest

    def compute_file_hash(p: Path) -> str:
        sha256 = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(4096), b""):
                sha256.update(b)
        return sha256.hexdigest()

    def compute_dir_hash(p: Path) -> str:
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
            sha256.update(compute_file_hash(full).encode("utf-8"))
        return sha256.hexdigest()

    manifest_data: Dict[str, Any] = {"version": "vTest_Q1", "source_date": "2025-01-01", "checksums": {}}

    manifest_data["checksums"]["vocab.duckdb"] = compute_file_hash(db_path)

    vectors_path = pack_dir / "vectors.lance"
    if vectors_path.exists():
        manifest_data["checksums"]["vectors.lance"] = compute_dir_hash(vectors_path)

    manifest_path = pack_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)

    yield pack_dir


@pytest.fixture
def source_csvs(tmp_path: Path) -> Path:
    src = tmp_path / "athena_src"
    src.mkdir()

    # Create CONCEPT.csv
    # Headers: concept_id,concept_name,domain_id,vocabulary_id,
    # concept_class_id,standard_concept,concept_code,invalid_reason
    with open(src / "CONCEPT.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "concept_id",
                "concept_name",
                "domain_id",
                "vocabulary_id",
                "concept_class_id",
                "standard_concept",
                "concept_code",
                "invalid_reason",
            ]
        )
        writer.writerow(
            [312327, "Acute myocardial infarction", "Condition", "SNOMED", "Clinical Finding", "S", "22298006", ""]
        )
        writer.writerow([1503297, "Metformin", "Drug", "RxNorm", "Ingredient", "S", "6809", ""])

    # Create CONCEPT_ANCESTOR.csv
    with open(src / "CONCEPT_ANCESTOR.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["ancestor_concept_id", "descendant_concept_id", "min_levels_of_separation", "max_levels_of_separation"]
        )
        writer.writerow([312327, 312327, 0, 0])

    # Create CONCEPT_RELATIONSHIP.csv
    with open(src / "CONCEPT_RELATIONSHIP.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["concept_id_1", "concept_id_2", "relationship_id", "valid_start_date", "valid_end_date", "invalid_reason"]
        )
        writer.writerow([312327, 312327, "Maps to", "1970-01-01", "2099-12-31", ""])

    return src
