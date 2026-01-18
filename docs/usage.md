# Usage Guide

This guide provides detailed instructions on how to use the `coreason-codex` library components.

## 1. Setup and Initialization

Before using any component, you must load the Codex Pack.

```python
from pathlib import Path
from coreason_codex.loader import CodexLoader

# Point to your Codex Pack directory
pack_path = Path("./codex_pack_v2025_Q1")

# Initialize the Loader
loader = CodexLoader(pack_path)

# Load the connections (verifies checksums automatically)
duckdb_conn, lancedb_conn = loader.load_codex()
```

## 2. Semantic Normalizer

The `CodexNormalizer` is used to map free text to Standard OMOP Concepts.

```python
from coreason_codex.normalizer import CodexNormalizer
from coreason_codex.embedders import SapBertEmbedder

# Initialize Embedder (SapBERT)
embedder = SapBertEmbedder()

# Initialize Normalizer
normalizer = CodexNormalizer(
    embedder=embedder,
    duckdb_conn=duckdb_conn,
    lancedb_conn=lancedb_conn
)

# Usage: Normalize text
matches = normalizer.normalize(
    text="patient felt queasy",
    k=5,                  # Top 5 results
    domain_filter=None    # Optional: "Condition", "Drug", etc.
)

for m in matches:
    print(f"Matched: {m.match_concept.concept_name} (ID: {m.match_concept.concept_id})")
    print(f"Standard: {m.is_standard}")
    if not m.is_standard and m.mapped_standard_id:
        print(f"Maps to Standard ID: {m.mapped_standard_id}")
```

## 3. Hierarchy Engine

The `CodexHierarchy` allows you to traverse the OMOP hierarchy, specifically to find descendants.

```python
from coreason_codex.hierarchy import CodexHierarchy

# Initialize Hierarchy
hierarchy = CodexHierarchy(duckdb_conn)

# Usage: Get all descendants of 'Statin' (Example ID: 1545958)
statin_id = 1545958
descendants = hierarchy.get_descendants(statin_id)

print(f"Found {len(descendants)} descendants for Statin.")
# descendants is a list of integers: [1545958, 1539403, ...]
```

## 4. Cross-Walker

The `CodexCrossWalker` handles translation between vocabularies and relationship checks.

```python
from coreason_codex.crosswalker import CodexCrossWalker

# Initialize CrossWalker
crosswalker = CodexCrossWalker(duckdb_conn)

# Usage A: Translate Code (e.g., SNOMED to ICD-10)
# Example: Translate 'Acute myocardial infarction' (312327) to ICD-10
icd10_concepts = crosswalker.translate_code(
    source_id=312327,
    target_vocabulary_id="ICD10",
    relationship_id="Maps to" # Note: direction matters, usually 'Maps to' or inverse
)

for concept in icd10_concepts:
    print(f"ICD-10 Code: {concept.concept_code} - {concept.concept_name}")


# Usage B: Check Relationship
# "Does Metformin (1503297) treat Diabetes (201820)?"
is_indicated = crosswalker.check_relationship(
    concept_id_1=1503297,
    concept_id_2=201820,
    relationship_id="Indication - Drug" # Requires specific relationship ID from OMOP
)

print(f"Metformin treats Diabetes: {is_indicated}")
```
