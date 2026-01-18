# The Architecture and Utility of coreason-codex

### 1. The Philosophy (The Why)

In the high-stakes domain of Bio-Pharma AI, fluency is not enough; precision is paramount. A Large Language Model (LLM) might understand that "Heart Attack" and "Myocardial Infarction" are similar, but in a regulatory context, they must be recognized as the exact same entity—ConceptID: 312327. This is the **"Semantic Precision Gap"** that `coreason-codex` exists to bridge.

The core insight driving this package is simple but profound: **"Text is ambiguous. Codes are absolute. All roads lead to Standard Concepts."**

Existing terminology servers are often live, mutable databases. If a medical dictionary updates, your study's results might silently change. `coreason-codex` rejects this volatility in favor of a **"Frozen Lake" architectural pattern**. It treats vocabulary data as **Immutable Artifacts** ("Codex Packs"). A specific version of the vocabulary (e.g., `v2025_Q1`) is a static, cryptographically verified asset. This ensures strict **GxP compliance** and **reproducibility**: a clinical query run five years from now using the same artifact will yield the exact same results, bit-for-bit.

### 2. Under the Hood (The Dependencies & logic)

The package constructs a high-performance, zero-copy architecture by leveraging a modern "embedded" data stack:

*   **DuckDB (OLAP Storage):** Instead of loading millions of vocabulary rows into Python memory (which is slow and expensive), `coreason-codex` uses DuckDB to query relational data (`CONCEPT`, `CONCEPT_ANCESTOR`) directly on disk. This enables O(1) lookups and complex joins over massive datasets with negligible RAM overhead.
*   **LanceDB (Vector Storage):** To handle the messy reality of clinical text ("patient felt queasy"), the system uses LanceDB for vector storage. It is optimized for local SSD random access, allowing for rapid similarity search without the latency of an external vector DB service.
*   **Sentence-Transformers (SapBERT):** The semantic engine is powered by `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`, a model fine-tuned specifically for medical entity alignment. This ensures that "tummy ache" maps correctly to "Abdominal Pain" rather than just looking for keyword overlaps.
*   **Pydantic:** Given the critical nature of the data, `coreason-codex` enforces strict type validation. Concept IDs are strictly integers, and models like `CodexMatch` and `Concept` ensure that data integrity is maintained throughout the pipeline.

**The Logic Flow:**
When initialized, the `CodexLoader` first performs a SHA-256 integrity check against the pack's `manifest.json`. Once verified, it establishes read-only connections to the DuckDB and LanceDB files. The `CodexNormalizer` then orchestrates the translation: it embeds input text via SapBERT, queries LanceDB for nearest neighbors, and "hydrates" the resulting IDs with rich metadata (Domain, Standard Status) from DuckDB. Finally, the `CodexHierarchy` module exploits pre-computed transitive closure tables to enable instant graph reasoning, such as finding all 50,000 specific conditions that fall under "Cardiac Disease."

### 3. In Practice (The How)

Using `coreason-codex` is designed to be seamless for Python developers, abstracting away the complexity of the underlying data engines.

```python
from coreason_codex.pipeline import (
    initialize,
    codex_normalize,
    codex_get_descendants
)

# 1. Initialize the Frozen Lake
# Point the system to a verified, immutable artifact pack.
# This mounts the DuckDB and LanceDB files in read-only mode.
initialize(pack_path="./vocab_2025_q1")

# 2. Semantic Normalization (Text -> Code)
# The system embeds "severe hepatic impairment", finds the nearest vector in LanceDB,
# and returns the Standard OMOP Concept.
matches = codex_normalize("Exclusion Criteria: severe hepatic impairment")

top_match = matches[0]
print(f"Match: {top_match.match_concept.concept_name}")
print(f"ID:    {top_match.match_concept.concept_id}")
print(f"Score: {top_match.similarity_score:.4f}")
# Output:
# Match: Severe liver failure
# ID:    4066237
# Score: 0.9850

# 3. Hierarchical Reasoning (Code -> Graph)
# Retrieve all specific sub-types of "Statin" (ConceptID: 1539403) using
# the pre-computed ancestor tables in DuckDB.
statin_descendants = codex_get_descendants(concept_id=1539403)

print(f"Found {len(statin_descendants)} specific Statin drugs.")
# Output: Found 1245 specific Statin drugs.
```

In this example, the developer interacts with a clean, functional API (`codex_normalize`, `codex_get_descendants`), while the system handles the complexities of vector embedding, approximate nearest neighbor search, and recursive SQL queries behind the scenes.
