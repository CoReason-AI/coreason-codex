# coreason-codex

**The Terminology Server for Bio-Pharma AI**

`coreason-codex` acts as the "Universal Translator" for the platform. It bridges the "Semantic Precision Gap" in Bio-Pharma AI by enforcing the use of **Standardized Vocabularies** (OMOP CDM).

[![CI](https://github.com/CoReason-AI/coreason_codex/actions/workflows/ci.yml/badge.svg)](https://github.com/CoReason-AI/coreason_codex/actions/workflows/ci.yml)

## Executive Summary

While Large Language Models are fluent, they can be imprecise. `coreason-codex` ensures that when an Agent reads "Heart Attack", it records it as `ConceptID: 312327` (Data), enabling precise retrieval, graph grounding, and regulatory reporting.

It provides tools for Agents to lookup, validate, and translate medical concepts using a "Frozen Lake" pattern for GxP compliance.

For detailed requirements, see the [Product Requirements Document](PRD.md).

## Getting Started

### Prerequisites

- Python 3.12+
- Poetry

### Installation

1.  Clone the repository:
    ```sh
    git clone https://github.com/CoReason-AI/coreason_codex.git
    cd coreason_codex
    ```
2.  Install dependencies:
    ```sh
    poetry install
    ```

### Usage

Here is a quick example of how to use `coreason-codex` to normalize text to a standard concept.

```python
from pathlib import Path
from coreason_codex.loader import CodexLoader
from coreason_codex.normalizer import CodexNormalizer
from coreason_codex.embedders import SapBertEmbedder

# 1. Initialize Loader with path to your Codex Pack
# Ensure you have a valid Codex Pack at this location
pack_path = Path("./codex_pack_v1")
loader = CodexLoader(pack_path)
duckdb_conn, lancedb_conn = loader.load_codex()

# 2. Initialize Embedder and Normalizer
embedder = SapBertEmbedder() # Uses cambridgeltl/SapBERT-from-PubMedBERT-fulltext
normalizer = CodexNormalizer(embedder, duckdb_conn, lancedb_conn)

# 3. Normalize Text
matches = normalizer.normalize("Heart Attack")

for match in matches:
    print(f"Concept: {match.match_concept.concept_name} (ID: {match.match_concept.concept_id})")
    print(f"Score: {match.similarity_score}")
```

## Documentation

Detailed documentation is available in the `docs/` directory:

- [Architecture](docs/architecture.md): Overview of the system design, including the Frozen Lake pattern and Zero-Copy architecture.
- [Usage Guide](docs/usage.md): Detailed instructions on using the Loader, Normalizer, Hierarchy, and CrossWalker components.
- [Vignettes](docs/vignette.md): Walkthroughs of key user stories (Semantic Tagging, Lateral Logic, Audit Replay).

## Development

-   Run the linter:
    ```sh
    poetry run pre-commit run --all-files
    ```
-   Run the tests:
    ```sh
    poetry run pytest
    ```
