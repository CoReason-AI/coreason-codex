# Welcome to coreason-codex

**coreason-codex** is the centralized Terminology Server for the platform. It bridges the gap between ambiguous free text and standardized medical concepts (OMOP CDM), serving as a foundational component for precise Bio-Pharma AI.

## Documentation Contents

*   **[System Architecture](architecture.md)**
    *   Understand the "Frozen Lake" pattern for GxP compliance.
    *   Learn about the Zero-Copy architecture using DuckDB and LanceDB.
    *   Explore the roles of the Loader, Normalizer, Hierarchy, and CrossWalker.

*   **[Usage Guide](usage.md)**
    *   Step-by-step instructions for initializing the system.
    *   Code examples for normalizing text, querying hierarchies, and translating codes.

*   **[Vignettes & User Stories](vignette.md)**
    *   See `coreason-codex` in action through real-world scenarios:
        *   **Story A:** Semantic Tagging of ingestion data.
        *   **Story B:** Logical reasoning and validation.
        *   **Story C:** Audit replay and reproducibility.

## Quick Start

```python
from pathlib import Path
from coreason_codex.loader import CodexLoader

# Initialize with your Codex Pack
loader = CodexLoader(Path("./codex_pack_v1"))
duckdb_conn, lancedb_conn = loader.load_codex()
```

For more details, please refer to the **[Usage Guide](usage.md)**.
