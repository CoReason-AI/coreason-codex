# System Architecture

`coreason-codex` is designed as a high-performance Terminology Server for Bio-Pharma AI. Its primary architectural goal is to serve as a reliable "Universal Translator" that converts ambiguous text into precise, standardized medical concepts (OMOP CDM).

## Core Philosophies

### 1. The "Frozen Lake" Pattern (GxP Compliance)

To ensure reproducibility and GxP compliance, `codex` operates on an **Immutable Vocabulary Artifact** (a "Codex Pack").

*   **No Live Database:** Unlike traditional systems that might connect to a mutable SQL server, `codex` loads versioned artifacts.
*   **Reproducibility:** A clinical study can be re-run years later using the exact same version of the vocabulary (e.g., `v2025_Q1`) to produce identical results.
*   **Auditability:** Artifacts are cryptographically verified (SHA-256) upon loading.

### 2. Zero-Copy Architecture

Efficiency is paramount. We avoid loading millions of rows into Python memory.

*   **DuckDB:** Used for OLAP storage of relational data (Concepts, Relationships, Ancestors). It queries Parquet/DB files directly on disk.
*   **LanceDB:** Used for vector storage. It is optimized for local SSD usage and zero-copy reads via Apache Arrow.

### 3. OMOP Standardization

We adopt the **OHDSI OMOP Common Data Model** as the lingua franca.

*   **Standard Concepts:** All diverse source vocabularies map to Standard Concepts.
*   **Hierarchy:** We leverage the `CONCEPT_ANCESTOR` table for O(1) semantic expansion.

## Component Roles

The system is composed of four main components:

### 1. Artifact Loader (`loader.py`)
*   **Role:** The Bootstrapper.
*   **Responsibility:** Mounts the read-only data engine.
*   **Action:** Verifies the manifest and checksums of the Codex Pack, then initializes DuckDB and LanceDB connections.

### 2. Semantic Normalizer (`normalizer.py`)
*   **Role:** The Mapper.
*   **Responsibility:** Maps free text to Standard Concept IDs.
*   **Action:**
    1.  Embeds text using `SapBERT`.
    2.  Performs vector search in LanceDB.
    3.  Hydrates concept details from DuckDB.
    4.  Optionally filters by Domain.

### 3. Hierarchy Engine (`hierarchy.py`)
*   **Role:** The Ancestor.
*   **Responsibility:** Exploits the OMOP hierarchy for reasoning.
*   **Action:** Queries the `CONCEPT_ANCESTOR` table to find descendants (e.g., "Find all forms of Heart Failure").

### 4. Cross-Walker (`crosswalker.py`)
*   **Role:** The Translator.
*   **Responsibility:** Maps between vocabularies and validates relationships.
*   **Action:** Uses the `CONCEPT_RELATIONSHIP` table to translate codes (e.g., ICD-10 to SNOMED) or check logical links (e.g., "Does Drug A treat Condition B?").
