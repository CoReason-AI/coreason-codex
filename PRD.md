# **Product Requirements Document: coreason-codex**

Domain: Ontology Management, Medical Coding, & Semantic Interoperability
Architectural Role: The "Universal Translator" / The Terminology Server
Core Philosophy: "Text is ambiguous. Codes are absolute. All roads lead to Standard Concepts."
Dependencies: coreason-refinery (Consumer), coreason-graph-nexus (Consumer), duckdb (OLAP Storage), lancedb (Vector Storage), mcp (Protocol)

## ---

**1\. Executive Summary**

coreason-codex is the centralized **Terminology Server** for the platform. It solves the "Semantic Precision Gap" in Bio-Pharma AI by enforcing the use of **Standardized Vocabularies** (OMOP CDM).

While Large Language Models are fluent, they are imprecise (e.g., treating "Heart Attack" and "Myocardial Infarction" as strings rather than identical entities). coreason-codex acts as an **MCP Server** that provides tools for Agents to lookup, validate, and translate medical concepts. It ensures that when an Agent reads "Heart Attack" (Text), it records it as ConceptID: 312327 (Data), enabling precise retrieval, graph grounding, and regulatory reporting.

## **2\. Functional Philosophy**

The agent must implement the **Ingest-Standardize-Serve Loop**:

1. **The "Frozen Lake" Pattern (GxP Compliance):**
   * To ensure reproducibility, codex does **not** connect to a live, mutable database.
   * It loads an **Immutable Vocabulary Artifact** (a "Codex Pack" containing DuckDB \+ LanceDB files) versioned by release date (e.g., v2025\_Q1).
   * This allows a clinical study to be re-run 5 years later using the *exact* terminology definitions active at the time.
2. **OMOP Standardization:**
   * We adopt the **OHDSI OMOP Common Data Model**.
   * **Standard Concepts:** All source terms (ICD-10, proprietary lab codes) map to a single "Standard Concept" (usually SNOMED/RxNorm).
   * **Hierarchy:** We utilize the pre-computed CONCEPT\_ANCESTOR table to enable O(1) semantic expansion (e.g., "Find all descendants of 'Statin'").
3. **Hybrid Search (SOTA):**
   * **Vector:** Embeds concept synonyms using **SapBERT** (SOTA for medical entities) to handle messy text input ("tummy ache").
   * **Exact:** Uses SQL indices for precise code lookups ("I21.9").

## ---

**3\. Core Functional Requirements (Component Level)**

### **3.1 The Artifact Loader (The Bootstrapper)**

**Concept:** Spins up the Read-Only Data Engine.

* **Input:** A path to a **Codex Pack** directory containing:
  * vocab.duckdb (The SQL Relations: Concept, Relationship, Ancestor).
  * vectors.lance (The Embeddings: Synonym Vectors).
  * manifest.json (Meta: Version, Checksum, Source Date).
* **Mechanism:**
  * Mounts the DuckDB file in READ\_ONLY mode via the Python API.
  * Initializes the LanceDB connection via PyArrow.
* **Validation:** Verifies the SHA-256 hash of the artifacts against the manifest. If the hash mismatches, the server refuses to start (Data Integrity).

### **3.2 The Semantic Normalizer (The Mapper)**

**Concept:** Maps free text to Standard Concept IDs.

* **Logic:**
  1. **Vector Search:** Input text ("Patient felt queasy") is embedded via SapBERT.
  2. **LanceDB Query:** Finds nearest neighbors in the synonym vector space. Matches "Nausea" (ConceptID: 31967).
  3. **Domain Filtering:** Optional filter by Domain (Condition, Drug, Measurement) to prevent homonym errors (e.g., "Cold" the temperature vs. "Cold" the virus).
* **MCP Tool:** codex\_normalize(text: str, domain\_filter: Optional\[str\]).

### **3.3 The Hierarchy Engine (The Ancestor)**

**Concept:** Exploits the pre-computed OMOP hierarchy for reasoning.

* **Logic:** Uses DuckDB to query the CONCEPT\_ANCESTOR table.
* **Use Case:**
  * **Query Expansion:** User asks for "Cardiac Events." Cortex calls codex\_get\_descendants(321588). Codex returns 50,000 specific IDs (MI, Atrial Fib, etc.). Cortex uses these IDs to filter chunks in coreason-refinery.
* **MCP Tool:** codex\_get\_descendants(concept\_id: int).

### **3.4 The Cross-Walker (The Translator)**

**Concept:** Maps between vocabularies for regulatory reporting.

* **Logic:** Uses the CONCEPT\_RELATIONSHIP table.
  * *Path:* SNOMED ID $\\to$ Maps to (inverse) $\\to$ ICD-10 Code.
* **MCP Tool:** codex\_translate\_code(source\_id: int, target\_vocabulary: str).

## ---

**4\. Integration Requirements**

* **coreason-mcp (The Host):**
  * codex is designed as a Python package that coreason-mcp imports. It exposes its functions as MCP Tools to the Cortex.
* **coreason-refinery (The Tagger):**
  * During document ingestion, Refinery calls codex.normalize() on extracted entities.
  * It tags the vector chunks with concept\_id: \[123, 456\] alongside the text. This enables "Concept-Based Retrieval" (100% Recall) rather than just "Keyword Retrieval."
* **coreason-nexus (The Graph):**
  * The Nodes in the Knowledge Graph utilize Codex IDs as their primary keys.
  * (:Patient)-\[:HAS\_CONDITION\]-\>(:Condition {id: 312327, label: "Acute MI"}).

## ---

**5\. User Stories**

### **Story A: The "Semantic Tagging" (Ingestion)**

Context: Ingesting a clinical trial protocol PDF.
Text: "Exclusion Criteria: Patients with severe hepatic impairment."
Codex Action:

1. Vector Search "severe hepatic impairment".
2. Matches ConceptID: 4066237 (Severe liver failure).
   Result: The chunk is tagged with metadata exclude\_concept\_id: 4066237\.

### **Story B: The "Lateral Logic" (Reasoning)**

Context: Agent needs to validate a medical claim.
Query: "Does Metformin treat Diabetes?"
Codex Action:

1. Lookup Metformin (Concept: 1503297).
2. Lookup Diabetes (Concept: 201820).
3. Check CONCEPT\_RELATIONSHIP table for an "Indication \- Drug" link between these IDs.
   Result: Returns True (Link exists).

### **Story C: The "Audit Replay" (Versioning)**

Context: FDA Auditor asks to reproduce a search from 2023\.
Action: The system spins up codex pointing to the /vocab\_2023\_q4/ artifact pack.
Result: The search yields the exact same results as it did in 2023, even though the medical dictionary has changed since then.

## ---

**6\. Data Schema**

### **Concept (OMOP Standard)**

Python

class Concept(BaseModel):
    concept\_id: int             \# e.g., 312327
    concept\_name: str           \# "Acute myocardial infarction"
    domain\_id: str              \# "Condition"
    vocabulary\_id: str          \# "SNOMED"
    concept\_class\_id: str       \# "Clinical Finding"
    standard\_concept: Optional\[str\] \# "S" (Standard) or NULL (Non-standard)
    concept\_code: str           \# "22298006" (Original source code)

### **CodexMatch**

Python

class CodexMatch(BaseModel):
    input\_text: str             \# "tummy ache"
    match\_concept: Concept
    similarity\_score: float     \# 0.95
    is\_standard: bool           \# False (Tylenol is non-standard)
    mapped\_standard\_id: Optional\[int\] \# 1125315 (Acetaminophen)

## ---

**7\. Implementation Directives for the Coding Agent**

1. **Zero-Copy Architecture:**
   * Use **DuckDB**'s ability to query Parquet/DB files directly on disk. Do **NOT** load the millions of vocabulary rows into Python/Pandas RAM.
   * Use **LanceDB** for vector storage; it is optimized for local SSD usage and zero-copy reads via Apache Arrow.
2. **Offline Builder Script:**
   * Create a separate utility module codex.build. This script takes raw Athena CSVs (downloaded by the user) and compiles the .duckdb and .lance artifacts. This is an ETL step, not a runtime step.
3. **Embeddings (SOTA):**
   * Use cambridgeltl/SapBERT-from-PubMedBERT-fulltext via sentence-transformers. This model is specifically fine-tuned for medical entity alignment and outperforms generic BERT/OpenAI embeddings for this task.
4. **Strict Typing:**
   * Concept IDs in OMOP are **Integers**. Ensure Pydantic models enforce int types to prevent "stringly typed" errors during graph joins.