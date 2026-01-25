# Setup & Deployment Guide

This guide provides step-by-step instructions for deploying the `coreason-codex` Terminology Server. It is designed for developers and data scientists who need to integrate the service into their infrastructure.

## Context & Architecture

`coreason-codex` is a **Terminology Server** that normalizes free text (e.g., "Heart Attack") into Standard OMOP Concepts (e.g., ID 312327).

### The "Frozen Lake" Pattern
Unlike typical microservices, Codex does **not** ship with a pre-populated database. It uses a "Bring Your Own Data" model. You must build a read-only artifact called a **"Codex Pack"** (containing DuckDB and LanceDB files) before the server can start.

### Zero-Copy Design
The system is designed to run locally or in a container without loading millions of rows into RAM. It queries the Codex Pack on disk using DuckDB and LanceDB, enabling high performance with low memory footprint.

---

## 1. Prerequisites (The "Athena" Step)

!!! warning "CRITICAL: You MUST download the vocabulary data yourself"
    Due to licensing restrictions (specifically SNOMED CT and others), CoReason cannot distribute the raw vocabulary files. You must download them directly from OHDSI Athena.

### Step 1.1: Download Vocabulary Files
1.  Go to **[OHDSI Athena](https://athena.ohdsi.org)**.
2.  Log in (or create an account).
3.  Select the vocabularies you need (Standard recommendation: **SNOMED**, **RxNorm**, **LOINC**, **ICD10**, **RxNorm Extension**).
4.  Click **Download**.

### Step 1.2: Verify Required Files
When you unzip the download, ensure you have the following **exact CSV files**:

*   `CONCEPT.csv`
*   `CONCEPT_RELATIONSHIP.csv`
*   `CONCEPT_ANCESTOR.csv`
*   `CONCEPT_SYNONYM.csv`

Unzip these files into a source directory, for example: `./raw_athena_data`.

---

## 2. Installation

You can install `coreason-codex` using pip or poetry.

### Using Pip
```bash
pip install coreason-codex
```

### Using Poetry
```bash
poetry add coreason-codex
```

---

## 3. The Build Step (ETL)

Before the server can run, you must compile the raw CSVs into the optimized "Codex Pack".

Run the following command:

```bash
codex build --source ./raw_athena_data --output ./my_codex_pack_v1 --device cpu
```

**What this does:**
*   **`vocab.duckdb`**: Creates an optimized SQL database for concept lookups, hierarchy, and relationships.
*   **`vectors.lance`**: Generates vector embeddings for all concepts to enable semantic search.
*   **`manifest.json`**: Creates a manifest file with checksums for integrity verification.

---

## 4. Running the Service (Deployment)

You can run the service locally using the CLI or deploy it using Docker (recommended for production).

### Method A: Local CLI

Set the `CODEX_PACK_PATH` environment variable to point to your built pack, then start the server.

```bash
export CODEX_PACK_PATH=./my_codex_pack_v1
uvicorn coreason_codex.server:app --host 0.0.0.0 --port 8000
```

### Method B: Docker (Production)

This is the preferred method for production deployments.

1.  **Pull the image** (or build it):
    ```bash
    docker build -t coreason-codex .
    ```

2.  **Run the container**:
    **Crucial:** You must mount the volume containing your Codex Pack so the container can access it.

    ```bash
    docker run -d \
      -p 8000:8000 \
      -v $(pwd)/my_codex_pack_v1:/data/codex_pack \
      -e CODEX_PACK_PATH=/data/codex_pack \
      --name codex-server \
      coreason-codex
    ```

---

## 5. Verification

Once the server is running, you can verify it using `curl`.

### Health Check
```bash
curl http://localhost:8000/health
# Expected: {"status": "ready"}
```

### Normalize Text
Test the normalization engine with a sample query:

```bash
curl -X POST "http://localhost:8000/normalize" \
     -H "Content-Type: application/json" \
     -d '{"text": "Severe hepatic impairment"}'
```

**Expected Response:**
```json
[
  {
    "input_text": "Severe hepatic impairment",
    "match_concept": {
      "concept_id": 43530807,
      "concept_name": "Severe hepatic impairment",
       ...
    },
    "similarity_score": 0.98,
    ...
  }
]
```
