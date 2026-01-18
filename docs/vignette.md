# Vignettes & User Stories

This section demonstrates how `coreason-codex` is used in real-world scenarios to solve complex biomedical terminology challenges.

## Story A: The "Semantic Tagging" (Ingestion)

**Context:** An ingestion pipeline is processing a clinical trial protocol PDF and encounters the text: *"Exclusion Criteria: Patients with severe hepatic impairment."*

**Goal:** Tag the text chunk with precise, machine-readable concept IDs.

**Codex Action:**
1.  **Vector Search:** The system embeds "severe hepatic impairment".
2.  **Match:** `CodexNormalizer` finds the nearest neighbor.
    *   **Result:** `ConceptID: 4066237` ("Severe liver failure").

**Outcome:**
The system tags the chunk with `metadata: { exclude_concept_id: 4066237 }`. Later, when a researcher searches for "Liver Failure", this document is retrieved with 100% recall, even though the exact words "Liver Failure" never appeared in the text.

```python
# Code Representation
results = normalizer.normalize("severe hepatic impairment")
top_match = results[0]

print(f"Tagging chunk with Concept ID: {top_match.match_concept.concept_id}")
# Output: Tagging chunk with Concept ID: 4066237
```

## Story B: The "Lateral Logic" (Reasoning)

**Context:** An AI Agent is validating a medical claim and needs to verify if a drug is appropriate for a condition.

**Query:** *"Does Metformin treat Diabetes?"*

**Codex Action:**
1.  **Lookup:** Agent resolves "Metformin" to `ConceptID: 1503297`.
2.  **Lookup:** Agent resolves "Diabetes" to `ConceptID: 201820`.
3.  **Check:** Agent uses `CodexCrossWalker` to check for a relationship.

**Outcome:**
The system queries the `CONCEPT_RELATIONSHIP` table and confirms an "Indication - Drug" link exists. The Agent can confidentially answer "Yes" based on ground-truth medical ontology, reducing hallucinations.

```python
# Code Representation
is_valid = crosswalker.check_relationship(
    concept_id_1=1503297, # Metformin
    concept_id_2=201820,  # Diabetes
    relationship_id="Indication - Drug"
)

if is_valid:
    print("Verification Successful: Relationship confirmed.")
```

## Story C: The "Audit Replay" (Versioning)

**Context:** An FDA Auditor asks the organization to reproduce a specific search result from a study conducted in 2023.

**Challenge:** Medical vocabularies change over time. Concepts are deprecated, and new ones are added. A live API might give different results today.

**Codex Action:**
1.  **Load Legacy Artifact:** The system initializes `CodexLoader` pointing to the `/vocab_2023_q4/` artifact pack (the specific version used in the study).
2.  **Execute:** The search is re-run against this immutable artifact.

**Outcome:**
The search yields the **exact same results** as it did in 2023. The "Frozen Lake" architecture ensures complete reproducibility for regulatory compliance.

```python
# Code Representation
# Pointing to the specific archived version
archive_path = Path("/archives/codex_pack_2023_q4")
loader = CodexLoader(archive_path)
loader.verify_integrity() # Ensures the data hasn't been tampered with since 2023
```
