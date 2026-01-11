# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_codex

import math

import pytest
from pydantic import ValidationError

from coreason_codex.models import CodexMatch, Concept


def test_concept_instantiation() -> None:
    """Test valid creation of a Concept."""
    concept = Concept(
        concept_id=312327,
        concept_name="Acute myocardial infarction",
        domain_id="Condition",
        vocabulary_id="SNOMED",
        concept_class_id="Clinical Finding",
        standard_concept="S",
        concept_code="22298006",
    )
    assert concept.concept_id == 312327
    assert concept.concept_name == "Acute myocardial infarction"
    assert concept.standard_concept == "S"


def test_concept_defaults() -> None:
    """Test defaults for optional fields in Concept."""
    concept = Concept(
        concept_id=123,
        concept_name="Test",
        domain_id="Test",
        vocabulary_id="Test",
        concept_class_id="Test",
        concept_code="123",
    )
    assert concept.standard_concept is None


def test_codex_match_instantiation() -> None:
    """Test valid creation of a CodexMatch."""
    concept = Concept(
        concept_id=31967,
        concept_name="Nausea",
        domain_id="Condition",
        vocabulary_id="SNOMED",
        concept_class_id="Clinical Finding",
        standard_concept="S",
        concept_code="422587007",
    )
    match = CodexMatch(
        input_text="tummy ache",
        match_concept=concept,
        similarity_score=0.95,
        is_standard=True,
        mapped_standard_id=31967,
    )
    assert match.input_text == "tummy ache"
    assert match.match_concept.concept_id == 31967
    assert match.similarity_score == 0.95
    assert match.is_standard is True
    assert match.mapped_standard_id == 31967


def test_concept_validation_error() -> None:
    """Test that Concept requires an integer for concept_id."""
    with pytest.raises(ValidationError):
        Concept(
            concept_id="not-an-integer",  # type: ignore[arg-type] # noqa: F821
            concept_name="Test",
            domain_id="Test",
            vocabulary_id="Test",
            concept_class_id="Test",
            concept_code="123",
        )


def test_codex_match_defaults() -> None:
    """Test defaults for optional fields in CodexMatch."""
    concept = Concept(
        concept_id=123,
        concept_name="Test",
        domain_id="Test",
        vocabulary_id="Test",
        concept_class_id="Test",
        concept_code="123",
    )
    match = CodexMatch(
        input_text="test",
        match_concept=concept,
        similarity_score=1.0,
        is_standard=False,
    )
    assert match.mapped_standard_id is None


def test_concept_serialization() -> None:
    """Test JSON roundtrip serialization for Concept."""
    concept = Concept(
        concept_id=999,
        concept_name="Serialization Test",
        domain_id="Test",
        vocabulary_id="Test",
        concept_class_id="Test",
        standard_concept="C",
        concept_code="999",
    )
    json_data = concept.model_dump_json()
    restored_concept = Concept.model_validate_json(json_data)
    assert restored_concept == concept
    assert restored_concept.concept_id == 999


def test_codex_match_nested_validation() -> None:
    """Test that CodexMatch validation fails if the nested concept is invalid."""
    with pytest.raises(ValidationError):
        CodexMatch(
            input_text="test",
            match_concept={  # type: ignore[arg-type] # noqa: F821
                "concept_id": "invalid",  # Not an int
                "concept_name": "Test",
            },
            similarity_score=0.5,
            is_standard=True,
        )


def test_concept_boundary_values() -> None:
    """Test Concept with boundary values like large integers and empty strings."""
    large_int = 2**63 - 1
    concept = Concept(
        concept_id=large_int,
        concept_name="",  # Empty name allowed
        domain_id="",
        vocabulary_id="",
        concept_class_id="",
        concept_code="",
    )
    assert concept.concept_id == large_int
    assert concept.concept_name == ""


def test_codex_match_complex_scenario() -> None:
    """
    Test a complex scenario: Non-standard source code mapping to a Standard Concept.
    Example: ICD-10 'R11' (Nausea) -> Standard SNOMED 'Nausea'.
    """
    # 1. Define the Standard Concept (Target)
    standard_concept = Concept(
        concept_id=31967,
        concept_name="Nausea",
        domain_id="Condition",
        vocabulary_id="SNOMED",
        concept_class_id="Clinical Finding",
        standard_concept="S",
        concept_code="422587007",
    )

    # 2. Define the Match Result (Matched against the Standard Concept)
    # Even though the input was likely "R11", the system matched it to the Standard Concept directly
    # or the system matched "R11" (Non-standard) and then mapped it.
    # The CodexMatch object represents the *result* of the lookup.

    # Scenario: The user searched for "R11", found the ICD-10 concept (Non-Standard),
    # and the system mapped it to SNOMED.

    icd10_concept = Concept(
        concept_id=12345,
        concept_name="Nausea and vomiting",
        domain_id="Condition",
        vocabulary_id="ICD10",
        concept_class_id="ICD10 Code",
        standard_concept=None,  # Non-standard
        concept_code="R11",
    )

    match = CodexMatch(
        input_text="R11",
        match_concept=icd10_concept,  # The immediate match
        similarity_score=1.0,  # Exact code match
        is_standard=False,
        mapped_standard_id=standard_concept.concept_id,  # The map
    )

    assert match.is_standard is False
    assert match.mapped_standard_id == 31967
    assert match.match_concept.vocabulary_id == "ICD10"


def test_unicode_support() -> None:
    """Test handling of Unicode characters in string fields."""
    concept = Concept(
        concept_id=1,
        concept_name="Naïve T-cell β-receptor",
        domain_id="Spec®ial",
        vocabulary_id="Test",
        concept_class_id="Test",
        concept_code="µ-123",
    )
    assert concept.concept_name == "Naïve T-cell β-receptor"
    assert concept.domain_id == "Spec®ial"


def test_nan_infinity_similarity() -> None:
    """Test that similarity_score accepts NaN and Infinity."""
    concept = Concept(
        concept_id=1,
        concept_name="Test",
        domain_id="Test",
        vocabulary_id="Test",
        concept_class_id="Test",
        concept_code="Test",
    )
    # Test NaN
    match_nan = CodexMatch(
        input_text="test",
        match_concept=concept,
        similarity_score=float("nan"),
        is_standard=False,
    )
    assert math.isnan(match_nan.similarity_score)

    # Test Infinity
    match_inf = CodexMatch(
        input_text="test",
        match_concept=concept,
        similarity_score=float("inf"),
        is_standard=False,
    )
    assert match_inf.similarity_score == float("inf")


def test_missing_fields() -> None:
    """Test that missing required fields raises ValidationError."""
    with pytest.raises(ValidationError) as excinfo:
        Concept(
            concept_id=1,
            # concept_name missing
            domain_id="Test",
            vocabulary_id="Test",
            concept_class_id="Test",
            concept_code="123",
        )  # type: ignore[call-arg]
    assert "concept_name" in str(excinfo.value)
    assert "Field required" in str(excinfo.value)
