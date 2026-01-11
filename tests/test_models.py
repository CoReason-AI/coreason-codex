# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_codex

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
            concept_id="not-an-integer",  # type: ignore
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
