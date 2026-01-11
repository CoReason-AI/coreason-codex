# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_codex

from typing import Optional

from pydantic import BaseModel


class Concept(BaseModel):
    """
    OMOP Standard Concept.

    Represents a single concept in the OMOP Common Data Model.
    """

    concept_id: int
    concept_name: str
    domain_id: str
    vocabulary_id: str
    concept_class_id: str
    standard_concept: Optional[str] = None
    concept_code: str


class CodexMatch(BaseModel):
    """
    Result of a semantic search or exact match against the Codex.

    Contains the original input text, the matched concept, similarity score,
    and information about whether the match is a standard concept.
    """

    input_text: str
    match_concept: Concept
    similarity_score: float
    is_standard: bool
    mapped_standard_id: Optional[int] = None
