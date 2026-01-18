# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_codex

from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel


class Concept(BaseModel):
    concept_id: int
    concept_name: str
    domain_id: str
    vocabulary_id: str
    concept_class_id: str
    standard_concept: Optional[str] = None
    concept_code: str

    @classmethod
    def from_row(cls, row: Tuple[Any, ...]) -> "Concept":
        """
        Creates a Concept instance from a DuckDB row tuple.
        Assumes row order: id, name, domain, vocab, class, standard, code.
        """
        return cls(
            concept_id=row[0],
            concept_name=row[1],
            domain_id=row[2],
            vocabulary_id=row[3],
            concept_class_id=row[4],
            standard_concept=row[5],
            concept_code=row[6],
        )


class CodexMatch(BaseModel):
    input_text: str
    match_concept: Concept
    similarity_score: float
    is_standard: bool
    mapped_standard_id: Optional[int] = None


class Manifest(BaseModel):
    version: str
    source_date: str
    checksums: Dict[str, str]
