# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_codex

from typing import Dict

from pydantic import BaseModel, Field


class Manifest(BaseModel):
    """
    Manifest for a Codex Pack.

    Contains metadata and integrity information (checksums) for the
    vocabulary and vector artifacts.
    """

    version: str = Field(..., description="Version of the Codex Pack (e.g., v2025_Q1)")
    source_date: str = Field(..., description="Date of the source data (e.g., 2025-01-01)")
    checksums: Dict[str, str] = Field(..., description="Map of filename to SHA-256 hash")
