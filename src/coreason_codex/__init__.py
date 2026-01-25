# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason-codex

"""
coreason-codex
"""

__version__ = "0.3.0"
__author__ = "Gowtham A Rao"
__email__ = "gowtham.rao@coreason.ai"

from .crosswalker import CodexCrossWalker
from .hierarchy import CodexHierarchy
from .loader import CodexLoader
from .normalizer import CodexNormalizer
from .pipeline import (
    codex_check_relationship,
    codex_get_descendants,
    codex_normalize,
    codex_translate_code,
    initialize,
)

__all__ = [
    "CodexLoader",
    "CodexNormalizer",
    "CodexHierarchy",
    "CodexCrossWalker",
    "initialize",
    "codex_normalize",
    "codex_get_descendants",
    "codex_translate_code",
    "codex_check_relationship",
]
