# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_codex

import coreason_codex


def test_public_api_exposure() -> None:
    """
    Verify that the core functions and classes are exposed at the package level.
    """
    expected_symbols = [
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

    for symbol in expected_symbols:
        assert hasattr(coreason_codex, symbol), f"{symbol} not exposed in coreason_codex"


def test_initialize_callable() -> None:
    """Verify initialize is a callable function."""
    assert callable(coreason_codex.initialize)


def test_codex_normalize_callable() -> None:
    """Verify codex_normalize is a callable function."""
    assert callable(coreason_codex.codex_normalize)


def test_codex_get_descendants_callable() -> None:
    """Verify codex_get_descendants is a callable function."""
    assert callable(coreason_codex.codex_get_descendants)


def test_codex_translate_code_callable() -> None:
    """Verify codex_translate_code is a callable function."""
    assert callable(coreason_codex.codex_translate_code)


def test_codex_check_relationship_callable() -> None:
    """Verify codex_check_relationship is a callable function."""
    assert callable(coreason_codex.codex_check_relationship)


def test_version_exposure() -> None:
    """Verify version is exposed."""
    assert hasattr(coreason_codex, "__version__")
    assert isinstance(coreason_codex.__version__, str)
