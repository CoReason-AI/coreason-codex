# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_codex

from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest
from coreason_identity.models import UserContext
from coreason_identity.types import SecretStr

from coreason_codex.pipeline import (
    CodexContext,
    CodexPipeline,
    codex_check_relationship,
    codex_get_descendants,
    codex_normalize,
    codex_translate_code,
    initialize,
)
from coreason_codex.schemas import CodexMatch, Concept


# We mock CodexContext initialization to avoid real DB loading in unit tests
@pytest.fixture
def mock_context(synthetic_codex_pack: Any) -> Generator[Any, None, None]:
    # Mock Loader
    with patch("coreason_codex.pipeline.CodexLoader") as MockLoader:
        mock_loader_instance = MockLoader.return_value
        # Mock connections
        mock_duck = MagicMock()
        mock_lance = MagicMock()
        mock_loader_instance.load_codex.return_value = (mock_duck, mock_lance)

        # Mock Embedder (to avoid download)
        with patch("coreason_codex.pipeline.SapBertEmbedder"):
            # Initialize
            initialize(str(synthetic_codex_pack))

            yield CodexContext.get_instance()

            # Teardown
            CodexContext._instance = None


def test_pipeline_initialize(synthetic_codex_pack: Any) -> None:
    # Reset singleton
    CodexContext._instance = None

    with patch("coreason_codex.pipeline.CodexLoader") as MockLoader:
        # We must return mock connections from load_codex
        mock_instance = MockLoader.return_value
        mock_instance.load_codex.return_value = (MagicMock(), MagicMock())

        with patch("coreason_codex.pipeline.SapBertEmbedder"):
            initialize(str(synthetic_codex_pack))
            assert CodexContext._instance is not None


def test_codex_normalize_proxy(mock_context: Any) -> None:
    # normalize returns List[CodexMatch]
    expected_match = CodexMatch(
        input_text="test",
        match_concept=Concept(
            concept_id=1, concept_name="C", domain_id="D", vocabulary_id="V", concept_class_id="C", concept_code="1"
        ),
        similarity_score=1.0,
        is_standard=True,
    )
    mock_context.normalizer.normalize = MagicMock(return_value=[expected_match])
    res = codex_normalize("test")
    assert res == [expected_match]
    mock_context.normalizer.normalize.assert_called_with("test", domain_filter=None)


def test_codex_get_descendants_proxy(mock_context: Any) -> None:
    mock_context.hierarchy.get_descendants = MagicMock(return_value=[123])
    res = codex_get_descendants(1)
    assert res == [123]
    mock_context.hierarchy.get_descendants.assert_called_with(1)


def test_codex_translate_code_proxy(mock_context: Any) -> None:
    expected_concept = Concept(
        concept_id=1, concept_name="C", domain_id="D", vocabulary_id="V", concept_class_id="C", concept_code="1"
    )
    mock_context.crosswalker.translate_code = MagicMock(return_value=[expected_concept])
    res = codex_translate_code(1, target_vocabulary="SNOMED")
    assert res == [expected_concept]
    mock_context.crosswalker.translate_code.assert_called_with(1, target_vocabulary_id="SNOMED")


def test_codex_check_relationship_proxy(mock_context: Any) -> None:
    mock_context.crosswalker.check_relationship = MagicMock(return_value=True)
    res = codex_check_relationship(1, 2, "Rel")
    assert res is True
    mock_context.crosswalker.check_relationship.assert_called_with(1, 2, "Rel")


def test_codex_pipeline_run(mock_context: Any) -> None:
    # Test run method calls build
    with patch("coreason_codex.pipeline.CodexBuilder") as MockBuilder:
        mock_builder = MockBuilder.return_value
        context = UserContext(user_id=SecretStr("u"), roles=[])

        pipeline = CodexPipeline()
        pipeline.run(Path("s"), Path("o"), context=context)

        MockBuilder.assert_called_with(Path("s"), Path("o"))
        mock_builder.build_graph.assert_called_with(context=context, device="cpu")


def test_codex_pipeline_search(mock_context: Any) -> None:
    # Test search method calls normalizer
    context = UserContext(user_id=SecretStr("u"), roles=[])
    pipeline = CodexPipeline()

    with patch.object(mock_context.normalizer, "normalize", return_value=[]) as mock_norm:
        pipeline.search("query", k=5, context=context)
        mock_norm.assert_called_with("query", k=5, domain_filter=None)
