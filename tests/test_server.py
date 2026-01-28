# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_codex

from typing import Generator, Tuple
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from coreason_codex.schemas import CodexMatch, Concept
from coreason_codex.server import app


# Mock CodexMatch object for responses
def create_mock_match(text: str, score: float = 0.9) -> CodexMatch:
    concept = Concept(
        concept_id=123,
        concept_name="Test Concept",
        domain_id="Condition",
        vocabulary_id="SNOMED",
        concept_class_id="Clinical Finding",
        standard_concept="S",
        concept_code="C123",
    )
    return CodexMatch(input_text=text, match_concept=concept, similarity_score=score, is_standard=True)


@pytest.fixture
def mock_codex_context() -> Generator[Tuple[MagicMock, MagicMock], None, None]:
    # Setup the mock instance
    mock_instance = MagicMock()
    mock_normalizer = MagicMock()
    mock_instance.normalizer = mock_normalizer

    # Patch in pipeline (for CodexPipeline)
    with patch("coreason_codex.pipeline.CodexContext") as mock_ctx_pipeline:
        mock_ctx_pipeline.get_instance.return_value = mock_instance

        # Patch in server (for lifespan)
        with patch("coreason_codex.server.CodexContext") as mock_ctx_server:
            mock_ctx_server.get_instance.return_value = mock_instance

            # We yield mock_ctx_server to match usage in tests that check initialize call in server
            yield mock_ctx_server, mock_normalizer


def test_health_check(mock_codex_context: Tuple[MagicMock, MagicMock]) -> None:
    """
    Test that health check works and verify lifespan startup.
    """
    mock_ctx_cls, _ = mock_codex_context
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ready"}

        # Verify initialization happened
        mock_ctx_cls.initialize.assert_called_once()


def test_normalize_endpoint(mock_codex_context: Tuple[MagicMock, MagicMock]) -> None:
    _, mock_normalizer = mock_codex_context
    # Mock return value
    expected_match = create_mock_match("input text")
    mock_normalizer.normalize.return_value = [expected_match]

    with TestClient(app) as client:
        response = client.post("/normalize", json={"text": "input text", "domain": "Condition"})

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["input_text"] == "input text"
        assert data[0]["match_concept"]["concept_id"] == 123

        # Verify call arguments
        mock_normalizer.normalize.assert_called_with("input text", domain_filter="Condition")


def test_search_endpoint(mock_codex_context: Tuple[MagicMock, MagicMock]) -> None:
    _, mock_normalizer = mock_codex_context
    # Mock return value
    expected_match = create_mock_match("query text")
    mock_normalizer.normalize.return_value = [expected_match]

    # Prepare request payload with both request and context
    # FastAPI expects embedded body when multiple body params are present
    payload = {
        "request": {"query": "query text", "limit": 5},
        "context": {
            "user_id": "test-user",
            "roles": ["tester"],
            "metadata": {"source": "test"},
        },
    }

    with TestClient(app) as client:
        response = client.post("/search", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1

        # Verify call arguments
        # Note: CodexPipeline.search passes domain_filter=None by default
        # And it uses named arguments
        mock_normalizer.normalize.assert_called_with("query text", k=5, domain_filter=None)


def test_startup_failure(mock_codex_context: Tuple[MagicMock, MagicMock]) -> None:
    mock_ctx_cls, _ = mock_codex_context
    mock_ctx_cls.initialize.side_effect = Exception("Startup failed")

    with pytest.raises(RuntimeError, match="Server initialization failed"):
        with TestClient(app):
            pass
