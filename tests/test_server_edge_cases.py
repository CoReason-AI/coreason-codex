# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_codex

import os
from typing import Generator, Tuple
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from coreason_codex.server import app


@pytest.fixture
def mock_codex_context() -> Generator[Tuple[MagicMock, MagicMock], None, None]:
    # Patch where CodexContext is imported in server.py
    with patch("coreason_codex.server.CodexContext") as mock_ctx:
        # Setup the mock instance
        mock_instance = MagicMock()
        mock_ctx.get_instance.return_value = mock_instance

        # Setup normalizer
        mock_normalizer = MagicMock()
        mock_instance.normalizer = mock_normalizer

        yield mock_ctx, mock_normalizer


def test_invalid_payload(mock_codex_context: Tuple[MagicMock, MagicMock]) -> None:
    """Test that missing required fields return 422."""
    with TestClient(app) as client:
        # Missing 'text'
        response = client.post("/normalize", json={"domain": "Condition"})
        assert response.status_code == 422

        # Missing 'query'
        response = client.post("/search", json={"limit": 5})
        assert response.status_code == 422


def test_empty_string_input(mock_codex_context: Tuple[MagicMock, MagicMock]) -> None:
    """Test that empty string input is handled (normalizer called)."""
    _, mock_normalizer = mock_codex_context
    mock_normalizer.normalize.return_value = []

    with TestClient(app) as client:
        response = client.post("/normalize", json={"text": ""})
        assert response.status_code == 200
        assert response.json() == []
        mock_normalizer.normalize.assert_called_with("", domain_filter=None)


def test_no_matches_found(mock_codex_context: Tuple[MagicMock, MagicMock]) -> None:
    """Test when the normalizer finds nothing."""
    _, mock_normalizer = mock_codex_context
    mock_normalizer.normalize.return_value = []

    with TestClient(app) as client:
        response = client.post("/normalize", json={"text": "unknown term"})
        assert response.status_code == 200
        assert response.json() == []


def test_internal_server_error(mock_codex_context: Tuple[MagicMock, MagicMock]) -> None:
    """Test that exceptions in the normalizer result in 500."""
    _, mock_normalizer = mock_codex_context
    mock_normalizer.normalize.side_effect = ValueError("Unexpected error")

    with TestClient(app) as client:
        # FastAPI handles exceptions by returning 500 if not handled
        with pytest.raises(ValueError):
            # TestClient raises the exception directly in the test process for unhandled app exceptions
            client.post("/normalize", json={"text": "crash me"})


def test_env_var_configuration(mock_codex_context: Tuple[MagicMock, MagicMock]) -> None:
    """Test that the server respects the CODEX_PACK_PATH env var."""
    mock_ctx_cls, _ = mock_codex_context
    custom_path = "/custom/path/to/pack"

    with patch.dict(os.environ, {"CODEX_PACK_PATH": custom_path}):
        with TestClient(app) as client:
            client.get("/health")

    mock_ctx_cls.initialize.assert_called_with(custom_path)


def test_redundant_workflow(mock_codex_context: Tuple[MagicMock, MagicMock]) -> None:
    """
    Complex redundant test:
    1. Check Health
    2. Perform Search
    3. Perform Normalization
    Verifies the state persists and works across requests.
    """
    _, mock_normalizer = mock_codex_context
    mock_normalizer.normalize.return_value = []

    with TestClient(app) as client:
        # 1. Health
        r1 = client.get("/health")
        assert r1.status_code == 200
        assert r1.json() == {"status": "ready"}

        # 2. Search
        r2 = client.post("/search", json={"query": "aspirin", "limit": 5})
        assert r2.status_code == 200
        assert isinstance(r2.json(), list)

        # 3. Normalize
        r3 = client.post("/normalize", json={"text": "aspirin"})
        assert r3.status_code == 200
        assert isinstance(r3.json(), list)
