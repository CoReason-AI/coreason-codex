import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from coreason_codex.loader import load_repository, AccessDeniedError
from coreason_codex.main import Codex
from coreason_identity.models import UserContext
from coreason_codex.pipeline import CodexContext

@pytest.fixture
def mock_user_context():
    return UserContext(
        user_id="test_user",
        groups=["dev_team"],
        downstream_token="fake_token",
        email="test@example.com"
    )

@patch("subprocess.run")
def test_load_repository_success(mock_run, mock_user_context):
    mock_run.return_value.returncode = 0

    # Mock mkdtemp to return a path that exists (it does by default)
    # But subprocess clones into it.

    path = load_repository("https://github.com/org/repo.git", mock_user_context)

    assert path.exists()
    assert mock_run.called
    args = mock_run.call_args[0][0]
    # Check if token is injected.
    # load_repository constructs: -c http.extraHeader=Authorization: Bearer {token}
    expected_header = "http.extraHeader=Authorization: Bearer fake_token"
    assert expected_header in args
    assert "https://github.com/org/repo.git" in args

@patch("subprocess.run")
def test_load_repository_no_token(mock_run):
    # UserContext is frozen, so create one without token
    user_context = UserContext(
        user_id="test_user",
        email="test@example.com",
        groups=["dev_team"],
        downstream_token=None
    )
    with pytest.raises(AccessDeniedError):
        load_repository("https://github.com/org/repo.git", user_context)

@patch("coreason_codex.pipeline.load_repository")
@patch("coreason_codex.pipeline.CodexContext")
def test_index_repository(mock_ctx_cls, mock_load, mock_user_context, tmp_path):
    # Mock CodexContext instance
    mock_instance = MagicMock()
    mock_ctx_cls.get_instance.return_value = mock_instance
    mock_instance.lancedb_conn = MagicMock()
    # Mock embedder
    mock_instance.embedder.embed_batch.return_value = [[0.1]*768]

    # Mock load_repository to return a temp path with some files
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "test.py").write_text("print('hello')")
    mock_load.return_value = repo_dir

    result = Codex.index_repository("https://github.com/org/repo.git", mock_user_context)

    assert result["status"] == "indexed"
    mock_load.assert_called_with("https://github.com/org/repo.git", mock_user_context)

    # Check if create_table or open_table was called on lancedb
    db = mock_instance.lancedb_conn
    # It might call open_table or create_table
    assert db.open_table.called or db.create_table.called

@patch("coreason_codex.main.CodexContext")
def test_query_filter(mock_ctx_cls, mock_user_context):
    mock_instance = MagicMock()
    mock_ctx_cls.get_instance.return_value = mock_instance

    # Mock search results
    mock_instance.search_code.return_value = [
        {"text": "mine", "owner_id": "test_user", "allowed_groups": []},
        {"text": "others", "owner_id": "other_user", "allowed_groups": []},
        {"text": "shared", "owner_id": "other_user", "allowed_groups": ["dev_team"]},
        {"text": "secret", "owner_id": "other_user", "allowed_groups": ["admins"]},
    ]

    results = Codex.query("something", mock_user_context)

    assert len(results) == 2
    texts = [r["text"] for r in results]
    assert "mine" in texts
    assert "shared" in texts
    assert "others" not in texts
    assert "secret" not in texts
