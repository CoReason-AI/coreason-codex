from pathlib import Path
from unittest.mock import patch

import pytest
from coreason_identity.models import UserContext

from coreason_codex.build import CodexBuilder


def test_build_graph_success(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()

    # Create required files to pass _verify_source_files if it weren't mocked
    # But we will mock the internal calls so we don't need real files if we mock build_vocab

    builder = CodexBuilder(source, output)
    context = UserContext(user_id="test", email="t@c.ai", groups=[], scopes=[], claims={})

    # Mock internal methods to avoid real heavy lifting
    with patch.object(builder, "build_vocab") as mock_vocab:
        with patch.object(builder, "build_vectors") as mock_vectors:
            with patch.object(builder, "generate_manifest") as mock_manifest:
                # Patch the module where SapBertEmbedder is defined, OR where it is imported.
                # In build.py, it is imported INSIDE the function:
                # `from coreason_codex.embedders import SapBertEmbedder`
                # So we must patch `coreason_codex.embedders.SapBertEmbedder`
                with patch("coreason_codex.embedders.SapBertEmbedder") as MockEmbedder:
                    builder.build_graph(context=context, device="cpu")

                    mock_vocab.assert_called_once()
                    mock_vectors.assert_called_once()
                    mock_manifest.assert_called_once()
                    MockEmbedder.assert_called_once_with(device="cpu")


def test_build_graph_missing_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    # Missing required files

    builder = CodexBuilder(source, output)
    context = UserContext(user_id="test", email="t@c.ai", groups=[], scopes=[], claims={})

    with pytest.raises(FileNotFoundError):
        builder.build_graph(context=context)


def test_build_graph_no_source_dir(tmp_path: Path) -> None:
    source = tmp_path / "missing_source"
    output = tmp_path / "output"

    builder = CodexBuilder(source, output)
    context = UserContext(user_id="test", email="t@c.ai", groups=[], scopes=[], claims={})

    with pytest.raises(FileNotFoundError):
        builder.build_graph(context=context)


def test_build_graph_no_context(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"

    builder = CodexBuilder(source, output)

    with pytest.raises(ValueError, match="UserContext is required"):
        builder.build_graph(context=None)
