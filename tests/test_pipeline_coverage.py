from pathlib import Path

import pytest

from coreason_codex.pipeline import CodexPipeline


def test_pipeline_run_no_context(tmp_path: Path) -> None:
    pipeline = CodexPipeline()
    with pytest.raises(ValueError, match="UserContext is required"):
        pipeline.run(tmp_path, tmp_path, context=None)


def test_pipeline_search_no_context() -> None:
    pipeline = CodexPipeline()
    with pytest.raises(ValueError, match="UserContext is required"):
        pipeline.search("query", context=None)
