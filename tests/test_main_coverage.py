from unittest.mock import patch

from typer.testing import CliRunner

from coreason_codex.main import app

runner = CliRunner()


def test_cli_serve_success(tmp_path) -> None:
    pack_dir = tmp_path / "codex_pack"
    pack_dir.mkdir()

    # Patch where uvicorn is imported in main.py
    with patch("uvicorn.run") as mock_run:
        with patch("coreason_codex.main.os.environ", {}) as mock_env:
            result = runner.invoke(app, ["serve", "--port", "9000", "--pack", str(pack_dir)])
            assert result.exit_code == 0
            mock_run.assert_called_once()
            assert mock_env["CODEX_PACK_PATH"] == str(pack_dir)


def test_cli_serve_failure(tmp_path) -> None:
    pack_dir = tmp_path / "codex_pack"
    pack_dir.mkdir()

    # Patch where uvicorn is imported in main.py
    with patch("uvicorn.run") as mock_run:
        mock_run.side_effect = RuntimeError("Server crash")
        result = runner.invoke(app, ["serve", "--pack", str(pack_dir)])
        assert result.exit_code == 1
