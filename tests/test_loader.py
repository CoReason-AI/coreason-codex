# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_codex

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import duckdb
import pytest

from coreason_codex.loader import CodexLoader


@pytest.fixture
def codex_pack(tmp_path: Path) -> Path:
    # Create valid dummy files
    pack_dir = tmp_path / "codex_v1"
    pack_dir.mkdir()

    # 1. vocab.duckdb
    # Create a real duckdb file
    duck_path = pack_dir / "vocab.duckdb"
    con = duckdb.connect(str(duck_path))
    con.execute("CREATE TABLE test (id INTEGER)")
    con.close()

    # 2. vectors.lance (directory)
    lance_path = pack_dir / "vectors.lance"
    lance_path.mkdir()
    (lance_path / "data.lance").write_bytes(b"dummy content")

    # Compute hashes
    def compute_file_hash(p: Path) -> str:
        sha = hashlib.sha256()
        with open(p, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha.update(byte_block)
        return sha.hexdigest()

    duck_hash = compute_file_hash(duck_path)

    # Use loader's static method logic for directory hash to ensure match
    # Or instantiate a loader just for hashing
    temp_loader = CodexLoader(pack_dir)
    vector_hash = temp_loader._compute_sha256(lance_path)

    manifest = {
        "version": "v1",
        "source_date": "2025-01-01",
        "checksums": {"vocab.duckdb": duck_hash, "vectors.lance": vector_hash},
    }

    with open(pack_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)

    return pack_dir


def test_loader_success(codex_pack: Path) -> None:
    """Test successful loading of valid artifacts."""
    loader = CodexLoader(codex_pack)
    con, lancedb_con = loader.load_codex()

    # Verify DuckDB connection
    assert con is not None
    # Check if we can query the table created in fixture
    res = con.execute("SELECT count(*) FROM test").fetchone()
    if res:
        assert res[0] == 0
    else:
        pytest.fail("Query failed")
    con.close()

    # Verify LanceDB connection
    assert lancedb_con is not None


def test_loader_checksum_mismatch(codex_pack: Path) -> None:
    """Test that modification of an artifact triggers integrity error."""
    # Modify duckdb file
    duck_path = codex_pack / "vocab.duckdb"
    with open(duck_path, "ab") as f:
        f.write(b"corruption")

    loader = CodexLoader(codex_pack)
    with pytest.raises(ValueError, match="Integrity check failed"):
        loader.load_codex()


def test_loader_missing_file(codex_pack: Path) -> None:
    """Test that missing artifact triggers error."""
    (codex_pack / "vocab.duckdb").unlink()
    loader = CodexLoader(codex_pack)
    with pytest.raises(FileNotFoundError):
        loader.load_codex()


def test_loader_manifest_missing(tmp_path: Path) -> None:
    """Test that missing manifest triggers error."""
    loader = CodexLoader(tmp_path)
    with pytest.raises(FileNotFoundError):
        loader.load_codex()


def test_loader_invalid_json(tmp_path: Path) -> None:
    """Test handling of invalid JSON in manifest."""
    manifest_path = tmp_path / "manifest.json"
    with open(manifest_path, "w") as f:
        f.write("{invalid_json")

    loader = CodexLoader(tmp_path)
    with pytest.raises(ValueError, match="Invalid JSON"):
        loader.load_manifest()


def test_directory_hashing(tmp_path: Path) -> None:
    """Test directory hashing consistency."""
    d = tmp_path / "dir"
    d.mkdir()
    (d / "a.txt").write_text("a")
    (d / "b.txt").write_text("b")

    loader = CodexLoader(tmp_path)
    hash1 = loader._compute_sha256(d)

    # Add file, hash should change
    (d / "c.txt").write_text("c")
    hash2 = loader._compute_sha256(d)
    assert hash1 != hash2

    # Remove file, hash should revert (if deterministic order)
    # Actually hash1 was for a+b. hash2 for a+b+c.
    # Removing c should revert to hash1?
    (d / "c.txt").unlink()
    hash3 = loader._compute_sha256(d)
    assert hash1 == hash3


def test_load_manifest_generic_exception(tmp_path: Path) -> None:
    """Test handling of generic exceptions during manifest loading."""
    (tmp_path / "manifest.json").write_text("{}")
    loader = CodexLoader(tmp_path)

    # Patch open to raise a generic exception
    # We need to patch where it is used. Since it's builtins.open, we patch it globally
    # but only within the scope.
    # Note: pathlib.Path.open (if used) might need different patch, but loader uses open()
    with patch("builtins.open", side_effect=Exception("Generic error")):
        with pytest.raises(ValueError, match="Failed to parse manifest: Generic error"):
            loader.load_manifest()


def test_verify_integrity_runtime_error(codex_pack: Path) -> None:
    """Test unreachable state where manifest is not loaded."""
    loader = CodexLoader(codex_pack)
    # Mock load_manifest to do nothing and not set self.manifest
    with patch.object(loader, "load_manifest"):
        with pytest.raises(RuntimeError, match="Failed to load manifest"):
            loader.verify_integrity()


def test_loader_pack_not_found(tmp_path: Path) -> None:
    """Test initializing loader with non-existent path."""
    with pytest.raises(FileNotFoundError, match="Codex Pack not found"):
        CodexLoader(tmp_path / "non_existent")
