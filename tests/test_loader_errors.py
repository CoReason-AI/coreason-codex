# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_codex

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from coreason_codex.loader import CodexLoader


def test_loader_pack_not_found() -> None:
    with pytest.raises(FileNotFoundError, match="Codex Pack not found"):
        CodexLoader("/non/existent/path")


def test_loader_manifest_not_found(tmp_path: Path) -> None:
    loader = CodexLoader(tmp_path)
    with pytest.raises(FileNotFoundError, match="Manifest not found"):
        loader.load_manifest()


def test_loader_invalid_json(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text("{invalid_json")
    loader = CodexLoader(tmp_path)
    with pytest.raises(ValueError, match="Invalid JSON"):
        loader.load_manifest()


def test_loader_manifest_parse_error(tmp_path: Path) -> None:
    # Valid JSON but missing fields (Manifest validation error)
    (tmp_path / "manifest.json").write_text("{}")
    loader = CodexLoader(tmp_path)
    # Pydantic raises ValidationError which wraps into ValueError in our code or propagates?
    # Loader code: except Exception as e: raise ValueError(f"Failed to parse manifest: {e}")
    with pytest.raises(ValueError, match="Failed to parse manifest"):
        loader.load_manifest()


def test_loader_verify_checksum_mismatch(synthetic_codex_pack: Path) -> None:
    # Tamper with vocab.duckdb
    db_path = synthetic_codex_pack / "vocab.duckdb"
    with open(db_path, "wb") as f:
        f.write(b"corrupted_data")

    loader = CodexLoader(synthetic_codex_pack)
    with pytest.raises(ValueError, match="Integrity check failed"):
        loader.verify_integrity()


def test_loader_verify_artifact_missing(synthetic_codex_pack: Path) -> None:
    # Delete vocab.duckdb
    (synthetic_codex_pack / "vocab.duckdb").unlink()

    loader = CodexLoader(synthetic_codex_pack)
    with pytest.raises(FileNotFoundError, match="Artifact not found"):
        loader.verify_integrity()


def test_loader_duckdb_connect_fail(synthetic_codex_pack: Path) -> None:
    loader = CodexLoader(synthetic_codex_pack)

    with patch("duckdb.connect", side_effect=Exception("Connection failed")):
        with pytest.raises(ValueError, match="Failed to initialize DuckDB"):
            loader.load_codex()


def test_loader_lancedb_connect_fail(synthetic_codex_pack: Path) -> None:
    loader = CodexLoader(synthetic_codex_pack)

    # Mock duckdb success
    with patch("duckdb.connect"):
        with patch("lancedb.connect", side_effect=Exception("Lance fail")):
            with pytest.raises(ValueError, match="Failed to initialize LanceDB"):
                loader.load_codex()


def test_loader_security_path_traversal(synthetic_codex_pack: Path) -> None:
    # Modify manifest to point outside
    manifest_path = synthetic_codex_pack / "manifest.json"
    with open(manifest_path, "r") as f:
        data = json.load(f)

    data["checksums"]["../evil.txt"] = "hash"

    with open(manifest_path, "w") as f:
        json.dump(data, f)

    loader = CodexLoader(synthetic_codex_pack)
    with pytest.raises(ValueError, match="Security Violation"):
        loader.verify_integrity()


def test_loader_security_symlink(synthetic_codex_pack: Path) -> None:
    # Create a symlink in the pack
    target = synthetic_codex_pack / "target"
    target.touch()
    link = synthetic_codex_pack / "symlink"
    link.symlink_to(target)

    # Add symlink to manifest
    manifest_path = synthetic_codex_pack / "manifest.json"
    with open(manifest_path, "r") as f:
        data = json.load(f)

    data["checksums"]["symlink"] = "somehash"

    with open(manifest_path, "w") as f:
        json.dump(data, f)

    loader = CodexLoader(synthetic_codex_pack)
    with pytest.raises(ValueError, match="Symlinks not allowed"):
        loader.verify_integrity()
