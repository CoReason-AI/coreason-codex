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
import os
from pathlib import Path
from typing import Any, Tuple, Union

import duckdb
import lancedb
from loguru import logger

from coreason_codex.schemas import Manifest


class CodexLoader:
    """
    Responsible for loading and verifying the Codex Pack artifacts.
    """

    def __init__(self, pack_path: Union[str, Path]):
        self.pack_path = Path(pack_path)
        if not self.pack_path.exists():
            raise FileNotFoundError(f"Codex Pack not found at: {self.pack_path}")

        self.manifest_path = self.pack_path / "manifest.json"
        self.manifest: Manifest | None = None

    def load_manifest(self) -> Manifest:
        """Loads and parses the manifest.json."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at: {self.manifest_path}")

        try:
            with open(self.manifest_path, "r") as f:
                data = json.load(f)
            self.manifest = Manifest(**data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in manifest: {e}") from e
        except Exception as e:
            raise ValueError(f"Failed to parse manifest: {e}") from e

        return self.manifest

    def _compute_file_sha256(self, file_path: Path) -> str:
        """Computes SHA256 for a single file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _compute_directory_sha256(self, dir_path: Path) -> str:
        """
        Computes a deterministic SHA256 for a directory.

        Hashes all files in the directory (sorted by relative path) and updates
        the master hash with the relative path and file content hash.
        """
        sha256_hash = hashlib.sha256()

        # Walk directory
        file_paths: list[Tuple[str, Path]] = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                full_path = Path(root) / file
                rel_path = full_path.relative_to(dir_path)
                file_paths.append((str(rel_path), full_path))

        # Sort by relative path to ensure deterministic order
        file_paths.sort(key=lambda x: x[0])

        for rel_path_str, full_path in file_paths:
            # Update with path
            sha256_hash.update(rel_path_str.encode("utf-8"))
            # Update with file content
            file_hash = self._compute_file_sha256(full_path)
            sha256_hash.update(file_hash.encode("utf-8"))

        return sha256_hash.hexdigest()

    def _compute_sha256(self, path: Path) -> str:
        if path.is_dir():
            return self._compute_directory_sha256(path)
        return self._compute_file_sha256(path)

    def verify_integrity(self) -> bool:
        """
        Verifies the checksums of all artifacts listed in the manifest.
        Raises ValueError if integrity check fails.
        """
        if not self.manifest:
            self.load_manifest()

        if not self.manifest:
            raise RuntimeError("Failed to load manifest")

        logger.info(f"Verifying integrity for Codex Pack: {self.manifest.version}")

        for filename, expected_hash in self.manifest.checksums.items():
            file_path = self.pack_path / filename

            # Security checks
            # 1. Prevent path traversal
            try:
                # Resolve path and check if it is relative to pack root
                resolved_path = file_path.resolve()
                pack_root = self.pack_path.resolve()
                if not resolved_path.is_relative_to(pack_root):
                    raise ValueError(f"Security Violation: Path traversal detected in {filename}")
            except Exception as e:
                # Check specifically if validation failed or some other error
                if "Security Violation" in str(e):
                    raise
                # Fallback for weird path issues
                raise ValueError(f"Invalid path for artifact {filename}: {e}") from e

            if not file_path.exists():
                raise FileNotFoundError(f"Artifact not found: {filename}")

            # 2. Disallow symlinks
            if file_path.is_symlink():
                raise ValueError(f"Security Violation: Symlinks not allowed for artifact {filename}")

            calculated_hash = self._compute_sha256(file_path)
            if calculated_hash != expected_hash:
                logger.error(f"Checksum mismatch for {filename}. Expected {expected_hash}, got {calculated_hash}")
                raise ValueError(f"Integrity check failed for {filename}")

        logger.info("Integrity check passed.")
        return True

    def load_codex(self) -> Tuple[duckdb.DuckDBPyConnection, Any]:
        """
        Verifies integrity and returns connections to DuckDB and LanceDB.

        Returns:
            Tuple[duckdb_connection, lancedb_db_object]
        """
        self.verify_integrity()

        # Load DuckDB
        duckdb_path = self.pack_path / "vocab.duckdb"
        logger.info(f"Connecting to DuckDB at {duckdb_path}")
        try:
            con = duckdb.connect(str(duckdb_path), read_only=True)
            # Verify it's a valid DB by running a simple query
            con.execute("SELECT 1")
        except Exception as e:
            logger.error(f"Failed to connect to DuckDB: {e}")
            raise ValueError(f"Failed to initialize DuckDB connection: {e}") from e

        # Load LanceDB
        # Connecting to the pack directory, assuming vectors are there
        logger.info(f"Connecting to LanceDB at {self.pack_path}")
        try:
            lance_db = lancedb.connect(str(self.pack_path))
        except Exception as e:
            logger.error(f"Failed to connect to LanceDB: {e}")
            raise ValueError(f"Failed to initialize LanceDB connection: {e}") from e

        return con, lance_db
