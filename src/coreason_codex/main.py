# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_codex

import argparse
import sys
from pathlib import Path

from loguru import logger

from coreason_codex.build import CodexBuilder
from coreason_codex.embedders import SapBertEmbedder


def build(args: argparse.Namespace) -> None:
    """Executes the build command."""
    source_dir = Path(args.source)
    output_dir = Path(args.output)
    device = args.device

    logger.info(f"Starting Codex Build from {source_dir} to {output_dir}")

    try:
        builder = CodexBuilder(source_dir, output_dir)

        # 1. Build Vocab
        builder.build_vocab()

        # 2. Build Vectors
        # Initialize embedder
        logger.info(f"Initializing embedder on {device}...")
        embedder = SapBertEmbedder(device=device)
        builder.build_vectors(embedder)

        # 3. Generate Manifest
        builder.generate_manifest()

        logger.info("Codex Build Completed Successfully.")

    except Exception:
        logger.exception("Codex Build Failed")
        sys.exit(1)


def main() -> None:
    """Entry point for the CLI."""
    parser = argparse.ArgumentParser(description="coreason-codex CLI")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to execute")

    # Build Command
    build_parser = subparsers.add_parser("build", help="Build Codex artifacts from raw CSVs")
    build_parser.add_argument("--source", "-s", required=True, help="Path to source CSV directory")
    build_parser.add_argument("--output", "-o", required=True, help="Path to output directory")
    build_parser.add_argument("--device", "-d", default="cpu", help="Device for embedding (cpu/cuda)")
    build_parser.set_defaults(func=build)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()  # pragma: no cover
