# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_codex

import sys
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from coreason_codex import __version__
from coreason_codex.build import CodexBuilder
from coreason_codex.embedders import SapBertEmbedder

app = typer.Typer(
    name="coreason-codex",
    help="CLI for coreason-codex: The Universal Terminology Server.",
    add_completion=False,
)


@app.command()  # type: ignore
def build(
    source: Annotated[Path, typer.Option("--source", "-s", help="Path to source CSV directory", exists=True)],
    output: Annotated[Path, typer.Option("--output", "-o", help="Path to output directory")],
    device: Annotated[str, typer.Option("--device", "-d", help="Device for embedding (cpu/cuda)")] = "cpu",
) -> None:
    """
    Build Codex artifacts from raw Athena CSVs.
    """
    logger.info(f"Starting Codex Build from {source} to {output}")

    try:
        builder = CodexBuilder(source, output)

        # 1. Build Vocab
        builder.build_vocab()

        # 2. Build Vectors
        logger.info(f"Initializing embedder on {device}...")
        embedder = SapBertEmbedder(device=device)
        builder.build_vectors(embedder)

        # 3. Generate Manifest
        builder.generate_manifest()

        logger.info("Codex Build Completed Successfully.")

    except Exception:
        logger.exception("Codex Build Failed")
        # Typer handles exit codes gracefully, but we want explicit non-zero on failure
        sys.exit(1)


@app.command()  # type: ignore
def version() -> None:
    """Print the version of coreason-codex."""
    typer.echo(f"coreason-codex v{__version__}")


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()  # pragma: no cover
