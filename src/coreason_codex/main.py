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
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
import uvicorn
from coreason_identity.models import UserContext
from coreason_identity.types import SecretStr
from loguru import logger

from coreason_codex import __version__
from coreason_codex.pipeline import CodexPipeline, initialize

app = typer.Typer(
    name="coreason-codex",
    help="CLI for coreason-codex: The Universal Terminology Server.",
    add_completion=False,
)


@app.command()
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
        system_context = UserContext(
            user_id=SecretStr("cli-user"), roles=["system"], metadata={"source": "cli"}
        )

        pipeline = CodexPipeline()
        pipeline.run(source, output, device, context=system_context)

        logger.info("Codex Build Completed Successfully.")

    except Exception:
        logger.exception("Codex Build Failed")
        # Typer handles exit codes gracefully, but we want explicit non-zero on failure
        sys.exit(1)


@app.command()
def normalize(
    text: Annotated[str, typer.Argument(help="Input text to normalize")],
    pack: Annotated[Path, typer.Option("--pack", "-p", help="Path to Codex Pack directory", exists=True)],
    domain: Annotated[Optional[str], typer.Option("--domain", "-d", help="Optional domain filter")] = None,
) -> None:
    """
    Normalize text to Standard Concepts.
    """
    try:
        initialize(str(pack))
        system_context = UserContext(
            user_id=SecretStr("cli-user"), roles=["system"], metadata={"source": "cli"}
        )

        pipeline = CodexPipeline()
        results = pipeline.search(text, k=10, domain_filter=domain, context=system_context)
        for match in results:
            typer.echo(match.model_dump_json(indent=2))
    except Exception:
        logger.exception("Normalization Failed")
        sys.exit(1)


@app.command()
def serve(
    host: str = "0.0.0.0",
    port: int = 8000,
    pack: Annotated[Path, typer.Option("--pack", "-p", help="Path to Codex Pack directory", exists=True)] = Path(
        "./data/codex_pack"
    ),
) -> None:
    """
    Start the Codex Server.
    """
    try:
        os.environ["CODEX_PACK_PATH"] = str(pack)

        system_context = UserContext(
            user_id=SecretStr("cli-user"), roles=["system"], metadata={"source": "cli"}
        )
        logger.info(
            "Starting Codex Server", user_id=system_context.user_id.get_secret_value()
        )

        uvicorn.run("coreason_codex.server:app", host=host, port=port, reload=False)
    except Exception:
        logger.exception("Server Failed")
        sys.exit(1)


@app.command()
def version() -> None:
    """Print the version of coreason-codex."""
    typer.echo(f"coreason-codex v{__version__}")


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()  # pragma: no cover
