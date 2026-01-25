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
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Optional, cast

from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel

from coreason_codex.normalizer import CodexNormalizer
from coreason_codex.pipeline import CodexContext
from coreason_codex.schemas import CodexMatch


# Pydantic Models for Requests
class NormalizationRequest(BaseModel):
    text: str
    domain: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    limit: int = 10


# Lifespan Management
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager to load the Codex models on startup.
    """
    pack_path = os.getenv("CODEX_PACK_PATH", "./data/codex_pack")
    logger.info(f"Initializing Codex Server with pack: {pack_path}")

    try:
        CodexContext.initialize(pack_path)
        app.state.normalizer = CodexContext.get_instance().normalizer
        logger.info("Codex Inference Engine Loaded Successfully.")
    except Exception as e:
        logger.exception("Failed to initialize Codex Inference Engine.")
        # We raise to ensure the server doesn't start in a broken state
        raise RuntimeError(f"Server initialization failed: {e}") from e

    yield

    logger.info("Shutting down Codex Server.")


app = FastAPI(title="Coreason Codex API", lifespan=lifespan)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """
    Health check endpoint. Returns status ready if models are loaded.
    """
    return {"status": "ready"}


@app.post("/normalize", response_model=List[CodexMatch])
async def normalize(request: NormalizationRequest) -> List[CodexMatch]:
    """
    Normalize text to Standard Concepts.
    """
    normalizer = cast(CodexNormalizer, app.state.normalizer)
    return normalizer.normalize(request.text, domain_filter=request.domain)


@app.post("/search", response_model=List[CodexMatch])
async def search(request: SearchRequest) -> List[CodexMatch]:
    """
    Expose vector search to find nearest concepts.
    """
    normalizer = cast(CodexNormalizer, app.state.normalizer)
    return normalizer.normalize(request.query, k=request.limit)
