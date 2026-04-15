"""
FastAPI application entry point.
Mounts the API router and configures middleware.
Single responsibility: wire up the ASGI app.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.router import api_router
from backend.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Run startup/shutdown logic around the app lifetime."""
    # Future: initialise Qdrant collection, warm Redis pool, etc.
    yield
    # Future: flush caches, close connections.


def build_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="DeepSearch API",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix="/api/v1")

    return app


app = build_app()


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
