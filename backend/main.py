"""
FastAPI application entry point.
Mounts the API router and configures middleware.
Single responsibility: wire up the ASGI app.
"""

import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from time import perf_counter
from typing import Any, AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAIError
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from backend.agent import run_agent
from backend.cache import cache_lookup, cache_store
from backend.config import get_settings
from backend.router import api_router


request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class JsonFormatter(logging.Formatter):
    """Format log records as JSON for production log search."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "msg": record.getMessage(),
            "request_id": request_id_var.get(),
            "logger": record.name,
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


root = logging.getLogger()
root.handlers.clear()
handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
root.addHandler(handler)
root.setLevel(logging.INFO)
logger = logging.getLogger("deepsearch")


class SearchRequest(BaseModel):
    question: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Run startup/shutdown logic around the app lifetime."""
    # Future: initialise Qdrant collection, warm Redis pool, etc.
    yield
    # Future: flush caches, close connections.


def _sse(event: str, data: dict[str, Any]) -> dict[str, str]:
    """Build an SSE event payload with JSON data."""
    return {"event": event, "data": json.dumps(data)}


def _error_message(exc: Exception) -> str:
    """Return a readable error message for the UI."""
    return str(exc).strip() or exc.__class__.__name__


def _stream_response(question: str) -> EventSourceResponse:
    """Return a configured SSE response for one search question."""
    settings = get_settings()
    return EventSourceResponse(
        _search_events(question),
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
        ping=settings.sse_ping_seconds,
    )


def _cors_origins(raw_origins: str) -> list[str]:
    """Parse comma-separated CORS origins from configuration."""
    return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]


async def _safe_cache_lookup(question: str) -> str | None:
    """Read cache without letting cache/OpenAI failures break streaming."""
    try:
        return await cache_lookup(question)
    except OpenAIError as exc:
        logger.warning("Skipping cache lookup: %s", exc)
        return None
    except Exception as exc:
        logger.warning("Skipping cache lookup: %s", exc)
        return None


async def _safe_cache_store(question: str, answer: str) -> None:
    """Write cache opportunistically after a successful stream."""
    try:
        await cache_store(question, answer)
    except OpenAIError as exc:
        logger.warning("Skipping cache store: %s", exc)
    except Exception as exc:
        logger.warning("Skipping cache store: %s", exc)


async def _search_events(question: str) -> AsyncGenerator[dict[str, str], None]:
    """Stream cache, agent, and completion events for one search question."""
    cached = await _safe_cache_lookup(question)
    if cached is not None:
        yield _sse("cached", {"answer": cached})
        yield _sse("done", {"cached": True})
        return

    started = perf_counter()
    first_token_sent = False
    answer_parts: list[str] = []
    yield _sse("status", {"message": "Researching…"})

    try:
        async for event in run_agent(question):
            match event["type"]:
                case "tool_call":
                    yield _sse("tool_call", {"name": event["name"]})
                case "text":
                    token = str(event["content"])
                    answer_parts.append(token)
                    data: dict[str, Any] = {"token": token}
                    if not first_token_sent:
                        data["ttft_ms"] = round((perf_counter() - started) * 1000)
                        first_token_sent = True
                    yield _sse("token", data)
    except Exception as exc:
        logger.exception("Search stream failed")
        yield _sse("error", {"message": _error_message(exc)})
        return

    answer = "".join(answer_parts)
    if answer:
        await _safe_cache_store(question, answer)
    yield _sse("done", {"cached": False})


def build_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="DeepSearch API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_logging(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        rid = str(uuid.uuid4())[:8]
        token = request_id_var.set(rid)
        started = perf_counter()
        logger.info("request.start path=%s method=%s", request.url.path, request.method)
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("request.error")
            request_id_var.reset(token)
            raise

        duration_ms = int((perf_counter() - started) * 1000)
        logger.info(
            "request.end status=%s duration_ms=%s",
            response.status_code,
            duration_ms,
        )
        request_id_var.reset(token)
        response.headers["X-Request-Id"] = rid
        return response

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix="/api/v1")

    @app.post("/search")
    async def search(request: SearchRequest) -> EventSourceResponse:
        """Stream a DeepSearch answer as server-sent events."""
        return _stream_response(request.question)

    @app.get("/search/stream")
    async def search_stream(question: str) -> EventSourceResponse:
        """Stream a DeepSearch answer over a native EventSource endpoint."""
        return _stream_response(question)

    return app


app = build_app()


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
