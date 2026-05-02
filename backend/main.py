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
from fastapi.responses import StreamingResponse
from openai import OpenAIError
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from backend.agent import run_agent, run_chat
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


class ChatRequest(BaseModel):
    """Body of ``POST /chat``.

    ``messages`` is already in OpenAI chat-completion format — the Next.js
    proxy is responsible for converting AI SDK UIMessages into this shape via
    ``convertToModelMessages`` before forwarding here.

    ``model`` is the OpenRouter model id picked by the user from the chat
    dropdown (e.g. ``openai/gpt-5.5``, ``google/gemini-3.1-pro``). When
    omitted or unknown, the agent falls back to its complexity-based router
    (``flash_model`` for simple queries, ``pro_model`` for complex).
    """

    id: str
    messages: list[dict[str, Any]]
    model: str | None = None


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


def _ui_part(obj: dict[str, Any]) -> bytes:
    """Encode an AI SDK UI Message Stream Protocol part as an SSE line.

    The protocol is plain SSE — each part is one ``data: {json}\\n\\n`` chunk.
    """
    return f"data: {json.dumps(obj, separators=(',', ':'))}\n\n".encode()


async def _ui_message_stream(
    messages: list[dict[str, Any]],
    model: str | None = None,
) -> AsyncGenerator[bytes, None]:
    """Translate ``run_chat`` events into AI SDK UI Message Stream Protocol.

    The output is consumed by the Next.js ``useChat`` hook on the frontend.
    Part types we emit:
      - ``start`` once at the beginning with the assistant message id
      - ``tool-input-available`` when the agent calls a tool (input ready)
      - ``tool-output-available`` when a tool returns
      - ``data-artifact`` for ``create_artifact`` payloads (rendered by the
        side-panel artifact component on the frontend)
      - ``text-start`` / ``text-delta`` / ``text-end`` for the streamed answer
      - ``finish`` once at the end (success path)
      - ``error`` if the agent raises before completion

    Stream is terminated with ``data: [DONE]``.
    """
    msg_id = f"msg_{uuid.uuid4().hex}"
    text_id = f"text_{uuid.uuid4().hex}"
    text_started = False
    artifact_emitted = False
    text_buffer: list[str] = []
    # When we auto-promote streamed text into an artifact, we want the
    # title to reflect what the user actually asked rather than a generic
    # "Research notes". Pull a summary of the last user message up front.
    fallback_title = _derive_artifact_title(messages)

    yield _ui_part({"type": "start", "messageId": msg_id})

    try:
        async for event in run_chat(messages, model=model):
            etype = event.get("type")
            if etype == "tool_call":
                args_raw = event.get("args") or "{}"
                try:
                    parsed_args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except json.JSONDecodeError:
                    parsed_args = {"_raw": args_raw}
                yield _ui_part({
                    "type": "tool-input-available",
                    "toolCallId": event["call_id"],
                    "toolName": event["name"],
                    "input": parsed_args,
                })
            elif etype == "tool_result":
                yield _ui_part({
                    "type": "tool-output-available",
                    "toolCallId": event["call_id"],
                    "output": event.get("content", ""),
                })
            elif etype == "artifact":
                artifact_emitted = True
                yield _ui_part({
                    "type": "data-artifact",
                    "id": event["artifact_id"],
                    "data": {
                        "id": event["artifact_id"],
                        "kind": event.get("kind", "text"),
                        "title": event.get("title", "Untitled"),
                        "content": event.get("content", ""),
                    },
                })
            elif etype == "citations":
                # [N] → URL mapping from the most recent retrieve_chunks.
                # Frontend uses this to make bracketed citations in the
                # streamed answer clickable. Multiple events are allowed
                # — the frontend keeps the latest one.
                citations_id = f"cite_{uuid.uuid4().hex}"
                yield _ui_part({
                    "type": "data-citations",
                    "id": citations_id,
                    "data": event.get("items", []),
                })
            elif etype == "text":
                token = str(event.get("content", ""))
                if not token:
                    continue
                text_buffer.append(token)
                if not text_started:
                    yield _ui_part({"type": "text-start", "id": text_id})
                    text_started = True
                yield _ui_part({
                    "type": "text-delta",
                    "id": text_id,
                    "delta": token,
                })
    except Exception as exc:
        logger.exception("Chat stream failed")
        if text_started:
            yield _ui_part({"type": "text-end", "id": text_id})
            text_started = False
        yield _ui_part({"type": "error", "errorText": _error_message(exc)})
        yield b"data: [DONE]\n\n"
        return

    if text_started:
        yield _ui_part({"type": "text-end", "id": text_id})

    # Safety net: if the agent answered substantively but didn't call
    # create_artifact (sometimes the model just forgets despite the
    # prompt), promote the streamed answer into a text artifact so the
    # user always gets the polished side-panel view they expect from
    # DeepSearch. The threshold is conservative — short factual lookups
    # ("capital of France?") stay inline-only.
    if not artifact_emitted:
        full_text = "".join(text_buffer).strip()
        if len(full_text.split()) >= 120:
            artifact_id = str(uuid.uuid4())
            yield _ui_part({
                "type": "data-artifact",
                "id": artifact_id,
                "data": {
                    "id": artifact_id,
                    "kind": "text",
                    "title": fallback_title,
                    "content": full_text,
                },
            })

    yield _ui_part({"type": "finish"})
    yield b"data: [DONE]\n\n"


def _derive_artifact_title(messages: list[dict[str, Any]]) -> str:
    """Best-effort title for an auto-promoted artifact.

    Pulls the most recent user message text and trims it to a short
    title-like phrase. Falls back to a generic label if no user text is
    available.
    """
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = " ".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        else:
            continue
        text = text.strip()
        if not text:
            continue
        # Compact whitespace and cap at ~60 chars on a word boundary.
        text = " ".join(text.split())
        if len(text) <= 60:
            return text
        head = text[:60]
        last_space = head.rfind(" ")
        return (head[:last_space] if last_space > 30 else head) + "…"
    return "Research notes"


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

    @app.post("/chat")
    async def chat(request: ChatRequest) -> StreamingResponse:
        """Stream a DeepSearch answer in AI SDK UI Message Stream format.

        The Next.js ``/api/chat`` route forwards messages to this endpoint
        after Clerk auth + DB persistence, then pipes the response body
        directly to the ``useChat`` hook.
        """
        return StreamingResponse(
            _ui_message_stream(request.messages, model=request.model),
            media_type="text/event-stream",
            headers={
                "x-vercel-ai-ui-message-stream": "v1",
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app


app = build_app()


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
