"""
FastAPI application entry point.
Mounts the API router and configures middleware.
Single responsibility: wire up the ASGI app.
"""

import asyncio
import json
import logging
import os
import re
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

from backend.agent import normalise_messages_for_openrouter, run_chat
from backend.cache import cache_lookup, cache_store
from backend.config import get_settings
from backend.research.db import close_pool as close_research_pool
from backend.research.worker import main_loop as research_worker_main_loop
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


class ChatRequest(BaseModel):
    """Body of ``POST /chat``.

    ``messages`` is already in OpenAI chat-completion format — the Next.js
    proxy is responsible for converting AI SDK UIMessages into this shape via
    ``convertToModelMessages`` before forwarding here.

    ``model`` is the OpenRouter model id picked by the user from the chat
    dropdown (e.g. ``openai/gpt-5.5``, ``google/gemini-3.1-pro``). When
    omitted or unknown, the agent falls back to its complexity-based router
    (``flash_model`` for simple queries, ``pro_model`` for complex).

    ``session_id`` and ``user_id`` are forwarded to OpenRouter as session
    tracking metadata so each conversation's generations are grouped
    under one session in the OpenRouter dashboard. ``session_id`` is the
    chat row id; ``user_id`` is the Clerk user id.
    """

    id: str
    messages: list[dict[str, Any]]
    model: str | None = None
    session_id: str | None = None
    user_id: str | None = None


def _inline_worker_disabled() -> bool:
    """Production deployments run a single Railway service that hosts
    both the HTTP server and the research worker — the worker runs as
    an asyncio task spawned in the lifespan below. Set
    ``DISABLE_INLINE_WORKER=true`` to opt out (e.g. when running a
    dedicated worker process on a separate service so you don't get
    duplicate claim contention)."""
    return os.environ.get("DISABLE_INLINE_WORKER", "").lower() in {"1", "true", "yes"}


# Mutable status object the lifespan keeps updated so the
# /api/v1/research-worker-status endpoint can report whether the
# worker is alive, when it last claimed work, and the last error
# message if it crashed. The HTTP server keeps serving even if the
# worker's dead, so without this surface a stuck deploy looks
# identical to "no queued work" from the outside.
_RESEARCH_WORKER_STATE: dict[str, Any] = {
    "configured": False,
    "running": False,
    "started_at": None,
    "stopped_at": None,
    "error": None,
    "error_at": None,
}


def get_research_worker_state() -> dict[str, Any]:
    """Snapshot of the inline worker's state — used by the status route."""
    return dict(_RESEARCH_WORKER_STATE)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Run startup/shutdown logic around the app lifetime.

    Spawns the research worker (``backend.research.worker.main_loop``)
    as a background asyncio task by default so research runs get
    picked up without needing a separate Railway service. The task
    is co-operatively cancelled on shutdown via a stop-event so the
    current run drains cleanly before SIGTERM expires.

    A done-callback surfaces silent crashes (``DATABASE_URL`` missing,
    Postgres unreachable, etc.) into the application log AND into
    ``_RESEARCH_WORKER_STATE`` so the frontend's research-status
    probe can show an honest "worker offline" state instead of the
    "Waiting for the worker…" placeholder spinning forever.
    """
    stop_event: asyncio.Event | None = None
    worker_task: asyncio.Task[None] | None = None

    if _inline_worker_disabled():
        logger.info("research worker: inline mode disabled via env")
        _RESEARCH_WORKER_STATE.update({
            "configured": False,
            "running": False,
        })
    else:
        from datetime import datetime, timezone

        stop_event = asyncio.Event()
        _RESEARCH_WORKER_STATE.update({
            "configured": True,
            "running": True,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "stopped_at": None,
            "error": None,
            "error_at": None,
        })

        def _on_done(task: asyncio.Task[None]) -> None:
            ts = datetime.now(timezone.utc).isoformat()
            _RESEARCH_WORKER_STATE["running"] = False
            _RESEARCH_WORKER_STATE["stopped_at"] = ts
            if task.cancelled():
                logger.info("research worker: task cancelled")
                return
            exc = task.exception()
            if exc is None:
                logger.warning("research worker: task exited cleanly (unexpected mid-run)")
                return
            # `RuntimeError("DATABASE_URL/POSTGRES_URL not set...")` is
            # the most common silent-killer in production. Surface
            # the message so it's visible in Railway logs AND in the
            # status endpoint.
            error_message = f"{exc.__class__.__name__}: {exc}"
            logger.error(
                "research worker: task crashed — %s",
                error_message,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            _RESEARCH_WORKER_STATE["error"] = error_message
            _RESEARCH_WORKER_STATE["error_at"] = ts

        worker_task = asyncio.create_task(
            research_worker_main_loop(stop_event),
            name="research-worker",
        )
        worker_task.add_done_callback(_on_done)
        logger.info("research worker: started in lifespan")

    try:
        yield
    finally:
        if stop_event is not None and worker_task is not None:
            logger.info("research worker: shutting down")
            stop_event.set()
            try:
                # Bounded wait so a hung worker can't hold up
                # container shutdown indefinitely. Railway sends
                # SIGKILL after ~30s; we want to drain well within
                # that.
                await asyncio.wait_for(worker_task, timeout=15)
            except asyncio.TimeoutError:
                logger.warning(
                    "research worker: didn't drain in 15s; cancelling"
                )
                worker_task.cancel()
                try:
                    await worker_task
                except (asyncio.CancelledError, Exception):
                    pass
            try:
                await close_research_pool()
            except Exception:
                logger.exception("research worker: failed to close DB pool")


def _error_message(exc: Exception) -> str:
    """Return a readable error message for the UI."""
    return str(exc).strip() or exc.__class__.__name__


def _cors_origins(raw_origins: str) -> list[str]:
    """Parse comma-separated CORS origins from configuration."""
    return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]


async def _safe_cache_lookup(question: str) -> str | None:
    """Read cache without letting cache/OpenAI failures break streaming.

    Logs full tracebacks on failure so a silent miss from a broken
    Upstash/embedding call is visible in production logs instead of
    being swallowed by the broad except.
    """
    try:
        result = await cache_lookup(question)
        logger.info(
            "cache.lookup query=%r hit=%s", question[:80], result is not None
        )
        return result
    except OpenAIError:
        logger.exception("cache.lookup failed (OpenAI)")
        return None
    except Exception:
        logger.exception("cache.lookup failed")
        return None


async def _safe_cache_store(question: str, answer: str) -> None:
    """Write cache opportunistically after a successful stream.

    Logs full tracebacks on failure so a silent store failure is
    visible in production logs instead of being swallowed.
    """
    try:
        await cache_store(question, answer)
        logger.info(
            "cache.store query=%r answer_len=%d", question[:80], len(answer)
        )
    except OpenAIError:
        logger.exception("cache.store failed (OpenAI)")
    except Exception:
        logger.exception("cache.store failed")


def _ui_part(obj: dict[str, Any]) -> bytes:
    """Encode an AI SDK UI Message Stream Protocol part as an SSE line.

    The protocol is plain SSE — each part is one ``data: {json}\\n\\n`` chunk.
    """
    return f"data: {json.dumps(obj, separators=(',', ':'))}\n\n".encode()


_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
_SOURCES_HEADER_PATTERN = re.compile(r"(?im)^#+\s*sources?\b")


def _domain_label(url: str) -> str:
    """Best-effort short label for a source URL — e.g. 'arxiv.org'."""
    try:
        without_scheme = re.sub(r"^https?://(?:www\.)?", "", url)
        return without_scheme.split("/")[0] or url
    except Exception:
        return url


def _inject_citations(content: str, citations: list[dict[str, Any]]) -> str:
    """Make bracketed citations clickable and ensure a Sources section.

    Two passes:

    1. Replace each ``[N]`` in the body with a markdown link
       ``[\\[N\\]](url)`` so the artifact renderer turns it into a real
       hyperlink while still showing ``[N]`` as the visible label.
    2. Append a ``## Sources`` markdown block at the end if the model
       didn't already write one. This is the safety-net the system
       prompt asks the model to handle but doesn't always remember to,
       especially on follow-up turns.
    """
    if not citations:
        return content

    url_map: dict[int, str] = {}
    for citation in citations:
        idx = citation.get("index")
        url = citation.get("url")
        if isinstance(idx, int) and isinstance(url, str) and url:
            url_map[idx] = url

    if not url_map:
        return content

    def _replace_marker(match: re.Match[str]) -> str:
        idx = int(match.group(1))
        url = url_map.get(idx)
        if not url:
            return match.group(0)
        return f"[\\[{idx}\\]]({url})"

    processed = _CITATION_PATTERN.sub(_replace_marker, content)

    if _SOURCES_HEADER_PATTERN.search(processed):
        return processed

    sources_lines = ["", "", "## Sources", ""]
    for citation in sorted(citations, key=lambda c: c.get("index", 0)):
        idx = citation.get("index")
        url = citation.get("url")
        if not (isinstance(idx, int) and isinstance(url, str) and url):
            continue
        sources_lines.append(f"{idx}. [{_domain_label(url)}]({url})")
    return processed + "\n".join(sources_lines)


def _extract_latest_user_text(messages: list[dict[str, Any]]) -> str:
    """Extract plain text from the most recent user message for cache keying."""
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return " ".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ).strip()
    return ""


def _is_cache_eligible(messages: list[dict[str, Any]]) -> bool:
    """Return True when this is a standalone first-turn text-only query.

    Multi-turn follow-ups are skipped because the cached answer for the same
    text might be wrong in a different conversation context.
    """
    user_turns = [m for m in messages if m.get("role") == "user"]
    if len(user_turns) != 1:
        return False
    content = user_turns[0].get("content")
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        has_attachment = any(
            isinstance(p, dict) and p.get("type") in ("image_url", "file")
            for p in content
        )
        return not has_attachment
    return False


async def _replay_cached_payload(
    cached: str,
    *,
    msg_id: str,
    text_id: str,
) -> AsyncGenerator[bytes, None]:
    """Replay a cached answer as AI SDK UI Message Stream parts.

    The cached string is the JSON payload we wrote in
    :func:`_ui_message_stream`: ``{"text": str, "artifacts": [...], "citations": [...]}``.
    On a hit we emit ``start``, the optional text body, citations (so the
    frontend has the [N] → URL mapping before artifacts render), each
    artifact (with a fresh id so each chat persists its own Document row
    instead of versioning the original), then ``finish`` + ``[DONE]``.
    """
    yield _ui_part({"type": "start", "messageId": msg_id})

    payload = json.loads(cached)

    cached_text = (payload.get("text") or "").strip()
    if cached_text:
        yield _ui_part({"type": "text-start", "id": text_id})
        yield _ui_part({"type": "text-delta", "id": text_id, "delta": cached_text})
        yield _ui_part({"type": "text-end", "id": text_id})

    cached_citations = payload.get("citations") or []
    if cached_citations:
        yield _ui_part({
            "type": "data-citations",
            "id": f"cite_{uuid.uuid4().hex}",
            "data": cached_citations,
        })

    for artifact in payload.get("artifacts") or []:
        if not isinstance(artifact, dict):
            continue
        # Re-issue a fresh artifact id per replay so each cached chat
        # persists its own Document row instead of stacking versions
        # under the original artifact's id.
        replay_artifact = {**artifact, "id": str(uuid.uuid4())}
        yield _ui_part({
            "type": "data-artifact",
            "id": replay_artifact["id"],
            "data": replay_artifact,
        })

    yield _ui_part({"type": "finish"})
    yield b"data: [DONE]\n\n"


async def _ui_message_stream(
    messages: list[dict[str, Any]],
    model: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
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
    # The Next.js proxy ships AI SDK ModelMessage parts (provider-neutral
    # ``image``/``file`` shapes). Translate to OpenAI Chat Completions
    # format up-front so cache eligibility, text extraction, and the
    # downstream agent loop all see one consistent wire format.
    messages = normalise_messages_for_openrouter(messages)

    msg_id = f"msg_{uuid.uuid4().hex}"
    text_id = f"text_{uuid.uuid4().hex}"
    text_started = False
    artifact_emitted = False
    text_buffer: list[str] = []
    # Artifacts emitted during this stream — captured so the cache payload
    # can replay them later. Without this the artifact's substantive
    # content is invisible to the cache and we end up storing only the
    # short text preamble (or nothing at all) for an answer the user
    # actually saw as a full side-panel artifact.
    artifacts_emitted: list[dict[str, Any]] = []
    # Latest citations from retrieve_chunks. Used to (a) linkify [N]
    # markers inside the artifact's body, and (b) append a Sources
    # section if the model forgot one. We keep the most recent set
    # because that's what the model used to ground the answer.
    latest_citations: list[dict[str, Any]] = []
    # When we auto-promote streamed text into an artifact, we want the
    # title to reflect what the user actually asked rather than a generic
    # "Research notes". Pull a summary of the last user message up front.
    fallback_title = _derive_artifact_title(messages)
    # Determine once whether this request is eligible for semantic caching.
    # Used for both the early-return hit path and the post-stream store path.
    query_text = _extract_latest_user_text(messages) if _is_cache_eligible(messages) else ""

    if query_text:
        cached_payload_str = await _safe_cache_lookup(query_text)
        if cached_payload_str is not None:
            async for chunk in _replay_cached_payload(
                cached_payload_str, msg_id=msg_id, text_id=text_id
            ):
                yield chunk
            return

    yield _ui_part({"type": "start", "messageId": msg_id})

    try:
        async for event in run_chat(
            messages, model=model, session_id=session_id, user_id=user_id
        ):
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
                kind = event.get("kind", "text")
                content = event.get("content", "")
                # Linkify [N] and append a Sources block — text artifacts
                # only. Code/sheet kinds are content-shaped where bracket
                # markers are not citations and inserting a markdown
                # Sources block would corrupt the artifact.
                if kind == "text" and latest_citations:
                    content = _inject_citations(content, latest_citations)
                artifact_data = {
                    "id": event["artifact_id"],
                    "kind": kind,
                    "title": event.get("title", "Untitled"),
                    "content": content,
                }
                artifacts_emitted.append(artifact_data)
                yield _ui_part({
                    "type": "data-artifact",
                    "id": event["artifact_id"],
                    "data": artifact_data,
                })
            elif etype == "citations":
                # [N] → URL mapping from the most recent retrieve_chunks.
                # Frontend uses this to make bracketed citations in the
                # streamed answer clickable. Multiple events are allowed
                # — the frontend keeps the latest one.
                items = event.get("items", [])
                if items:
                    latest_citations = items
                citations_id = f"cite_{uuid.uuid4().hex}"
                yield _ui_part({
                    "type": "data-citations",
                    "id": citations_id,
                    "data": items,
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

    full_text = "".join(text_buffer).strip()

    # Safety net: if the agent answered substantively but didn't call
    # create_artifact (sometimes the model just forgets despite the
    # prompt), promote the streamed answer into a text artifact so the
    # user always gets the polished side-panel view they expect from
    # DeepSearch. The threshold is conservative — short factual lookups
    # ("capital of France?") stay inline-only.
    if not artifact_emitted and len(full_text.split()) >= 120:
        artifact_id = str(uuid.uuid4())
        promoted_artifact = {
            "id": artifact_id,
            "kind": "text",
            "title": fallback_title,
            "content": full_text,
        }
        artifacts_emitted.append(promoted_artifact)
        yield _ui_part({
            "type": "data-artifact",
            "id": artifact_id,
            "data": promoted_artifact,
        })

    # Cache a structured payload (text + artifacts + citations) so a hit
    # can replay the full UI shape, not just a wall of text. The previous
    # code cached only ``full_text``, which was empty whenever the agent
    # put the answer in an artifact (the common path per the system
    # prompt) — so the cache stayed empty and every query re-ran the
    # full pipeline.
    if query_text and (full_text or artifacts_emitted):
        cache_payload = json.dumps({
            "text": full_text,
            "artifacts": artifacts_emitted,
            "citations": latest_citations,
        })
        await _safe_cache_store(query_text, cache_payload)

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

    @app.get("/api/v1/research-worker-status")
    async def research_worker_status() -> dict[str, Any]:
        """Diagnostic — is the inline research worker alive?

        The frontend's research artifact card probes this when a run
        sits in 'queued' for too long. Returns the live state object
        the lifespan keeps updated. ``configured=true running=false``
        with a non-null ``error`` is the diagnostic for "worker
        crashed on startup, set DATABASE_URL on Railway".
        """
        return get_research_worker_state()

    @app.post("/chat")
    async def chat(request: ChatRequest) -> StreamingResponse:
        """Stream a DeepSearch answer in AI SDK UI Message Stream format.

        The Next.js ``/api/chat`` route forwards messages to this endpoint
        after Clerk auth + DB persistence, then pipes the response body
        directly to the ``useChat`` hook.
        """
        # Default the OpenRouter session id to the chat id when the
        # caller didn't pass one explicitly. Keeps every generation in a
        # given chat thread grouped together in the OR dashboard even
        # for legacy clients that haven't been updated.
        session_id = request.session_id or request.id
        return StreamingResponse(
            _ui_message_stream(
                request.messages,
                model=request.model,
                session_id=session_id,
                user_id=request.user_id,
            ),
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
