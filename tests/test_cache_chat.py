"""Tests that the /chat endpoint reads from and writes to the Upstash semantic cache."""

from __future__ import annotations

import inspect
import os
import sys
import types
from collections.abc import AsyncGenerator
from typing import Any

import httpx
import pytest

os.environ.setdefault("OPENROUTER_API_KEY", "test-openrouter-key")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("SERPER_API_KEY", "test-serper-key")
os.environ.setdefault("QDRANT_URL", "http://qdrant.test")
os.environ.setdefault("UPSTASH_REDIS_REST_URL", "http://redis.test")
os.environ.setdefault("UPSTASH_REDIS_REST_TOKEN", "test-redis-token")


async def _fake_run_chat(
    messages: list[dict[str, Any]],
    model: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    yield {"type": "text", "content": "freshly computed answer"}


async def _fake_run_agent(question: str) -> AsyncGenerator[dict[str, Any], None]:
    yield {"type": "text", "content": "hello"}


class _FakeDeepSearchAgent:
    async def run(self, query: str, use_cache: bool = True) -> dict[str, object]:
        return {"answer": "hello", "sources": [], "cached": False, "confidence": 1.0}


_agent_stub = types.ModuleType("backend.agent")
_agent_stub.run_agent = _fake_run_agent  # type: ignore[attr-defined]
_agent_stub.run_chat = _fake_run_chat  # type: ignore[attr-defined]
_agent_stub.DeepSearchAgent = _FakeDeepSearchAgent  # type: ignore[attr-defined]
sys.modules.setdefault("backend.agent", _agent_stub)

from backend import main  # noqa: E402
from backend import cache  # noqa: E402


_SINGLE_TURN_BODY: dict[str, Any] = {
    "id": "test-chat-id",
    "messages": [{"role": "user", "content": "What is machine learning?"}],
}

_MULTI_TURN_BODY: dict[str, Any] = {
    "id": "test-chat-id-2",
    "messages": [
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "ML is a field of AI."},
        {"role": "user", "content": "Can you elaborate?"},
    ],
}


# ---------------------------------------------------------------------------
# Cache-hit tests
# ---------------------------------------------------------------------------


async def _hit_lookup(query: str) -> str | None:
    return "cached answer from upstash"


async def _noop_store(query: str, answer: str) -> None:
    return None


async def _miss_lookup(query: str) -> str | None:
    return None


@pytest.mark.asyncio
async def test_chat_cache_hit_returns_cached_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cache hit: response contains cached answer and run_chat is not called."""
    run_chat_called = False

    async def _sentinel_run_chat(
        messages: list[dict[str, Any]], **kwargs: Any
    ) -> AsyncGenerator[dict[str, Any], None]:
        nonlocal run_chat_called
        run_chat_called = True
        yield {"type": "text", "content": "should not appear"}

    monkeypatch.setattr(main, "cache_lookup", _hit_lookup)
    monkeypatch.setattr(main, "cache_store", _noop_store)
    monkeypatch.setattr(main, "run_chat", _sentinel_run_chat)

    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/chat", json=_SINGLE_TURN_BODY)

    assert response.status_code == 200
    assert "cached answer from upstash" in response.text
    assert not run_chat_called, "run_chat must not be called on a cache hit"


@pytest.mark.asyncio
async def test_chat_cache_hit_emits_valid_stream_parts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cache hit stream must contain start, text-delta, finish, and [DONE]."""
    monkeypatch.setattr(main, "cache_lookup", _hit_lookup)
    monkeypatch.setattr(main, "cache_store", _noop_store)
    monkeypatch.setattr(main, "run_chat", _fake_run_chat)

    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/chat", json=_SINGLE_TURN_BODY)

    body = response.text
    assert '"type":"start"' in body
    assert '"type":"text-delta"' in body
    assert '"type":"finish"' in body
    assert "[DONE]" in body


# ---------------------------------------------------------------------------
# Cache-miss / store tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_cache_miss_calls_run_chat_and_stores_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cache miss: run_chat runs and the final answer is stored in cache.

    The stored payload is a JSON string with shape
    ``{"text": ..., "artifacts": [...], "citations": [...]}`` so a hit
    can replay the full UI shape, not just inline text.
    """
    import json as _json

    stored: dict[str, str] = {}

    async def _capture_store(query: str, answer: str) -> None:
        stored["query"] = query
        stored["answer"] = answer

    monkeypatch.setattr(main, "cache_lookup", _miss_lookup)
    monkeypatch.setattr(main, "cache_store", _capture_store)
    monkeypatch.setattr(main, "run_chat", _fake_run_chat)

    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/chat", json=_SINGLE_TURN_BODY)

    assert response.status_code == 200
    assert "freshly computed answer" in response.text
    assert stored.get("query") == "What is machine learning?"
    payload = _json.loads(stored["answer"])
    assert payload["text"] == "freshly computed answer"
    assert payload["artifacts"] == []
    assert payload["citations"] == []


@pytest.mark.asyncio
async def test_chat_artifact_only_response_is_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: when the agent answers only via create_artifact (no
    streamed text), the artifact must still land in the cache. Previously
    the cache stored only ``"".join(text_buffer)`` which was empty in this
    path, so the cache stayed empty and identical follow-ups re-ran the
    full pipeline.
    """
    import json as _json

    async def _artifact_only_run_chat(
        messages: list[dict[str, Any]], **kwargs: Any
    ) -> AsyncGenerator[dict[str, Any], None]:
        yield {
            "type": "artifact",
            "artifact_id": "11111111-1111-1111-1111-111111111111",
            "kind": "text",
            "title": "ML overview",
            "content": "Machine learning is...",
        }

    stored: dict[str, str] = {}

    async def _capture_store(query: str, answer: str) -> None:
        stored["query"] = query
        stored["answer"] = answer

    monkeypatch.setattr(main, "cache_lookup", _miss_lookup)
    monkeypatch.setattr(main, "cache_store", _capture_store)
    monkeypatch.setattr(main, "run_chat", _artifact_only_run_chat)

    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/chat", json=_SINGLE_TURN_BODY)

    assert response.status_code == 200
    assert stored.get("query") == "What is machine learning?"
    payload = _json.loads(stored["answer"])
    assert payload["text"] == ""
    assert len(payload["artifacts"]) == 1
    assert payload["artifacts"][0]["title"] == "ML overview"
    assert payload["artifacts"][0]["content"] == "Machine learning is..."


@pytest.mark.asyncio
async def test_chat_cache_hit_replays_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cached structured payload with an artifact must replay as a
    data-artifact stream part, not as inline text."""
    import json as _json

    async def _artifact_payload_lookup(query: str) -> str | None:
        return _json.dumps({
            "text": "",
            "artifacts": [
                {
                    "id": "22222222-2222-2222-2222-222222222222",
                    "kind": "text",
                    "title": "Cached ML overview",
                    "content": "ML is a subfield of AI.",
                }
            ],
            "citations": [],
        })

    monkeypatch.setattr(main, "cache_lookup", _artifact_payload_lookup)
    monkeypatch.setattr(main, "cache_store", _noop_store)
    monkeypatch.setattr(main, "run_chat", _fake_run_chat)

    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/chat", json=_SINGLE_TURN_BODY)

    body = response.text
    assert response.status_code == 200
    assert '"type":"data-artifact"' in body
    assert "Cached ML overview" in body
    assert "ML is a subfield of AI." in body


@pytest.mark.asyncio
async def test_multi_turn_chat_skips_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multi-turn conversations must not read from or write to the cache."""
    lookup_called = False
    store_called = False

    async def _tracking_lookup(query: str) -> str | None:
        nonlocal lookup_called
        lookup_called = True
        return None

    async def _tracking_store(query: str, answer: str) -> None:
        nonlocal store_called
        store_called = True

    monkeypatch.setattr(main, "cache_lookup", _tracking_lookup)
    monkeypatch.setattr(main, "cache_store", _tracking_store)
    monkeypatch.setattr(main, "run_chat", _fake_run_chat)

    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/chat", json=_MULTI_TURN_BODY)

    assert response.status_code == 200
    assert not lookup_called, "cache_lookup must not be called for multi-turn chats"
    assert not store_called, "cache_store must not be called for multi-turn chats"


# ---------------------------------------------------------------------------
# cache.py unit tests
# ---------------------------------------------------------------------------


def test_stable_key_is_deterministic() -> None:
    """The same query always produces the same Redis key across calls."""
    key1 = cache._stable_key("What is machine learning?")
    key2 = cache._stable_key("What is machine learning?")
    assert key1 == key2


def test_stable_key_uses_correct_prefix() -> None:
    """Module-level stable key must use the primary ds:cache: prefix."""
    key = cache._stable_key("some query")
    assert key.startswith("ds:cache:")


@pytest.mark.parametrize("query_a,query_b", [
    ("What is AI?", "What is ML?"),
    ("hello", "world"),
    ("foo", ""),
])
def test_stable_key_differs_for_distinct_queries(query_a: str, query_b: str) -> None:
    """Different query strings must produce different keys."""
    assert cache._stable_key(query_a) != cache._stable_key(query_b)


def test_semantic_cache_does_not_use_embed_query() -> None:
    """SemanticCache.lookup must not reference the non-existent embed_query symbol."""
    source = inspect.getsource(cache)
    assert "embed_query" not in source, (
        "embed_query does not exist in embedder.py; use embed() instead"
    )


def test_cache_module_exports_expected_public_api() -> None:
    """Public API surface must include the two module-level functions and the class."""
    assert callable(cache.cache_lookup)
    assert callable(cache.cache_store)
    assert isinstance(cache.SemanticCache, type)
