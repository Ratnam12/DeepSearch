"""Verify the public search stream endpoints emit server-sent events."""

from __future__ import annotations

import os
import sys
import types
from collections.abc import AsyncGenerator

import httpx
import pytest

os.environ.setdefault("OPENROUTER_API_KEY", "test-openrouter-key")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("SERPER_API_KEY", "test-serper-key")
os.environ.setdefault("QDRANT_URL", "http://qdrant.test")
os.environ.setdefault("UPSTASH_REDIS_REST_URL", "http://redis.test")
os.environ.setdefault("UPSTASH_REDIS_REST_TOKEN", "test-redis-token")


async def _fake_agent(question: str) -> AsyncGenerator[dict[str, str], None]:
    yield {"type": "tool_call", "name": "retrieve_chunks"}
    yield {"type": "text", "content": "hello"}


class _FakeDeepSearchAgent:
    async def run(self, query: str, use_cache: bool = True) -> dict[str, object]:
        return {"answer": "hello", "sources": [], "cached": False, "confidence": 1.0}


agent_stub = types.ModuleType("backend.agent")
agent_stub.run_agent = _fake_agent
agent_stub.DeepSearchAgent = _FakeDeepSearchAgent
sys.modules.setdefault("backend.agent", agent_stub)

from backend import main


async def _no_cached_answer(question: str) -> str | None:
    return None


async def _noop_cache_store(question: str, answer: str) -> None:
    return None


@pytest.mark.asyncio
async def test_search_stream_returns_sse_events(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "cache_lookup", _no_cached_answer)
    monkeypatch.setattr(main, "cache_store", _noop_cache_store)
    monkeypatch.setattr(main, "run_agent", _fake_agent)

    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/search/stream", params={"question": "hello?"})

    body = response.text
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache, no-transform"
    assert response.headers["x-accel-buffering"] == "no"
    assert response.headers["x-request-id"]
    assert "event: status" in body
    assert "event: tool_call" in body
    assert "event: token" in body
    assert "event: done" in body
