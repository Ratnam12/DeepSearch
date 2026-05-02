"""Unit tests for normalise_messages_for_openrouter().

Verifies that the AI SDK ModelMessage wire format (provider-neutral
``image``/``file`` parts) is correctly translated to the OpenAI
Chat Completions shape that OpenRouter expects.
"""

from __future__ import annotations

import os
import sys
import types
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Stub out all heavy backend imports so the test module can import agent.py
# without a running Qdrant, Redis, or OpenAI key.
# ---------------------------------------------------------------------------
_STUBS = [
    "backend.chunker",
    "backend.embedder",
    "backend.retriever",
    "backend.scraper",
    "backend.llm",
    "backend.model_router",
    "backend.security",
    "backend.dspy_modules",
]

os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("SERPER_API_KEY", "test-key")
os.environ.setdefault("QDRANT_URL", "http://qdrant.test")
os.environ.setdefault("UPSTASH_REDIS_REST_URL", "http://redis.test")
os.environ.setdefault("UPSTASH_REDIS_REST_TOKEN", "test-token")

for _mod in _STUBS:
    if _mod not in sys.modules:
        _stub = types.ModuleType(_mod)
        sys.modules[_mod] = _stub

# backend.chunker — needs chunk_documents and chunk_text
_chunker_stub = sys.modules["backend.chunker"]
if not hasattr(_chunker_stub, "chunk_documents"):
    _chunker_stub.chunk_documents = lambda *_: []  # type: ignore[attr-defined]
if not hasattr(_chunker_stub, "chunk_text"):
    _chunker_stub.chunk_text = lambda *_, **__: []  # type: ignore[attr-defined]

# backend.embedder — needs embed, embed_batch, and cosine_similarity
# (cache.py imports cosine_similarity; agent.py imports embed + embed_batch)
_embedder_stub = sys.modules["backend.embedder"]
if not hasattr(_embedder_stub, "embed"):
    async def _fake_embed(text: str) -> list[float]:  # noqa: D103
        return [0.0]
    _embedder_stub.embed = _fake_embed  # type: ignore[attr-defined]
if not hasattr(_embedder_stub, "embed_batch"):
    async def _fake_embed_batch(texts: list[str]) -> list[list[float]]:  # noqa: D103
        return [[0.0]] * len(texts)
    _embedder_stub.embed_batch = _fake_embed_batch  # type: ignore[attr-defined]
if not hasattr(_embedder_stub, "cosine_similarity"):
    def _fake_cosine(a: list[float], b: list[float]) -> float:  # noqa: D103
        return 0.0
    _embedder_stub.cosine_similarity = _fake_cosine  # type: ignore[attr-defined]

# backend.retriever
_ret_stub = sys.modules["backend.retriever"]
for _sym in ("hybrid_search", "retrieve_chunks", "upsert_chunks"):
    if not hasattr(_ret_stub, _sym):
        async def _noop(*_a: Any, **_k: Any) -> list[Any]:  # noqa: D103
            return []
        setattr(_ret_stub, _sym, _noop)

# backend.scraper
_scraper_stub = sys.modules["backend.scraper"]
for _sym in ("scrape_url", "scrape_urls", "search_web"):
    if not hasattr(_scraper_stub, _sym):
        async def _noop_scraper(*_a: Any, **_k: Any) -> Any:  # noqa: D103
            return ""
        setattr(_scraper_stub, _sym, _noop_scraper)

# backend.llm
_llm_stub = sys.modules["backend.llm"]
if not hasattr(_llm_stub, "synthesise_answer"):
    async def _fake_synth(*_a: Any, **_k: Any) -> tuple[str, float, list[Any]]:  # noqa: D103
        return ("", 0.0, [])
    _llm_stub.synthesise_answer = _fake_synth  # type: ignore[attr-defined]

# backend.model_router — needs FLASH + helpers
_mr_stub = sys.modules["backend.model_router"]
if not hasattr(_mr_stub, "FLASH"):
    _mr_stub.FLASH = "openai/gpt-5.4-mini"  # type: ignore[attr-defined]
if not hasattr(_mr_stub, "log_cost"):
    async def _noop_cost(*_a: Any, **_k: Any) -> None:  # noqa: D103
        return
    _mr_stub.log_cost = _noop_cost  # type: ignore[attr-defined]
if not hasattr(_mr_stub, "route_model"):
    _mr_stub.route_model = lambda _q: "openai/gpt-5.4-mini"  # type: ignore[attr-defined]

# backend.security
_sec_stub = sys.modules["backend.security"]
if not hasattr(_sec_stub, "sanitize"):
    _sec_stub.sanitize = lambda url, text: {"safe_text": text, "is_suspicious": False, "patterns": []}  # type: ignore[attr-defined]
if not hasattr(_sec_stub, "verify_api_key"):
    _sec_stub.verify_api_key = lambda: None  # type: ignore[attr-defined]

# backend.dspy_modules
_dspy_stub = sys.modules["backend.dspy_modules"]
if not hasattr(_dspy_stub, "generate_candidate"):
    _dspy_stub.generate_candidate = lambda **_k: type("R", (), {"answer": ""})()  # type: ignore[attr-defined]

from backend.agent import normalise_messages_for_openrouter  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _user(content: Any) -> dict[str, Any]:
    return {"role": "user", "content": content}


# ---------------------------------------------------------------------------
# String-content messages pass through unchanged
# ---------------------------------------------------------------------------


def test_string_content_is_unchanged() -> None:
    messages = [_user("Hello world")]
    result = normalise_messages_for_openrouter(messages)
    assert result == messages


def test_non_user_role_string_passes_through() -> None:
    messages = [{"role": "system", "content": "You are helpful."}]
    result = normalise_messages_for_openrouter(messages)
    assert result == messages


# ---------------------------------------------------------------------------
# Image part: AI SDK ``image`` → OpenAI ``image_url``
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("url", [
    "https://example.com/photo.jpg",
    "data:image/png;base64,abc123",
])
def test_image_part_converted_to_image_url(url: str) -> None:
    messages = [_user([{"type": "image", "image": url}])]
    result = normalise_messages_for_openrouter(messages)
    parts = result[0]["content"]
    assert len(parts) == 1
    assert parts[0]["type"] == "image_url"
    assert parts[0]["image_url"]["url"] == url


# ---------------------------------------------------------------------------
# File part: AI SDK ``file`` (data/url fields) → OpenAI ``file.file_data``
# ---------------------------------------------------------------------------


def test_file_part_converted_to_openrouter_shape() -> None:
    messages = [_user([{
        "type": "file",
        "data": "https://blob.vercel.io/doc.pdf",
        "mediaType": "application/pdf",
        "name": "doc.pdf",
    }])]
    result = normalise_messages_for_openrouter(messages)
    parts = result[0]["content"]
    assert len(parts) == 1
    part = parts[0]
    assert part["type"] == "file"
    assert isinstance(part["file"], dict)
    assert part["file"]["file_data"] == "https://blob.vercel.io/doc.pdf"
    assert part["file"]["filename"] == "doc.pdf"


def test_file_part_url_field_used_as_fallback() -> None:
    """``url`` key (older AI SDK shape) is accepted when ``data`` is absent."""
    messages = [_user([{
        "type": "file",
        "url": "https://blob.vercel.io/image.png",
        "name": "image.png",
    }])]
    result = normalise_messages_for_openrouter(messages)
    part = result[0]["content"][0]
    assert part["file"]["file_data"] == "https://blob.vercel.io/image.png"


def test_image_file_part_converted_to_image_url() -> None:
    """AI SDK file parts with image media types use OpenRouter image_url."""
    messages = [_user([{
        "type": "file",
        "data": "data:image/png;base64,abc123",
        "mediaType": "image/png",
        "filename": "chart.png",
    }])]
    result = normalise_messages_for_openrouter(messages)
    part = result[0]["content"][0]
    assert part["type"] == "image_url"
    assert part["image_url"]["url"] == "data:image/png;base64,abc123"


# ---------------------------------------------------------------------------
# Idempotency: already-OpenAI shapes pass through unchanged
# ---------------------------------------------------------------------------


def test_already_openai_image_url_passes_through() -> None:
    openai_part = {"type": "image_url", "image_url": {"url": "https://x.com/img.jpg"}}
    messages = [_user([openai_part])]
    result = normalise_messages_for_openrouter(messages)
    assert result[0]["content"][0] == openai_part


def test_already_openai_file_passes_through() -> None:
    """A part with ``type=file`` and a nested ``file`` dict should not be double-wrapped."""
    openai_part = {"type": "file", "file": {"filename": "a.pdf", "file_data": "https://..."}}
    messages = [_user([openai_part])]
    result = normalise_messages_for_openrouter(messages)
    assert result[0]["content"][0] == openai_part


# ---------------------------------------------------------------------------
# Text parts are left alone
# ---------------------------------------------------------------------------


def test_text_part_unchanged() -> None:
    text_part = {"type": "text", "text": "Summarise this document."}
    messages = [_user([text_part])]
    result = normalise_messages_for_openrouter(messages)
    assert result[0]["content"][0] == text_part


# ---------------------------------------------------------------------------
# Mixed message: text + image + file all in one content array
# ---------------------------------------------------------------------------


def test_mixed_content_array() -> None:
    messages = [_user([
        {"type": "text", "text": "What does this PDF say about the image?"},
        {"type": "image", "image": "https://example.com/chart.png"},
        {"type": "file", "data": "https://blob.vercel.io/report.pdf", "name": "report.pdf"},
    ])]
    result = normalise_messages_for_openrouter(messages)
    parts = result[0]["content"]
    assert parts[0] == {"type": "text", "text": "What does this PDF say about the image?"}
    assert parts[1]["type"] == "image_url"
    assert parts[2]["type"] == "file"
    assert isinstance(parts[2]["file"], dict)


# ---------------------------------------------------------------------------
# Multi-message conversation
# ---------------------------------------------------------------------------


def test_multi_message_only_touches_list_content() -> None:
    messages = [
        {"role": "system", "content": "You are helpful."},
        _user("First turn"),
        {"role": "assistant", "content": "Got it."},
        _user([{"type": "image", "image": "https://x.com/img.jpg"}]),
    ]
    result = normalise_messages_for_openrouter(messages)
    # System, first user, assistant unchanged
    assert result[0]["content"] == "You are helpful."
    assert result[1]["content"] == "First turn"
    assert result[2]["content"] == "Got it."
    # Last user message translated
    assert result[3]["content"][0]["type"] == "image_url"
