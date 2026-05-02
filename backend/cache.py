"""
Semantic cache backed by Redis.
Single responsibility: store and retrieve previous answers by embedding
similarity so identical (or near-identical) queries skip the full pipeline.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from upstash_redis.asyncio import Redis

from backend.config import get_settings
from backend.embedder import cosine_similarity, embed

_KEY_PREFIX = "ds:cache:"
_LEGACY_KEY_PREFIX = "deepsearch:cache:"
_redis: Redis | None = None


def _get_redis() -> Redis:
    """Return the Upstash REST client configured from environment settings."""
    global _redis
    if _redis is None:
        settings = get_settings()
        _redis = Redis(
            url=settings.upstash_redis_rest_url,
            token=settings.upstash_redis_rest_token,
        )
    return _redis


def _stable_key(query: str, prefix: str = _KEY_PREFIX) -> str:
    """Return a deterministic, collision-resistant Redis key for a query string."""
    digest = hashlib.sha256(query.encode()).hexdigest()[:24]
    return f"{prefix}{digest}"


async def cache_lookup(query: str) -> str | None:
    """Return the cached answer whose stored embedding is closest to *query*.

    Scans ``ds:cache:*`` and ``deepsearch:cache:*`` keys, computes cosine
    similarity against each stored embedding, and returns the answer when the
    best score meets or exceeds ``cache_similarity_threshold``.
    Returns ``None`` on a miss.
    """
    settings = get_settings()
    redis = _get_redis()
    query_embedding = await embed(query)

    keys: list[str] = []
    for prefix in (_KEY_PREFIX, _LEGACY_KEY_PREFIX):
        keys.extend(await redis.keys(f"{prefix}*"))

    best_score = 0.0
    best_answer: str | None = None

    for key in keys:
        raw = await redis.get(key)
        if not raw:
            continue
        entry: dict[str, Any] = json.loads(raw)
        embedding = entry.get("embedding")
        if not embedding:
            continue
        score = cosine_similarity(query_embedding, embedding)
        if score > best_score:
            best_score = score
            # Simple payload stores "answer" directly; legacy SemanticCache
            # path nests the answer inside a "result" dict.
            if "answer" in entry:
                best_answer = entry["answer"]
            elif isinstance(entry.get("result"), dict):
                best_answer = entry["result"].get("answer")

    if best_score >= settings.cache_similarity_threshold:
        return best_answer
    return None


async def cache_store(query: str, answer: str) -> None:
    """Embed *query* and persist a JSON payload in Redis with a TTL.

    Key format: ``ds:cache:{sha256(query)[:24]}``
    Payload fields: query, answer, embedding (list[float]), stored_at (ISO-8601).
    """
    settings = get_settings()
    redis = _get_redis()
    embedding = await embed(query)
    key = _stable_key(query)
    payload = json.dumps({
        "query": query,
        "answer": answer,
        "embedding": embedding,
        "stored_at": datetime.now(timezone.utc).isoformat(),
    })
    await redis.setex(key, settings.cache_ttl_seconds, payload)


# ---------------------------------------------------------------------------
# Legacy class — kept for backwards-compatibility with agent.py / router.py.
# ---------------------------------------------------------------------------


class SemanticCache:
    """
    Stores (embedding, result) pairs in Redis with a TTL.
    On lookup, computes cosine similarity against all cached embeddings
    and returns the result dict if the best match exceeds the threshold.
    """

    def __init__(self) -> None:
        self._key_prefix = _LEGACY_KEY_PREFIX

    async def lookup(self, query: str) -> dict[str, Any] | None:
        """Return a cached result dict if a sufficiently similar query exists."""
        redis = _get_redis()
        settings = get_settings()
        query_vec = await embed(query)
        keys = await redis.keys(f"{self._key_prefix}*")

        best_score = 0.0
        best_result: dict[str, Any] | None = None

        for key in keys:
            raw = await redis.get(key)
            if not raw:
                continue
            entry = json.loads(raw)
            embedding = entry.get("embedding")
            if not embedding:
                continue
            score = cosine_similarity(query_vec, embedding)
            if score > best_score:
                best_score = score
                best_result = entry.get("result")

        if best_score >= settings.cache_similarity_threshold:
            return best_result
        return None

    async def store(
        self,
        query: str,
        embedding: list[float],
        result: dict[str, Any],
    ) -> None:
        """Persist a query embedding + result dict with a TTL."""
        redis = _get_redis()
        settings = get_settings()
        key = _stable_key(query, self._key_prefix)
        payload = json.dumps({"embedding": embedding, "result": result})
        await redis.setex(key, settings.cache_ttl_seconds, payload)
