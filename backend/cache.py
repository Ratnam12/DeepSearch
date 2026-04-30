"""
Semantic cache backed by Redis.
Single responsibility: store and retrieve previous answers by embedding
similarity so identical (or near-identical) queries skip the full pipeline.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
from upstash_redis.asyncio import Redis

from backend.config import get_settings
from backend.embedder import cosine_similarity, embed

_KEY_PREFIX = "ds:cache:"
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


async def cache_lookup(query: str) -> str | None:
    """Return the cached answer whose stored embedding is closest to *query*.

    Scans all ``ds:cache:*`` keys, computes cosine similarity against each
    stored embedding, and returns the answer when the best score meets or
    exceeds ``cache_similarity_threshold``.  Returns ``None`` on a miss.
    """
    settings = get_settings()
    redis = _get_redis()
    query_embedding = await embed(query)
    keys: list[str] = await redis.keys(f"{_KEY_PREFIX}*")

    best_score = 0.0
    best_answer: str | None = None

    for key in keys:
        raw = await redis.get(key)
        if not raw:
            continue
        entry: dict[str, Any] = json.loads(raw)
        score = cosine_similarity(query_embedding, entry["embedding"])
        if score > best_score:
            best_score = score
            best_answer = entry["answer"]

    if best_score >= settings.cache_similarity_threshold:
        return best_answer
    return None


async def cache_store(query: str, answer: str) -> None:
    """Embed *query* and persist a JSON payload in Redis with a TTL.

    Key format: ``ds:cache:{hash(query) % 10**10}``
    Payload fields: query, answer, embedding (list[float]), stored_at (ISO-8601).
    """
    settings = get_settings()
    redis = _get_redis()
    embedding = await embed(query)
    key = f"{_KEY_PREFIX}{hash(query) % 10 ** 10}"
    payload = json.dumps({
        "query": query,
        "answer": answer,
        "embedding": embedding,
        "stored_at": datetime.now(timezone.utc).isoformat(),
    })
    await redis.setex(key, settings.cache_ttl_seconds, payload)


# ---------------------------------------------------------------------------
# Legacy class — kept for backwards-compatibility with agent.py / tests.
# ---------------------------------------------------------------------------


class SemanticCache:
    """
    Stores (embedding, result) pairs in Redis with a TTL.
    On lookup, computes cosine similarity against all cached embeddings
    and returns the result if the best match exceeds the threshold.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._redis = Redis(
            url=self._settings.upstash_redis_rest_url,
            token=self._settings.upstash_redis_rest_token,
        )
        self._key_prefix = "deepsearch:cache:"

    async def lookup(self, query: str) -> dict[str, Any] | None:
        """Return a cached result if a sufficiently similar query exists."""
        from backend.embedder import embed_query

        query_vec = np.array(await embed_query(query), dtype=np.float32)
        keys = await self._redis.keys(f"{self._key_prefix}*")

        best_score = 0.0
        best_result: dict[str, Any] | None = None

        for key in keys:
            raw = await self._redis.get(key)
            if not raw:
                continue
            entry = json.loads(raw)
            cached_vec = np.array(entry["embedding"], dtype=np.float32)
            score = float(_cosine_similarity(query_vec, cached_vec))

            if score > best_score:
                best_score = score
                best_result = entry["result"]

        if best_score >= self._settings.cache_similarity_threshold:
            return best_result

        return None

    async def store(
        self,
        query: str,
        embedding: list[float],
        result: dict[str, Any],
    ) -> None:
        """Persist a query embedding + result with a TTL."""
        key = f"{self._key_prefix}{hash(query)}"
        payload = json.dumps({"embedding": embedding, "result": result})
        await self._redis.setex(
            key, self._settings.cache_ttl_seconds, payload
        )


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
