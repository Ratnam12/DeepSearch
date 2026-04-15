"""
Semantic cache backed by Redis.
Single responsibility: store and retrieve previous answers by embedding
similarity so identical (or near-identical) queries skip the full pipeline.
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import redis.asyncio as aioredis

from backend.config import get_settings


class SemanticCache:
    """
    Stores (embedding, result) pairs in Redis with a TTL.
    On lookup, computes cosine similarity against all cached embeddings
    and returns the result if the best match exceeds the threshold.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._redis: aioredis.Redis = aioredis.from_url(
            self._settings.redis_url, decode_responses=True
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
