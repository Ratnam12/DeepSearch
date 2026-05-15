"""
Semantic cache backed by Redis.
Single responsibility: store and retrieve previous answers by embedding
similarity so identical (or near-identical) queries skip the full pipeline.

Above-threshold candidates are gated through a relevance judge (a small LLM
via OpenRouter) before being served, so high cosine similarity alone never
returns a stale or off-scope answer. See backend.config for the judge
model, timeout, and enable flag.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

from openai import AsyncOpenAI
from upstash_redis.asyncio import Redis

from backend.config import get_settings
from backend.embedder import cosine_similarity, embed

logger = logging.getLogger("deepsearch.cache")

_KEY_PREFIX = "ds:cache:"
_LEGACY_KEY_PREFIX = "deepsearch:cache:"
_redis: Redis | None = None
_judge_client: AsyncOpenAI | None = None


_JUDGE_SYSTEM_PROMPT = """You are a strict relevance gate for a semantic cache.

A user has asked a NEW question. The cache returned a previously-stored question and answer.
Your job: decide whether serving that cached answer would truly answer the NEW question.

A meaningful mismatch makes the cached answer insufficient. Mismatches include:
- different time period or recency (e.g. 2023 vs 2026, "yesterday" vs "now")
- different version of a product / software / framework
- different geographic, demographic, or organizational scope
- different entity (especially same-name collisions)
- different aspect or sub-topic of the same entity
- different level of depth or technical audience
- the new question asks for something the cached answer doesn't actually contain

If the cached answer fully and accurately answers the new question, return:
  {"sufficient": true, "reason": "<one short sentence>"}

If there is any meaningful mismatch, return:
  {"sufficient": false, "reason": "<one short sentence naming the mismatch>"}

When in genuine doubt, prefer false. Output JSON only — no preamble."""


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


def _get_judge_client() -> AsyncOpenAI:
    """OpenRouter client used only for the relevance-judge calls."""
    global _judge_client
    if _judge_client is None:
        settings = get_settings()
        _judge_client = AsyncOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
        )
    return _judge_client


def _stable_key(query: str, prefix: str = _KEY_PREFIX) -> str:
    """Return a deterministic, collision-resistant Redis key for a query string."""
    digest = hashlib.sha256(query.encode()).hexdigest()[:24]
    return f"{prefix}{digest}"


async def _judge_cache_relevance(
    probe_query: str,
    cached_query: str,
    cached_answer: str,
) -> bool:
    """Ask the relevance judge whether the cached answer truly answers the probe.

    Returns True only when the judge explicitly approves. Timeouts, parse
    errors, network failures, and the disabled flag all degrade safely:
    timeouts / errors return False (treat as miss — never serve a candidate
    the judge could not validate); the disabled flag returns True (preserve
    pre-judge behaviour for tests or emergency rollback).
    """
    settings = get_settings()
    if not settings.cache_judge_enabled:
        return True

    user_msg = (
        f"NEW question: {probe_query}\n"
        f"CACHED question: {cached_query or '(unknown — legacy cache entry)'}\n"
        f"CACHED answer: {cached_answer}"
    )

    t0 = time.perf_counter()
    try:
        client = _get_judge_client()
        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model=settings.cache_judge_model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            ),
            timeout=settings.cache_judge_timeout_seconds,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        raw = resp.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        verdict = bool(parsed.get("sufficient"))
        logger.info(
            "cache.judge verdict=%s latency_ms=%.0f probe=%r cached_q=%r reason=%r",
            verdict,
            latency_ms,
            probe_query[:80],
            (cached_query or "")[:80],
            str(parsed.get("reason", ""))[:140],
        )
        return verdict
    except asyncio.TimeoutError:
        logger.warning(
            "cache.judge timeout after %.1fs — treating as miss probe=%r",
            settings.cache_judge_timeout_seconds,
            probe_query[:80],
        )
        return False
    except Exception:
        logger.exception("cache.judge failed — treating as miss probe=%r", probe_query[:80])
        return False


def _extract_answer(entry: dict[str, Any]) -> str | None:
    """Pull the answer text out of either cache schema (new or legacy)."""
    if "answer" in entry:
        return entry["answer"]
    if isinstance(entry.get("result"), dict):
        return entry["result"].get("answer")
    return None


async def cache_lookup(query: str) -> str | None:
    """Return the cached answer for *query* when both gates pass:
    (a) cosine similarity to the best stored embedding meets the threshold, and
    (b) the relevance judge approves serving it.
    Returns ``None`` on miss, rejection, or any judge failure.
    """
    settings = get_settings()
    redis = _get_redis()
    query_embedding = await embed(query)

    keys: list[str] = []
    for prefix in (_KEY_PREFIX, _LEGACY_KEY_PREFIX):
        keys.extend(await redis.keys(f"{prefix}*"))

    best_score = 0.0
    best_query: str = ""
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
            best_query = entry.get("query") or ""
            best_answer = _extract_answer(entry)

    if best_score < settings.cache_similarity_threshold or best_answer is None:
        return None

    logger.info(
        "cache.candidate score=%.4f probe=%r cached_q=%r",
        best_score, query[:80], best_query[:80],
    )
    if await _judge_cache_relevance(query, best_query, best_answer):
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
    and returns the result dict if the best match exceeds the threshold
    AND the relevance judge approves the candidate.
    """

    def __init__(self) -> None:
        self._key_prefix = _LEGACY_KEY_PREFIX

    async def lookup(self, query: str) -> dict[str, Any] | None:
        """Return a cached result dict if a sufficiently similar query exists
        AND the relevance judge approves serving its answer."""
        redis = _get_redis()
        settings = get_settings()
        query_vec = await embed(query)
        keys = await redis.keys(f"{self._key_prefix}*")

        best_score = 0.0
        best_query: str = ""
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
                best_query = entry.get("query") or ""
                best_result = entry.get("result")

        if best_score < settings.cache_similarity_threshold or best_result is None:
            return None

        # Judge against the candidate's answer text, if extractable.
        candidate_answer = (
            best_result.get("answer", "") if isinstance(best_result, dict) else ""
        )
        logger.info(
            "cache.candidate (legacy) score=%.4f probe=%r cached_q=%r",
            best_score, query[:80], best_query[:80],
        )
        if await _judge_cache_relevance(query, best_query, candidate_answer):
            return best_result
        return None

    async def store(
        self,
        query: str,
        embedding: list[float],
        result: dict[str, Any],
    ) -> None:
        """Persist a query embedding + result dict with a TTL.

        Also records the raw ``query`` string in the payload so the relevance
        judge can compare cached-intent vs new-intent at lookup time. Older
        entries without this field still work (the judge sees an empty
        cached_query and reasons from the answer alone).
        """
        redis = _get_redis()
        settings = get_settings()
        key = _stable_key(query, self._key_prefix)
        payload = json.dumps({
            "query": query,
            "embedding": embedding,
            "result": result,
        })
        await redis.setex(key, settings.cache_ttl_seconds, payload)
