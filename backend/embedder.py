"""Embedding utilities using OpenAI text-embedding-3-small.
Single responsibility: convert text → dense float vectors.

OpenAI's embedding models are not available on OpenRouter, so this module
uses the OpenAI client directly with OPENAI_API_KEY.
"""

from __future__ import annotations

import numpy as np
from openai import AsyncOpenAI

from backend.config import get_settings

# Single shared client instance, initialised on first use so tests can patch
# get_settings() before the client is constructed.
_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=get_settings().openai_api_key)
    return _client


async def embed(text: str) -> list[float]:
    """Return the embedding vector for a single text string.

    Input is silently truncated to embed_max_chars characters before the API
    call — text-embedding-3-small has an 8 191-token context window and
    truncating by character is a safe, cheap approximation.
    """
    settings = get_settings()
    truncated = text[: settings.embed_max_chars]
    response = await _get_client().embeddings.create(
        model=settings.embedding_model,
        input=truncated,
    )
    return response.data[0].embedding


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed every string in `texts` in a single API call.

    Results are sorted by the index field returned by the API so the output
    order always matches the input order, regardless of server-side reordering.
    """
    settings = get_settings()
    truncated = [t[: settings.embed_max_chars] for t in texts]
    response = await _get_client().embeddings.create(
        model=settings.embedding_model,
        input=truncated,
    )
    return [e.embedding for e in sorted(response.data, key=lambda e: e.index)]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return the cosine similarity between two embedding vectors."""
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    norm_a = float(np.linalg.norm(va))
    norm_b = float(np.linalg.norm(vb))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))
