"""
Embedding utilities using OpenAI text-embedding-3-small.
Single responsibility: convert text → dense float vectors.

OpenAI's embedding models are not available on OpenRouter, so this module
uses the OpenAI client directly with OPENAI_API_KEY.
"""

from __future__ import annotations

from openai import AsyncOpenAI

from backend.config import get_settings

_EMBEDDING_MODEL = "text-embedding-3-small"


def _get_client() -> AsyncOpenAI:
    settings = get_settings()
    return AsyncOpenAI(api_key=settings.openai_api_key)


async def embed_query(text: str) -> list[float]:
    """Return the embedding vector for a single query string."""
    client = _get_client()
    response = await client.embeddings.create(
        model=_EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


async def embed_chunks(
    chunks: list[dict],
) -> list[dict]:
    """
    Add an "embedding" key to each chunk dict in-place and return the list.
    Batches all texts in a single API call to minimise latency and cost.
    """
    client = _get_client()
    texts = [chunk["text"] for chunk in chunks]

    response = await client.embeddings.create(
        model=_EMBEDDING_MODEL,
        input=texts,
    )

    for chunk, embedding_obj in zip(chunks, response.data):
        chunk["embedding"] = embedding_obj.embedding

    return chunks
