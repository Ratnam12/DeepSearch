"""
Vector retrieval via Qdrant.
Single responsibility: upsert and query the vector store.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from backend.config import get_settings

_COLLECTION = "deepsearch_chunks"
_VECTOR_SIZE = 1536  # text-embedding-3-small output dimension


def _get_client() -> AsyncQdrantClient:
    settings = get_settings()
    return AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
    )


async def ensure_collection() -> None:
    """Create the Qdrant collection if it does not already exist."""
    client = _get_client()
    existing = await client.get_collections()
    names = [c.name for c in existing.collections]

    if _COLLECTION not in names:
        await client.create_collection(
            collection_name=_COLLECTION,
            vectors_config=VectorParams(size=_VECTOR_SIZE, distance=Distance.COSINE),
        )


async def upsert_chunks(chunks: list[dict[str, Any]]) -> None:
    """Write embedded chunks into Qdrant."""
    await ensure_collection()
    client = _get_client()

    points = [
        PointStruct(
            id=str(uuid4()),
            vector=chunk["embedding"],
            payload={"text": chunk["text"], "url": chunk["url"]},
        )
        for chunk in chunks
        if "embedding" in chunk
    ]

    if points:
        await client.upsert(collection_name=_COLLECTION, points=points)


async def retrieve_chunks(
    query_embedding: list[float],
) -> list[dict[str, Any]]:
    """Return the top-k most similar chunks for a query embedding."""
    settings = get_settings()
    client = _get_client()

    results = await client.search(
        collection_name=_COLLECTION,
        query_vector=query_embedding,
        limit=settings.top_k_final,
        with_payload=True,
    )

    return [
        {"text": hit.payload["text"], "url": hit.payload["url"], "score": hit.score}
        for hit in results
    ]
