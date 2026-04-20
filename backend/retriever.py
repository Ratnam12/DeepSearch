"""
Vector retrieval via Qdrant.
Single responsibility: upsert and query the vector store.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any
from uuid import uuid4

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    Fusion,
    FusionQuery,
    PointStruct,
    Prefetch,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from backend.config import get_settings
from backend.embedder import embed, embed_batch

_COLLECTION = "deepsearch"
_VECTOR_SIZE = 1536   # text-embedding-3-small output dimension
_SPARSE_DIM = 65536   # hash space for sparse word-frequency vectors

_client: AsyncQdrantClient | None = None


def _get_client() -> AsyncQdrantClient:
    global _client
    if _client is None:
        settings = get_settings()
        _client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
    return _client


async def ensure_collection() -> None:
    """Create the Qdrant collection if it does not already exist.

    The collection uses two named vector spaces:
    - ``dense``  — 1536-dim float vectors with cosine distance.
    - ``sparse`` — sparse word-frequency vectors for hybrid search.
    """
    client = _get_client()
    existing = await client.get_collections()
    names = [c.name for c in existing.collections]

    if _COLLECTION not in names:
        await client.create_collection(
            collection_name=_COLLECTION,
            vectors_config={
                "dense": VectorParams(size=_VECTOR_SIZE, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(),
            },
        )


def _build_sparse_vector(text: str) -> SparseVector:
    """Return a sparse vector built from word frequencies using hash bucketing.

    Each unique word is bucketed via ``hash(word) % 65536``.  Collisions are
    handled by accumulating frequencies into the same bucket index.
    """
    words = re.findall(r"\w+", text.lower())
    counts = Counter(words)

    bucket: dict[int, float] = {}
    for word, freq in counts.items():
        idx = hash(word) % _SPARSE_DIM
        bucket[idx] = bucket.get(idx, 0.0) + float(freq)

    indices = sorted(bucket.keys())
    values = [bucket[i] for i in indices]
    return SparseVector(indices=indices, values=values)


async def upsert_chunks(chunks: list[dict[str, Any]]) -> None:
    """Embed *chunks* and upsert them into Qdrant with dense + sparse vectors.

    Each chunk dict is stored verbatim as the point payload so all fields
    (``text``, ``source_url``, etc.) are retrievable after search.
    """
    if not chunks:
        return

    await ensure_collection()
    client = _get_client()

    texts = [chunk.get("text", "") for chunk in chunks]
    dense_vectors = await embed_batch(texts)

    points = [
        PointStruct(
            id=str(uuid4()),
            vector={
                "dense": dense_vectors[i],
                "sparse": _build_sparse_vector(texts[i]),
            },
            payload=chunk,
        )
        for i, chunk in enumerate(chunks)
    ]

    await client.upsert(collection_name=_COLLECTION, points=points)


async def retrieve_chunks(
    query_embedding: list[float],
) -> list[dict[str, Any]]:
    """Return the top-k most similar chunks for a query embedding."""
    settings = get_settings()
    client = _get_client()

    results = await client.search(
        collection_name=_COLLECTION,
        query_vector=("dense", query_embedding),
        limit=settings.top_k_final,
        with_payload=True,
    )

    return [
        {
            **hit.payload,
            "score": hit.score,
        }
        for hit in results
    ]


async def hybrid_search(query: str) -> list[dict[str, Any]]:
    """Retrieve chunks via hybrid dense + sparse search with RRF fusion.

    Two ``Prefetch`` legs run in parallel inside Qdrant:
    - **dense**  — ANN over the ``dense`` vector space (semantic similarity).
    - **sparse** — exact dot-product over the ``sparse`` vector space (lexical).

    Qdrant merges the two candidate sets with Reciprocal Rank Fusion before
    returning the final top-``top_k_final`` results.
    """
    settings = get_settings()
    client = _get_client()

    dense_vec = await embed(query)
    sparse_vec = _build_sparse_vector(query)

    response = await client.query_points(
        collection_name=_COLLECTION,
        prefetch=[
            Prefetch(
                query=dense_vec,
                using="dense",
                limit=settings.top_k_retrieval,
            ),
            Prefetch(
                query=sparse_vec,
                using="sparse",
                limit=settings.top_k_retrieval,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=settings.top_k_final,
        with_payload=True,
    )

    return [
        {
            "text": hit.payload.get("text", ""),
            "source_url": hit.payload.get("source_url", ""),
            "title": hit.payload.get("title", ""),
            "score": hit.score,
        }
        for hit in response.points
    ]
