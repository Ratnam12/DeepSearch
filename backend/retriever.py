"""
Vector retrieval via Qdrant.
Single responsibility: upsert and query the vector store.

Retrieval stack (post-BM25 / BGE upgrade):
- Dense: ``text-embedding-3-small`` (1536-dim, cosine).
- Sparse: ``Qdrant/bm25`` via FastEmbed — proper Okapi BM25 with IDF and no
  hash collisions. Replaces the prior hash-bucketed word-frequency vector
  which was effectively noise at ~76% collision load.
- Fusion: Reciprocal Rank Fusion inside Qdrant ``query_points``.
- Rerank: ``BAAI/bge-reranker-v2-m3`` (Apache-2.0, 568M params) — ~+0.10
  nDCG@10 over the prior ms-marco-MiniLM-L-6-v2 on BEIR.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any
from uuid import uuid4

from fastembed import SparseTextEmbedding
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
from sentence_transformers import CrossEncoder

from backend.config import get_settings
from backend.embedder import embed, embed_batch

_COLLECTION = "deepsearch"
_VECTOR_SIZE = 1536   # text-embedding-3-small output dimension
_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
_BM25_MODEL = "Qdrant/bm25"

_client: AsyncQdrantClient | None = None
_reranker: CrossEncoder | None = None
_bm25: SparseTextEmbedding | None = None
# Serialize lazy-init across threads. Without this, concurrent ``to_thread``
# calls race on first use and tqdm (which fastembed's HF downloader uses)
# trips over its uninitialised class ``_lock`` attribute. The locks are also
# defensive against the reranker download having the same shape.
_bm25_lock = threading.Lock()
_reranker_lock = threading.Lock()


def _select_reranker_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                _reranker = CrossEncoder(_RERANKER_MODEL, device=_select_reranker_device())
    return _reranker


def _get_bm25() -> SparseTextEmbedding:
    """Lazy-init the BM25 sparse model. First call triggers a ~6MB download.

    Double-checked locking — the first concurrent caller pays the download
    cost; the rest wait on the lock and then hit the cached instance. The
    lock is necessary because FastEmbed downloads via HuggingFace's snapshot
    downloader, which uses tqdm in a thread pool; without serialisation the
    first run races on tqdm's class state.
    """
    global _bm25
    if _bm25 is None:
        with _bm25_lock:
            if _bm25 is None:
                _bm25 = SparseTextEmbedding(_BM25_MODEL)
    return _bm25


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


async def reset_collection() -> None:
    """Delete and recreate the Qdrant collection used by DeepSearch."""
    client = _get_client()
    existing = await client.get_collections()
    names = [c.name for c in existing.collections]
    if _COLLECTION in names:
        await client.delete_collection(collection_name=_COLLECTION)
    await ensure_collection()


def _build_sparse_doc_vectors(texts: list[str]) -> list[SparseVector]:
    """Build BM25 sparse vectors for document indexing (IDF-weighted)."""
    bm25 = _get_bm25()
    return [
        SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
        for e in bm25.embed(texts)
    ]


def _build_sparse_query_vector(text: str) -> SparseVector:
    """Build a BM25 sparse vector for a single query (term-presence weighted).

    ``query_embed`` skips IDF on the query side per the Okapi BM25 spec —
    the IDF lives in the indexed document vectors.
    """
    bm25 = _get_bm25()
    [embedding] = list(bm25.query_embed([text]))
    return SparseVector(
        indices=embedding.indices.tolist(),
        values=embedding.values.tolist(),
    )


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
    # BM25 embedding is CPU-only and synchronous — offload to a thread so it
    # doesn't block the event loop on large batches.
    sparse_vectors = await asyncio.to_thread(_build_sparse_doc_vectors, texts)

    points = [
        PointStruct(
            id=str(uuid4()),
            vector={
                "dense": dense_vectors[i],
                "sparse": sparse_vectors[i],
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


async def _candidates_for_query(
    query: str,
    settings: Any,
    client: AsyncQdrantClient,
) -> list[tuple[Any, dict[str, Any]]]:
    """Run one hybrid retrieval round and return ``(point_id, chunk_dict)`` pairs.

    Returns the raw fused candidates pre-rerank. Splitting this out lets
    :func:`hybrid_search_multi` collect candidates from several sub-queries
    and dedupe by point id before paying for a single rerank pass over the
    union.
    """
    dense_vec = await embed(query)
    sparse_vec = await asyncio.to_thread(_build_sparse_query_vector, query)

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
        limit=settings.top_k_retrieval,
        with_payload=True,
    )
    return [
        (
            hit.id,
            {
                "text": hit.payload.get("text", ""),
                "source_url": hit.payload.get("source_url", ""),
                "title": hit.payload.get("title", ""),
                "score": hit.score,
            },
        )
        for hit in response.points
    ]


async def hybrid_search(query: str) -> list[dict[str, Any]]:
    """Retrieve chunks via hybrid dense + sparse search, then cross-encoder reranking.

    Pipeline:
    1. Two ``Prefetch`` legs (dense ANN + sparse BM25) run in parallel inside
       Qdrant and are merged with Reciprocal Rank Fusion. The fused recall set
       is sized to ``top_k_retrieval`` so the reranker sees a wide candidate
       pool.
    2. The cross-encoder scores every (query, chunk) pair in one batch call
       and the results are sorted by ``rerank_score`` descending.
    3. Only the top ``top_k_final`` chunks are returned.
    """
    settings = get_settings()
    client = _get_client()

    id_chunk_pairs = await _candidates_for_query(query, settings, client)
    candidates = [c for _, c in id_chunk_pairs]

    if not candidates:
        return []

    reranker = _get_reranker()
    pairs = [(query, c["text"]) for c in candidates]

    # predict() is CPU-bound; offload to a thread to avoid blocking the loop.
    rerank_scores: list[float] = await asyncio.to_thread(reranker.predict, pairs)

    for chunk, rerank_score in zip(candidates, rerank_scores):
        chunk["rerank_score"] = float(rerank_score)

    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    return candidates[: settings.top_k_final]


async def hybrid_search_multi(
    queries: list[str],
    primary_query: str,
) -> list[dict[str, Any]]:
    """Retrieve via hybrid search across several sub-queries with one shared rerank pass.

    Each sub-query in ``queries`` produces its own dense+sparse fused candidate
    list. The union (deduped by Qdrant point id) is scored by the cross-encoder
    against ``primary_query`` — typically the user's original question — so the
    reranker concentrates precision on what the user actually asked, while the
    sub-queries cast a wider recall net.

    This is the multi-claim path for synthesis questions whose ground truths
    span orthogonal subspaces (e.g. "primary failure modes of a RAG pipeline").
    A single embedding cannot span those subspaces; running retrieval per
    decomposed sub-query and reranking the union solves that.
    """
    if not queries:
        return []
    settings = get_settings()
    client = _get_client()

    # Run all hybrid retrievals concurrently.
    candidate_lists = await asyncio.gather(
        *[_candidates_for_query(q, settings, client) for q in queries]
    )

    seen_ids: set = set()
    candidates: list[dict[str, Any]] = []
    for pairs in candidate_lists:
        for point_id, chunk in pairs:
            if point_id in seen_ids:
                continue
            seen_ids.add(point_id)
            candidates.append(chunk)

    if not candidates:
        return []

    reranker = _get_reranker()
    rerank_pairs = [(primary_query, c["text"]) for c in candidates]
    rerank_scores: list[float] = await asyncio.to_thread(reranker.predict, rerank_pairs)

    for chunk, rerank_score in zip(candidates, rerank_scores):
        chunk["rerank_score"] = float(rerank_score)

    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    return candidates[: settings.top_k_final]
