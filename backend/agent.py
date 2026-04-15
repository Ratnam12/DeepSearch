"""
Orchestration layer: runs the full DeepSearch pipeline end-to-end.
Single responsibility: coordinate scraper → chunker → embedder →
retriever → cache → LLM synthesis in the correct order.
"""

from __future__ import annotations

from typing import Any

from backend.cache import SemanticCache
from backend.chunker import chunk_documents
from backend.config import get_settings
from backend.embedder import embed_query
from backend.retriever import retrieve_chunks
from backend.scraper import scrape_urls
from backend.llm import synthesise_answer


class DeepSearchAgent:
    """Stateless agent — safe to instantiate per request."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._cache = SemanticCache()

    async def run(self, query: str, use_cache: bool = True) -> dict[str, Any]:
        """
        Execute the full pipeline for a user query.

        Returns a dict with keys: answer, sources, cached, confidence.
        """
        if use_cache:
            cached = await self._cache.lookup(query)
            if cached:
                return {**cached, "cached": True}

        query_embedding = await embed_query(query)
        chunks = await retrieve_chunks(query_embedding)

        if not chunks:
            urls = await self._fetch_fresh_urls(query)
            raw_docs = await scrape_urls(urls)
            new_chunks = chunk_documents(raw_docs)
            await self._store_chunks(new_chunks)
            chunks = await retrieve_chunks(query_embedding)

        answer, confidence, sources = await synthesise_answer(query, chunks)

        result: dict[str, Any] = {
            "answer": answer,
            "sources": sources,
            "cached": False,
            "confidence": confidence,
        }

        if confidence >= self._settings.confidence_threshold:
            await self._cache.store(query, query_embedding, result)

        return result

    async def _fetch_fresh_urls(self, query: str) -> list[str]:
        """Call Serper to get candidate URLs for a query."""
        from backend.scraper import search_web
        return await search_web(query)

    async def _store_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """Embed and upsert chunks into Qdrant."""
        from backend.embedder import embed_chunks
        from backend.retriever import upsert_chunks
        embedded = await embed_chunks(chunks)
        await upsert_chunks(embedded)
