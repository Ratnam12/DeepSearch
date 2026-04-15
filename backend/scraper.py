"""
Web scraping and search utilities.
Single responsibility: turn a query into raw text documents.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from backend.config import get_settings


async def search_web(query: str) -> list[str]:
    """
    Call Serper's Google Search API and return a list of result URLs.
    Returns at most top_k_retrieval URLs.
    """
    settings = get_settings()

    payload = {"q": query, "num": settings.top_k_retrieval}
    headers = {
        "X-API-KEY": settings.serper_api_key,
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.post(
            "https://google.serper.dev/search",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()

    return [item["link"] for item in data.get("organic", [])]


async def scrape_urls(urls: list[str]) -> list[dict[str, str]]:
    """
    Fetch each URL concurrently and return a list of
    {"url": ..., "text": ...} dicts. Failed fetches are skipped silently.
    """
    tasks = [_fetch_one(url) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    return [r for r in results if isinstance(r, dict)]


async def _fetch_one(url: str) -> dict[str, str]:
    """Fetch a single URL and strip it down to plain text."""
    async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
        response = await client.get(url, headers={"User-Agent": "DeepSearch/0.1"})
        response.raise_for_status()
        return {"url": url, "text": response.text}
