"""Web scraping and search utilities.
Single responsibility: turn a URL or list of URLs into cleaned plain text.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from playwright.async_api import Browser, Error as PlaywrightError
from playwright.async_api import Route, async_playwright
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from backend.config import get_settings

_BLOCKED_RESOURCE_TYPES: frozenset[str] = frozenset({"image", "font", "media"})

# CSS selector for low-signal structural and ad elements to strip before extraction.
_NOISE_SELECTOR = (
    "nav, header, footer, aside,"
    " [class*='ad-'], [class*='-ad'], [id*='ad-'], [id*='-ad'],"
    " [class*='banner'], [class*='cookie'], [class*='popup']"
)

_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    """Return the shared semaphore, initialising it on first call."""
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(get_settings().scrape_concurrency)
    return _semaphore


async def scrape_url(url: str) -> str:
    """Scrape `url` with headless Chromium and return filtered plain text."""
    settings = get_settings()
    async with _get_semaphore():
        async with async_playwright() as pw:
            browser: Browser = await pw.chromium.launch(headless=True)
            try:
                return await _scrape_page(browser, url, settings.scrape_timeout_seconds)
            finally:
                await browser.close()


async def scrape_urls(urls: list[str]) -> list[str]:
    """Scrape all URLs concurrently (semaphore-capped) and return texts in order."""
    return list(await asyncio.gather(*[scrape_url(url) for url in urls]))


async def search_web(query: str) -> list[str]:
    """Call Serper's Google Search API and return a list of result URLs."""
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


async def _scrape_page(browser: Browser, url: str, timeout_s: int) -> str:
    """Open a new page, block noise resources, strip DOM clutter, return text."""
    settings = get_settings()
    page = await browser.new_page()
    await page.route("**/*", _block_resource)

    try:
        await page.goto(url, timeout=timeout_s * 1000, wait_until="domcontentloaded")
    except (PlaywrightTimeoutError, PlaywrightError):
        await page.close()
        return ""

    await page.evaluate(
        f"() => document.querySelectorAll(`{_NOISE_SELECTOR}`).forEach(el => el.remove())"
    )

    text = await page.inner_text("body")
    await page.close()
    return _filter_lines(text, settings.scrape_min_line_length)


async def _block_resource(route: Route) -> None:
    """Abort image, font, and media requests to speed up page loads."""
    if route.request.resource_type in _BLOCKED_RESOURCE_TYPES:
        await route.abort()
    else:
        await route.continue_()


def _filter_lines(text: str, min_length: int) -> str:
    """Keep only stripped lines exceeding `min_length` characters."""
    return "\n".join(
        line.strip()
        for line in text.splitlines()
        if len(line.strip()) > min_length
    )
