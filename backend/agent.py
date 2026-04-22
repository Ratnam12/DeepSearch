"""Orchestration layer: runs the full DeepSearch agentic pipeline end-to-end.
Single responsibility: orchestrate tool-calling loop between the LLM and
the scraper / chunker / retriever modules.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from openai import AsyncOpenAI

from backend.cache import SemanticCache
from backend.chunker import chunk_documents, chunk_text
from backend.config import get_settings
from backend.embedder import embed, embed_batch
from backend.retriever import hybrid_search, retrieve_chunks, upsert_chunks
from backend.scraper import scrape_url, scrape_urls
from backend.llm import synthesise_answer
from backend.model_router import log_cost, route_model
from backend.security import sanitize

# Module-level client — base_url and api_key are read once from config.
_settings = get_settings()
_openai_client = AsyncOpenAI(
    base_url=_settings.openrouter_base_url,
    api_key=_settings.openrouter_api_key,
)

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web via Serper and return organic results with URLs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query to look up."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scrape_and_index",
            "description": "Fetch a URL with headless Chromium, chunk the text, and upsert into the vector store.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to scrape and index."},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_chunks",
            "description": "Hybrid-search the vector index for chunks most relevant to a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Query for chunk retrieval."},
                },
                "required": ["query"],
            },
        },
    },
]

_SYSTEM_PROMPT = (
    "You are DeepSearch, a research assistant backed by a live knowledge base.\n"
    "Rules:\n"
    "1. Cite every factual claim inline using bracketed source indices, e.g. [1], [2].\n"
    "2. Never assert facts not present in the retrieved chunks.\n"
    "3. If the retrieved context is insufficient, call web_search to find relevant URLs,\n"
    "   then scrape_and_index to ingest each one, then retrieve_chunks before answering.\n"
    "4. Always call retrieve_chunks at least once before composing your final answer."
)


async def _run_web_search(query: str) -> str:
    """POST to Serper and return formatted organic results."""
    settings = get_settings()
    headers = {"X-API-KEY": settings.serper_api_key, "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=15) as http:
        resp = await http.post(
            "https://google.serper.dev/search",
            json={"q": query, "num": 10},
            headers=headers,
        )
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
    organic = data.get("organic", [])
    lines = [
        f"[{i + 1}] {r.get('title', '')} — {r.get('link', '')}\n{r.get('snippet', '')}"
        for i, r in enumerate(organic)
    ]
    return "\n\n".join(lines) or "No results found."


async def _run_scrape_and_index(url: str) -> str:
    """Scrape *url*, sanitize against prompt injection, chunk, and upsert."""
    text = await scrape_url(url)
    if not text:
        return f"No content extracted from {url}."
    result = sanitize(url, text)
    if result["is_suspicious"]:
        print(
            f"[security] injection patterns detected in {url}: "
            f"{result['patterns']}"
        )
    chunks = chunk_text(result["safe_text"], source_url=url)
    await upsert_chunks(chunks)
    return f"Indexed {len(chunks)} chunks from {url}."


def check_confidence(chunks: list[dict]) -> str | None:
    """Return a refusal string when retrieval quality is too low, else None.

    Returns None immediately when *chunks* is empty (no retrieval occurred at
    all — the caller should handle the empty case separately).  When chunks are
    present, compares the best rerank_score against config.confidence_threshold:
    a score below the threshold means the retrieved context is not trustworthy
    enough to pass to the LLM.
    """
    if not chunks:
        return None
    settings = get_settings()
    max_score = max(c.get("rerank_score", c.get("score", 0.0)) for c in chunks)
    threshold = settings.confidence_threshold
    if max_score < threshold:
        return (
            f"I wasn't able to find sufficiently relevant information to answer "
            f"confidently. (best relevance score: {max_score:.3f}, "
            f"required threshold: {threshold:.3f})"
        )
    return None


def _format_chunks(chunks: list[dict]) -> str:
    """Format raw chunk dicts into a string suitable for the LLM context window."""
    if not chunks:
        return "No relevant chunks found."
    lines = [
        f"[{i + 1}] score={c.get('rerank_score', c.get('score', 0)):.3f} "
        f"src={c.get('source_url', '')}\n{c['text'][:600]}"
        for i, c in enumerate(chunks)
    ]
    return "\n\n".join(lines)


async def _run_retrieve_chunks(query: str) -> str:
    """Hybrid-search and return top chunks formatted for the LLM context window."""
    chunks = await hybrid_search(query)
    return _format_chunks(chunks)


async def _dispatch_tool(name: str, arguments: str) -> str:
    """Route a tool call by name and return its string result."""
    args: dict[str, Any] = json.loads(arguments)
    match name:
        case "web_search":
            return await _run_web_search(args["query"])
        case "scrape_and_index":
            return await _run_scrape_and_index(args["url"])
        case "retrieve_chunks":
            return await _run_retrieve_chunks(args["query"])
        case _:
            return f"Unknown tool: {name}"


async def _execute_tool_calls(
    message: Any,
    history: list[dict[str, Any]],
) -> AsyncGenerator[dict[str, Any], None]:
    """Execute every tool_call on *message*, mutate *history*, and yield events.

    For ``retrieve_chunks`` calls the raw chunk list is inspected via
    ``check_confidence`` before formatting.  A ``"low_confidence"`` event is
    yielded when the best score falls below the configured threshold so that
    ``run_agent`` can abort the pipeline early.
    """
    history.append(message.model_dump(exclude_none=True))
    for tool_call in message.tool_calls or []:
        fn_name = tool_call.function.name
        fn_args = tool_call.function.arguments
        yield {"type": "tool_call", "name": fn_name, "args": fn_args}

        if fn_name == "retrieve_chunks":
            query = json.loads(fn_args).get("query", "")
            raw_chunks = await hybrid_search(query)
            refusal = check_confidence(raw_chunks)
            if refusal:
                yield {"type": "low_confidence", "message": refusal}
            result = _format_chunks(raw_chunks)
        else:
            result = await _dispatch_tool(fn_name, fn_args)

        history.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
        yield {"type": "tool_result", "name": fn_name, "content": result}


async def run_agent(question: str) -> AsyncGenerator[dict[str, Any], None]:
    """Agentic loop: tool-call until the model signals stop, then stream the answer.

    Yields dicts with key ``type`` set to one of:
    - ``"tool_call"``   — the model is invoking a tool.
    - ``"tool_result"`` — the tool execution result.
    - ``"text"``        — a streamed chunk of the final answer.
    """
    settings = get_settings()
    history: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    model = route_model(question)
    while True:
        response = await _openai_client.chat.completions.create(
            model=model,
            messages=history,
            tools=TOOLS,
            tool_choice="auto",
        )
        choice = response.choices[0]
        usage = response.usage
        if usage:
            await log_cost(model, usage.prompt_tokens, usage.completion_tokens)

        if choice.finish_reason == "tool_calls":
            low_confidence_message: str | None = None
            async for event in _execute_tool_calls(choice.message, history):
                if event["type"] == "low_confidence":
                    low_confidence_message = event["message"]
                    continue
                yield event
            if low_confidence_message is not None:
                yield {"type": "text", "content": low_confidence_message}
                return
            continue

        if choice.finish_reason == "stop":
            stream = await _openai_client.chat.completions.create(
                model=model,
                messages=history,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield {"type": "text", "content": delta}
            break

        # Guard: exit on unexpected finish reasons (e.g. "length", "content_filter").
        
        break
        


# ---------------------------------------------------------------------------
# Legacy pipeline — kept for backwards-compatibility with router.py / tests.
# ---------------------------------------------------------------------------


class DeepSearchAgent:
    """Stateless agent — safe to instantiate per request."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._cache = SemanticCache()

    async def run(self, query: str, use_cache: bool = True) -> dict[str, Any]:
        """Execute the full pipeline for a user query.

        Returns a dict with keys: answer, sources, cached, confidence.
        """
        if use_cache:
            cached = await self._cache.lookup(query)
            if cached:
                return {**cached, "cached": True}

        query_embedding = await embed(query)
        chunks = await retrieve_chunks(query_embedding)

        if not chunks:
            urls = await self._fetch_fresh_urls(query)
            texts = await scrape_urls(urls)
            raw_docs = []
            for url, text in zip(urls, texts):
                if not text:
                    continue
                san = sanitize(url, text)
                if san["is_suspicious"]:
                    print(
                        f"[security] injection patterns detected in {url}: "
                        f"{san['patterns']}"
                    )
                raw_docs.append({"url": url, "text": san["safe_text"]})
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
        embeddings = await embed_batch([c["text"] for c in chunks])
        for chunk, vector in zip(chunks, embeddings):
            chunk["embedding"] = vector
        await upsert_chunks(chunks)
