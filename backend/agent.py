"""Orchestration layer: runs the full DeepSearch agentic pipeline end-to-end.
Single responsibility: orchestrate tool-calling loop between the LLM and
the scraper / chunker / retriever modules.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from collections.abc import AsyncGenerator
from time import perf_counter
from typing import Any

import httpx
from openai import AsyncOpenAI
from types import SimpleNamespace

from backend.cache import SemanticCache
from backend.chunker import chunk_documents, chunk_text
from backend.config import get_settings
from backend.embedder import embed, embed_batch
from backend.retriever import hybrid_search, retrieve_chunks, upsert_chunks
from backend.scraper import scrape_url, scrape_urls
from backend.llm import synthesise_answer
from backend.model_router import FLASH, log_cost, route_model
from backend.security import sanitize
from backend.dspy_modules import generate_candidate as _dspy_candidate


logger = logging.getLogger("deepsearch")

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
    {
        "type": "function",
        "function": {
            "name": "create_artifact",
            "description": (
                "Render a side-panel artifact for long-form output. Use this when the "
                "answer is best presented as a standalone document rather than inline "
                "chat text. Pick the kind:\n"
                "- 'text' for research reports, summaries, or any markdown-formatted "
                "long-form answer (>300 words).\n"
                "- 'code' for code samples or scripts. Include the language as the "
                "first line of content as a markdown fence header.\n"
                "- 'sheet' for tabular data — content must be valid CSV.\n"
                "Do not use for short conversational replies. After calling this tool, "
                "your inline reply should be a 1-2 sentence summary referring the user "
                "to the artifact."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["text", "code", "sheet"]},
                    "title": {"type": "string", "description": "Short descriptive title."},
                    "content": {
                        "type": "string",
                        "description": (
                            "Full artifact content. Markdown for 'text', raw source "
                            "for 'code', CSV for 'sheet'."
                        ),
                    },
                },
                "required": ["kind", "title", "content"],
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
    "4. Always call retrieve_chunks at least once before composing your final answer.\n"
    "5. For long-form output (research reports >300 words, code samples, or tabular\n"
    "   data), call create_artifact instead of dumping it all inline. Keep the inline\n"
    "   reply to a 1-2 sentence summary that points to the artifact."
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
        f"src={c.get('source_url', '')}\n{c['text'][:1500]}"
        for i, c in enumerate(chunks)
    ]
    return "\n\n".join(lines)


def _chunk_contexts(chunks: list[dict[str, Any]]) -> list[str]:
    """Return raw chunk texts for downstream evaluation against used evidence."""
    return [str(c["text"]) for c in chunks if c.get("text")]


async def _run_retrieve_chunks(query: str) -> str:
    """Hybrid-search and return top chunks formatted for the LLM context window."""
    chunks = await hybrid_search(query)
    return _format_chunks(chunks)


async def _retrieve_chunks_with_contexts(query: str) -> tuple[str, list[str]]:
    """Hybrid-search and return both formatted and raw chunk text contexts."""
    chunks = await hybrid_search(query)
    return _format_chunks(chunks), _chunk_contexts(chunks)


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
    ``check_confidence`` before formatting. Low-confidence retrieval is passed
    back to the model as an insufficiency signal so it can search and scrape
    fresh sources before answering.

    The ``create_artifact`` tool emits an extra ``artifact`` event containing
    the structured artifact payload (kind/title/content). The UI bridge
    converts this to an AI SDK ``data-artifact`` part so the side panel can
    render it; the model itself just receives an acknowledgement string as the
    tool result.
    """
    history.append(message.model_dump(exclude_none=True))
    for tool_call in message.tool_calls or []:
        fn_name = tool_call.function.name
        fn_args = tool_call.function.arguments
        call_id = tool_call.id
        yield {"type": "tool_call", "call_id": call_id, "name": fn_name, "args": fn_args}

        started = perf_counter()
        logger.info("tool.start name=%s", fn_name)
        contexts: list[str] = []
        try:
            if fn_name == "retrieve_chunks":
                query = json.loads(fn_args).get("query", "")
                raw_chunks = await hybrid_search(query)
                refusal = check_confidence(raw_chunks)
                result = _format_chunks(raw_chunks)
                contexts = _chunk_contexts(raw_chunks)
                if refusal:
                    result = (
                        f"{result}\n\n"
                        f"Retrieval confidence warning: {refusal}\n"
                        "The retrieved context is insufficient. Use web_search, "
                        "scrape_and_index, and retrieve_chunks again before answering."
                    )
                    contexts = []
            elif fn_name == "create_artifact":
                args = json.loads(fn_args)
                artifact_id = str(uuid.uuid4())
                yield {
                    "type": "artifact",
                    "artifact_id": artifact_id,
                    "kind": args.get("kind", "text"),
                    "title": args.get("title", "Untitled"),
                    "content": args.get("content", ""),
                }
                result = (
                    f"Artifact rendered (id={artifact_id}, "
                    f"kind={args.get('kind')}, title={args.get('title')!r})."
                )
            else:
                result = await _dispatch_tool(fn_name, fn_args)
        except Exception:
            duration_ms = int((perf_counter() - started) * 1000)
            logger.exception("tool.error name=%s duration_ms=%s", fn_name, duration_ms)
            raise

        duration_ms = int((perf_counter() - started) * 1000)
        logger.info("tool.end name=%s duration_ms=%s", fn_name, duration_ms)

        history.append({"role": "tool", "tool_call_id": call_id, "content": result})
        yield {
            "type": "tool_result",
            "call_id": call_id,
            "name": fn_name,
            "content": result,
            "contexts": contexts,
        }


# ---------------------------------------------------------------------------
# Complexity routing
# ---------------------------------------------------------------------------


def complexity_score(question: str) -> int:
    """Return 1 for simple queries (FLASH-routed) or 3 for complex ones (PRO-routed).

    Mirrors the same heuristics used by route_model() so the two are always
    consistent — a score of 1 means the single-pass path is sufficient, while
    3 triggers multi-candidate synthesis + LLM judging.
    """
    return 1 if route_model(question) == FLASH else 3


# ---------------------------------------------------------------------------
# Multi-candidate synthesis helpers
# ---------------------------------------------------------------------------


async def _generate_candidates(question: str, contexts: str) -> list[str]:
    """Run SynthesizeAnswer 3 times in parallel and return the answer strings.

    DSPy predict calls are synchronous; asyncio.to_thread keeps them off the
    event loop so all three run concurrently without blocking.
    """

    async def _one() -> str:
        try:
            pred = await asyncio.to_thread(
                _dspy_candidate, question=question, contexts=contexts
            )
            answer = getattr(pred, "answer", "") or ""
            return answer
        except Exception:
            return ""

    results = await asyncio.gather(_one(), _one(), _one(), return_exceptions=True)
    candidates: list[str] = []
    for r in results:
        if isinstance(r, BaseException):
            candidates.append("")
        else:
            candidates.append(r)  # type: ignore[arg-type]
    return candidates


_JUDGE_PROMPT = """\
You are an impartial judge evaluating candidate answers to a research question.

Question: {question}

Retrieved context:
{contexts}

Pick the answer that is best supported by the retrieved context.
Reject candidates that include claims not present in the retrieved context, even if
they are useful or likely true. A longer answer can win, but only if every factual
claim is grounded in the context. Prefer concise answers when grounding is tied.

Candidates:
{candidates}

Respond with exactly one line:  BEST: <number>
"""


async def llm_judge(question: str, candidates: list[str], contexts: str) -> str:
    """Use the flash model to pick the best candidate via a scoring rubric.

    Returns the winning candidate string.  Falls back to candidates[0] if
    parsing fails or the list is empty.
    """
    if not candidates:
        return ""
    if len(candidates) == 1:
        return candidates[0]

    numbered = "\n\n".join(
        f"[{i + 1}]\n{c}" for i, c in enumerate(candidates) if c
    )
    prompt = _JUDGE_PROMPT.format(
        question=question,
        contexts=contexts,
        candidates=numbered,
    )

    settings = get_settings()
    response = await _openai_client.chat.completions.create(
        model=settings.flash_model,
        messages=[{"role": "user", "content": prompt}],
    )
    text = (response.choices[0].message.content or "").strip()

    match = re.search(r"BEST\s*:\s*(\d+)", text, re.IGNORECASE)
    if match:
        idx = int(match.group(1)) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx]

    return candidates[0]


def _latest_user_text(messages: list[dict[str, Any]]) -> str:
    """Extract plain-text content from the most recent user message.

    Handles both string content and OpenAI multimodal content arrays
    (``[{"type": "text", "text": "..."}, {"type": "image_url", ...}]``) by
    concatenating the text parts. Used for complexity routing and model
    selection — vision parts are forwarded to the LLM but don't influence
    routing.
    """
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ).strip()
    return ""


async def run_chat(
    messages: list[dict[str, Any]],
    model: str | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Multi-turn entry point: run the agent loop over a full messages array.

    Messages must already be in OpenAI chat-completion format. The Next.js
    proxy is responsible for translating AI SDK UIMessages into this format
    via ``convertToModelMessages`` before forwarding.

    ``model`` overrides the complexity-based router with a specific
    OpenRouter model id (e.g. ``openai/gpt-5.5``,
    ``google/gemini-3.1-pro``). When omitted or empty, falls back to the
    default routing.

    Yields the same event dicts as :func:`run_agent`. Routing (complexity,
    model selection) keys off the latest user message's text content.
    """
    if not messages:
        return

    question_text = _latest_user_text(messages)
    history: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        *messages,
    ]
    score = complexity_score(question_text) if question_text else 1
    if model:
        chosen_model = model
        # When the user explicitly picks a model, treat the request as
        # complex enough to engage the multi-candidate synthesis path
        # (the user is opting into the more expensive model on purpose).
        if score == 1 and ":" in model:  # heuristic: branded model id
            score = 1  # leave routing alone — only the *which* model
                       # changes, not the strategy
    else:
        chosen_model = route_model(question_text) if question_text else FLASH

    async for event in _agent_loop(history, score, chosen_model, question_text):
        yield event


async def run_agent(question: str) -> AsyncGenerator[dict[str, Any], None]:
    """Single-question entry (legacy path used by ``/search`` SSE endpoint).

    Wraps :func:`run_chat` with a one-element messages array. New callers
    should use :func:`run_chat` directly.
    """
    async for event in run_chat([{"role": "user", "content": question}]):
        yield event


async def _stream_completion(
    model: str,
    history: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Make a streaming chat-completion call to OpenRouter.

    Yields ``{"type": "text", "content": <delta>}`` for each text token as
    it arrives, then a final ``{"type": "_done", ...}`` event with an
    accumulated message-like object whose surface matches what
    :func:`_execute_tool_calls` expects from ``ChatCompletionMessage``:
    ``.role``, ``.content``, ``.tool_calls`` (each with ``.id``, ``.type``,
    ``.function.{name,arguments}``), and ``.model_dump(exclude_none=...)``.

    Tool-call deltas are merged by index across chunks; OpenAI's streaming
    protocol splits a single tool call's arguments into multiple chunks
    keyed by ``index`` rather than ``id``.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": history,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    response_stream = await _openai_client.chat.completions.create(**kwargs)

    accumulated_content = ""
    tool_calls_by_index: dict[int, dict[str, Any]] = {}
    finish_reason: str | None = None
    role = "assistant"
    usage = None

    async for chunk in response_stream:
        if getattr(chunk, "usage", None):
            usage = chunk.usage
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        delta = choice.delta

        if getattr(delta, "role", None):
            role = delta.role

        delta_content = getattr(delta, "content", None)
        if delta_content:
            accumulated_content += delta_content
            yield {"type": "text", "content": delta_content}

        delta_tool_calls = getattr(delta, "tool_calls", None) or []
        for tc_delta in delta_tool_calls:
            idx = tc_delta.index
            entry = tool_calls_by_index.setdefault(
                idx,
                {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
            )
            if tc_delta.id:
                entry["id"] = tc_delta.id
            if tc_delta.type:
                entry["type"] = tc_delta.type
            fn_delta = getattr(tc_delta, "function", None)
            if fn_delta:
                if fn_delta.name:
                    entry["function"]["name"] += fn_delta.name
                if fn_delta.arguments:
                    entry["function"]["arguments"] += fn_delta.arguments

        if choice.finish_reason:
            finish_reason = choice.finish_reason

    if usage is not None:
        await log_cost(model, usage.prompt_tokens, usage.completion_tokens)

    tool_calls_sorted = [tool_calls_by_index[i] for i in sorted(tool_calls_by_index)]
    tool_call_objs = [
        SimpleNamespace(
            id=tc["id"],
            type=tc["type"],
            function=SimpleNamespace(
                name=tc["function"]["name"],
                arguments=tc["function"]["arguments"],
            ),
        )
        for tc in tool_calls_sorted
    ]

    final_content = accumulated_content if accumulated_content else None

    def _model_dump(exclude_none: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {"role": role}
        if final_content is not None:
            out["content"] = final_content
        if tool_calls_sorted:
            out["tool_calls"] = tool_calls_sorted
        if exclude_none:
            return {k: v for k, v in out.items() if v is not None}
        return out

    message = SimpleNamespace(
        role=role,
        content=final_content,
        tool_calls=tool_call_objs or None,
        model_dump=_model_dump,
    )

    yield {"type": "_done", "message": message, "finish_reason": finish_reason}


async def _agent_loop(
    history: list[dict[str, Any]],
    score: int,
    model: str,
    question_text: str,
) -> AsyncGenerator[dict[str, Any], None]:
    """Agentic loop: tool-call until the model signals stop, then synthesize.

    For simple queries (complexity_score == 1) the model's final answer is
    streamed token-by-token as it arrives — the user sees text fill in
    rather than appearing all at once.  For complex queries (score == 3)
    three candidate answers are generated in parallel via the
    SynthesizeAnswer DSPy module and the best is selected by llm_judge;
    that winner is then word-streamed.

    Yields dicts with key ``type`` set to one of:
    - ``"tool_call"``   — the model is invoking a tool.
    - ``"tool_result"`` — the tool execution result.
    - ``"artifact"``    — a structured artifact created via create_artifact.
    - ``"text"``        — a streamed/yielded chunk of the final answer.
    """
    settings = get_settings()
    # Track the most recent retrieve_chunks result so the multi-candidate path
    # can pass grounded context into each SynthesizeAnswer call.
    last_context: str = ""

    while True:
        # For score==3 we suppress the first-pass tokens because the
        # multi-candidate path will replace them — letting the model's
        # answer briefly stream then yanking it would be jarring.
        suppress_text = score == 3
        streamed_message = None
        finish_reason: str | None = None

        async for event in _stream_completion(
            model=model, history=history, tools=TOOLS
        ):
            if event.get("type") == "_done":
                streamed_message = event["message"]
                finish_reason = event["finish_reason"]
                continue
            if event.get("type") == "text" and not suppress_text:
                yield event

        if streamed_message is None:
            break

        choice = SimpleNamespace(
            message=streamed_message,
            finish_reason=finish_reason,
        )

        if choice.finish_reason == "tool_calls":
            async for event in _execute_tool_calls(choice.message, history):
                if event["type"] == "tool_result" and event["name"] == "retrieve_chunks":
                    new_ctx = event["content"]
                    has_raw_contexts = bool(event.get("contexts"))
                    if has_raw_contexts:
                        # Use the final successful retrieval as the synthesis
                        # context. Earlier retrievals are often exploratory and
                        # lower context precision when carried into evaluation.
                        last_context = new_ctx[:settings.max_dspy_context_chars]
                yield event
            continue

        if choice.finish_reason == "stop":
            content = choice.message.content or ""

            if score == 3 and not last_context and not content:
                # The model skipped all tools and returned empty content on a
                # complex query — proactively retrieve context so the
                # multi-candidate path has grounded material to work with.
                last_context, forced_contexts = await _retrieve_chunks_with_contexts(question_text)
                forced_call_id = f"forced-retrieve-{uuid.uuid4().hex[:8]}"
                yield {
                    "type": "tool_call",
                    "call_id": forced_call_id,
                    "name": "retrieve_chunks",
                    "args": json.dumps({"query": question_text}),
                }
                yield {
                    "type": "tool_result",
                    "call_id": forced_call_id,
                    "name": "retrieve_chunks",
                    "content": last_context,
                    "contexts": forced_contexts,
                }

            if score == 3 and last_context:
                # ── Multi-candidate path (complex queries) ──────────────────
                candidates = await _generate_candidates(question_text, last_context)
                # The judge sees the same retrieved context as the generators,
                # so long answers can win when every claim is grounded.
                if content:
                    candidates.append(content)
                winner = await llm_judge(question_text, candidates, last_context)
                for word in re.split(r"(\s+)", winner):
                    if word:
                        yield {"type": "text", "content": word}
            # Single-pass path: text already streamed inside the
            # _stream_completion call above — nothing more to yield here.
            break

        # Unexpected finish reason (e.g. "length", "content_filter"): the
        # streamed call already emitted whatever partial content arrived
        # before the cut-off, so we just exit the loop.
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
