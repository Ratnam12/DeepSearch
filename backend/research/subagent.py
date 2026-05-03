"""One sub-agent — answers a single sub-question and returns a cited finding.

Phase-3 entry point for the actual research work. The supervisor
(:mod:`backend.research.supervisor`) spawns N of these in parallel,
one per ``ResearchSubQuestion`` in the approved plan.

A sub-agent runs a bounded tool-calling loop using the same primitives
the chat agent already wires up — :func:`_run_web_search`,
:func:`_run_scrape_and_index`, :func:`hybrid_search` — but with a
focused system prompt that produces a self-contained markdown finding
rather than an artifact + chat reply.

Stays offline-friendly: when ``DISABLE_SUBAGENT_LLM=true`` the
sub-agent returns a deterministic stub finding so integration tests
can exercise the whole pipeline without an OpenRouter key.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx
from openai import APIError, AsyncOpenAI

from backend.agent import (
    _run_scrape_and_index,
    _run_web_search,
)
from backend.config import get_settings
from backend.research.db import get_pool
from backend.research.events import append_event
from backend.research.state import is_cancelled
from backend.retriever import hybrid_search

logger = logging.getLogger("deepsearch.research.subagent")


# Bounded so a runaway sub-agent can't burn budget on infinite tool
# calls. ``MAX_TOOL_ITERATIONS`` counts LLM ↔ tool round-trips; the
# sub-agent has to converge on a finding within that many steps or the
# loop forcibly extracts whatever finding it has.
MAX_TOOL_ITERATIONS = int(os.environ.get("RESEARCH_SUBAGENT_MAX_ITERATIONS", "6"))


# Subset of agent.py's tools — the chat-only ``create_artifact`` is
# deliberately excluded so the sub-agent never tries to render a
# side-panel artifact (the writer in phase 4 owns the final report).
SUBAGENT_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web (Serper) and return organic results with URLs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scrape_and_index",
            "description": (
                "Fetch a URL with headless Chromium, chunk the text, and upsert "
                "into the vector store so retrieve_chunks can find it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to scrape."},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_chunks",
            "description": (
                "Hybrid-search the vector index for chunks most relevant to a "
                "query. Returns text + source URLs you can cite as [N]."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Retrieval query."},
                },
                "required": ["query"],
            },
        },
    },
]


_SYSTEM_PROMPT = """You are a focused research sub-agent. You answer ONE specific
sub-question that's part of a larger research plan; other sub-agents are
handling other angles in parallel — stay tightly scoped.

Tools (call them, don't describe them):
- web_search(query): organic search results with URLs + snippets
- scrape_and_index(url): fetch a URL and store its chunks for retrieval
- retrieve_chunks(query): hybrid-search the indexed chunks; results are
  numbered [1], [2], … and include source URLs

Workflow (you don't have to use every tool — stop as soon as you can
answer well):
1. web_search once or twice to find 5–10 relevant URLs
2. scrape_and_index 2–4 of the most promising URLs
3. retrieve_chunks to pull the most relevant context
4. Synthesise a focused markdown finding (300–700 words)

Your final answer (after the tool calls) must be plain markdown with:
- A 1–2 sentence summary at the top
- The body of the finding with bracketed citations like [1], [2] tied
  to the most recent retrieve_chunks indices
- A short "## Sources" section at the end listing the URLs you cited

Do NOT call any tool after producing the final answer — finish with
markdown content only. Keep the output focused on this sub-question.
Do NOT speculate beyond what the retrieved chunks support."""


# ── Output shape ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SubagentSource:
    url: str
    title: str | None = None
    snippet: str | None = None


@dataclass(frozen=True)
class SubagentResult:
    sub_agent_id: str
    sub_question: str
    finding_md: str
    sources: list[SubagentSource]
    model: str
    used_stub: bool

    def sources_json(self) -> list[dict[str, Any]]:
        return [
            {"url": s.url, "title": s.title, "snippet": s.snippet} for s in self.sources
        ]


# ── Stub fallback ────────────────────────────────────────────────────────


def _llm_disabled() -> bool:
    return os.environ.get("DISABLE_SUBAGENT_LLM", "").lower() in {"1", "true", "yes"}


def _model() -> str:
    explicit = os.environ.get("RESEARCH_SUBAGENT_MODEL")
    if explicit:
        return explicit
    # Prefer the cheaper flash model for sub-agents — they each do a
    # bounded amount of work and we run several in parallel, so the
    # per-sub-agent cost dominates the run total.
    return get_settings().flash_model


def _client() -> AsyncOpenAI:
    settings = get_settings()
    return AsyncOpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
    )


def _stub_result(
    sub_agent_id: str, sub_question: str, *, reason: str = "no LLM"
) -> SubagentResult:
    finding = (
        f"## Stub finding ({reason})\n\n"
        f"This is a placeholder finding for sub-question: **{sub_question}**.\n\n"
        "The real sub-agent would search the web, scrape relevant pages, "
        "and synthesise a cited answer here. Phase-3 scaffolding is "
        "exercising the rest of the pipeline (parallel orchestration, "
        "event streaming, persistence) using this stub as a stand-in.\n\n"
        "## Sources\n"
        "1. https://example.com/stub-source-a\n"
        "2. https://example.com/stub-source-b\n"
    )
    sources = [
        SubagentSource(
            url="https://example.com/stub-source-a",
            title="Stub source A",
            snippet="Placeholder snippet a.",
        ),
        SubagentSource(
            url="https://example.com/stub-source-b",
            title="Stub source B",
            snippet="Placeholder snippet b.",
        ),
    ]
    return SubagentResult(
        sub_agent_id=sub_agent_id,
        sub_question=sub_question,
        finding_md=finding,
        sources=sources,
        model="stub",
        used_stub=True,
    )


# ── LLM-backed sub-agent ─────────────────────────────────────────────────


async def _safe_run_web_search(query: str) -> str:
    """Wrapper that catches Serper failures so they show up as a tool
    error in the model's history rather than crashing the sub-agent."""
    try:
        return await _run_web_search(query)
    except Exception as exc:
        logger.exception("web_search failed q=%r", query)
        return f"web_search failed: {exc.__class__.__name__}: {exc}"


async def _safe_scrape_and_index(url: str) -> str:
    """Same defensive wrap for scrape_and_index — a single broken page
    shouldn't kill the sub-agent."""
    try:
        return await _run_scrape_and_index(url)
    except Exception as exc:
        logger.exception("scrape_and_index failed url=%s", url)
        return f"scrape_and_index failed for {url}: {exc.__class__.__name__}"


async def _safe_retrieve_chunks(query: str) -> tuple[str, list[dict[str, Any]]]:
    """Run hybrid_search and format. Returns (formatted_text, raw_chunks)."""
    try:
        chunks = await hybrid_search(query)
    except Exception as exc:
        logger.exception("retrieve_chunks failed q=%r", query)
        return (f"retrieve_chunks failed: {exc.__class__.__name__}", [])
    if not chunks:
        return ("No relevant chunks found yet — try web_search or scrape_and_index first.", [])
    lines = [
        f"[{i + 1}] score={c.get('rerank_score', c.get('score', 0)):.3f} "
        f"src={c.get('source_url', '')}\n{c['text'][:1500]}"
        for i, c in enumerate(chunks)
    ]
    return ("\n\n".join(lines), chunks)


async def _emit_progress(
    run_id: str, sub_agent_id: str, action: str, detail: str
) -> None:
    """Append a per-sub-agent progress event so the SSE timeline can
    render the sub-agent's tool-by-tool activity."""
    await append_event(
        run_id,
        "subagent_progress",
        {"id": sub_agent_id, "action": action, "detail": detail},
    )


async def _run_subagent_llm(
    run_id: str,
    sub_agent_id: str,
    sub_question: str,
    model: str,
) -> SubagentResult:
    """Run a bounded tool-calling loop for one sub-question."""
    client = _client()
    history: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": sub_question},
    ]
    # Track every URL the sub-agent touches (search results scraped,
    # chunks retrieved). Phase 4's writer dedups across sub-agents.
    sources_by_url: dict[str, SubagentSource] = {}
    # Snapshot of the most recent retrieve_chunks result — the model's
    # final [N] citations are anchored to this set.
    last_retrieved_urls: list[str] = []

    final_text: str = ""

    for iteration in range(MAX_TOOL_ITERATIONS):
        if await is_cancelled(run_id):
            raise asyncio.CancelledError(f"run {run_id} cancelled mid-subagent")

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=history,
                tools=SUBAGENT_TOOLS,
                tool_choice="auto",
                temperature=0.3,
            )
        except (APIError, httpx.HTTPError) as exc:
            logger.exception("subagent LLM call failed run=%s sub=%s", run_id, sub_agent_id)
            await _emit_progress(
                run_id,
                sub_agent_id,
                "error",
                f"LLM error: {exc.__class__.__name__}",
            )
            # Surface a partial finding so the writer in phase 4 still
            # has something for this sub-question.
            final_text = (
                f"_Sub-agent for **{sub_question}** failed mid-research "
                f"(`{exc.__class__.__name__}`). Findings may be incomplete._"
            )
            break

        msg = response.choices[0].message
        history.append(msg.model_dump(exclude_none=True))

        if msg.content:
            final_text = msg.content

        if not msg.tool_calls:
            # Model finished without more tool calls — we have the answer.
            break

        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            fn_args_raw = tool_call.function.arguments or "{}"
            try:
                fn_args = json.loads(fn_args_raw)
            except json.JSONDecodeError:
                fn_args = {}

            tool_result: str = ""
            if fn_name == "web_search":
                query = str(fn_args.get("query", ""))
                await _emit_progress(run_id, sub_agent_id, "search", query)
                tool_result = await _safe_run_web_search(query)
            elif fn_name == "scrape_and_index":
                url = str(fn_args.get("url", ""))
                await _emit_progress(run_id, sub_agent_id, "scrape", url)
                tool_result = await _safe_scrape_and_index(url)
                if url and url not in sources_by_url:
                    sources_by_url[url] = SubagentSource(url=url)
            elif fn_name == "retrieve_chunks":
                query = str(fn_args.get("query", ""))
                await _emit_progress(run_id, sub_agent_id, "retrieve", query)
                tool_result, raw_chunks = await _safe_retrieve_chunks(query)
                last_retrieved_urls = []
                for chunk in raw_chunks:
                    url = str(chunk.get("source_url") or "")
                    if not url:
                        continue
                    last_retrieved_urls.append(url)
                    if url not in sources_by_url:
                        sources_by_url[url] = SubagentSource(
                            url=url,
                            snippet=str(chunk.get("text", ""))[:240] or None,
                        )
            else:
                tool_result = f"Unknown tool: {fn_name}"

            history.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                }
            )
    else:
        # Loop hit MAX_TOOL_ITERATIONS without breaking. Force one
        # final non-tool call so the model emits its best summary
        # given everything it has gathered.
        history.append(
            {
                "role": "user",
                "content": (
                    "Iteration limit reached. Stop calling tools and "
                    "produce your best markdown finding now using the "
                    "context you already have."
                ),
            }
        )
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=history,
                temperature=0.3,
            )
            final_text = response.choices[0].message.content or final_text
        except (APIError, httpx.HTTPError):
            logger.exception(
                "subagent forced-summary call failed run=%s sub=%s",
                run_id,
                sub_agent_id,
            )

    # If the model never produced text at all, generate a minimal
    # fallback so the writer in phase 4 has something to work with.
    if not final_text.strip():
        final_text = (
            f"_Sub-agent for **{sub_question}** could not produce a finding._"
        )

    # Order sources: URLs cited in the last retrieve_chunks first
    # (those are what [N] markers most likely refer to), then any
    # other URLs the sub-agent touched.
    ordered: list[SubagentSource] = []
    seen: set[str] = set()
    for url in last_retrieved_urls:
        if url in seen:
            continue
        ordered.append(sources_by_url.get(url) or SubagentSource(url=url))
        seen.add(url)
    for url, src in sources_by_url.items():
        if url in seen:
            continue
        ordered.append(src)
        seen.add(url)

    return SubagentResult(
        sub_agent_id=sub_agent_id,
        sub_question=sub_question,
        finding_md=final_text.strip(),
        sources=ordered,
        model=model,
        used_stub=False,
    )


# ── Public entry point + persistence ─────────────────────────────────────


async def _persist_subagent_row(
    run_id: str, result: SubagentResult, *, status: str
) -> None:
    """Insert (or update) the ResearchSubagent row for this finding."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO "ResearchSubagent"
                ("runId", "id", "subQuestion", "status", "model",
                 "startedAt", "finishedAt", "findingMd", "sources")
            VALUES ($1::uuid, $2, $3, $4, $5, now(), now(), $6, $7::json)
            ON CONFLICT ("runId", "id") DO UPDATE
            SET "status" = EXCLUDED."status",
                "model" = EXCLUDED."model",
                "finishedAt" = EXCLUDED."finishedAt",
                "findingMd" = EXCLUDED."findingMd",
                "sources" = EXCLUDED."sources"
            """,
            run_id,
            result.sub_agent_id,
            result.sub_question,
            status,
            result.model,
            result.finding_md,
            json.dumps(result.sources_json()),
        )


async def _persist_subagent_started(
    run_id: str, sub_agent_id: str, sub_question: str, model: str
) -> None:
    """Insert (or upsert) a ``running`` row before the sub-agent begins.

    The status flips to ``done``/``failed`` later in
    :func:`_persist_subagent_row` once the LLM loop finishes. Having
    the row appear at start time is what lets the UI render a
    placeholder card per sub-agent right away.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO "ResearchSubagent"
                ("runId", "id", "subQuestion", "status", "model", "startedAt")
            VALUES ($1::uuid, $2, $3, 'running', $4, now())
            ON CONFLICT ("runId", "id") DO UPDATE
            SET "status" = 'running',
                "model" = EXCLUDED."model",
                "startedAt" = COALESCE("ResearchSubagent"."startedAt", now())
            """,
            run_id,
            sub_agent_id,
            sub_question,
            model,
        )


async def run_subagent(
    run_id: str,
    sub_agent_id: str,
    sub_question: str,
) -> SubagentResult:
    """Public entry — spawns one sub-agent, persists its row, returns the result.

    Catches and reports per-sub-agent failures so a single bad sub-agent
    can't take down the whole run; the caller (the supervisor) gets a
    SubagentResult either way (with finding text describing the failure
    if the LLM call collapsed).
    """
    used_stub = _llm_disabled()
    chosen_model = "stub" if used_stub else _model()

    await _persist_subagent_started(
        run_id, sub_agent_id, sub_question, chosen_model
    )
    await append_event(
        run_id,
        "subagent_started",
        {
            "id": sub_agent_id,
            "subQuestion": sub_question,
            "model": chosen_model,
            "stub": used_stub,
        },
    )

    settings = get_settings()
    if used_stub:
        result = _stub_result(sub_agent_id, sub_question, reason="LLM disabled")
    elif not settings.openrouter_api_key:
        logger.warning("subagent: OPENROUTER_API_KEY missing; using stub")
        result = _stub_result(
            sub_agent_id, sub_question, reason="no OPENROUTER_API_KEY"
        )
    else:
        try:
            result = await _run_subagent_llm(
                run_id, sub_agent_id, sub_question, chosen_model
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(
                "subagent crashed run=%s sub=%s", run_id, sub_agent_id
            )
            await _persist_subagent_row(
                run_id,
                SubagentResult(
                    sub_agent_id=sub_agent_id,
                    sub_question=sub_question,
                    finding_md=(
                        f"_Sub-agent crashed: {exc.__class__.__name__}_"
                    ),
                    sources=[],
                    model=chosen_model,
                    used_stub=False,
                ),
                status="failed",
            )
            await append_event(
                run_id,
                "subagent_failed",
                {"id": sub_agent_id, "error": str(exc) or exc.__class__.__name__},
            )
            raise

    await _persist_subagent_row(run_id, result, status="done")
    await append_event(
        run_id,
        "subagent_finished",
        {
            "id": sub_agent_id,
            "findingMd": result.finding_md,
            "sourceCount": len(result.sources),
            "model": result.model,
            "stub": result.used_stub,
        },
    )
    return result
