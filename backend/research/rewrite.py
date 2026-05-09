"""Standalone-question rewrite for DeepSearch follow-ups.

The chat-mode pipeline sees the full conversation history natively
(every turn is sent to the agent loop). The DeepSearch worker only sees
``ResearchRun.query`` — a single string — so a follow-up like "give me
inventions done by him" reaches the planner with no referent for "him".

This module rewrites that follow-up into a self-contained research
question by inlining the entities and topics from the prior turns. One
cheap Flash call; falls back to the original query on any failure so a
flaky classifier can never block a research run from starting.
"""

from __future__ import annotations

import asyncio

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from backend.config import get_settings

# Cap how much of the prior conversation we feed the rewriter. The last
# few turns are almost always enough to resolve a pronoun or topic
# reference, and capping keeps the Flash call cheap and fast.
_HISTORY_TURN_CAP = 8
_HISTORY_CHAR_CAP = 6_000
_RESPONSE_TIMEOUT_SECONDS = 5.0
_MAX_OUTPUT_TOKENS = 256

_SYSTEM_PROMPT = (
    "You are a query rewriter for a deep-research assistant. You will be "
    "given a prior chat conversation and a follow-up question. Rewrite "
    "the follow-up into a SELF-CONTAINED research question that names "
    "every entity, topic, and pronoun explicitly using the prior "
    "conversation as context. Preserve the user's intent exactly — do "
    "not add new requirements or change the scope. If the follow-up is "
    "already self-contained, return it unchanged. Reply with just the "
    "rewritten question, no preamble or explanation."
)


class HistoryMessage(BaseModel):
    role: str
    content: str


class RewriteRequest(BaseModel):
    query: str = Field(..., min_length=1)
    history: list[HistoryMessage] = Field(default_factory=list)


class RewriteResponse(BaseModel):
    query: str
    rewritten: bool


def _format_history(history: list[HistoryMessage]) -> str:
    """Trim and serialise the history for the rewrite prompt."""
    trimmed = history[-_HISTORY_TURN_CAP:]
    lines: list[str] = []
    running = 0
    for msg in trimmed:
        line = f"{msg.role}: {msg.content.strip()}"
        running += len(line)
        if running > _HISTORY_CHAR_CAP:
            break
        lines.append(line)
    return "\n".join(lines)


async def rewrite_for_research(
    query: str, history: list[HistoryMessage]
) -> RewriteResponse:
    """Return a standalone version of *query* using *history* as context.

    Returns ``rewritten=False`` (and the original query) when there's no
    prior context to draw from or when the classifier call fails.
    """
    if not history:
        return RewriteResponse(query=query, rewritten=False)

    settings = get_settings()
    client = AsyncOpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
    )
    history_text = _format_history(history)
    if not history_text:
        return RewriteResponse(query=query, rewritten=False)

    user_prompt = (
        f"PRIOR CONVERSATION:\n{history_text}\n\n"
        f"FOLLOW-UP QUESTION: {query}\n\n"
        "STANDALONE QUESTION:"
    )

    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=settings.flash_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=_MAX_OUTPUT_TOKENS,
            ),
            timeout=_RESPONSE_TIMEOUT_SECONDS,
        )
        rewritten = (response.choices[0].message.content or "").strip()
        if not rewritten:
            return RewriteResponse(query=query, rewritten=False)
        # Strip leading/trailing quote marks Flash sometimes wraps the
        # answer in despite the prompt telling it not to.
        rewritten = rewritten.strip("\"'`")
        if rewritten == query.strip():
            return RewriteResponse(query=query, rewritten=False)
        print(f"[rewrite] {query!r} -> {rewritten!r}")
        return RewriteResponse(query=rewritten, rewritten=True)
    except Exception as exc:  # noqa: BLE001 — any failure → return original
        print(f"[rewrite] failed: {exc} — using original query")
        return RewriteResponse(query=query, rewritten=False)
