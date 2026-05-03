"""Research supervisor — fans out sub-agents in parallel, collects findings.

The supervisor is the phase-3 replacement for the stub
``_run_researching`` from earlier phases. It reads the most recent
approved ``ResearchPlan`` for the run, spawns one sub-agent per
sub-question (bounded concurrency), and emits aggregate progress
events around them.

It does not write the final report — that's phase 4's writer. The
supervisor only ensures every sub-question has a populated
``ResearchSubagent`` row by the time it returns.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from backend.research.db import get_pool
from backend.research.events import append_event
from backend.research.state import is_cancelled
from backend.research.subagent import (
    SubagentResult,
    run_subagent,
)

logger = logging.getLogger("deepsearch.research.supervisor")


# How many sub-agents can run at once. Each sub-agent does several
# tool calls + LLM round-trips, so this caps the OpenRouter parallel
# fan-out and the concurrent Playwright instances. Tunable per-env;
# default 3 is a sweet spot between speed and rate limits.
DEFAULT_CONCURRENCY = int(
    os.environ.get("RESEARCH_SUBAGENT_CONCURRENCY", "3")
)


async def _load_latest_plan(run_id: str) -> dict[str, Any] | None:
    """Pull the most recent plan version for this run.

    Returns the row as a dict (or None) — we don't need a strong
    schema here because the user-edited form is permissive (the API
    layer already validated it).
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT "version", "subQuestions", "outline", "approvedAt"
            FROM "ResearchPlan"
            WHERE "runId" = $1::uuid
            ORDER BY "version" DESC
            LIMIT 1
            """,
            run_id,
        )
    if not row:
        return None
    return {
        "version": row["version"],
        # asyncpg returns JSON columns as Python dicts/lists already,
        # but `subQuestions` and `outline` could come back as the
        # raw json text on some drivers — coerce defensively.
        "subQuestions": _coerce_json_array(row["subQuestions"]),
        "outline": _coerce_json_array(row["outline"]),
        "approvedAt": row["approvedAt"],
    }


def _coerce_json_array(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, list):
        return [v for v in value if isinstance(v, dict)]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [v for v in parsed if isinstance(v, dict)]
    return []


async def run_research(run: dict[str, Any]) -> list[SubagentResult]:
    """Phase-3 researching step.

    Reads the approved plan, runs every sub-agent in parallel (with a
    semaphore), and returns the list of results in the original
    sub-question order. Per-sub-agent failures are isolated — the
    returned list contains whatever findings the failing sub-agent
    persisted (a placeholder text + empty sources) so the writer in
    phase 4 still sees a row for every sub-question.

    Raises :class:`asyncio.CancelledError` if the user cancels the run
    mid-flight; in that case the partial findings stay on disk.
    """
    run_id = str(run["id"])
    plan = await _load_latest_plan(run_id)
    if plan is None or not plan["subQuestions"]:
        logger.warning("run %s entered researching with no plan/sub-questions", run_id)
        await append_event(
            run_id,
            "research_skipped",
            {"reason": "no plan or empty subQuestions"},
        )
        return []

    sub_questions: list[dict[str, Any]] = plan["subQuestions"]
    await append_event(
        run_id,
        "research_dispatch",
        {
            "subagentCount": len(sub_questions),
            "concurrency": DEFAULT_CONCURRENCY,
        },
    )

    semaphore = asyncio.Semaphore(DEFAULT_CONCURRENCY)
    results: list[SubagentResult | BaseException] = [None] * len(sub_questions)  # type: ignore[list-item]

    async def _bounded(index: int, sq: dict[str, Any]) -> None:
        sub_id = str(sq.get("id") or f"sq{index + 1}")
        question = str(sq.get("question") or "").strip()
        if not question:
            logger.warning(
                "run %s sub-question %s has empty text; skipping", run_id, sub_id
            )
            return
        async with semaphore:
            if await is_cancelled(run_id):
                raise asyncio.CancelledError(run_id)
            try:
                results[index] = await run_subagent(run_id, sub_id, question)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # ``run_subagent`` already persisted a 'failed' row +
                # emitted an event before re-raising; just record the
                # exception in the results slot for the summary below.
                results[index] = exc

    tasks = [
        asyncio.create_task(_bounded(i, sq)) for i, sq in enumerate(sub_questions)
    ]
    try:
        await asyncio.gather(*tasks, return_exceptions=False)
    except asyncio.CancelledError:
        for task in tasks:
            task.cancel()
        # Let outstanding cancellations settle so we don't leak tasks.
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    succeeded = sum(1 for r in results if isinstance(r, SubagentResult))
    failed = sum(1 for r in results if isinstance(r, BaseException))
    await append_event(
        run_id,
        "research_complete",
        {
            "subagentCount": len(sub_questions),
            "succeeded": succeeded,
            "failed": failed,
        },
    )
    # Drop None / exception slots from the returned list — callers only
    # care about successful findings (the writer reads the
    # ResearchSubagent table directly anyway, so this is mostly for
    # tests).
    return [r for r in results if isinstance(r, SubagentResult)]
