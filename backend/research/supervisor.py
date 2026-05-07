"""Research supervisor — fans out sub-agents in parallel, collects findings.

Phase-3 entry point. Reads the latest approved ``ResearchPlan``,
dispatches one sub-agent per sub-question (bounded concurrency),
collects findings, then runs the evaluator over wave-1 findings to
decide whether a second targeted wave is worth dispatching.

Wave 2 is the dynamic-workflow piece the survey
(`backend/research/evaluator.py` cites Huang et al. 2025) describes
as the missing ingredient between static "plan→research→write"
pipelines and industrial DR systems (OpenAI DR, Gemini DR, Perplexity
DR). The evaluator decides whether the wave-1 findings cover the
brief; if not, it returns 2-4 focused gap questions that we dispatch
as a second wave. Capped at ``RESEARCH_MAX_WAVES`` to bound runtime.

The supervisor does not write the final report — that's phase 4's
writer. It only ensures every sub-question (and any evaluator-
spawned gap question) has a populated ``ResearchSubagent`` row by
the time it returns.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from backend.research.db import get_pool
from backend.research.events import append_event
from backend.research.evaluator import EvaluationResult, evaluate_findings
from backend.research.state import is_cancelled
from backend.research.subagent import (
    SubagentResult,
    run_subagent,
)

logger = logging.getLogger("deepsearch.research.supervisor")


# How many sub-agents can run at once. Each sub-agent does several
# tool calls + LLM round-trips, so this caps the OpenRouter parallel
# fan-out and the concurrent Playwright instances. Default 4 is a
# sweet spot — with 8–12 sub-questions per plan, two waves of 4 keeps
# total wall time in the 15–25 min range typical of deep-research
# products without saturating Serper / OpenRouter rate limits.
DEFAULT_CONCURRENCY = int(
    os.environ.get("RESEARCH_SUBAGENT_CONCURRENCY", "4")
)


# Hard ceiling on research waves. Wave 1 = original plan; wave 2 =
# evaluator-driven gap fill. We don't allow wave 3+ today — the
# survey notes that's where unbounded loops and runaway cost come
# from. Override via env if a specific deployment wants more depth.
MAX_WAVES = int(os.environ.get("RESEARCH_MAX_WAVES", "2"))


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


async def _dispatch_wave(
    *,
    run_id: str,
    user_id: str | None,
    sub_questions: list[dict[str, Any]],
    wave: int,
    concurrency: int,
) -> list[SubagentResult | BaseException]:
    """Run one wave of sub-agents in parallel, returning per-question
    results (or exceptions) in input order.

    Per-sub-agent failures are isolated — the returned list contains
    whatever findings the failing sub-agent persisted (a placeholder
    text + empty sources). Cancellation propagates out so the
    pipeline's outer try/except can mark the run cancelled cleanly.
    """
    semaphore = asyncio.Semaphore(concurrency)
    results: list[SubagentResult | BaseException] = [None] * len(sub_questions)  # type: ignore[list-item]

    async def _bounded(index: int, sq: dict[str, Any]) -> None:
        sub_id = str(sq.get("id") or f"sq{wave}-{index + 1}")
        question = str(sq.get("question") or "").strip()
        if not question:
            logger.warning(
                "run %s wave %d sub-question %s has empty text; skipping",
                run_id, wave, sub_id,
            )
            return
        async with semaphore:
            if await is_cancelled(run_id):
                raise asyncio.CancelledError(run_id)
            try:
                results[index] = await run_subagent(
                    run_id, sub_id, question, user_id=user_id,
                )
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
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    return results


async def _load_subagent_findings(run_id: str) -> list[dict[str, Any]]:
    """Load every persisted sub-agent row for evaluator input.

    The evaluator wants ``{subQuestion, findingMd, sources}`` per row;
    we read straight from ``ResearchSubagent`` rather than rebuilding
    from in-memory ``SubagentResult`` so wave-1 + wave-2 are uniformly
    represented (and a resumed run still works).
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT "id", "subQuestion", "findingMd", "sources", "status"
            FROM "ResearchSubagent"
            WHERE "runId" = $1::uuid
            ORDER BY "startedAt" ASC
            """,
            run_id,
        )
    findings: list[dict[str, Any]] = []
    for row in rows:
        sources = row["sources"]
        if isinstance(sources, str):
            try:
                sources = json.loads(sources)
            except json.JSONDecodeError:
                sources = []
        findings.append({
            "id": row["id"],
            "subQuestion": row["subQuestion"],
            "findingMd": row["findingMd"],
            "sources": sources if isinstance(sources, list) else [],
            "status": row["status"],
        })
    return findings


async def _count_unique_sources(run_id: str) -> int:
    """Cheap source-pool size for evaluator context."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            'SELECT COUNT(*)::int AS n FROM "ResearchSource" WHERE "runId" = $1::uuid',
            run_id,
        )
    return int(row["n"]) if row else 0


async def run_research(run: dict[str, Any]) -> list[SubagentResult]:
    """Phase-3 researching step.

    Two-wave dynamic workflow:

    1. Wave 1: dispatch every sub-question from the approved plan in
       parallel (concurrency capped by ``DEFAULT_CONCURRENCY``).
    2. Evaluator: read all persisted findings; decide whether to
       proceed to the writer or dispatch a focused second wave.
    3. Wave 2 (optional): up to ``MAX_WAVES - 1`` more, fed by the
       evaluator's gap questions.

    Returns the merged list of successful sub-agent results in
    dispatch order. Raises ``asyncio.CancelledError`` if the user
    cancels the run mid-flight; partial findings stay on disk.
    """
    run_id = str(run["id"])
    user_id = run.get("userId")
    query = str(run.get("query") or "").strip()
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
    brief_md: str | None = None
    # The plan dict from `_load_latest_plan` doesn't carry briefMd;
    # it's only used by the evaluator and the writer reloads it
    # itself, so we just pull it once here.
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            'SELECT "briefMd" FROM "ResearchPlan" WHERE "runId" = $1::uuid ORDER BY "version" DESC LIMIT 1',
            run_id,
        )
        if row:
            brief_md = row["briefMd"]

    # ── Wave 1 ───────────────────────────────────────────────────────────
    await append_event(
        run_id,
        "research_dispatch",
        {
            "wave": 1,
            "subagentCount": len(sub_questions),
            "concurrency": DEFAULT_CONCURRENCY,
        },
    )
    wave_results: list[list[SubagentResult | BaseException]] = []
    wave_results.append(
        await _dispatch_wave(
            run_id=run_id,
            user_id=user_id,
            sub_questions=sub_questions,
            wave=1,
            concurrency=DEFAULT_CONCURRENCY,
        )
    )

    # ── Evaluator + optional wave 2 ──────────────────────────────────────
    if MAX_WAVES > 1 and not await is_cancelled(run_id):
        await append_event(run_id, "evaluation_started", {"wave": 1})
        try:
            findings = await _load_subagent_findings(run_id)
            citation_count = await _count_unique_sources(run_id)
            evaluation: EvaluationResult = await evaluate_findings(
                run_id=run_id,
                user_id=user_id,
                query=query,
                brief_md=brief_md,
                sub_questions=sub_questions,
                findings=findings,
                citation_count=citation_count,
            )
        except Exception as exc:
            # Don't let an evaluator hiccup take down the whole run —
            # the pipeline still has wave-1 findings to write from.
            logger.exception("evaluator crashed run=%s; skipping wave 2", run_id)
            evaluation = EvaluationResult(
                ready_for_writer=True,
                gaps=[],
                contradictions=[],
                rationale=f"Evaluator crashed: {exc.__class__.__name__}",
                model="error",
                used_stub=True,
            )
        await append_event(
            run_id,
            "evaluation_complete",
            {
                "ready_for_writer": evaluation.ready_for_writer,
                "gapCount": len(evaluation.gaps),
                "contradictionCount": len(evaluation.contradictions),
                "rationale": evaluation.rationale,
                "model": evaluation.model,
                "stub": evaluation.used_stub,
            },
        )

        if (
            not evaluation.ready_for_writer
            and evaluation.gaps
            and not await is_cancelled(run_id)
        ):
            # Cap gap dispatch at concurrency — opening wave 2 with 6
            # gap-fill agents adds little over 4 in our budget.
            gap_questions = [
                {"id": g["id"], "question": g["question"]}
                for g in evaluation.gaps[:DEFAULT_CONCURRENCY]
            ]
            await append_event(
                run_id,
                "gap_dispatch",
                {
                    "wave": 2,
                    "subagentCount": len(gap_questions),
                    "rationale": evaluation.rationale,
                },
            )
            await append_event(
                run_id,
                "research_dispatch",
                {
                    "wave": 2,
                    "subagentCount": len(gap_questions),
                    "concurrency": DEFAULT_CONCURRENCY,
                },
            )
            wave_results.append(
                await _dispatch_wave(
                    run_id=run_id,
                    user_id=user_id,
                    sub_questions=gap_questions,
                    wave=2,
                    concurrency=DEFAULT_CONCURRENCY,
                )
            )

    # ── Roll up ──────────────────────────────────────────────────────────
    flat: list[SubagentResult | BaseException] = [
        r for wave in wave_results for r in wave
    ]
    succeeded = sum(1 for r in flat if isinstance(r, SubagentResult))
    failed = sum(1 for r in flat if isinstance(r, BaseException))
    await append_event(
        run_id,
        "research_complete",
        {
            "waves": len(wave_results),
            "subagentCount": len(flat),
            "succeeded": succeeded,
            "failed": failed,
        },
    )
    return [r for r in flat if isinstance(r, SubagentResult)]
