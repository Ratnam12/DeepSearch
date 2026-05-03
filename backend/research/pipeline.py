"""Phase-1 stub orchestrator for the deep-research pipeline.

Walks a run through ``scoping → planning → awaiting_approval →
researching → writing → done``, emitting one or two events per phase
and writing a placeholder plan/report so the surrounding scaffolding
(SSE replay, plan-approval API, report-view skeleton) can be exercised
end-to-end without any LLM calls.

Phase 2 onward replaces each phase function with the real agent
implementation; the surrounding worker, status machine, event log, and
SSE plumbing don't need to change.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from backend.research.events import append_event
from backend.research.planner import create_plan
from backend.research.state import (
    is_cancelled,
    transition_status,
)
from backend.research.db import get_pool

logger = logging.getLogger("deepsearch.research.pipeline")


# Visible to the worker for resume logic and to tests for assertions.
RUN_PHASES = (
    "scoping",
    "planning",
    "awaiting_approval",
    "researching",
    "writing",
    "done",
)


def _phase_delay() -> float:
    """Per-phase sleep, env-tunable so tests can run fast."""
    return float(os.environ.get("RESEARCH_PHASE_DELAY_S", "1.5"))


def _auto_approve_after() -> float:
    """If > 0, the awaiting_approval phase auto-approves after this many seconds."""
    return float(os.environ.get("RESEARCH_AUTO_APPROVE_AFTER_S", "0"))


# ── Cancellation primitive ───────────────────────────────────────────────


class Cancelled(Exception):
    """Raised when the run row was flipped to status='cancelled'."""


async def _check_cancelled(run_id: str) -> None:
    if await is_cancelled(run_id):
        raise Cancelled(run_id)


# ── Phase implementations (stubs) ────────────────────────────────────────


async def _run_scoping(run: dict[str, Any]) -> None:
    """Phase 1: scoping.

    Lightweight acknowledgement step — emits a ``scoping_started``
    event so the SSE timeline shows the agent is engaging with the
    query. The actual scope analysis is folded into the planner call
    in phase 2 (one LLM round-trip is enough for both).
    """
    run_id = run["id"]
    await asyncio.sleep(_phase_delay())
    await _check_cancelled(run_id)
    await append_event(
        run_id,
        "scoping_started",
        {"query": run["query"]},
    )


async def _run_planning(run: dict[str, Any]) -> None:
    """Phase 2: planning.

    Calls :func:`create_plan` once to produce ``briefMd`` +
    ``subQuestions`` + ``outline`` from the user's query. The result
    is persisted as a ``ResearchPlan`` row (version 1) and announced
    via two timeline events:

    - ``brief_drafted`` carries just ``briefMd`` so the UI can render
      the framing paragraph as soon as it's ready, even before the
      full plan card lands.
    - ``plan_proposed`` carries the full plan and is what the
      approval card listens for.

    Failures inside the planner itself are caught there (it falls
    back to a deterministic stub plan with a clear ``stub: true`` flag
    on the event payload), so this function only fails if the DB
    write does — which we want to surface as a normal pipeline
    failure.
    """
    run_id = run["id"]
    await _check_cancelled(run_id)

    plan = await create_plan(run["query"])

    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO "ResearchPlan" ("runId", "version", "briefMd", "subQuestions", "outline")
            VALUES ($1::uuid, $2, $3, $4::json, $5::json)
            """,
            run_id,
            1,
            plan.brief_md,
            json.dumps(plan.sub_questions),
            json.dumps(plan.outline),
        )

    await append_event(
        run_id,
        "brief_drafted",
        {"briefMd": plan.brief_md, "model": plan.model, "stub": plan.used_stub},
    )
    await append_event(
        run_id,
        "plan_proposed",
        {
            "version": 1,
            "subQuestions": plan.sub_questions,
            "outline": plan.outline,
            "model": plan.model,
            "stub": plan.used_stub,
        },
    )


async def _wait_for_plan_approval(run: dict[str, Any]) -> None:
    """Phase 3: awaiting_approval. Block until /plan POSTs.

    The plan endpoint flips the run's status to 'researching' once the
    user approves. We poll the row until then (cancellation is also
    detected via the same poll). For the stub, also auto-approve after
    ``RESEARCH_AUTO_APPROVE_AFTER_S`` seconds if the env var is set —
    convenient for the integration test which doesn't have a UI in the
    loop.
    """
    run_id = run["id"]
    auto_approve_after = _auto_approve_after()

    started = asyncio.get_event_loop().time()
    pool = await get_pool()
    while True:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                'SELECT "status" FROM "ResearchRun" WHERE "id" = $1::uuid',
                run_id,
            )
        if not row:
            raise RuntimeError(f"run {run_id} disappeared while awaiting approval")
        status = row["status"]
        if status == "researching":
            return
        if status == "cancelled":
            raise Cancelled(run_id)
        if status != "awaiting_approval":
            # Some other concurrent transition — bail safely.
            raise RuntimeError(
                f"run {run_id} left awaiting_approval to unexpected status={status}"
            )

        if (
            auto_approve_after > 0
            and (asyncio.get_event_loop().time() - started) >= auto_approve_after
        ):
            await _auto_approve_plan(run_id)
            # The next loop iteration will see status='researching'.

        await asyncio.sleep(0.5)


async def _auto_approve_plan(run_id: str) -> None:
    """Test-only helper: approve the latest plan and flip to researching."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            plan = await conn.fetchrow(
                """
                SELECT "version", "subQuestions", "outline"
                FROM "ResearchPlan"
                WHERE "runId" = $1::uuid
                ORDER BY "version" DESC
                LIMIT 1
                """,
                run_id,
            )
            if not plan:
                return
            await conn.execute(
                """
                UPDATE "ResearchPlan"
                SET "approvedAt" = now()
                WHERE "runId" = $1::uuid AND "version" = $2
                """,
                run_id,
                plan["version"],
            )
            await conn.execute(
                'UPDATE "ResearchRun" SET "status" = \'researching\' WHERE "id" = $1::uuid',
                run_id,
            )
    await append_event(
        run_id,
        "plan_approved",
        {
            "version": plan["version"],
            "subQuestions": json.loads(plan["subQuestions"])
            if isinstance(plan["subQuestions"], str)
            else plan["subQuestions"],
            "outline": json.loads(plan["outline"])
            if isinstance(plan["outline"], str)
            else plan["outline"],
            "editedByUser": False,
            "auto": True,
        },
    )


async def _run_researching(run: dict[str, Any]) -> None:
    """Phase 4: researching. Stub — pretends two sub-agents finished."""
    run_id = run["id"]
    pool = await get_pool()

    # Stub sub-agent rows + per-agent events.
    for idx in (1, 2):
        await _check_cancelled(run_id)
        sub_id = f"sa{idx}"
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO "ResearchSubagent"
                    ("runId", "id", "subQuestion", "status", "model",
                     "startedAt", "finishedAt", "findingMd", "sources")
                VALUES ($1::uuid, $2, $3, 'done', $4, now(), now(), $5, $6::json)
                ON CONFLICT ("runId", "id") DO NOTHING
                """,
                run_id,
                sub_id,
                f"Stub sub-question #{idx} for: {run['query']}",
                "stub-model",
                f"Stub finding for sub-agent {idx}.",
                json.dumps([]),
            )
        await append_event(
            run_id,
            "subagent_started",
            {"id": sub_id, "subQuestion": f"Stub sub-question #{idx}"},
        )
        await asyncio.sleep(_phase_delay() / 2)
        await append_event(
            run_id,
            "subagent_finished",
            {"id": sub_id, "findingMd": f"Stub finding for sub-agent {idx}."},
        )


async def _run_writing(run: dict[str, Any]) -> None:
    """Phase 5: writing. Stub — writes a placeholder report."""
    run_id = run["id"]
    await asyncio.sleep(_phase_delay())
    await _check_cancelled(run_id)

    markdown = (
        f"# Research report: {run['query']}\n\n"
        "## Background\n"
        "*Stub background section — phase-1 scaffolding.*\n\n"
        "## Findings\n"
        "1. Stub finding from sub-agent 1.\n"
        "2. Stub finding from sub-agent 2.\n\n"
        "## Open questions\n"
        "Phase 1 doesn't run real agents yet, so this report is a placeholder.\n"
    )
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO "ResearchReport" ("runId", "version", "markdown", "sections")
            VALUES ($1::uuid, $2, $3, $4::json)
            """,
            run_id,
            1,
            markdown,
            json.dumps([
                {"id": "s1", "title": "Background"},
                {"id": "s2", "title": "Findings"},
                {"id": "s3", "title": "Open questions"},
            ]),
        )
    await append_event(
        run_id,
        "report_written",
        {"version": 1, "characters": len(markdown)},
    )


# ── Public entry point ───────────────────────────────────────────────────


async def run_pipeline(run: dict[str, Any]) -> None:
    """Run a freshly-claimed run all the way through to terminal status.

    The status machine is the source of truth; this function reads the
    *current* status to decide which phase to pick up first, so it
    safely resumes after a worker restart. It does **not** assume the
    caller already moved the run out of 'queued' — but the worker's
    :func:`claim_queued_run` does that for new runs, so the typical
    entry status is 'scoping'.
    """
    run_id = run["id"]
    try:
        # Resume support: figure out where to start from.
        current_status = run["status"]
        if current_status not in {"scoping", "planning", "awaiting_approval", "researching", "writing"}:
            # Either already terminal (caller will skip) or queued (caller
            # should have claimed first). Defensive: just bail.
            logger.warning(
                "run %s entered run_pipeline with status=%s; nothing to do",
                run_id,
                current_status,
            )
            return

        if current_status == "scoping":
            await _run_scoping(run)
            await transition_status(run_id, "planning")
            current_status = "planning"

        if current_status == "planning":
            await _run_planning(run)
            await transition_status(run_id, "awaiting_approval")
            current_status = "awaiting_approval"

        if current_status == "awaiting_approval":
            await _wait_for_plan_approval(run)
            # _wait_for_plan_approval returns once status='researching';
            # transition_status would double-write, so we just record
            # the phase-start event explicitly.
            await append_event(run_id, "research_started", {})
            current_status = "researching"

        if current_status == "researching":
            await _run_researching(run)
            await transition_status(run_id, "writing")
            current_status = "writing"

        if current_status == "writing":
            await _run_writing(run)
            await transition_status(
                run_id,
                "done",
                extra_event=("run_completed", {}),
            )

    except Cancelled:
        # The user cancelled mid-flight. Status was already set to
        # 'cancelled' by the API; just record the agent-side
        # acknowledgement so the SSE reader sees a clean terminal.
        await append_event(run_id, "run_cancelled", {"reason": "agent_observed"})
        logger.info("run %s cancelled", run_id)
