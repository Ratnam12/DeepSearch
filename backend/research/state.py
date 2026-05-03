"""Status transitions + run-claiming for the research worker.

All status writes go through this module so phase transitions are
auditable and consistent — the worker pipeline never updates the
``ResearchRun.status`` column directly.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from backend.research.db import get_pool
from backend.research.events import append_event, notify_channel

logger = logging.getLogger("deepsearch.research.state")


# Mirror of RESEARCH_RUN_STATUSES in frontend/lib/db/schema.ts.
RUN_STATUSES = (
    "queued",
    "scoping",
    "planning",
    "awaiting_approval",
    "researching",
    "writing",
    "done",
    "failed",
    "cancelled",
)

TERMINAL_STATUSES = frozenset({"done", "failed", "cancelled"})

# Statuses the worker should pick back up after a crash/restart. A run
# stuck in `awaiting_approval` is *not* resumable autonomously — it's
# blocked on user input via POST /api/research/[id]/plan.
RESUMABLE_STATUSES = frozenset({"scoping", "planning", "researching", "writing"})


async def get_run(run_id: str) -> Optional[dict[str, Any]]:
    """Fetch a run row as a plain dict (or None if missing)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            'SELECT * FROM "ResearchRun" WHERE "id" = $1::uuid',
            run_id,
        )
    return dict(row) if row else None


async def claim_queued_run() -> Optional[dict[str, Any]]:
    """Atomically claim one queued run and flip it to 'scoping'.

    Uses ``UPDATE ... WHERE status='queued' ... RETURNING`` so two
    workers can't both claim the same row even without explicit row
    locking. Returns the claimed run dict, or None if the queue is
    empty.

    The implicit ``startedAt`` write here is the run's first wall-clock
    timestamp — useful for surfacing latency in the UI.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            WITH next_run AS (
                SELECT "id"
                FROM "ResearchRun"
                WHERE "status" = 'queued'
                ORDER BY "createdAt" ASC
                LIMIT 1
                FOR UPDATE SKIP LOCKED
            )
            UPDATE "ResearchRun" r
            SET "status" = 'scoping',
                "startedAt" = COALESCE(r."startedAt", now())
            FROM next_run
            WHERE r."id" = next_run."id"
            RETURNING r.*
            """
        )
    if not row:
        return None
    run = dict(row)
    await append_event(run["id"], "run_started", {"query": run["query"]})
    logger.info("claimed run id=%s", run["id"])
    return run


async def fetch_resumable_runs() -> list[dict[str, Any]]:
    """Find runs the worker should resume after a (crash) restart.

    Statuses in ``awaiting_approval`` are skipped — those are blocked
    on user input, not on the worker. ``queued`` rows are picked up by
    :func:`claim_queued_run` instead.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT * FROM "ResearchRun"
            WHERE "status" = ANY($1::text[])
            ORDER BY "createdAt" ASC
            """,
            list(RESUMABLE_STATUSES),
        )
    return [dict(r) for r in rows]


async def transition_status(
    run_id: str,
    new_status: str,
    *,
    error: str | None = None,
    extra_event: Optional[tuple[str, dict[str, Any]]] = None,
) -> None:
    """Move a run to a new status and append a phase-transition event.

    The phase-transition event makes the run history self-describing —
    a reader replaying the event log can reconstruct the run's status
    timeline without hitting ``ResearchRun``. ``extra_event`` lets a
    caller append a richer companion event (e.g. ``plan_proposed``) in
    the same logical step.
    """
    if new_status not in RUN_STATUSES:
        raise ValueError(f"unknown status: {new_status}")

    pool = await get_pool()
    async with pool.acquire() as conn:
        if new_status in TERMINAL_STATUSES:
            await conn.execute(
                """
                UPDATE "ResearchRun"
                SET "status" = $2,
                    "error" = $3,
                    "finishedAt" = now()
                WHERE "id" = $1::uuid
                """,
                run_id,
                new_status,
                error,
            )
        else:
            await conn.execute(
                """
                UPDATE "ResearchRun"
                SET "status" = $2,
                    "error" = COALESCE($3, "error")
                WHERE "id" = $1::uuid
                """,
                run_id,
                new_status,
                error,
            )

    await append_event(
        run_id,
        "status_changed",
        {
            "status": new_status,
            "at": datetime.now(timezone.utc).isoformat(),
            "error": error,
        },
    )
    if extra_event is not None:
        evt_type, evt_payload = extra_event
        await append_event(run_id, evt_type, evt_payload)
    # Make sure SSE listeners that missed the per-event NOTIFY (rare,
    # but possible if a connection had just dropped) wake up on the
    # status change.
    await notify_channel(run_id)


async def finalize_failed(run_id: str, error_message: str) -> None:
    """Convenience helper for the worker's outer try/except."""
    await transition_status(run_id, "failed", error=error_message)
    await append_event(run_id, "run_failed", {"error": error_message})


async def is_cancelled(run_id: str) -> bool:
    """Cheap status probe — used as a co-operative cancellation point."""
    run = await get_run(run_id)
    return bool(run and run["status"] == "cancelled")


def remaining_phases_for(status: str) -> Iterable[str]:
    """Yield the phases still to run after the given current status.

    Used when resuming a crashed worker — start from the next phase
    after whatever the run was doing when it died, not from the top.
    """
    timeline = ("scoping", "planning", "awaiting_approval", "researching", "writing", "done")
    seen = False
    for phase in timeline:
        if phase == status:
            seen = True
            continue
        if seen:
            yield phase
