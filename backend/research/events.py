"""ResearchEvent log writer + Postgres NOTIFY fanout.

Every step the agent takes is appended here as a row in
``ResearchEvent``. The frontend's SSE endpoint replays the table from
``Last-Event-ID`` and listens for live updates on the
:data:`NOTIFY_CHANNEL` channel — so this module is the only place the
two ends meet.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from backend.research.db import get_pool

logger = logging.getLogger("deepsearch.research.events")


# Single shared channel for all runs. Listeners filter by run id in the
# payload. Per-run channels would mean churn (LISTEN/UNLISTEN every
# time a client opens a stream) and at our scale the broadcast
# bandwidth is negligible.
NOTIFY_CHANNEL = "research_events"


async def append_event(
    run_id: str,
    event_type: str,
    payload: dict[str, Any] | None = None,
) -> tuple[int, Any]:
    """Append one event row and broadcast a NOTIFY to live SSE clients.

    The seq number is allocated atomically inside the same statement
    that does the INSERT so concurrent writes can't collide on
    ``(runId, seq)``. Returns the assigned ``(seq, ts)`` so callers
    can log meaningful timestamps.
    """
    payload = payload or {}
    payload_json = json.dumps(payload, default=str)
    # ``pg_notify``'s second argument is ``text``, and asyncpg returns
    # UUID objects from RETURNING clauses — so callers might pass us a
    # UUID instance. Coerce here to keep the boundary consistent.
    run_id_str = str(run_id)

    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                """
                INSERT INTO "ResearchEvent" ("runId", "seq", "type", "payload")
                VALUES (
                    $1::uuid,
                    COALESCE(
                        (SELECT MAX("seq") FROM "ResearchEvent" WHERE "runId" = $1::uuid),
                        0
                    ) + 1,
                    $2,
                    $3::json
                )
                RETURNING "seq", "ts"
                """,
                run_id_str,
                event_type,
                payload_json,
            )
            assert row is not None, "INSERT...RETURNING produced no row"
            # NOTIFY only delivers on commit, which is fine — the
            # frontend re-queries the table after notification anyway.
            # Use pg_notify() because the NOTIFY statement takes
            # literal string args, not bind parameters.
            await conn.execute(
                "SELECT pg_notify($1, $2)",
                NOTIFY_CHANNEL,
                run_id_str,
            )
    seq = int(row["seq"])
    logger.info("event run=%s seq=%d type=%s", run_id_str, seq, event_type)
    return seq, row["ts"]


async def notify_channel(run_id: str) -> None:
    """Send a bare NOTIFY (no event row) — used to kick the SSE loop."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "SELECT pg_notify($1, $2)", NOTIFY_CHANNEL, str(run_id)
        )
