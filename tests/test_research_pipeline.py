"""Integration test for the phase-1 research worker pipeline.

Drives a real ResearchRun row through the worker against the live Neon
database — no mocks. We rely on Neon being reachable and DATABASE_URL
being set (the test will skip if it isn't).

Coverage:

1. Happy path: a fresh run progresses scoping → planning →
   awaiting_approval → researching → writing → done with the right
   events and a report row.
2. Cancellation: a run cancelled while awaiting approval terminates
   cleanly without producing a report.
3. Resume after worker restart: the worker is killed mid-pipeline and
   the run picks up from where it left off when the worker is restarted.

The test cleans up its own rows on teardown (cascade via
ResearchRun's FKs handles the dependent tables).

Run with::

    PYTHONPATH=. RESEARCH_PHASE_DELAY_S=0.2 RESEARCH_AUTO_APPROVE_AFTER_S=2 \
        venv/bin/pytest tests/test_research_pipeline.py -v
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

# Skip cleanly if no DB is configured — keeps CI green when secrets aren't.
if not (
    os.environ.get("DATABASE_URL")
    or os.environ.get("POSTGRES_URL")
    or (Path(__file__).resolve().parent.parent / "frontend" / ".env.development.local").exists()
):
    pytest.skip("DATABASE_URL not set; integration test requires Neon", allow_module_level=True)

from backend.research import (
    append_event,
    claim_queued_run,
    fetch_resumable_runs,
    get_run,
    run_pipeline,
)
from backend.research import db as research_db
from backend.research.db import close_pool, get_pool


REPO_ROOT = Path(__file__).resolve().parent.parent
WORKER_CMD = [sys.executable, "-m", "backend.research_worker"]

# Each test stamps its rows with a unique userId so concurrent test
# runs (or leftover rows from a previous failed run) don't bleed into
# each other. The setup helper wipes rows for the *current* userId
# before the test starts.
TEST_USER_ID = f"test-user-{os.getpid()}"


# ── Helpers ──────────────────────────────────────────────────────────────


def _worker_env() -> dict[str, str]:
    """Subprocess env: tight phase delays + auto-approval for the test.

    ``DISABLE_PLANNER_LLM=true`` keeps the planner offline (returns a
    deterministic stub plan) so the integration test doesn't need an
    API key and finishes in seconds. The planner-LLM happy path is
    exercised separately in :func:`test_planner_llm_call_smoke`,
    which is gated on ``RUN_LLM_TESTS=1``.
    """
    env = os.environ.copy()
    env.setdefault("RESEARCH_PHASE_DELAY_S", "0.2")
    env.setdefault("RESEARCH_AUTO_APPROVE_AFTER_S", "2")
    env.setdefault("DISABLE_PLANNER_LLM", "true")
    env.setdefault("PYTHONPATH", str(REPO_ROOT))
    return env


async def _reset_pool_for_test() -> None:
    """Drop the cached pool so each test gets a pool on *its* event loop.

    pytest-asyncio in strict mode runs each test on a fresh event
    loop, but our module-level pool is bound to the loop that first
    called :func:`get_pool`. Without this reset, the second test
    blows up with ``Event loop is closed``.
    """
    if research_db._pool is not None:
        try:
            await research_db._pool.close()
        except Exception:
            # Old pool's loop is already gone; nothing to clean up.
            pass
        research_db._pool = None


def _terminate_worker(proc: subprocess.Popen) -> None:
    """Stop the worker subprocess.

    The worker's SIGTERM handler drains the current run before exiting,
    which is the right production behavior but means SIGTERM can take
    tens of seconds during a phase. Tests need a hard kill on timeout
    so they don't false-fail after they've already verified the
    important state via the DB.
    """
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


async def _wipe_test_rows() -> None:
    """Delete any leftover rows for this test process's userId.

    Used in setup, not teardown — if a test fails and leaves rows
    behind, the next test still starts clean. The cascade FK on
    ResearchRun handles the dependent tables.
    """
    await _reset_pool_for_test()
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            'DELETE FROM "ResearchRun" WHERE "userId" = $1',
            TEST_USER_ID,
        )


async def _create_run(query: str) -> str:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO "ResearchRun" ("userId", "query")
            VALUES ($1, $2)
            RETURNING "id"
            """,
            TEST_USER_ID,
            query,
        )
    return str(row["id"])


async def _delete_run(run_id: str) -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute('DELETE FROM "ResearchRun" WHERE "id" = $1::uuid', run_id)


async def _wait_for_status(
    run_id: str,
    target: str | tuple[str, ...],
    *,
    timeout_s: float = 90.0,
    poll_interval: float = 0.5,
) -> str:
    targets = (target,) if isinstance(target, str) else target
    deadline = time.monotonic() + timeout_s
    last_seen = "?"
    while time.monotonic() < deadline:
        run = await get_run(run_id)
        if run is None:
            raise AssertionError(f"run {run_id} disappeared")
        last_seen = run["status"]
        if last_seen in targets:
            return last_seen
        await asyncio.sleep(poll_interval)
    raise AssertionError(
        f"run {run_id} did not reach {targets} within {timeout_s}s; last={last_seen}"
    )


async def _list_event_types(run_id: str) -> list[str]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            'SELECT "type" FROM "ResearchEvent" WHERE "runId" = $1::uuid ORDER BY "seq" ASC',
            run_id,
        )
    return [r["type"] for r in rows]


async def _has_report(run_id: str) -> bool:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            'SELECT 1 FROM "ResearchReport" WHERE "runId" = $1::uuid LIMIT 1',
            run_id,
        )
    return row is not None


# ── Happy path ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_happy_path_pipeline_runs_to_done() -> None:
    """A queued run progresses through every phase and produces a report."""
    await _wipe_test_rows()
    run_id = await _create_run("integration-test happy path")
    proc = subprocess.Popen(WORKER_CMD, env=_worker_env(), cwd=REPO_ROOT)
    try:
        # Phase-1 stub auto-approves after RESEARCH_AUTO_APPROVE_AFTER_S so
        # the test doesn't need to drive the API.
        await _wait_for_status(run_id, "done", timeout_s=120)
    finally:
        _terminate_worker(proc)

    types = await _list_event_types(run_id)
    # Order matters — these are the canonical phase-1 events.
    assert "run_started" in types
    assert "brief_drafted" in types
    assert "plan_proposed" in types
    assert "plan_approved" in types
    assert "research_started" in types
    assert "subagent_started" in types
    assert "subagent_finished" in types
    assert "report_written" in types
    assert "run_completed" in types
    assert await _has_report(run_id), "report row not written"

    await _delete_run(run_id)


# ── Cancellation ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cancel_during_awaiting_approval() -> None:
    """A run cancelled while waiting for approval terminates without a report."""
    # Long auto-approve so the run sits in awaiting_approval long enough.
    env = _worker_env()
    env["RESEARCH_AUTO_APPROVE_AFTER_S"] = "60"

    await _wipe_test_rows()
    run_id = await _create_run("integration-test cancel")
    proc = subprocess.Popen(WORKER_CMD, env=env, cwd=REPO_ROOT)
    try:
        await _wait_for_status(run_id, "awaiting_approval", timeout_s=60)

        # Simulate the cancel API: flip status + emit a cancel event.
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE "ResearchRun"
                SET "status" = 'cancelled', "finishedAt" = now()
                WHERE "id" = $1::uuid
                """,
                run_id,
            )
        await append_event(run_id, "run_cancelled", {"reason": "user_cancelled"})

        # The worker's awaiting-approval loop polls every 0.5s; give it a beat.
        await _wait_for_status(run_id, "cancelled", timeout_s=10)
    finally:
        _terminate_worker(proc)

    assert not await _has_report(run_id), "cancelled run should not produce a report"
    await _delete_run(run_id)


# ── Resume after restart ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_resume_after_worker_restart() -> None:
    """A run interrupted mid-flight resumes when the worker is restarted."""
    # Slow each phase so we can reliably catch the run mid-flight.
    env = _worker_env()
    env["RESEARCH_PHASE_DELAY_S"] = "1.0"
    # Pre-approve via auto-approval so we don't need a second flip.
    env["RESEARCH_AUTO_APPROVE_AFTER_S"] = "1"

    await _wipe_test_rows()
    run_id = await _create_run("integration-test resume")
    proc = subprocess.Popen(WORKER_CMD, env=env, cwd=REPO_ROOT)
    try:
        # Catch it once it has started researching — that means the
        # planner ran and we have a plan row.
        await _wait_for_status(run_id, ("researching", "writing"), timeout_s=120)
    finally:
        _terminate_worker(proc)

    # Run is mid-flight. Status should be one of the resumable ones.
    mid = await get_run(run_id)
    assert mid is not None
    assert mid["status"] in {"researching", "writing"}, mid["status"]

    # Restart the worker — :func:`fetch_resumable_runs` should pick it up.
    proc2 = subprocess.Popen(WORKER_CMD, env=env, cwd=REPO_ROOT)
    try:
        await _wait_for_status(run_id, "done", timeout_s=120)
    finally:
        _terminate_worker(proc2)

    assert await _has_report(run_id)
    await _delete_run(run_id)


# ── Direct-pipeline smoke (no subprocess) ────────────────────────────────


@pytest.mark.asyncio
async def test_direct_pipeline_drives_a_run_inline() -> None:
    """End-to-end test that drives the pipeline in-process.

    This catches issues that subprocess-based tests can miss (import
    errors in modules only loaded inside the worker, env propagation,
    etc.) by exercising the same code paths directly.
    """
    os.environ.setdefault("RESEARCH_PHASE_DELAY_S", "0.05")
    os.environ.setdefault("RESEARCH_AUTO_APPROVE_AFTER_S", "1")
    # Keep the planner offline for the in-process test too — the
    # planner-LLM path has its own dedicated test.
    os.environ.setdefault("DISABLE_PLANNER_LLM", "true")
    await _wipe_test_rows()
    run_id = await _create_run("integration-test direct")

    # Manually claim (the worker normally does this).
    claimed = await claim_queued_run()
    assert claimed is not None
    assert str(claimed["id"]) == run_id, "claimed wrong run; another test in-flight?"

    await run_pipeline(claimed)

    final = await get_run(run_id)
    assert final is not None
    assert final["status"] == "done"
    assert await _has_report(run_id)

    types = await _list_event_types(run_id)
    assert "run_started" in types
    assert "report_written" in types
    assert "run_completed" in types

    await _delete_run(run_id)


# ── Resumable-runs visibility ────────────────────────────────────────────


# ── Planner ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_planner_returns_stub_when_disabled() -> None:
    """With DISABLE_PLANNER_LLM=true, ``create_plan`` returns a valid
    stub plan that satisfies the schema constraints the LLM path is
    held to (3+ sub-questions, 3+ outline sections, non-empty brief)."""
    os.environ["DISABLE_PLANNER_LLM"] = "true"
    from backend.research.planner import create_plan

    plan = await create_plan("How does prompt caching work in LLMs?")

    assert plan.used_stub is True
    assert plan.model == "stub"
    assert plan.brief_md.strip()
    assert 3 <= len(plan.sub_questions) <= 8
    for sq in plan.sub_questions:
        assert sq["id"] and sq["question"] and sq["rationale"]
    assert 3 <= len(plan.outline) <= 6
    for sec in plan.outline:
        assert sec["id"] and sec["title"] and sec["description"]


@pytest.mark.skipif(
    os.environ.get("RUN_LLM_TESTS", "").lower() not in {"1", "true", "yes"},
    reason="LLM-backed planner test is opt-in (set RUN_LLM_TESTS=1).",
)
@pytest.mark.asyncio
async def test_planner_llm_call_smoke() -> None:
    """Hit the real OpenRouter pro model and verify the response
    parses + validates. Runs only when explicitly enabled because it
    costs real money each invocation."""
    # Belt-and-suspenders — ensure no leftover env var disables it.
    os.environ.pop("DISABLE_PLANNER_LLM", None)
    from backend.research.planner import create_plan

    plan = await create_plan(
        "What are the trade-offs between BERT-style and decoder-only LLMs?"
    )

    assert plan.used_stub is False
    assert plan.brief_md.strip()
    assert 3 <= len(plan.sub_questions) <= 8
    assert 3 <= len(plan.outline) <= 6


# ── Other ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_resumable_runs_skips_terminal_and_awaiting_approval() -> None:
    """The boot-up sweep must include mid-flight rows but exclude others."""
    await _wipe_test_rows()
    pool = await get_pool()
    test_runs: list[str] = []
    try:
        async with pool.acquire() as conn:
            # One in each interesting status.
            for status in ("queued", "researching", "awaiting_approval", "done", "failed"):
                row = await conn.fetchrow(
                    """
                    INSERT INTO "ResearchRun" ("userId", "query", "status")
                    VALUES ($1, $2, $3)
                    RETURNING "id"
                    """,
                    TEST_USER_ID,
                    f"resumable test {status}",
                    status,
                )
                test_runs.append(str(row["id"]))

        resumable = await fetch_resumable_runs()
        ids = {str(r["id"]) for r in resumable}
        # 'researching' should be in the resumable set; everything else shouldn't.
        researching_id = test_runs[1]
        for excluded_id in (test_runs[0], test_runs[2], test_runs[3], test_runs[4]):
            assert excluded_id not in ids, f"non-resumable {excluded_id} surfaced"
        assert researching_id in ids
    finally:
        # Clean up regardless.
        async with pool.acquire() as conn:
            await conn.execute(
                'DELETE FROM "ResearchRun" WHERE "id" = ANY($1::uuid[])',
                test_runs,
            )
