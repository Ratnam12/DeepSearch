"""Research-worker main loop.

Run as a separate process so HTTP latency on the main FastAPI service
isn't affected by the long-running agent loop. The worker:

1. On boot, picks up any runs left in resumable mid-flight states (e.g.
   the previous worker crashed mid-pipeline) and re-runs them through
   :func:`run_pipeline`, which is idempotent on phase transitions.
2. Then enters its main loop: claim a queued run, run it, repeat.

Cancellation is co-operative — :func:`run_pipeline` polls the row
status at every safe point.
"""

from __future__ import annotations

import asyncio
import logging
import signal

from backend.research.db import close_pool
from backend.research.pipeline import run_pipeline
from backend.research.state import (
    claim_queued_run,
    fetch_resumable_runs,
    finalize_failed,
)

logger = logging.getLogger("deepsearch.research.worker")


# How long to wait between empty polls for queued runs. Short enough
# to feel responsive in dev, long enough not to hammer Neon when idle.
IDLE_POLL_SECONDS = 1.0


async def _run_one(run: dict) -> None:
    """Wrap :func:`run_pipeline` in a try/except so a single failing
    run can't take down the whole worker."""
    run_id = run["id"]
    try:
        await run_pipeline(run)
    except Exception as exc:
        logger.exception("pipeline failed run=%s", run_id)
        try:
            await finalize_failed(run_id, str(exc) or exc.__class__.__name__)
        except Exception:
            logger.exception("failed to mark run failed; row may be stuck")


async def _resume_inflight() -> None:
    """On boot, pick up any runs the previous worker left mid-flight."""
    runs = await fetch_resumable_runs()
    if not runs:
        return
    logger.info("resuming %d in-flight run(s)", len(runs))
    # Run them one at a time on the same loop. Phase 2+ may want to
    # parallelise, but the stub orchestrator finishes in seconds.
    for run in runs:
        await _run_one(run)


async def main_loop(stop_event: asyncio.Event) -> None:
    """Drain queued runs until ``stop_event`` is set."""
    await _resume_inflight()
    while not stop_event.is_set():
        try:
            run = await claim_queued_run()
        except Exception:
            logger.exception("claim_queued_run failed; backing off")
            await asyncio.sleep(IDLE_POLL_SECONDS * 5)
            continue

        if run is None:
            # Sleep with a wake-up on stop so we exit promptly on
            # SIGTERM rather than at the end of the next idle tick.
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=IDLE_POLL_SECONDS)
            except asyncio.TimeoutError:
                pass
            continue

        await _run_one(run)


def install_signal_handlers(loop: asyncio.AbstractEventLoop, stop_event: asyncio.Event) -> None:
    """Trip ``stop_event`` on SIGTERM / SIGINT so the loop exits cleanly."""

    def _stop() -> None:
        if not stop_event.is_set():
            logger.info("received stop signal; draining current run then exiting")
            stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _stop)
        except NotImplementedError:
            # Windows / non-main-thread fallback.
            signal.signal(sig, lambda *_: _stop())


async def run_forever() -> None:
    """Top-level coroutine started by the entrypoint script."""
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    install_signal_handlers(loop, stop_event)
    try:
        await main_loop(stop_event)
    finally:
        await close_pool()
