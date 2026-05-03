"""Entrypoint for the research worker process.

Run with::

    python -m backend.research_worker

The worker connects to the same Neon database the Next.js side uses,
claims queued ResearchRun rows, and walks them through the pipeline.
It's intentionally a separate process from the FastAPI HTTP server so
crashes / hangs in the agent loop don't block ``/chat`` latency.
"""

from __future__ import annotations

import asyncio
import logging

from backend.research.worker import run_forever


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )


def main() -> None:
    _configure_logging()
    asyncio.run(run_forever())


if __name__ == "__main__":
    main()
