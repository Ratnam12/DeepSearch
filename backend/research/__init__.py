"""Deep research agent — long-running, durable, event-streamed pipeline.

The worker process (``backend/research_worker.py``) runs the
orchestrator (``pipeline.py``) for runs claimed from the
``ResearchRun`` table; events are appended via :mod:`events` and live
streaming uses Postgres LISTEN/NOTIFY on the ``research_events``
channel so the Next.js SSE endpoint can fan out to subscribed clients.
"""

from backend.research.events import append_event, notify_channel
from backend.research.pipeline import RUN_PHASES, run_pipeline
from backend.research.planner import ResearchPlan, create_plan
from backend.research.state import (
    claim_queued_run,
    fetch_resumable_runs,
    finalize_failed,
    get_run,
    is_cancelled,
    transition_status,
)
from backend.research.subagent import (
    SubagentResult,
    SubagentSource,
    run_subagent,
)
from backend.research.supervisor import run_research
from backend.research.writer import (
    Citation,
    WriterReport,
    dedup_sources,
    run_writer,
)

__all__ = [
    "RUN_PHASES",
    "Citation",
    "ResearchPlan",
    "SubagentResult",
    "SubagentSource",
    "WriterReport",
    "append_event",
    "claim_queued_run",
    "create_plan",
    "dedup_sources",
    "fetch_resumable_runs",
    "finalize_failed",
    "get_run",
    "is_cancelled",
    "notify_channel",
    "run_pipeline",
    "run_research",
    "run_subagent",
    "run_writer",
    "transition_status",
]
