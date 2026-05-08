"""asyncpg connection pool for the research worker.

The Next.js side already talks to Neon via Drizzle (``postgres-js``);
this module is the Python equivalent for the long-running worker
process. We keep a single ``asyncpg.Pool`` per worker process and lazily
construct it on first use.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import asyncio

import asyncpg

logger = logging.getLogger("deepsearch.research.db")


# Searched in order; first match wins. The frontend's Vercel-managed
# Neon dumps DATABASE_URL into ``frontend/.env.development.local``, so
# we read that as a fallback for local dev where the root ``.env``
# isn't kept in sync.
_ENV_FILE_CANDIDATES = (
    ".env",
    "frontend/.env.development.local",
    "frontend/.env.local",
)


_pool: Optional[asyncpg.Pool] = None


def _maybe_load_env_file(path: Path) -> None:
    """Minimal .env loader so we don't have to add python-dotenv.

    Only sets vars that aren't already set in the environment, so a
    real shell ``export`` always wins.
    """
    if not path.exists():
        return
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError as exc:
        logger.warning("env file read failed: %s (%s)", path, exc)


def _resolve_database_url() -> str:
    """Find a Postgres URL from env, walking the project root upwards."""
    direct = os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL")
    if direct:
        return direct

    # Walk up from this file looking for the project root, then try
    # each candidate env file. We use Path.parents so this works
    # whether the worker is started from the repo root, the backend
    # dir, or a deployment sandbox.
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        for candidate in _ENV_FILE_CANDIDATES:
            _maybe_load_env_file(parent / candidate)

    direct = os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL")
    if direct:
        return direct

    raise RuntimeError(
        "DATABASE_URL/POSTGRES_URL not set. The research worker needs a "
        "Neon Postgres URL to claim and update runs. Export it in the "
        "shell or add it to .env at the repo root."
    )


def _normalize_pg_url(url: str) -> str:
    """Strip query params asyncpg doesn't recognise.

    Neon's pooled connection string includes ``?sslmode=require`` and
    sometimes ``?pgbouncer=true``. asyncpg understands ``sslmode`` but
    not ``pgbouncer``; the latter is a server-side hint for postgres-js
    and is harmless to drop here.
    """
    if "?" not in url:
        return url
    base, _, query = url.partition("?")
    keep = []
    for pair in query.split("&"):
        if not pair:
            continue
        key = pair.split("=", 1)[0]
        if key in {"pgbouncer", "channel_binding", "options"}:
            continue
        keep.append(pair)
    return f"{base}?{'&'.join(keep)}" if keep else base


async def get_pool() -> asyncpg.Pool:
    """Return the lazily-initialised asyncpg pool for this process.

    Wrapped in a retry loop so a Neon cold-start (suspended compute
    waking on first connect) doesn't crash the worker on boot. Each
    attempt gets a generous ``timeout`` because Neon's wake-from-
    suspend can take 30-60s on its own; the default 60s
    ``connect_timeout`` would race the wake. We give it 90s per try
    and retry up to 3 times — enough headroom for any single wake
    event without holding worker startup hostage.
    """
    global _pool
    if _pool is not None:
        return _pool

    url = _normalize_pg_url(_resolve_database_url())
    # ``min_size=1, max_size=10`` is plenty for a single worker
    # process running ~3 sub-agents in parallel plus pg_notify.
    # ``ssl`` is auto-negotiated from the URL's sslmode parameter.
    last_exc: Exception | None = None
    for attempt in range(1, 4):
        try:
            _pool = await asyncpg.create_pool(
                url,
                min_size=1,
                max_size=10,
                # Per-connection establish timeout. asyncpg uses
                # this for each handshake inside create_pool; 90s
                # covers a full Neon cold-start.
                timeout=90,
            )
            return _pool
        except (asyncio.TimeoutError, asyncpg.exceptions.ConnectionFailureError) as exc:
            last_exc = exc
            logger.warning(
                "asyncpg pool init attempt %d/3 timed out (likely Neon cold-start); retrying",
                attempt,
            )
            await asyncio.sleep(2)
    # Exhausted retries — re-raise the last failure so the worker's
    # done-callback can surface it via /api/v1/research-worker-status.
    assert last_exc is not None
    raise last_exc


async def close_pool() -> None:
    """Close the pool — called from the worker's shutdown path."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def openrouter_session_meta(
    run_id: str | None,
    user_id: str | None,
) -> dict[str, str] | None:
    """Build the ``extra_body`` payload for OpenRouter session tracking.

    Mirrors the pattern in ``backend.agent._stream_completion`` so every
    LLM call inside a single research run groups under the same OR
    session in the dashboard — same view we already get for normal chat.
    Returns ``None`` when there's nothing to send so callers can skip
    the kwarg entirely.
    """
    eb: dict[str, str] = {}
    if run_id:
        # Prefix so research sessions don't collide with chat session
        # ids in the OR dashboard's session list.
        eb["session_id"] = f"research:{run_id}"
    if user_id:
        # OpenRouter caps `user` at 128 chars; Clerk ids are ~32.
        eb["user"] = str(user_id)[:128]
    return eb or None
