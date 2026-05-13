"""Model routing: heuristic + LLM-classifier router for the Auto option.

Two routers live here:

* :func:`route_model` — sync, three-rule heuristic. Used by callers that
  need a synchronous decision (e.g. the dspy-candidate-count branch in
  ``agent.py``) and as the fallback when the LLM router fails.
* :func:`llm_route_model` — async, asks the cheap Flash model to classify
  the query into one of a small label set, then maps the label to a
  curated chat model id. This is what runs when the user picks "Auto"
  from the model dropdown.

Routing labels are kept small and orthogonal so a one-token Flash reply
is enough; the table below maps each label to the best-fit model from
the curated catalogue surfaced in the frontend dropdown.
"""

from __future__ import annotations

import asyncio

from openai import AsyncOpenAI

from backend.config import get_settings

_cfg = get_settings()

FLASH: str = _cfg.flash_model
PRO: str = _cfg.pro_model

# Label → OpenRouter model id. Mirrors the curated chat models in
# frontend/lib/ai/models.ts; update both lists together when adding
# routes. Unknown labels fall through to the heuristic router below.
ROUTING_TABLE: dict[str, str] = {
    "quick_lookup": FLASH,
    "research": PRO,
    "long_context": "google/gemini-3.1-pro",
    "citation_heavy": "anthropic/claude-opus-4.7",
    "coding": PRO,
    "creative": "anthropic/claude-sonnet-4.6",
    "general": PRO,
}

# Tight, single-shot classifier prompt. Keeping the label list compact
# means a one-token reply is usually enough — Flash latency stays under
# ~500ms and cost per routing call is well under $0.0001.
_ROUTER_SYSTEM_PROMPT = (
    "You are a model router for a research assistant that has multiple "
    "model backends. Read the user's query and pick the model best suited "
    "to it. Reply with ONE label from the list below — lowercase, nothing else.\n\n"
    "Labels:\n"
    "- quick_lookup: short factual question, expecting a one-paragraph answer\n"
    "- research: multi-hop comparison, synthesis across sources, or analysis\n"
    "- long_context: needs to digest large pasted documents or long passages\n"
    "- citation_heavy: must produce careful citations (legal, medical, academic)\n"
    "- coding: writing, debugging, or explaining code\n"
    "- creative: long-form writing, drafting, structured prose\n"
    "- general: anything else"
)

# Cap how long we'll wait for the classifier before giving up and falling
# back to the heuristic. The Auto option is supposed to feel
# instantaneous — better a slightly worse model pick than a 10-second
# stall before the real answer starts streaming.
_ROUTER_TIMEOUT_SECONDS = 5.0
# Routing decisions key off the latest user message only; trim very long
# pastes so we don't burn classifier tokens on body text the label
# already implies (long_context).
_ROUTER_QUERY_CHAR_CAP = 2_000

_SIMPLE_PREFIXES = (
    "what is",
    "who is",
    "when did",
    "define",
    "how many",
)

_RATES: dict[str, tuple[float, float]] = {
    "google/gemini-2.5-flash-lite-preview": (0.25, 1.50),
    "google/gemini-2.5-pro-preview": (2.00, 12.00),
    "google/gemini-3.1-flash-lite-preview": (0.25, 1.50),
    "google/gemini-3.1-pro-preview": (2.00, 12.00),
    "openai/gpt-5.5": (5.00, 30.00),
    "openai/gpt-5.4-mini": (0.25, 2.00),
}


def route_model(query: str) -> str:
    """Return FLASH for simple queries (≥2 of 3 conditions), PRO otherwise."""
    normalized = query.strip().lower()
    word_count = len(normalized.split())

    short_query = word_count < 8
    simple_prefix = any(normalized.startswith(p) for p in _SIMPLE_PREFIXES)
    single_short_question = query.count("?") == 1 and len(query) < 80

    score = sum([short_query, simple_prefix, single_short_question])
    return FLASH if score >= 2 else PRO


async def llm_route_model(query: str) -> str:
    """Pick a model by asking Flash to classify the query.

    One cheap classifier call returns a label from ``ROUTING_TABLE``;
    we map it to a curated OpenRouter model id. Falls back to the
    heuristic router on any failure (timeout, malformed reply, unknown
    label) so the Auto option always returns a usable model.
    """
    settings = get_settings()
    client = AsyncOpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
    )
    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=settings.flash_model,
                messages=[
                    {"role": "system", "content": _ROUTER_SYSTEM_PROMPT},
                    {"role": "user", "content": query[:_ROUTER_QUERY_CHAR_CAP]},
                ],
                temperature=0.0,
                max_tokens=8,
            ),
            timeout=_ROUTER_TIMEOUT_SECONDS,
        )
        raw = (response.choices[0].message.content or "").strip().lower()
        # Flash sometimes wraps the label in punctuation or tacks on a
        # short justification — take the first token-like run only.
        label = raw.split()[0].strip(".,:;'\"`*-_") if raw else ""
        chosen = ROUTING_TABLE.get(label)
        if chosen:
            print(f"[router] llm-classified label={label} model={chosen}")
            return chosen
        print(
            f"[router] llm-unknown label={label!r} raw={raw!r} — "
            f"falling back to heuristic"
        )
    except Exception:  # noqa: BLE001 — any classifier failure → fallback
        # Silent fallback by design. The classifier currently 400s on Azure
        # (max_tokens=8 < Azure's 16 floor) so this branch fires on every
        # Auto-routed call. Bumping max_tokens would activate LLM routing
        # but also route some queries to models with weaker tool-calling
        # reliability (e.g. gemini-3.1-pro), which regresses the artifact
        # capture path in tests/test_ragas.py. Once the routing-table
        # models are vetted for create_artifact reliability, re-enable
        # this log and bump max_tokens together in one change.
        pass
    return route_model(query)


async def log_cost(model: str, in_tokens: int, out_tokens: int) -> None:
    """Print the estimated USD cost for a single model call."""
    in_rate, out_rate = _RATES.get(model, (2.00, 12.00))
    cost = (in_tokens * in_rate + out_tokens * out_rate) / 1_000_000
    print(
        f"[cost] model={model} "
        f"in={in_tokens} out={out_tokens} "
        f"estimated=${cost:.6f}"
    )
