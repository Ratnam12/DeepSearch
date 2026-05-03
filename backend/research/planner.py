"""Research planner — turns a user query into a brief + sub-questions + outline.

Replaces the stub plan emitted in phase 1. The planner is one OpenRouter
call to the pro model (configurable) using structured output (JSON
Schema) so the response shape is enforced upstream and we don't have to
defend against the model freelancing.

The planner's job is small but load-bearing:

- ``briefMd`` — a 1–3 paragraph research brief in markdown that
  reframes the user's query into a clear scope statement, identifies
  what's known vs. what needs investigation, and notes any obvious
  ambiguity that the user should clarify.
- ``subQuestions`` — 3–8 independent, parallelisable sub-questions
  that together cover the brief. Each carries an ``id`` (used as the
  sub-agent identifier in phase 3+) and a short ``rationale``
  explaining why the question matters.
- ``outline`` — 3–6 sections for the final report, each with an ``id``
  used as the section anchor in phase 4's writer.

The whole output gets shown to the user as the plan-approval card; the
user can edit it before research kicks off, so the planner's job is to
propose, not decide.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx
from openai import APIError, AsyncOpenAI

from backend.config import get_settings

logger = logging.getLogger("deepsearch.research.planner")


# ── Output schema (mirrors ResearchSubQuestion / ResearchOutlineSection) ──

PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["briefMd", "subQuestions", "outline"],
    "properties": {
        "briefMd": {
            "type": "string",
            "description": (
                "A 1–3 paragraph research brief in markdown. Reframes the "
                "user's query as a scope statement, calls out the key "
                "unknowns, and notes ambiguity worth clarifying."
            ),
        },
        "subQuestions": {
            "type": "array",
            "minItems": 3,
            "maxItems": 8,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "question", "rationale"],
                "properties": {
                    "id": {
                        "type": "string",
                        "description": (
                            "Stable short id like 'sq1'. Used to key sub-agents."
                        ),
                    },
                    "question": {"type": "string"},
                    "rationale": {"type": "string"},
                },
            },
        },
        "outline": {
            "type": "array",
            "minItems": 3,
            "maxItems": 6,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "title", "description"],
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Stable short id like 's1'.",
                    },
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                },
            },
        },
    },
}


_SYSTEM_PROMPT = """You are the planning module of a deep-research agent.

Given a user's research query, produce a research plan that another
agent (or set of parallel sub-agents) can execute. Return JSON
matching the provided schema EXACTLY — no prose around it, no markdown
code fence, just the JSON object.

Constraints:

1. ``briefMd`` is 1–3 short paragraphs in markdown, not a wall of
   text. Reframe the user's query as a clear scope, surface obvious
   ambiguities, and flag anything the user might want to clarify.
2. ``subQuestions`` are 3–8 *independent* questions. Each one should
   be answerable on its own, without depending on another sub-
   question's findings. If two sub-questions can't be researched in
   parallel, merge them or drop one. Use ids ``sq1``, ``sq2``, …
3. ``outline`` is 3–6 report sections. Section titles should describe
   the section's content, not the user's question. Use ids
   ``s1``, ``s2``, …
4. Keep each ``rationale`` and ``description`` to one short sentence —
   they're hints, not paragraphs.
5. Do not invent facts. The plan is a research blueprint; the actual
   facts come from the sub-agents and the writer."""


# ── Public output shape ──────────────────────────────────────────────────


@dataclass(frozen=True)
class ResearchPlan:
    brief_md: str
    sub_questions: list[dict[str, str]]
    outline: list[dict[str, str]]
    model: str
    used_stub: bool

    def to_pipeline_event_payload(self) -> dict[str, Any]:
        return {
            "briefMd": self.brief_md,
            "subQuestions": self.sub_questions,
            "outline": self.outline,
            "model": self.model,
            "stub": self.used_stub,
        }


# ── Stub fallback ────────────────────────────────────────────────────────


def _stub_plan(query: str, *, reason: str = "no LLM") -> ResearchPlan:
    """Deterministic plan used when the LLM is disabled or unreachable.

    Lets the rest of the pipeline keep working in tests / offline
    environments, with a clearly-labelled brief so anyone looking at
    the artifact knows it didn't come from the model.
    """
    sub_questions = [
        {
            "id": "sq1",
            "question": f"What is the current state of: {query}?",
            "rationale": "Establish the baseline facts before going deep.",
        },
        {
            "id": "sq2",
            "question": f"What are the open debates around: {query}?",
            "rationale": "Surface the disagreements and live tensions.",
        },
        {
            "id": "sq3",
            "question": f"What are the practical implications of: {query}?",
            "rationale": "Translate findings into something the user can act on.",
        },
    ]
    outline = [
        {"id": "s1", "title": "Background", "description": "Context and definitions."},
        {
            "id": "s2",
            "title": "Findings",
            "description": "Per sub-question summaries.",
        },
        {
            "id": "s3",
            "title": "Open questions",
            "description": "What we still don't know.",
        },
    ]
    return ResearchPlan(
        brief_md=(
            f"**Research goal (stub plan, {reason}):** {query}\n\n"
            "*This brief is a fallback used when the planner LLM is "
            "disabled. The structure is real; the wording is generic.*"
        ),
        sub_questions=sub_questions,
        outline=outline,
        model="stub",
        used_stub=True,
    )


# ── LLM-backed plan ──────────────────────────────────────────────────────


def _llm_disabled() -> bool:
    return os.environ.get("DISABLE_PLANNER_LLM", "").lower() in {"1", "true", "yes"}


def _planner_model() -> str:
    """Override hook for tests/CI; default to the configured pro model."""
    explicit = os.environ.get("RESEARCH_PLANNER_MODEL")
    if explicit:
        return explicit
    return get_settings().pro_model


def _client() -> AsyncOpenAI:
    settings = get_settings()
    return AsyncOpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
    )


def _validate_plan_payload(payload: dict[str, Any]) -> None:
    """Cheap structural check — catches obvious failures the schema
    would normally catch when it's enforced server-side. We don't pull
    in jsonschema for this; a few asserts keep the dependency surface
    small.
    """
    if not isinstance(payload, dict):
        raise ValueError("plan payload is not a JSON object")
    for key in ("briefMd", "subQuestions", "outline"):
        if key not in payload:
            raise ValueError(f"plan payload missing key: {key}")

    if not isinstance(payload["briefMd"], str) or not payload["briefMd"].strip():
        raise ValueError("briefMd must be a non-empty string")

    sub_qs = payload["subQuestions"]
    if not isinstance(sub_qs, list) or not (3 <= len(sub_qs) <= 8):
        raise ValueError(
            f"subQuestions must be a list of 3–8 items (got {len(sub_qs) if isinstance(sub_qs, list) else 'non-list'})"
        )
    for sq in sub_qs:
        if not isinstance(sq, dict):
            raise ValueError("each sub-question must be an object")
        for k in ("id", "question", "rationale"):
            if not isinstance(sq.get(k), str) or not sq[k].strip():
                raise ValueError(f"sub-question missing/empty {k}")

    outline = payload["outline"]
    if not isinstance(outline, list) or not (3 <= len(outline) <= 6):
        raise ValueError("outline must be a list of 3–6 items")
    for sec in outline:
        if not isinstance(sec, dict):
            raise ValueError("each outline section must be an object")
        for k in ("id", "title", "description"):
            if not isinstance(sec.get(k), str) or not sec[k].strip():
                raise ValueError(f"outline section missing/empty {k}")


async def _call_llm(query: str, model: str) -> dict[str, Any]:
    """One OpenRouter call returning the plan as a dict.

    Uses ``response_format={"type": "json_object"}`` and reinforces the
    schema in the user prompt, since not all OpenRouter-served models
    support full ``json_schema`` strict mode. We validate the result
    ourselves either way.
    """
    client = _client()
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Research query:\n\n{query}\n\n"
                    "Return JSON matching this schema:\n\n"
                    f"{json.dumps(PLAN_SCHEMA, indent=2)}"
                ),
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.4,
        max_tokens=2500,
    )
    raw = response.choices[0].message.content or "{}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        # Surface the raw output in logs — the wrapped error otherwise
        # makes it impossible to tell whether the model returned no
        # JSON, malformed JSON, or a partial response.
        logger.error("planner returned non-JSON: %s", raw[:500])
        raise ValueError(f"planner returned non-JSON: {exc}") from exc


async def create_plan(query: str) -> ResearchPlan:
    """Produce a research plan for ``query``.

    Falls back to :func:`_stub_plan` when the planner is explicitly
    disabled, when the API key isn't configured, or when the LLM call
    fails — the upstream pipeline only blocks on the user's plan
    approval, so a degraded plan still lets the surrounding mechanics
    work.
    """
    if _llm_disabled():
        logger.info("planner LLM disabled via env; using stub")
        return _stub_plan(query, reason="LLM disabled")

    settings = get_settings()
    if not settings.openrouter_api_key:
        logger.warning("planner: OPENROUTER_API_KEY missing; using stub")
        return _stub_plan(query, reason="no OPENROUTER_API_KEY")

    model = _planner_model()
    try:
        payload = await _call_llm(query, model)
        _validate_plan_payload(payload)
    except (APIError, httpx.HTTPError, ValueError) as exc:
        logger.exception("planner LLM call failed; falling back to stub")
        return _stub_plan(query, reason=f"LLM error ({exc.__class__.__name__})")

    return ResearchPlan(
        brief_md=payload["briefMd"].strip(),
        sub_questions=payload["subQuestions"],
        outline=payload["outline"],
        model=model,
        used_stub=False,
    )
