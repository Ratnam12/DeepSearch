"""Evaluator — gap detection between research waves.

Sits between the supervisor's first wave of sub-agents and the writer.
Reads the brief + sub-questions + every sub-agent's finding and
returns a structured assessment:

- ``gaps``: angles the current findings cover thinly or not at all,
  each with a follow-up sub-question the supervisor can dispatch as a
  second-wave sub-agent
- ``contradictions``: points where two findings disagree on the same
  fact, paired with both sources
- ``ready_for_writer``: bool — when False the supervisor runs another
  research wave; when True (or after the wave-cap), the writer takes
  over

This is the missing "outer loop" that turns our static
plan→research→write pipeline into a dynamic one. Cribbed from the
pattern OpenAI DR / Gemini DR / Perplexity DR all use: critic reads
interim findings, decides whether to spawn more research, replans on
the fly. Bounded by ``RESEARCH_MAX_WAVES`` so a confused critic can't
loop forever.
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
from backend.research.db import openrouter_session_meta

logger = logging.getLogger("deepsearch.research.evaluator")


# Same model family as the planner — this is a planning-style judgement
# call, not a verbatim text generation. Override for tests.
def _evaluator_model() -> str:
    explicit = os.environ.get("RESEARCH_EVALUATOR_MODEL")
    if explicit:
        return explicit
    return get_settings().pro_model


def _llm_disabled() -> bool:
    return os.environ.get("DISABLE_EVALUATOR_LLM", "").lower() in {"1", "true", "yes"}


# ── Output schema (json-object enforced; we validate ourselves) ──────────

EVAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["ready_for_writer", "gaps", "contradictions", "rationale"],
    "properties": {
        "ready_for_writer": {
            "type": "boolean",
            "description": (
                "True when the current findings cover the brief well "
                "enough to write a substantive report. False when there "
                "are clear gaps or contradictions worth resolving with "
                "another research wave."
            ),
        },
        "gaps": {
            "type": "array",
            "maxItems": 6,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "question", "why_thin"],
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Stable id like 'gap1'. Used as sub-agent id when dispatched.",
                    },
                    "question": {
                        "type": "string",
                        "description": "A focused follow-up sub-question to dispatch as a second-wave sub-agent.",
                    },
                    "why_thin": {
                        "type": "string",
                        "description": "One sentence explaining why current findings don't cover this.",
                    },
                },
            },
        },
        "contradictions": {
            "type": "array",
            "maxItems": 6,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["claim_a", "claim_b", "why_matters"],
                "properties": {
                    "claim_a": {"type": "string"},
                    "claim_b": {"type": "string"},
                    "why_matters": {"type": "string"},
                },
            },
        },
        "rationale": {
            "type": "string",
            "description": "1-3 sentence summary of the assessment for the activity log.",
        },
    },
}


_SYSTEM_PROMPT = """You are the evaluator in a deep-research pipeline. A
team of sub-agents has just finished one wave of research. Your job is
to read their findings against the original research brief and decide:

1. Is this enough to write a substantive report? Or are there obvious
   gaps / thin areas that another research wave should fill?
2. Are there contradictions between findings worth flagging?

Be honest but pragmatic. The pipeline allows at most ONE more wave,
so prioritise the highest-value gaps — pick the 2-4 sub-questions
that would most lift report quality if researched. If the findings
genuinely cover the brief well, set ``ready_for_writer = true`` with
an empty ``gaps`` array. Don't invent gaps to look thorough.

Return JSON matching the provided schema EXACTLY — no prose around
it, no markdown fence. Each gap's ``question`` must be a focused,
self-contained research question another sub-agent could execute
without needing context from the others. Use ids ``gap1``, ``gap2``,
…"""


# ── Output dataclass ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class EvaluationResult:
    ready_for_writer: bool
    gaps: list[dict[str, str]]
    contradictions: list[dict[str, str]]
    rationale: str
    model: str
    used_stub: bool


def _stub_result(reason: str) -> EvaluationResult:
    """Default-safe result used when the evaluator LLM is disabled.

    Returns ``ready_for_writer = True`` so the pipeline doesn't dispatch
    a wasted second wave when we can't actually evaluate.
    """
    return EvaluationResult(
        ready_for_writer=True,
        gaps=[],
        contradictions=[],
        rationale=f"Evaluator skipped ({reason}); proceeding to writer.",
        model="stub",
        used_stub=True,
    )


# ── LLM call ─────────────────────────────────────────────────────────────


def _client() -> AsyncOpenAI:
    settings = get_settings()
    return AsyncOpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
    )


def _format_findings_for_prompt(findings: list[dict[str, Any]]) -> str:
    """Compact rendering of all sub-agent findings for the evaluator.

    Bound each finding to ~1500 chars so even a 12-sub-agent run fits
    well within the evaluator context. We're judging coverage, not
    nitpicking details — the truncated version is plenty.
    """
    lines: list[str] = []
    for f in findings:
        sub_q = (f.get("subQuestion") or "").strip() or "(no sub-question)"
        finding = (f.get("findingMd") or "").strip()
        if len(finding) > 1500:
            finding = finding[:1500].rstrip() + "…"
        sources = f.get("sources") or []
        n_sources = len(sources) if isinstance(sources, list) else 0
        lines.append(f"### {sub_q}\n_({n_sources} sources)_\n\n{finding}")
    return "\n\n".join(lines) if lines else "(no findings)"


def _validate_payload(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("evaluator payload is not an object")
    for k in ("ready_for_writer", "gaps", "contradictions", "rationale"):
        if k not in payload:
            raise ValueError(f"evaluator payload missing key: {k}")
    if not isinstance(payload["ready_for_writer"], bool):
        raise ValueError("ready_for_writer must be bool")
    if not isinstance(payload["gaps"], list):
        raise ValueError("gaps must be a list")
    for g in payload["gaps"]:
        if not isinstance(g, dict):
            raise ValueError("each gap must be an object")
        for k in ("id", "question", "why_thin"):
            if not isinstance(g.get(k), str) or not g[k].strip():
                raise ValueError(f"gap missing/empty {k}")
    if not isinstance(payload["contradictions"], list):
        raise ValueError("contradictions must be a list")
    if not isinstance(payload["rationale"], str):
        raise ValueError("rationale must be string")


async def evaluate_findings(
    *,
    run_id: str,
    user_id: str | None,
    query: str,
    brief_md: str | None,
    sub_questions: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    citation_count: int,
) -> EvaluationResult:
    """Run the evaluator LLM. Falls back to a permissive stub on error.

    On any failure (LLM unreachable, malformed JSON, validation), we
    return ``ready_for_writer = True`` with no gaps so the pipeline
    doesn't stall — the writer can still produce a report from
    whatever findings exist.
    """
    if _llm_disabled():
        return _stub_result("LLM disabled")

    settings = get_settings()
    if not settings.openrouter_api_key:
        logger.warning("evaluator: OPENROUTER_API_KEY missing; skipping")
        return _stub_result("no OPENROUTER_API_KEY")

    model = _evaluator_model()
    client = _client()

    sub_q_lines = "\n".join(
        f"- {sq.get('id','?')}: {sq.get('question','')}"
        for sq in sub_questions
    ) or "(no sub-questions)"

    user_prompt = (
        f"# Research query\n{query.strip()}\n\n"
        f"# Brief\n{(brief_md or '(no brief)').strip()}\n\n"
        f"# Sub-questions assigned to wave-1 sub-agents\n{sub_q_lines}\n\n"
        f"# Findings from wave-1 sub-agents\n"
        f"{_format_findings_for_prompt(findings)}\n\n"
        f"# Source pool\n{citation_count} unique sources cited so far.\n\n"
        f"Return JSON matching this schema:\n\n"
        f"{json.dumps(EVAL_SCHEMA, indent=2)}"
    )

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.3,
        "max_tokens": 2000,
    }
    extra_body = openrouter_session_meta(run_id, user_id)
    if extra_body:
        kwargs["extra_body"] = extra_body

    try:
        response = await client.chat.completions.create(**kwargs)
        raw = response.choices[0].message.content or "{}"
        payload = json.loads(raw)
        _validate_payload(payload)
    except (APIError, httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
        logger.exception("evaluator LLM call failed; using stub")
        return _stub_result(f"LLM error ({exc.__class__.__name__})")

    return EvaluationResult(
        ready_for_writer=bool(payload["ready_for_writer"]),
        gaps=payload["gaps"],
        contradictions=payload["contradictions"],
        rationale=str(payload["rationale"]).strip(),
        model=model,
        used_stub=False,
    )
