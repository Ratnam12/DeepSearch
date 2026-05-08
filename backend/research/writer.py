"""Research writer — turns sub-agent findings into a single cited report.

Phase-4 entry point. Replaces the stub ``_run_writing`` that emitted a
placeholder report in earlier phases.

The writer is one OpenRouter call against the configured pro model. It
takes:

- the run's query (becomes the report title),
- the planner's brief and outline,
- every ``ResearchSubagent`` row's finding + sources,
- a globally deduped citation list (URLs across all sub-agents
  collapsed to a single ``[N] → URL`` table that the writer cites
  with).

…and returns a polished markdown report with H1 title, executive
summary, one section per outline entry, inline ``[N]`` citations, and
a ``## Sources`` section.

Source dedup also writes ``ResearchSource`` rows so the frontend can
render a clickable bibliography panel without re-deriving the table.

Stays offline-friendly via ``DISABLE_WRITER_LLM=true``: the stub
writer stitches sub-agent findings under outline headings and emits a
clearly-labelled placeholder report so integration tests can exercise
the pipeline without touching OpenRouter.
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
from backend.research.db import get_pool, openrouter_session_meta
from backend.research.events import append_event
from backend.research.subagent import run_subagent

logger = logging.getLogger("deepsearch.research.writer")


# ── Output / input shapes ────────────────────────────────────────────────


@dataclass(frozen=True)
class Citation:
    citation_num: int
    url: str
    title: str | None = None
    snippet: str | None = None


@dataclass(frozen=True)
class WriterReport:
    markdown: str
    sections: list[dict[str, Any]]
    citations: list[Citation]
    model: str
    used_stub: bool


_SYSTEM_PROMPT = """You are the report writer for a deep-research agent. Multiple
sub-agents have already gathered findings; your job is to fuse them
into one polished, well-cited report — the same kind of output a
reader would expect from OpenAI Deep Research or Gemini Deep
Research.

Output requirements:

1. Format: GitHub-flavoured markdown only (no JSON, no prose around the
   markdown). Start with a single H1 of the report's title. Follow
   with a short *Executive summary* paragraph (2–4 sentences). Then
   one H2 per outline section in the order provided. End with a
   ``## Sources`` section.
2. Citations: cite specific factual claims with bracketed numbers
   like ``[3]``. **Use only the citation numbers from the
   "Available citations" table below — do not invent new ones, do
   not renumber.** Multiple citations on one claim look like
   ``[1][2]``. The same source can be cited many times.
3. Sources section: render the citation table in numeric order, each
   line as ``N. [domain](url) — short title``. The frontend turns
   bracketed citations into clickable links to this section, so the
   numbering must match exactly.
4. Use markdown tables when comparing alternatives or laying out
   structured data. Use a fenced ```mermaid``` block when a flowchart,
   sequence diagram, or timeline genuinely helps comprehension —
   don't force diagrams just to have one.
5. Faithfully represent the sub-agent findings. Don't fabricate facts
   that aren't in the findings. If sub-agents disagreed on a point,
   note the disagreement.
6. Length: as long as the content genuinely needs. Don't pad, but
   don't artificially truncate either — a deep-research report with
   8-12 sub-questions, multiple comparison tables, and a milestone
   roadmap typically runs 2000-4000 words and that's fine. Stop when
   the brief is fully covered, not when you hit a word count.
7. No markdown code fences around the entire response. Just the
   report content."""


# ── Dedup + persistence ──────────────────────────────────────────────────


async def dedup_sources(run_id: str) -> list[Citation]:
    """Build the global citation list and persist ``ResearchSource`` rows.

    Iterates ``ResearchSubagent`` rows in stable order, collects every
    source URL, dedupes by URL, and assigns 1-based citation numbers
    in first-seen order. Idempotent — safe to call twice (the
    ``ON CONFLICT DO UPDATE`` keeps the existing row but refreshes
    title/snippet if they got better on a re-run).
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT "sources"
            FROM "ResearchSubagent"
            WHERE "runId" = $1::uuid
            ORDER BY "createdAt" ASC, "id" ASC
            """,
            str(run_id),
        )

    seen: dict[str, Citation] = {}
    next_num = 1
    for row in rows:
        raw = row["sources"]
        if raw is None:
            continue
        if isinstance(raw, str):
            try:
                source_list = json.loads(raw)
            except json.JSONDecodeError:
                continue
        elif isinstance(raw, list):
            source_list = raw
        else:
            continue
        for src in source_list:
            if not isinstance(src, dict):
                continue
            url = str(src.get("url") or "").strip()
            if not url or url in seen:
                continue
            title_val = src.get("title")
            snippet_val = src.get("snippet")
            seen[url] = Citation(
                citation_num=next_num,
                url=url,
                title=str(title_val) if isinstance(title_val, str) else None,
                snippet=str(snippet_val) if isinstance(snippet_val, str) else None,
            )
            next_num += 1

    citations = list(seen.values())
    if not citations:
        return []

    async with pool.acquire() as conn:
        async with conn.transaction():
            for c in citations:
                await conn.execute(
                    """
                    INSERT INTO "ResearchSource"
                        ("runId", "citationNum", "url", "title", "snippet")
                    VALUES ($1::uuid, $2, $3, $4, $5)
                    ON CONFLICT ("runId", "citationNum") DO UPDATE
                    SET "url" = EXCLUDED."url",
                        "title" = COALESCE(EXCLUDED."title", "ResearchSource"."title"),
                        "snippet" = COALESCE(EXCLUDED."snippet", "ResearchSource"."snippet")
                    """,
                    str(run_id),
                    c.citation_num,
                    c.url,
                    c.title,
                    c.snippet,
                )
    return citations


def _domain_for(url: str) -> str:
    """Best-effort short label like 'arxiv.org' for citation rendering."""
    import re

    cleaned = re.sub(r"^https?://(?:www\.)?", "", url)
    return cleaned.split("/")[0] or url


# ── Stub fallback ────────────────────────────────────────────────────────


def _llm_disabled() -> bool:
    return os.environ.get("DISABLE_WRITER_LLM", "").lower() in {"1", "true", "yes"}


def _writer_model() -> str:
    explicit = os.environ.get("RESEARCH_WRITER_MODEL")
    if explicit:
        return explicit
    return get_settings().pro_model


def _client() -> AsyncOpenAI:
    settings = get_settings()
    return AsyncOpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
    )


def _stub_report(
    *,
    query: str,
    brief_md: str | None,
    outline: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    citations: list[Citation],
    reason: str,
) -> WriterReport:
    """Deterministic stitching of findings under outline headings.

    Deliberately marked as a stub so the UI can show the amber
    "stub report" indicator and not pretend the writer LLM ran.
    """
    parts: list[str] = []
    parts.append(f"# {query.strip() or 'Research report'}")
    parts.append("")
    parts.append(
        f"_Stub report ({reason}). The phase-4 writer LLM didn't run, "
        "so this is a deterministic concatenation of sub-agent findings._"
    )
    parts.append("")
    parts.append("## Executive summary")
    parts.append(brief_md.strip() if brief_md else "_No brief available._")
    parts.append("")

    if outline:
        for section in outline:
            title = str(section.get("title") or "Section").strip()
            parts.append(f"## {title}")
            section_desc = section.get("description")
            if isinstance(section_desc, str) and section_desc.strip():
                parts.append(section_desc.strip())
                parts.append("")

    parts.append("## Sub-agent findings")
    for sa in findings:
        sub_q = str(sa.get("subQuestion") or "(unknown sub-question)").strip()
        finding = str(sa.get("findingMd") or "_No finding._").strip()
        parts.append(f"### {sub_q}")
        parts.append(finding)
        parts.append("")

    if citations:
        parts.append("## Sources")
        for c in citations:
            label = c.title or _domain_for(c.url)
            parts.append(f"{c.citation_num}. [{label}]({c.url})")

    sections = [
        {"id": "executive-summary", "title": "Executive summary"},
        *[
            {
                "id": str(s.get("id") or f"section-{i}"),
                "title": str(s.get("title") or "Section"),
            }
            for i, s in enumerate(outline)
        ],
        {"id": "findings", "title": "Sub-agent findings"},
        {"id": "sources", "title": "Sources"},
    ]

    return WriterReport(
        markdown="\n".join(parts).strip() + "\n",
        sections=sections,
        citations=citations,
        model="stub",
        used_stub=True,
    )


# ── LLM-backed writer ────────────────────────────────────────────────────


def _format_citations_for_prompt(citations: list[Citation]) -> str:
    if not citations:
        return "(no sources gathered — note this in the report)"
    lines = []
    for c in citations:
        label = c.title or _domain_for(c.url)
        lines.append(f"[{c.citation_num}] {c.url} — {label}")
    return "\n".join(lines)


def _format_findings_for_prompt(findings: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for sa in findings:
        sub_q = str(sa.get("subQuestion") or "").strip() or "(unknown sub-question)"
        finding = str(sa.get("findingMd") or "").strip() or "_No finding produced._"
        sources_raw = sa.get("sources") or []
        if isinstance(sources_raw, str):
            try:
                sources_raw = json.loads(sources_raw)
            except json.JSONDecodeError:
                sources_raw = []
        urls = [
            str(s.get("url"))
            for s in sources_raw
            if isinstance(s, dict) and s.get("url")
        ]
        url_list = "; ".join(urls) if urls else "(no sources)"
        blocks.append(
            f"### Sub-question: {sub_q}\n"
            f"Sources used by this sub-agent (in retrieval order):\n  {url_list}\n\n"
            f"Sub-agent finding:\n{finding}"
        )
    return "\n\n---\n\n".join(blocks)


def _format_outline_for_prompt(outline: list[dict[str, Any]]) -> str:
    if not outline:
        return "(no outline; structure the report sensibly yourself)"
    lines = []
    for i, section in enumerate(outline, start=1):
        title = str(section.get("title") or f"Section {i}").strip()
        desc = section.get("description")
        if isinstance(desc, str) and desc.strip():
            lines.append(f"{i}. **{title}** — {desc.strip()}")
        else:
            lines.append(f"{i}. **{title}**")
    return "\n".join(lines)


async def _call_writer_llm(
    *,
    query: str,
    brief_md: str | None,
    outline: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    citations: list[Citation],
    model: str,
    run_id: str | None = None,
    user_id: str | None = None,
) -> str:
    """One non-streaming call to the pro model — returns markdown.

    ``run_id`` / ``user_id`` are forwarded to OpenRouter so this LLM
    call groups under the same session as the planner + sub-agent
    calls in the OR dashboard.
    """
    client = _client()

    user_prompt = (
        f"# Report query\n{query.strip()}\n\n"
        f"# Research brief\n{(brief_md or '(no brief)').strip()}\n\n"
        f"# Required outline (use these sections in order)\n"
        f"{_format_outline_for_prompt(outline)}\n\n"
        f"# Sub-agent findings\n{_format_findings_for_prompt(findings)}\n\n"
        f"# Available citations (use these numbers; do not invent new ones)\n"
        f"{_format_citations_for_prompt(citations)}\n\n"
        f"Now write the full markdown report following all the rules in "
        f"the system prompt. Begin with the H1 title."
    )

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.4,
    }
    # No max_tokens cap — the writer's whole job is to produce a
    # complete report, and a hard cap chops markdown tables /
    # roadmaps mid-sentence (the cap is per-token, not per-section,
    # so structured outputs blow through it faster than prose). The
    # model's own context-window ceiling is the real limit; if a
    # specific deployment ever needs a budget guard, override via
    # the standard OpenRouter request params.
    extra_body = openrouter_session_meta(run_id, user_id)
    if extra_body:
        kwargs["extra_body"] = extra_body
    response = await client.chat.completions.create(**kwargs)
    return (response.choices[0].message.content or "").strip()


def _validate_writer_markdown(markdown: str) -> None:
    """Light sanity checks. Heavy validation here would mostly cause
    pointless retries — better to surface oddities in the UI than
    block on them."""
    if not markdown.strip():
        raise ValueError("writer returned empty markdown")
    # Has at least one H1 — otherwise the title is missing.
    if not any(line.startswith("# ") for line in markdown.splitlines()):
        raise ValueError("writer markdown lacks an H1 title")


# ── Section extraction (lightweight, for sections JSON) ──────────────────


def _extract_sections(markdown: str) -> list[dict[str, str]]:
    """Pull H1 + H2 headings out of the markdown for the sections JSON.

    Used purely for the side-panel TOC and for tests asserting the
    writer's structure. We don't need to be exhaustive — H1/H2 cover
    the OpenAI-style report structure cleanly.
    """
    import re

    sections: list[dict[str, str]] = []
    for line in markdown.splitlines():
        m = re.match(r"^(#{1,2})\s+(.+?)\s*$", line)
        if not m:
            continue
        level = len(m.group(1))
        title = m.group(2).strip()
        if level not in (1, 2) or not title:
            continue
        slug = re.sub(r"[^\w\s-]", "", title).strip().lower()
        slug = re.sub(r"[\s_-]+", "-", slug) or f"section-{len(sections) + 1}"
        sections.append({"id": slug, "title": title, "level": str(level)})
    return sections


# ── Public entry point ───────────────────────────────────────────────────


async def _load_inputs(
    run_id: str,
) -> tuple[str | None, list[dict[str, Any]], list[dict[str, Any]]]:
    """Pull everything the writer needs from the DB.

    Returns ``(briefMd, outline, sub-agent rows)``. Each sub-agent row
    is a plain dict with ``id``, ``subQuestion``, ``findingMd``,
    ``sources``.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        plan_row = await conn.fetchrow(
            """
            SELECT "briefMd", "outline"
            FROM "ResearchPlan"
            WHERE "runId" = $1::uuid
            ORDER BY "version" DESC
            LIMIT 1
            """,
            run_id,
        )
        sa_rows = await conn.fetch(
            """
            SELECT "id", "subQuestion", "findingMd", "sources", "status"
            FROM "ResearchSubagent"
            WHERE "runId" = $1::uuid
            ORDER BY "createdAt" ASC, "id" ASC
            """,
            run_id,
        )

    brief = (plan_row or {}).get("briefMd") if plan_row else None

    raw_outline = (plan_row or {}).get("outline") if plan_row else None
    outline: list[dict[str, Any]] = []
    if isinstance(raw_outline, list):
        outline = [s for s in raw_outline if isinstance(s, dict)]
    elif isinstance(raw_outline, str):
        try:
            parsed = json.loads(raw_outline)
            outline = [s for s in parsed if isinstance(s, dict)] if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            outline = []

    findings: list[dict[str, Any]] = []
    for r in sa_rows:
        findings.append(
            {
                "id": r["id"],
                "subQuestion": r["subQuestion"],
                "findingMd": r["findingMd"],
                "sources": r["sources"],
                "status": r["status"],
            }
        )
    return brief, outline, findings


async def _persist_report(
    run_id: str,
    report: WriterReport,
) -> int:
    """Insert the report row, returning its version. Versions increment
    so a re-run of the writer (e.g. on resume) doesn't clobber prior
    output."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                'SELECT MAX("version") AS max_v FROM "ResearchReport" WHERE "runId" = $1::uuid',
                run_id,
            )
            next_version = int((row["max_v"] or 0)) + 1 if row else 1
            await conn.execute(
                """
                INSERT INTO "ResearchReport"
                    ("runId", "version", "markdown", "sections")
                VALUES ($1::uuid, $2, $3, $4::json)
                """,
                run_id,
                next_version,
                report.markdown,
                json.dumps(report.sections),
            )
    return next_version


# ── Self-critique / fact-check ───────────────────────────────────────────


# JSON-mode schema for the critique pass. Produces three signals:
# (1) ready_to_ship — gate for the optional refinement loop;
# (2) weak_sections — at most one becomes a follow-up sub-agent;
# (3) unsupported_claims — surfaced as events so the activity feed
# shows fact-check results, without rewriting the draft.
_CRITIQUE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "ready_to_ship",
        "weak_sections",
        "unsupported_claims",
        "rationale",
    ],
    "properties": {
        "ready_to_ship": {"type": "boolean"},
        "weak_sections": {
            "type": "array",
            "maxItems": 3,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["section_title", "why_weak", "suggested_query"],
                "properties": {
                    "section_title": {"type": "string"},
                    "why_weak": {"type": "string"},
                    "suggested_query": {
                        "type": "string",
                        "description": (
                            "Self-contained research question another sub-agent "
                            "could execute to strengthen this section."
                        ),
                    },
                },
            },
        },
        "unsupported_claims": {
            "type": "array",
            "maxItems": 6,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["claim", "why_unsupported"],
                "properties": {
                    "claim": {"type": "string"},
                    "why_unsupported": {"type": "string"},
                },
            },
        },
        "rationale": {"type": "string"},
    },
}


_CRITIQUE_SYSTEM_PROMPT = """You are a critical reviewer of a deep-research
report. Read the draft against the available sources and decide:

1. Is the draft ready to ship? Or is one section thin enough that a
   single follow-up sub-agent could materially strengthen it?
2. Are there specific claims in the draft that the available sources
   don't actually support — i.e. fact-check failures the writer
   should know about?

Be strict but pragmatic. The pipeline allows AT MOST one refinement
pass, so if you flag weak_sections, pick the single weakest section
and propose a focused, self-contained ``suggested_query`` for it.
For unsupported_claims, list the 3-6 most important specific
factual statements (numbers, dates, attributions, causal claims)
where the cited sources don't substantiate the claim — not stylistic
nits, not subjective phrasing.

Return JSON matching the provided schema EXACTLY — no prose around
it, no markdown fence."""


@dataclass(frozen=True)
class WriterCritique:
    ready_to_ship: bool
    weak_sections: list[dict[str, str]]
    unsupported_claims: list[dict[str, str]]
    rationale: str
    model: str
    used_stub: bool


def _stub_critique(reason: str) -> WriterCritique:
    return WriterCritique(
        ready_to_ship=True,
        weak_sections=[],
        unsupported_claims=[],
        rationale=f"Critique skipped ({reason}); shipping draft as-is.",
        model="stub",
        used_stub=True,
    )


async def _call_critique_llm(
    *,
    draft_markdown: str,
    findings: list[dict[str, Any]],
    citations: list[Citation],
    model: str,
    run_id: str | None,
    user_id: str | None,
) -> WriterCritique:
    """One non-streaming critique call. Stub-falls-back on any error
    so a critique hiccup can't break the run."""
    if _llm_disabled():
        return _stub_critique("LLM disabled")

    settings = get_settings()
    if not settings.openrouter_api_key:
        return _stub_critique("no OPENROUTER_API_KEY")

    client = _client()
    user_prompt = (
        f"# Available citations (the writer was told to cite only these)\n"
        f"{_format_citations_for_prompt(citations)}\n\n"
        f"# Sub-agent findings used to build the draft\n"
        f"{_format_findings_for_prompt(findings)}\n\n"
        f"# Draft report to review\n{draft_markdown}\n\n"
        f"Return JSON matching this schema:\n\n"
        f"{json.dumps(_CRITIQUE_SCHEMA, indent=2)}"
    )

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": _CRITIQUE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.3,
        "max_tokens": 1800,
    }
    extra_body = openrouter_session_meta(run_id, user_id)
    if extra_body:
        kwargs["extra_body"] = extra_body

    try:
        response = await client.chat.completions.create(**kwargs)
        raw = response.choices[0].message.content or "{}"
        payload = json.loads(raw)
        # Light validation — defaults on anything missing rather than
        # failing the run, since the writer already produced a draft.
        weak = payload.get("weak_sections") or []
        unsupported = payload.get("unsupported_claims") or []
        if not isinstance(weak, list):
            weak = []
        if not isinstance(unsupported, list):
            unsupported = []
        return WriterCritique(
            ready_to_ship=bool(payload.get("ready_to_ship", True)),
            weak_sections=[
                w for w in weak
                if isinstance(w, dict)
                and isinstance(w.get("suggested_query"), str)
                and w["suggested_query"].strip()
            ],
            unsupported_claims=[
                c for c in unsupported
                if isinstance(c, dict) and isinstance(c.get("claim"), str)
            ],
            rationale=str(payload.get("rationale", "")).strip(),
            model=model,
            used_stub=False,
        )
    except (APIError, httpx.HTTPError, json.JSONDecodeError, ValueError) as exc:
        logger.exception("critique call failed; treating draft as ready")
        return _stub_critique(f"LLM error ({exc.__class__.__name__})")


def _critique_model() -> str:
    """Override hook; defaults to writer model so token budgeting is one knob."""
    explicit = os.environ.get("RESEARCH_CRITIQUE_MODEL")
    if explicit:
        return explicit
    return _writer_model()


# ── Phase 4 entry point ──────────────────────────────────────────────────


async def run_writer(run: dict[str, Any]) -> WriterReport:
    """Phase-4 entry point. Read inputs, dedup sources, write a report,
    persist, and emit lifecycle events.

    Returns the :class:`WriterReport` for tests / direct callers.
    Pipeline callers don't need the return value — they just rely on
    ``ResearchReport`` being populated when this returns.
    """
    run_id = str(run["id"])
    user_id = run.get("userId")
    query = str(run.get("query") or "Research report").strip()

    await append_event(run_id, "writer_started", {})

    brief_md, outline, findings = await _load_inputs(run_id)
    citations = await dedup_sources(run_id)
    await append_event(
        run_id,
        "sources_deduped",
        {"count": len(citations)},
    )

    used_stub = _llm_disabled()
    if used_stub:
        report = _stub_report(
            query=query,
            brief_md=brief_md,
            outline=outline,
            findings=findings,
            citations=citations,
            reason="LLM disabled",
        )
    else:
        settings = get_settings()
        if not settings.openrouter_api_key:
            logger.warning("writer: OPENROUTER_API_KEY missing; using stub")
            report = _stub_report(
                query=query,
                brief_md=brief_md,
                outline=outline,
                findings=findings,
                citations=citations,
                reason="no OPENROUTER_API_KEY",
            )
        else:
            model = _writer_model()
            try:
                markdown = await _call_writer_llm(
                    query=query,
                    brief_md=brief_md,
                    outline=outline,
                    findings=findings,
                    citations=citations,
                    model=model,
                    run_id=run_id,
                    user_id=user_id,
                )
                _validate_writer_markdown(markdown)
                report = WriterReport(
                    markdown=markdown,
                    sections=_extract_sections(markdown),
                    citations=citations,
                    model=model,
                    used_stub=False,
                )
            except (APIError, httpx.HTTPError, ValueError) as exc:
                logger.exception("writer LLM call failed; falling back to stub")
                report = _stub_report(
                    query=query,
                    brief_md=brief_md,
                    outline=outline,
                    findings=findings,
                    citations=citations,
                    reason=f"LLM error ({exc.__class__.__name__})",
                )

    version = await _persist_report(run_id, report)
    await append_event(
        run_id,
        "report_written",
        {
            "version": version,
            "characters": len(report.markdown),
            "citationCount": len(citations),
            "model": report.model,
            "stub": report.used_stub,
        },
    )

    # ── Self-critique + optional one-shot refinement ────────────────────
    # Skipped when the draft itself was a stub (nothing for the critic
    # to usefully evaluate) or when the critique LLM is disabled.
    if not report.used_stub and not _llm_disabled():
        await append_event(run_id, "writer_critique_started", {})
        critique = await _call_critique_llm(
            draft_markdown=report.markdown,
            findings=findings,
            citations=citations,
            model=_critique_model(),
            run_id=run_id,
            user_id=user_id,
        )
        await append_event(
            run_id,
            "writer_critique_complete",
            {
                "ready_to_ship": critique.ready_to_ship,
                "weakSectionCount": len(critique.weak_sections),
                "unsupportedClaimCount": len(critique.unsupported_claims),
                "rationale": critique.rationale,
                "model": critique.model,
                "stub": critique.used_stub,
            },
        )

        # Surface fact-check signal as discrete events so the activity
        # feed can render them per-claim. Doesn't modify the draft —
        # just transparency for the user.
        for claim in critique.unsupported_claims:
            await append_event(
                run_id,
                "claim_unsupported",
                {
                    "claim": str(claim.get("claim", "")).strip()[:400],
                    "reason": str(claim.get("why_unsupported", "")).strip()[:300],
                },
            )

        # Single refinement cycle: dispatch ONE follow-up sub-agent
        # for the weakest flagged section, then re-run the writer
        # with the augmented findings. The hard cap matches the
        # paper's recommendation that unbounded critique loops are
        # where runaway cost comes from.
        if (
            not critique.ready_to_ship
            and critique.weak_sections
            and not critique.used_stub
        ):
            weak = critique.weak_sections[0]
            refine_query = str(weak.get("suggested_query", "")).strip()
            section_title = str(weak.get("section_title", "")).strip() or "(unknown)"
            if refine_query:
                refine_id = "refine-1"
                await append_event(
                    run_id,
                    "writer_refine_dispatch",
                    {
                        "section": section_title,
                        "query": refine_query,
                        "reason": str(weak.get("why_weak", "")).strip(),
                    },
                )
                try:
                    await run_subagent(
                        run_id, refine_id, refine_query, user_id=user_id,
                    )
                except Exception:
                    logger.exception(
                        "refinement sub-agent failed run=%s; keeping original draft",
                        run_id,
                    )
                else:
                    # Re-load findings (now includes the refinement
                    # sub-agent's row) and re-dedup sources before
                    # the second writer pass.
                    brief_md, outline, findings = await _load_inputs(run_id)
                    citations = await dedup_sources(run_id)
                    try:
                        refined_markdown = await _call_writer_llm(
                            query=query,
                            brief_md=brief_md,
                            outline=outline,
                            findings=findings,
                            citations=citations,
                            model=_writer_model(),
                            run_id=run_id,
                            user_id=user_id,
                        )
                        _validate_writer_markdown(refined_markdown)
                    except (APIError, httpx.HTTPError, ValueError):
                        logger.exception(
                            "refined writer call failed; keeping original draft"
                        )
                    else:
                        report = WriterReport(
                            markdown=refined_markdown,
                            sections=_extract_sections(refined_markdown),
                            citations=citations,
                            model=report.model,
                            used_stub=False,
                        )
                        version = await _persist_report(run_id, report)
                        await append_event(
                            run_id,
                            "report_refined",
                            {
                                "version": version,
                                "characters": len(report.markdown),
                                "citationCount": len(citations),
                                "section": section_title,
                            },
                        )

    return report
