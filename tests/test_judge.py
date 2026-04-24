"""
Smoke tests for backend.judge.judge_quality.

Verifies that the LLM judge scores behave as expected across three cases:
- a clearly good answer        → all dimensions 4–5
- a useless / evasive answer   → helpfulness 1–2
- a plausible but vague answer → groundedness_signal 2–3

Run with:
    pytest tests/test_judge.py -v
"""

from __future__ import annotations

import pytest

from backend.judge import judge_quality


@pytest.mark.asyncio
async def test_good_answer_scores_high() -> None:
    scores = await judge_quality(
        question="What is the capital of France?",
        answer=(
            "The capital of France is Paris. It is the largest city in the country "
            "and serves as its political, cultural, and economic center."
        ),
    )
    assert scores["helpfulness"] >= 4, f"helpfulness too low: {scores}"
    assert scores["clarity"] >= 4, f"clarity too low: {scores}"
    assert scores["groundedness_signal"] >= 4, f"groundedness_signal too low: {scores}"


@pytest.mark.asyncio
async def test_useless_answer_scores_low_helpfulness() -> None:
    scores = await judge_quality(
        question="What is the capital of France?",
        answer="I don't know, maybe something happened.",
    )
    assert scores["helpfulness"] <= 2, f"helpfulness should be low: {scores}"


@pytest.mark.asyncio
async def test_vague_answer_scores_low_groundedness() -> None:
    scores = await judge_quality(
        question="What caused the 2008 financial crisis?",
        answer=(
            "It was caused by some problems in the banking system and housing market. "
            "Various financial institutions were involved and things eventually collapsed."
        ),
    )
    assert scores["groundedness_signal"] <= 3, f"groundedness_signal should be low: {scores}"
