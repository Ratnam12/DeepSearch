"""
LLM-as-judge quality scorer using the flash model via OpenRouter.

Scores three dimensions on a 1–5 scale:
  - helpfulness:          Does the answer address what was asked?
  - clarity:              Is the answer easy to understand?
  - groundedness_signal:  Does it sound grounded in real sources vs. made up?

Returns those three scores plus an `overall` average.
"""

from __future__ import annotations

import json

from openai import AsyncOpenAI

from backend.config import get_settings

_SYSTEM_PROMPT = """\
You are a strict answer-quality evaluator. Given a question and an answer, \
score the answer on three dimensions using integers from 1 (very poor) to 5 (excellent):

1. helpfulness        – Does the answer directly address what was asked?
2. clarity            – Is the answer easy to understand and well-structured?
3. groundedness_signal – Does the answer sound like it is based on real, \
verifiable information, or does it read like speculation / hallucination?

Return ONLY valid JSON with exactly these keys:
{
  "helpfulness": <int 1-5>,
  "clarity": <int 1-5>,
  "groundedness_signal": <int 1-5>
}
No extra keys, no markdown fences, no commentary.\
"""


async def judge_quality(question: str, answer: str) -> dict[str, float]:
    """Score an answer on helpfulness, clarity, and groundedness_signal.

    Args:
        question: The original user question.
        answer:   The answer produced by the system under evaluation.

    Returns:
        A dict with keys ``helpfulness``, ``clarity``, ``groundedness_signal``,
        and ``overall`` (arithmetic mean of the three dimension scores).
    """
    settings = get_settings()

    client = AsyncOpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
    )

    response = await client.chat.completions.create(
        model=settings.flash_model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Question: {question}\n\nAnswer: {answer}",
            },
        ],
        temperature=0.0,
    )

    raw = response.choices[0].message.content or "{}"
    scores: dict[str, int] = json.loads(raw)

    helpfulness = int(scores["helpfulness"])
    clarity = int(scores["clarity"])
    groundedness_signal = int(scores["groundedness_signal"])
    overall = round((helpfulness + clarity + groundedness_signal) / 3, 2)

    return {
        "helpfulness": helpfulness,
        "clarity": clarity,
        "groundedness_signal": groundedness_signal,
        "overall": overall,
    }
