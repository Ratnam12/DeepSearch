"""
RAGAS-style evaluation against the golden set.
Measures faithfulness, answer relevancy, and context recall for each example.
Also runs an LLM-as-judge quality score (helpfulness, clarity, groundedness_signal)
for every eval row via backend.judge.judge_quality.

Run with:
    pytest tests/test_ragas.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from backend.agent import DeepSearchAgent
from backend.config import get_settings
from backend.judge import judge_quality

_GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"
_settings = get_settings()


def load_golden_set() -> list[dict[str, Any]]:
    with _GOLDEN_SET_PATH.open() as f:
        return json.load(f)


def _token_overlap(a: str, b: str) -> float:
    """Jaccard token overlap as a lightweight faithfulness proxy."""
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


@pytest.fixture(scope="module")
def agent() -> DeepSearchAgent:
    return DeepSearchAgent()


@pytest.mark.asyncio
@pytest.mark.parametrize("example", load_golden_set(), ids=[e["id"] for e in load_golden_set()])
async def test_answer_meets_confidence(
    example: dict[str, Any],
    agent: DeepSearchAgent,
) -> None:
    """Each answer must meet or exceed the example's min_confidence threshold."""
    result = await agent.run(query=example["query"], use_cache=False)

    assert result["confidence"] >= example["min_confidence"], (
        f"[{example['id']}] confidence {result['confidence']:.3f} < "
        f"expected {example['min_confidence']}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("example", load_golden_set(), ids=[e["id"] for e in load_golden_set()])
async def test_answer_faithfulness(
    example: dict[str, Any],
    agent: DeepSearchAgent,
) -> None:
    """Answer tokens must overlap with ground truth above a minimum threshold."""
    result = await agent.run(query=example["query"], use_cache=False)
    overlap = _token_overlap(result["answer"], example["ground_truth"])

    assert overlap >= 0.10, (
        f"[{example['id']}] token overlap {overlap:.3f} too low — "
        f"answer may be hallucinated or off-topic."
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("example", load_golden_set(), ids=[e["id"] for e in load_golden_set()])
async def test_sources_returned(
    example: dict[str, Any],
    agent: DeepSearchAgent,
) -> None:
    """At least one source URL must be returned."""
    result = await agent.run(query=example["query"], use_cache=False)
    assert len(result["sources"]) >= 1, (
        f"[{example['id']}] no sources returned — retrieval may have failed."
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("example", load_golden_set(), ids=[e["id"] for e in load_golden_set()])
async def test_judge_quality_scores(
    example: dict[str, Any],
    agent: DeepSearchAgent,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Log LLM-as-judge scores for every eval row.

    Asserts that the overall average is at least 2.0 (a very lenient floor —
    the primary value here is the logged per-dimension breakdown, not a hard gate).
    """
    result = await agent.run(query=example["query"], use_cache=False)
    scores = await judge_quality(question=example["query"], answer=result["answer"])

    with capsys.disabled():
        print(
            f"\n[judge] {example['id']} | "
            f"helpfulness={scores['helpfulness']} "
            f"clarity={scores['clarity']} "
            f"groundedness_signal={scores['groundedness_signal']} "
            f"overall={scores['overall']}"
        )

    assert scores["overall"] >= 2.0, (
        f"[{example['id']}] judge overall score {scores['overall']} is below floor of 2.0 — "
        f"scores: {scores}"
    )
