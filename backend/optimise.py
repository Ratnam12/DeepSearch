"""Optimise SynthesizeAnswer via DSPy MIPROv2 scored with RAGAS faithfulness.

Usage:
    python -m backend.optimise

Outputs:
    optimised_synthesizer.json  — compiled module weights saved to project root.
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import dspy

from backend.config import get_settings
from backend.dspy_modules import SynthesizeAnswer

# RAGAS looks for OPENAI_API_KEY as an env var internally — read from settings
# the same way every other module does, then expose it for RAGAS
_settings = get_settings()
os.environ["OPENAI_API_KEY"] = _settings.openai_api_key

_GOLDEN_PATH = Path(__file__).parent.parent / "tests" / "golden_set.json"
_OUTPUT_PATH = Path(__file__).parent.parent / "optimised_synthesizer.json"
_TARGET_N = 20  # desired training-set size; golden set is cycled when smaller

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Golden-set helpers
# ---------------------------------------------------------------------------


def _load_examples(path: Path) -> list[dspy.Example]:
    """Read golden_set.json and return DSPy Examples.

    Each item becomes an Example with:
      - question  (input)
      - contexts  (input) — ground_truth used as offline proxy context
      - answer    (output label)
    """
    with path.open() as fh:
        items: list[dict[str, Any]] = json.load(fh)

    return [
        dspy.Example(
            question=item["query"],
            contexts=item["ground_truth"],
            answer=item["ground_truth"],
        ).with_inputs("question", "contexts")
        for item in items
    ]


def _build_trainset(examples: list[dspy.Example], target: int) -> list[dspy.Example]:
    """Cycle *examples* until *target* is reached (or return as-is if longer)."""
    if len(examples) >= target:
        return examples[:target]
    reps = math.ceil(target / len(examples))
    cycled = (examples * reps)[:target]
    log.warning(
        "Golden set has %d items; cycled to %d training examples.",
        len(examples),
        target,
    )
    return cycled


# ---------------------------------------------------------------------------
# RAGAS faithfulness scorer
# ---------------------------------------------------------------------------


def _ragas_score(question: str, answer: str, contexts: list[str]) -> float:
    """Return a RAGAS faithfulness score in [0, 1].

    Falls back to 0.0 and logs a warning on any failure so that the optimiser
    loop is never hard-aborted by a scoring error.
    """
    try:
        import asyncio
        from openai import AsyncOpenAI
        from ragas.llms import llm_factory
        from ragas.metrics.collections.faithfulness import Faithfulness

        client = AsyncOpenAI(api_key=_settings.openai_api_key)
        llm = llm_factory("gpt-4o-mini", client=client)
        metric = Faithfulness(llm=llm)

        result = asyncio.run(metric.ascore(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
        ))
        return float(result.value)
    except Exception as exc:  # noqa: BLE001
        log.warning("RAGAS faithfulness scoring failed: %s — defaulting to 0.0", exc)
        return 0.0


# ---------------------------------------------------------------------------
# DSPy module
# ---------------------------------------------------------------------------


class SynthesizerModule(dspy.Module):
    """Thin Module wrapper around the SynthesizeAnswer signature."""

    def __init__(self) -> None:
        super().__init__()
        self.predict = dspy.Predict(SynthesizeAnswer)

    def forward(self, question: str, contexts: str = "") -> dspy.Prediction:
        return self.predict(question=question, contexts=contexts)


# ---------------------------------------------------------------------------
# MIPROv2 metric
# ---------------------------------------------------------------------------


def faithfulness_metric(
    example: dspy.Example,
    pred: dspy.Prediction,
    trace: Any = None,
) -> float:
    """Metric for MIPROv2: score predicted answer with RAGAS faithfulness.

    *example.contexts* (the ground-truth used as proxy) is passed as the
    retrieved-context list so RAGAS can measure whether the prediction is
    grounded in the available evidence.
    """
    contexts = [example.contexts] if isinstance(example.contexts, str) else example.contexts
    return _ragas_score(
        question=example.question,
        answer=getattr(pred, "answer", ""),
        contexts=contexts,
    )


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------


def _mean_faithfulness(module: dspy.Module, examples: list[dspy.Example]) -> float:
    """Run *module* over *examples* and return mean RAGAS faithfulness."""
    scores: list[float] = []
    for ex in examples:
        try:
            pred = module(question=ex.question, contexts=ex.contexts)
            score = _ragas_score(
                question=ex.question,
                answer=getattr(pred, "answer", ""),
                contexts=[ex.contexts] if isinstance(ex.contexts, str) else ex.contexts,
            )
            scores.append(score)
            log.info("  [%s]  faithfulness=%.4f", ex.question[:50], score)
        except Exception as exc:  # noqa: BLE001
            log.warning("  Scoring failed for '%s': %s", ex.question[:50], exc)
            scores.append(0.0)
    return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    raw_examples = _load_examples(_GOLDEN_PATH)
    log.info("Loaded %d examples from %s", len(raw_examples), _GOLDEN_PATH)

    train_examples = _build_trainset(raw_examples, _TARGET_N)
    log.info("Training set size: %d", len(train_examples))

    # ── Baseline score ───────────────────────────────────────────────────────
    baseline_module = SynthesizerModule()
    log.info("Scoring baseline faithfulness …")
    before_score = _mean_faithfulness(baseline_module, raw_examples)
    log.info("Baseline mean faithfulness:  %.4f", before_score)

    # ── MIPROv2 compilation ──────────────────────────────────────────────────
    log.info(
        "Compiling with MIPROv2 (num_candidates=10, auto='light') against %d examples …",
        len(train_examples),
    )
    optimizer = dspy.MIPROv2(
        metric=faithfulness_metric,
        num_candidates=10,
        auto="light",
    )

    module_to_compile = SynthesizerModule()
    compiled = optimizer.compile(
        module_to_compile,
        trainset=train_examples,
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    compiled.save(str(_OUTPUT_PATH))
    log.info("Compiled module saved → %s", _OUTPUT_PATH)

    # ── Post-optimisation score ───────────────────────────────────────────────
    log.info("Scoring optimised module faithfulness …")
    after_score = _mean_faithfulness(compiled, raw_examples)
    log.info("Optimised mean faithfulness: %.4f", after_score)

    # ── Summary ──────────────────────────────────────────────────────────────
    delta = after_score - before_score
    log.info(
        "Faithfulness delta: %+.4f  (%.4f → %.4f)",
        delta,
        before_score,
        after_score,
    )


if __name__ == "__main__":
    main()
