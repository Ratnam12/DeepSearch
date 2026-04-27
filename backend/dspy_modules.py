"""
DSPy signatures and predictors for the DeepSearch pipeline.

Configures a single DSPy LM backed by OpenRouter and exposes two
Predict modules — decompose_query and synthesize_answer — ready to be
called by the orchestration layer.
"""

from __future__ import annotations

import dspy

from backend.config import get_settings

# ---------------------------------------------------------------------------
# LM configuration
# ---------------------------------------------------------------------------

_cfg = get_settings()

_lm = dspy.LM(
    model=f"openrouter/{_cfg.pro_model}",
    api_key=_cfg.openrouter_api_key,
    api_base=_cfg.openrouter_base_url,
)

dspy.configure(lm=_lm)

# ---------------------------------------------------------------------------
# Signatures
# ---------------------------------------------------------------------------


class DecomposeQuery(dspy.Signature):
    """Break a broad research question into focused sub-queries."""

    question: str = dspy.InputField(
        description="user's research question"
    )
    queries: str = dspy.OutputField(
        description="3 specific search queries as a JSON array"
    )


class SynthesizeAnswer(dspy.Signature):
    """Compose a grounded answer from retrieved chunks."""

    question: str = dspy.InputField(
        description="user's research question"
    )
    contexts: str = dspy.InputField(
        description="retrieved text chunks with source URLs"
    )
    answer: str = dspy.OutputField(
        description="comprehensive cited answer with [1][2] inline citations"
    )
    citations: str = dspy.OutputField(
        description="list of source URLs used, or empty string if none"
    )


class GenerateCandidate(dspy.Signature):
    """Generate a grounded answer strictly from the provided context chunks."""

    question: str = dspy.InputField(
        description="user's research question"
    )
    contexts: str = dspy.InputField(
        description="retrieved text chunks with source URLs"
    )
    answer: str = dspy.OutputField(
        description=(
            "Answer based ONLY on information present in the provided context chunks. "
            "Do not add facts from prior knowledge not found in the contexts. "
            "Use [1][2] inline citations referring to the numbered chunks."
        )
    )


# ---------------------------------------------------------------------------
# Predictors  (ready to call; not invoked at import time)
# ---------------------------------------------------------------------------

decompose_query: dspy.Predict = dspy.Predict(DecomposeQuery)
synthesize_answer: dspy.Predict = dspy.Predict(SynthesizeAnswer)
generate_candidate: dspy.Predict = dspy.Predict(GenerateCandidate)
