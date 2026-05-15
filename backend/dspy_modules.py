"""
DSPy signatures and predictors for the DeepSearch pipeline.

Configures a single DSPy LM backed by OpenRouter and exposes two
Predict modules — decompose_query and synthesize_answer — ready to be
called by the orchestration layer.
"""

from __future__ import annotations

import dspy
import litellm

from backend.config import get_settings

# DSPy import activates litellm's in-memory request cache. On litellm 1.51.0
# (pinned by dspy 2.5.41) the cache-key builder reads __annotations__ off
# openai's TranscriptionCreateParams TypedDict, which raises AttributeError
# on Python 3.12 — once per completion via _get_relevant_args_to_use_for_cache_key
# and again via get_standard_logging_object_payload. The completion itself
# still succeeds (HTTP 200), but the error spams production logs. We don't
# need litellm's cache (we have our own semantic cache), so disable it.
litellm.cache = None

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
