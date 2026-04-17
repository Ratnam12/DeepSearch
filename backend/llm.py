"""
LLM synthesis via OpenRouter.
Single responsibility: turn retrieved chunks + a query into a grounded answer.
"""

from __future__ import annotations

from openai import AsyncOpenAI

from backend.config import get_settings


def _get_client() -> AsyncOpenAI:
    settings = get_settings()
    return AsyncOpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
    )


def _build_prompt(query: str, chunks: list[dict]) -> str:
    context_blocks = "\n\n---\n\n".join(
        f"Source: {c['source_url']}\n{c['text']}" for c in chunks
    )
    return (
        f"You are a precise research assistant. "
        f"Answer the question using ONLY the sources below. "
        f"If the sources do not contain enough information, say so.\n\n"
        f"SOURCES:\n{context_blocks}\n\n"
        f"QUESTION: {query}\n\nANSWER:"
    )


async def synthesise_answer(
    query: str,
    chunks: list[dict],
    use_pro: bool = False,
) -> tuple[str, float, list[str]]:
    """
    Call the configured OpenRouter model and return
    (answer_text, confidence_score, list_of_source_urls).

    Confidence is approximated from the model's logprob on the first token;
    falls back to 1.0 if logprobs are unavailable.
    """
    settings = get_settings()
    client = _get_client()
    model = settings.pro_model if use_pro else settings.flash_model

    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": _build_prompt(query, chunks)}],
        temperature=0.2,
        max_tokens=1024,
        logprobs=True,
        top_logprobs=1,
    )

    answer = response.choices[0].message.content or ""
    sources = list({c["source_url"] for c in chunks})

    try:
        top_lp = response.choices[0].logprobs.content[0].top_logprobs[0].logprob
        import math
        confidence = round(math.exp(top_lp), 4)
    except Exception:
        confidence = 1.0

    return answer, confidence, sources
