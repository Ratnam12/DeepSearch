"""Model routing: decides Flash vs Pro per query, and logs estimated cost."""

from backend.config import get_settings

_cfg = get_settings()

FLASH: str = _cfg.flash_model
PRO: str = _cfg.pro_model

_SIMPLE_PREFIXES = (
    "what is",
    "who is",
    "when did",
    "define",
    "how many",
)

_RATES: dict[str, tuple[float, float]] = {
    "google/gemini-2.5-flash-lite-preview": (0.25, 1.50),
    "google/gemini-2.5-pro-preview": (2.00, 12.00),
    "google/gemini-3.1-flash-lite-preview": (0.25, 1.50),
    "google/gemini-3.1-pro-preview": (2.00, 12.00),
}


def route_model(query: str) -> str:
    """Return FLASH for simple queries (≥2 of 3 conditions), PRO otherwise."""
    normalized = query.strip().lower()
    word_count = len(normalized.split())

    short_query = word_count < 8
    simple_prefix = any(normalized.startswith(p) for p in _SIMPLE_PREFIXES)
    single_short_question = query.count("?") == 1 and len(query) < 80

    score = sum([short_query, simple_prefix, single_short_question])
    return FLASH if score >= 2 else PRO


async def log_cost(model: str, in_tokens: int, out_tokens: int) -> None:
    """Print the estimated USD cost for a single model call."""
    in_rate, out_rate = _RATES.get(model, (2.00, 12.00))
    cost = (in_tokens * in_rate + out_tokens * out_rate) / 1_000_000
    print(
        f"[cost] model={model} "
        f"in={in_tokens} out={out_tokens} "
        f"estimated=${cost:.6f}"
    )
