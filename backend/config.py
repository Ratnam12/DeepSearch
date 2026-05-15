"""
Central configuration loaded from .env via pydantic-settings.
Every other module imports from here — no hardcoded values anywhere else.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── API credentials ────────────────────────────────────────────────────
    openrouter_api_key: str
    openai_api_key: str          # text-embedding-3-small only (not on OpenRouter)
    serper_api_key: str

    # ── Service URLs ───────────────────────────────────────────────────────
    qdrant_url: str
    qdrant_api_key: str = ""
    upstash_redis_rest_url: str
    upstash_redis_rest_token: str

    # ── HTTP app config ────────────────────────────────────────────────────
    cors_origins: str = "*"

    # ── OpenRouter config ──────────────────────────────────────────────────
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    pro_model: str = "openai/gpt-5.5"
    flash_model: str = "openai/gpt-5.4-mini"
    ragas_model: str = "openai/gpt-5.4-mini"

    # ── Embeddings ─────────────────────────────────────────────────────────
    embedding_model: str = "text-embedding-3-small"
    embed_max_chars: int = 8191

    # ── Chunking ───────────────────────────────────────────────────────────
    chunk_size: int = 800
    chunk_overlap: int = 200
    chunk_min_length: int = 80

    # ── Retrieval ──────────────────────────────────────────────────────────
    # top_k_final was 5 — RAGAS context_recall measured 0.48, meaning the
    # LLM was only seeing half the ground-truth claims. Bumped to 12 so
    # synthesis questions with 8-15 claim ground_truths can actually be
    # covered. top_k_retrieval widened to 50 to give the cross-encoder a
    # wider candidate pool to rank from, which preserves precision as
    # top_k_final grows.
    top_k_retrieval: int = 50
    top_k_final: int = 12
    confidence_threshold: float = 0.65

    # ── Multi-candidate synthesis ───────────────────────────────────────────
    # Max chars of accumulated retrieved context passed to DSPy GenerateCandidate.
    # Caps the concatenation of multiple retrieve_chunks rounds.
    max_dspy_context_chars: int = 9_000

    # ── Semantic cache ─────────────────────────────────────────────────────
    # Cosine threshold raised from 0.70 → 0.75 because text-embedding-3-small
    # placed semantically distinct queries (e.g. "DSPy in 2023" vs "DSPy in 2026")
    # at ~0.80. The relevance judge below is the real safeguard against the
    # remaining gray-zone false positives; this threshold just filters obvious
    # noise before invoking the judge.
    cache_similarity_threshold: float = 0.75
    cache_ttl_seconds: int = 3600

    # ── Semantic cache relevance judge ─────────────────────────────────────
    # Every above-threshold candidate is passed through this small LLM, which
    # decides whether the cached answer actually answers the new question.
    # Catches mismatches that cosine alone misses: time period, version, scope,
    # entity, aspect, audience, etc. See /tmp/eval_judge_100.py for the eval
    # that picked this model (Gemini 3.1 Flash Lite: 100/100, 0 FP, 0 FN).
    cache_judge_enabled: bool = True
    cache_judge_model: str = "google/gemini-3.1-flash-lite"
    cache_judge_timeout_seconds: float = 3.0

    # ── Server-sent events ──────────────────────────────────────────────────
    sse_ping_seconds: int = 10

    # ── Scraper ────────────────────────────────────────────────────────────
    scrape_timeout_seconds: int = 15
    scrape_concurrency: int = 3
    scrape_min_line_length: int = 50


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a singleton Settings instance (parsed once, cached forever)."""
    return Settings()
