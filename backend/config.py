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
    top_k_retrieval: int = 25
    top_k_final: int = 5
    confidence_threshold: float = 0.65

    # ── Multi-candidate synthesis ───────────────────────────────────────────
    # Max chars of accumulated retrieved context passed to DSPy GenerateCandidate.
    # Caps the concatenation of multiple retrieve_chunks rounds.
    max_dspy_context_chars: int = 9_000

    # ── Semantic cache ─────────────────────────────────────────────────────
    cache_similarity_threshold: float = 0.70
    cache_ttl_seconds: int = 3600

    # ── Scraper ────────────────────────────────────────────────────────────
    scrape_timeout_seconds: int = 15
    scrape_concurrency: int = 3
    scrape_min_line_length: int = 50


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a singleton Settings instance (parsed once, cached forever)."""
    return Settings()
