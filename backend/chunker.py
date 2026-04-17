"""Text chunking utilities.
Single responsibility: split raw document text into overlapping chunks
suitable for embedding and retrieval.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import get_settings

# Separator priority: paragraph → line → sentence → word → character.
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def chunk_text(
    text: str,
    source_url: str,
    title: str = "",
) -> list[dict[str, Any]]:
    """Split `text` into overlapping chunks and return enriched metadata dicts.

    Each dict contains: text, source_url, title, chunk_index, total_chunks,
    scraped_at. Chunks under chunk_min_length characters are dropped.
    """
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=_SEPARATORS,
    )
    raw_chunks = splitter.split_text(text)
    valid = [c.strip() for c in raw_chunks if len(c.strip()) >= settings.chunk_min_length]

    scraped_at = datetime.now(timezone.utc).isoformat()
    total = len(valid)

    return [
        {
            "text": chunk,
            "source_url": source_url,
            "title": title,
            "chunk_index": idx,
            "total_chunks": total,
            "scraped_at": scraped_at,
        }
        for idx, chunk in enumerate(valid)
    ]


def chunk_documents(documents: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Adapter: chunk a list of {url, text} dicts using chunk_text."""
    all_chunks: list[dict[str, Any]] = []
    for doc in documents:
        all_chunks.extend(chunk_text(doc["text"], source_url=doc["url"]))
    return all_chunks
