"""
Text chunking utilities.
Single responsibility: split raw document text into overlapping chunks
suitable for embedding and retrieval.
"""

from __future__ import annotations

from typing import Any

from backend.config import get_settings


def chunk_documents(documents: list[dict[str, str]]) -> list[dict[str, Any]]:
    """
    Split each document's text into overlapping windows.

    Returns a flat list of chunk dicts:
        {"text": str, "url": str, "chunk_index": int}
    """
    settings = get_settings()
    all_chunks: list[dict[str, Any]] = []

    for doc in documents:
        chunks = _split_text(
            text=doc["text"],
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "url": doc["url"],
                "chunk_index": idx,
            })

    return all_chunks


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Slide a window of `chunk_size` characters across `text`,
    stepping forward by (chunk_size - overlap) each time.
    Strips empty chunks.
    """
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("chunk_size must be greater than chunk_overlap")

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks
