"""
Request authentication and prompt-injection defence.

Two responsibilities:
- verify_api_key  : validate inbound API keys on protected routes.
- detect_injection / sanitize : guard scraped web content before it reaches
  the LLM context window.
"""

from __future__ import annotations

import re

from fastapi import Header, HTTPException, status

from backend.config import get_settings

# ---------------------------------------------------------------------------
# Prompt-injection detection
# ---------------------------------------------------------------------------

# Each tuple is (human-readable label, compiled pattern).
# All patterns are matched against the *lowercased* input so they are
# case-insensitive without the re.IGNORECASE overhead per call.
_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ignore previous instructions", re.compile(r"ignore\s+previous\s+instructions")),
    ("you are now",                  re.compile(r"\byou\s+are\s+now\b")),
    ("new system prompt",            re.compile(r"new\s+system\s+prompt")),
    ("forget your instructions",     re.compile(r"forget\s+your\s+instructions")),
    ("act as a different",           re.compile(r"act\s+as\s+a\s+different")),
    ("override your",                re.compile(r"override\s+your\b")),
    ("[INST] token",                 re.compile(r"\[inst\]")),
    ("<im_start> token",             re.compile(r"<im_start>")),
    ("<|im_start|> token",           re.compile(r"<\|im_start\|>")),
    ("<|system|> token",             re.compile(r"<\|system\|>")),
]


def detect_injection(text: str) -> tuple[bool, list[str]]:
    """Scan *text* for known prompt-injection patterns.

    Returns a ``(matched, labels)`` tuple where *matched* is True when at
    least one pattern fires and *labels* is the list of human-readable names
    for every pattern that matched.  The check runs on the lowercased text so
    patterns do not need the IGNORECASE flag.
    """
    lowered = text.lower()
    matched_labels: list[str] = [
        label for label, pattern in _INJECTION_PATTERNS if pattern.search(lowered)
    ]
    return bool(matched_labels), matched_labels


def sanitize(url: str, text: str) -> dict[str, object]:
    """Wrap scraped web content in an XML boundary and flag injections.

    The ``safe_text`` value encloses *text* in
    ``<untrusted_web_content source="…">`` tags.  This signals to the LLM
    that the content originates from an external, potentially hostile source
    and should never be treated as instructions.

    Returns a dict with:
    - ``safe_text``      — XML-wrapped text ready for chunking / context use.
    - ``is_suspicious``  — True when any injection pattern was detected.
    - ``patterns``       — List of matched pattern labels (empty if clean).
    """
    is_suspicious, patterns = detect_injection(text)
    safe_text = (
        f'<untrusted_web_content source="{url}">\n'
        f"{text}\n"
        f"</untrusted_web_content>"
    )
    return {
        "safe_text": safe_text,
        "is_suspicious": is_suspicious,
        "patterns": patterns,
    }


# ---------------------------------------------------------------------------
# API-key authentication
# ---------------------------------------------------------------------------


async def verify_api_key(x_api_key: str = Header(...)) -> None:
    """
    Dependency injected into protected routes.
    Raises 401 if the header is missing or the key does not match
    the OPENROUTER_API_KEY stored in settings (reused as the service
    key for simplicity in development; swap for a dedicated secret in prod).
    """
    settings = get_settings()
    expected = settings.openrouter_api_key

    if not x_api_key or x_api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header.",
        )
