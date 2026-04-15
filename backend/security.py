"""
Request authentication.
Single responsibility: validate that inbound requests carry a valid API key.
"""

from fastapi import Header, HTTPException, status

from backend.config import get_settings


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
