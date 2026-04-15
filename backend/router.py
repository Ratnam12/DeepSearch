"""
FastAPI router: declares all HTTP endpoints.
Single responsibility: map HTTP verbs + paths to agent calls.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from backend.agent import DeepSearchAgent
from backend.security import verify_api_key

api_router = APIRouter()


class SearchRequest(BaseModel):
    query: str
    use_cache: bool = True


class SearchResponse(BaseModel):
    answer: str
    sources: list[str]
    cached: bool
    confidence: float


def get_agent() -> DeepSearchAgent:
    """FastAPI dependency: returns a fresh agent instance per request."""
    return DeepSearchAgent()


@api_router.post(
    "/search",
    response_model=SearchResponse,
    dependencies=[Depends(verify_api_key)],
)
async def search(
    request: SearchRequest,
    agent: DeepSearchAgent = Depends(get_agent),
) -> SearchResponse:
    """Run a full deep-search pipeline and return a grounded answer."""
    try:
        result = await agent.run(query=request.query, use_cache=request.use_cache)
        return SearchResponse(**result)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


@api_router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
