"""Smoke test for both model routers.

Run with ``python test_routing.py`` to see what each router picks for
representative queries. The heuristic side is fully offline; the LLM
classifier needs OPENROUTER_API_KEY in the environment and makes one
real Flash call per query.
"""

import asyncio

from backend.model_router import llm_route_model, route_model

QUERIES = [
    "Who is Elon Musk?",
    "Compare the trade-offs between BM25 and dense retrieval in hybrid RAG systems",
    "Summarise this 30-page legal contract and flag any unusual indemnity clauses",
    "Write a Python function that deduplicates a list while preserving order",
    "Draft a cover letter for a senior backend engineer role at a fintech startup",
]


def main() -> None:
    print("# heuristic router (sync)")
    for q in QUERIES:
        print(f"  {route_model(q):<30} ← {q[:80]}")

    print("\n# llm-classifier router (async, one Flash call each)")

    async def _run() -> None:
        for q in QUERIES:
            chosen = await llm_route_model(q)
            print(f"  {chosen:<30} ← {q[:80]}")

    asyncio.run(_run())


if __name__ == "__main__":
    main()
