import asyncio
from backend.retriever import upsert_chunks, hybrid_search
from backend.config import get_settings

chunks = [
    {
        "text": "Quantum entanglement is a phenomenon where two particles become linked and the state of one instantly affects the other, regardless of distance.",
        "source_url": "https://en.wikipedia.org/wiki/Quantum_entanglement",
        "title": "Quantum Entanglement",
        "chunk_index": 0,
        "total_chunks": 1,
        "scraped_at": "2024-01-01"
    },
    {
        "text": "Quantum computing uses quantum bits or qubits which can exist in superposition, allowing them to process many calculations simultaneously.",
        "source_url": "https://en.wikipedia.org/wiki/Quantum_computing",
        "title": "Quantum Computing",
        "chunk_index": 0,
        "total_chunks": 1,
        "scraped_at": "2024-01-01"
    },
    {
        "text": "Quantum mechanics is the branch of physics that describes the behavior of particles at the smallest scales of energy levels of atoms and subatomic particles.",
        "source_url": "https://en.wikipedia.org/wiki/Quantum_mechanics",
        "title": "Quantum Mechanics",
        "chunk_index": 0,
        "total_chunks": 1,
        "scraped_at": "2024-01-01"
    },
]

async def main():
    print("Upserting chunks...")
    await upsert_chunks(chunks)
    print("Done upserting.")

    print("\nRunning hybrid_search...")
    results = await hybrid_search("quantum entanglement")

    for r in results:
        print(f"Title: {r['title']} | Score: {r['score']}")

    assert len(results) == get_settings().top_k_final, f"Expected {get_settings().top_k_final} results, got {len(results)}"
    print(f"\n✓ Got {len(results)} results as expected")

asyncio.run(main())