import asyncio
from backend.retriever import upsert_chunks

chunk = {"text": "hello world", "source_url": "test", "title": "test", "chunk_index": 0, "total_chunks": 1, "scraped_at": "2024-01-01"}

asyncio.run(upsert_chunks([chunk]))
print("done")