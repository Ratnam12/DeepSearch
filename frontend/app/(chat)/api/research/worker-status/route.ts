import { auth } from "@clerk/nextjs/server";
import { ChatbotError } from "@/lib/errors";

const BACKEND_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/+$/, "") ??
  "http://localhost:8000";

// Proxy for the FastAPI ``/api/v1/research-worker-status`` diagnostic.
// The artifact card probes this when a run sits in 'queued' too long
// so it can render an honest "worker offline" state with the actual
// error message (e.g. "DATABASE_URL not set") instead of an
// optimistic "Waiting for the worker…" forever.
export async function GET() {
  const { userId } = await auth();
  if (!userId) {
    return new ChatbotError("unauthorized:chat").toResponse();
  }
  try {
    const res = await fetch(`${BACKEND_URL}/api/v1/research-worker-status`, {
      cache: "no-store",
      signal: AbortSignal.timeout(8000),
    });
    if (!res.ok) {
      return Response.json(
        {
          configured: false,
          running: false,
          error: `Backend returned ${res.status}`,
          error_at: null,
          started_at: null,
          stopped_at: null,
        },
        { status: 200 }
      );
    }
    const body = await res.json();
    return Response.json(body, { status: 200 });
  } catch (err) {
    return Response.json(
      {
        configured: false,
        running: false,
        error:
          err instanceof Error
            ? `Cannot reach backend: ${err.message}`
            : "Cannot reach backend",
        error_at: null,
        started_at: null,
        stopped_at: null,
      },
      { status: 200 }
    );
  }
}
