import { auth } from "@clerk/nextjs/server";
import type { NextRequest } from "next/server";
import { z } from "zod";
import {
  createResearchRun,
  listResearchRunsByUserId,
} from "@/lib/db/queries-research";
import { ChatbotError } from "@/lib/errors";

// POST /api/research → create a new research run.
// The worker process polls ResearchRun for status='queued' and picks it
// up; the response returns immediately with the runId so the client can
// navigate to /research/<id> and subscribe to the SSE event stream.
const createBodySchema = z.object({
  query: z.string().trim().min(1).max(2000),
});

export async function POST(request: NextRequest) {
  const { userId } = await auth();
  if (!userId) {
    return new ChatbotError("unauthorized:chat").toResponse();
  }

  let body: { query: string };
  try {
    body = createBodySchema.parse(await request.json());
  } catch {
    return new ChatbotError(
      "bad_request:api",
      "Body must be { query: string } where query is non-empty"
    ).toResponse();
  }

  const run = await createResearchRun({ userId, query: body.query });
  return Response.json({ id: run.id, status: run.status }, { status: 201 });
}

// GET /api/research → list this user's research runs (most recent first).
export async function GET(request: NextRequest) {
  const { userId } = await auth();
  if (!userId) {
    return new ChatbotError("unauthorized:chat").toResponse();
  }

  const limitParam = request.nextUrl.searchParams.get("limit");
  const limit = Math.min(
    Math.max(Number.parseInt(limitParam ?? "20", 10) || 20, 1),
    100
  );

  const runs = await listResearchRunsByUserId({ userId, limit });
  return Response.json({ runs });
}
