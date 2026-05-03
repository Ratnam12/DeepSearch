import { auth } from "@clerk/nextjs/server";
import type { NextRequest } from "next/server";
import {
  getLatestResearchPlan,
  getLatestResearchReport,
  getResearchRunById,
  listResearchSources,
  listResearchSubagents,
} from "@/lib/db/queries-research";
import { ChatbotError } from "@/lib/errors";

// GET /api/research/[id] → full snapshot for a single run.
// Used by the /research/[id] page on initial load and on refresh — it
// returns enough state to render the report view without waiting for
// the SSE replay. The SSE endpoint is then opened for live updates.
export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const { userId } = await auth();
  if (!userId) {
    return new ChatbotError("unauthorized:chat").toResponse();
  }

  const run = await getResearchRunById({ id });
  if (!run) {
    return new ChatbotError("not_found:chat").toResponse();
  }
  if (run.userId !== userId) {
    return new ChatbotError("forbidden:chat").toResponse();
  }

  const [plan, subagents, sources, report] = await Promise.all([
    getLatestResearchPlan({ runId: id }),
    listResearchSubagents({ runId: id }),
    listResearchSources({ runId: id }),
    getLatestResearchReport({ runId: id }),
  ]);

  return Response.json({ run, plan, subagents, sources, report });
}
