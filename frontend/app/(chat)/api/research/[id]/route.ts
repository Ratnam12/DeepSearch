import { auth } from "@clerk/nextjs/server";
import type { NextRequest } from "next/server";
import {
  getLatestResearchPlan,
  getLatestResearchReport,
  getResearchRunById,
  listResearchEventsSince,
  listResearchSources,
  listResearchSubagents,
} from "@/lib/db/queries-research";
import { ChatbotError } from "@/lib/errors";

// GET /api/research/[id] → full snapshot for a single run.
// Returns enough state to render the report + activity view without
// the SSE stream. The SSE endpoint is still opened on top for live
// runs so newly-arriving events flow in. For terminal runs, the
// `events` array in this snapshot is the only history source — the
// artifact pane uses it to render the activity tab.
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

  const [plan, subagents, sources, report, events] = await Promise.all([
    getLatestResearchPlan({ runId: id }),
    listResearchSubagents({ runId: id }),
    listResearchSources({ runId: id }),
    getLatestResearchReport({ runId: id }),
    listResearchEventsSince({ runId: id, sinceSeq: 0 }),
  ]);

  return Response.json({ run, plan, subagents, sources, report, events });
}
