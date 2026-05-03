import { auth } from "@clerk/nextjs/server";
import type { NextRequest } from "next/server";
import {
  appendResearchEvent,
  cancelResearchRun,
  getResearchRunById,
} from "@/lib/db/queries-research";
import { ChatbotError } from "@/lib/errors";

// POST /api/research/[id]/cancel → mark a run cancelled.
// The worker checks the status row before each phase and bails out when
// it sees `cancelled`, so this is an at-most-one-step-behind cancel.
export async function POST(
  _request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id: runId } = await params;
  const { userId } = await auth();
  if (!userId) {
    return new ChatbotError("unauthorized:chat").toResponse();
  }

  const run = await getResearchRunById({ id: runId });
  if (!run) {
    return new ChatbotError("not_found:chat").toResponse();
  }
  if (run.userId !== userId) {
    return new ChatbotError("forbidden:chat").toResponse();
  }
  if (run.status === "done" || run.status === "cancelled" || run.status === "failed") {
    return Response.json({ ok: true, status: run.status });
  }

  const cancelled = await cancelResearchRun({ id: runId, userId });
  if (!cancelled) {
    return new ChatbotError(
      "bad_request:database",
      "Failed to cancel run"
    ).toResponse();
  }
  await appendResearchEvent({
    runId,
    type: "run_cancelled",
    payload: { reason: "user_cancelled" },
  });
  return Response.json({ ok: true, status: "cancelled" });
}
