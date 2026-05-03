import { auth } from "@clerk/nextjs/server";
import type { NextRequest } from "next/server";
import { z } from "zod";
import {
  appendResearchEvent,
  approveResearchPlan,
  getLatestResearchPlan,
  getResearchRunById,
  updateResearchRunStatus,
} from "@/lib/db/queries-research";
import { ChatbotError } from "@/lib/errors";
import type {
  ResearchOutlineSection,
  ResearchSubQuestion,
} from "@/lib/db/schema";

// POST /api/research/[id]/plan → approve (and optionally edit) the plan.
//
// The worker leaves the run in `awaiting_approval` once the planner has
// emitted a plan and a `plan_proposed` event. Calling this endpoint
// (a) optionally rewrites subQuestions/outline with the user's edits,
// (b) marks the plan version approved, (c) flips the run to
// `researching`, and (d) appends a `plan_approved` event so the worker
// (and any connected SSE clients) see the transition. The worker picks
// up the new status on its next poll.

const subQuestionSchema = z.object({
  id: z.string().min(1).max(64),
  question: z.string().min(1).max(2000),
  rationale: z.string().max(5000).optional(),
});

const outlineSectionSchema = z.object({
  id: z.string().min(1).max(64),
  title: z.string().min(1).max(500),
  description: z.string().max(2000).optional(),
});

const bodySchema = z
  .object({
    subQuestions: z.array(subQuestionSchema).min(1).max(20).optional(),
    outline: z.array(outlineSectionSchema).min(1).max(20).optional(),
  })
  .strict();

export async function POST(
  request: NextRequest,
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
  if (run.status !== "awaiting_approval") {
    return new ChatbotError(
      "bad_request:api",
      `Plan can only be approved while status='awaiting_approval' (current: ${run.status})`
    ).toResponse();
  }

  let edits: z.infer<typeof bodySchema>;
  try {
    edits = bodySchema.parse(await request.json().catch(() => ({})));
  } catch {
    return new ChatbotError("bad_request:api").toResponse();
  }

  const latest = await getLatestResearchPlan({ runId });
  if (!latest) {
    return new ChatbotError(
      "not_found:chat",
      "No plan exists for this run"
    ).toResponse();
  }

  const subQuestions = (edits.subQuestions ?? latest.subQuestions) as
    | ResearchSubQuestion[]
    | undefined;
  const outline = (edits.outline ?? latest.outline) as
    | ResearchOutlineSection[]
    | undefined;

  await approveResearchPlan({
    runId,
    version: latest.version,
    subQuestions,
    outline,
  });
  await updateResearchRunStatus({ id: runId, status: "researching" });
  await appendResearchEvent({
    runId,
    type: "plan_approved",
    payload: {
      version: latest.version,
      subQuestions,
      outline,
      editedByUser: Boolean(edits.subQuestions || edits.outline),
    },
  });

  return Response.json({ ok: true, status: "researching" });
}
