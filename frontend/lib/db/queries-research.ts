import "server-only";

import { and, asc, desc, eq, gt, sql } from "drizzle-orm";
import { drizzle } from "drizzle-orm/postgres-js";
import postgres from "postgres";
import { ChatbotError } from "../errors";
import {
  type ResearchEvent,
  researchEvent,
  type ResearchOutlineSection,
  type ResearchPlan,
  researchPlan,
  type ResearchReport,
  researchReport,
  type ResearchRun,
  type ResearchRunStatus,
  researchRun,
  type ResearchSource,
  researchSource,
  type ResearchSubagent,
  researchSubagent,
  type ResearchSubQuestion,
} from "./schema";

// All identity is owned by Clerk — `userId` parameters are Clerk user
// ids (e.g. "user_2abc..."). Authorisation is enforced at the route
// layer; queries here only accept inputs that have already been
// validated against the caller's `userId`.

const client = postgres(
  process.env.DATABASE_URL ?? process.env.POSTGRES_URL ?? ""
);
const db = drizzle(client);

// ── Run lifecycle ────────────────────────────────────────────────────────

export async function createResearchRun({
  userId,
  query,
}: {
  userId: string;
  query: string;
}): Promise<ResearchRun> {
  try {
    const [row] = await db
      .insert(researchRun)
      .values({ userId, query })
      .returning();
    return row;
  } catch (error) {
    console.error("createResearchRun failed", {
      userId,
      error: error instanceof Error ? error.message : String(error),
    });
    throw new ChatbotError(
      "bad_request:database",
      "Failed to create research run"
    );
  }
}

export async function getResearchRunById({
  id,
}: {
  id: string;
}): Promise<ResearchRun | null> {
  try {
    const [row] = await db
      .select()
      .from(researchRun)
      .where(eq(researchRun.id, id))
      .limit(1);
    return row ?? null;
  } catch (error) {
    console.error("getResearchRunById failed", { id, error });
    throw new ChatbotError(
      "bad_request:database",
      "Failed to get research run"
    );
  }
}

export async function listResearchRunsByUserId({
  userId,
  limit = 20,
}: {
  userId: string;
  limit?: number;
}): Promise<ResearchRun[]> {
  try {
    return await db
      .select()
      .from(researchRun)
      .where(eq(researchRun.userId, userId))
      .orderBy(desc(researchRun.createdAt))
      .limit(Math.min(Math.max(limit, 1), 100));
  } catch (error) {
    console.error("listResearchRunsByUserId failed", { userId, error });
    throw new ChatbotError(
      "bad_request:database",
      "Failed to list research runs"
    );
  }
}

export async function updateResearchRunStatus({
  id,
  status,
  error,
  startedAt,
  finishedAt,
}: {
  id: string;
  status: ResearchRunStatus;
  error?: string | null;
  startedAt?: Date | null;
  finishedAt?: Date | null;
}): Promise<void> {
  try {
    const patch: Partial<ResearchRun> = { status };
    if (error !== undefined) patch.error = error;
    if (startedAt !== undefined) patch.startedAt = startedAt;
    if (finishedAt !== undefined) patch.finishedAt = finishedAt;
    await db.update(researchRun).set(patch).where(eq(researchRun.id, id));
  } catch (err) {
    console.error("updateResearchRunStatus failed", { id, status, err });
    throw new ChatbotError(
      "bad_request:database",
      "Failed to update research run status"
    );
  }
}

export async function cancelResearchRun({
  id,
  userId,
}: {
  id: string;
  userId: string;
}): Promise<boolean> {
  try {
    const result = await db
      .update(researchRun)
      .set({ status: "cancelled", finishedAt: new Date() })
      .where(and(eq(researchRun.id, id), eq(researchRun.userId, userId)))
      .returning({ id: researchRun.id });
    return result.length > 0;
  } catch (err) {
    console.error("cancelResearchRun failed", { id, err });
    throw new ChatbotError(
      "bad_request:database",
      "Failed to cancel research run"
    );
  }
}

// ── Plan ──────────────────────────────────────────────────────────────────

export async function saveResearchPlan({
  runId,
  version,
  briefMd,
  subQuestions,
  outline,
}: {
  runId: string;
  version: number;
  briefMd: string | null;
  subQuestions: ResearchSubQuestion[];
  outline: ResearchOutlineSection[];
}): Promise<void> {
  try {
    await db.insert(researchPlan).values({
      runId,
      version,
      briefMd,
      subQuestions,
      outline,
    });
  } catch (err) {
    console.error("saveResearchPlan failed", { runId, version, err });
    throw new ChatbotError(
      "bad_request:database",
      "Failed to save research plan"
    );
  }
}

export async function getLatestResearchPlan({
  runId,
}: {
  runId: string;
}): Promise<ResearchPlan | null> {
  try {
    const [row] = await db
      .select()
      .from(researchPlan)
      .where(eq(researchPlan.runId, runId))
      .orderBy(desc(researchPlan.version))
      .limit(1);
    return row ?? null;
  } catch (err) {
    console.error("getLatestResearchPlan failed", { runId, err });
    throw new ChatbotError(
      "bad_request:database",
      "Failed to get research plan"
    );
  }
}

export async function approveResearchPlan({
  runId,
  version,
  subQuestions,
  outline,
}: {
  runId: string;
  version: number;
  subQuestions?: ResearchSubQuestion[];
  outline?: ResearchOutlineSection[];
}): Promise<void> {
  try {
    const patch: Partial<ResearchPlan> = { approvedAt: new Date() };
    if (subQuestions) patch.subQuestions = subQuestions;
    if (outline) patch.outline = outline;
    await db
      .update(researchPlan)
      .set(patch)
      .where(
        and(
          eq(researchPlan.runId, runId),
          eq(researchPlan.version, version)
        )
      );
  } catch (err) {
    console.error("approveResearchPlan failed", { runId, version, err });
    throw new ChatbotError(
      "bad_request:database",
      "Failed to approve research plan"
    );
  }
}

// ── Events ───────────────────────────────────────────────────────────────

// Single shared NOTIFY channel for all runs. Listeners filter on the
// runId in the payload. Has to match backend/research/events.py.
const RESEARCH_NOTIFY_CHANNEL = "research_events";

// Allocate the next seq inside the same statement that inserts the row
// so concurrent writes can't collide on the (runId, seq) primary key.
// Uses COALESCE so the very first event for a run starts at seq=1.
//
// pg_notify is sent in the same transaction so the SSE endpoint's
// LISTEN handler wakes up the moment the event is durable. Without
// the NOTIFY, plan-approval / cancellation events from the API
// wouldn't reach connected clients until the SSE poller's next tick.
export async function appendResearchEvent({
  runId,
  type,
  payload,
}: {
  runId: string;
  type: string;
  payload: Record<string, unknown>;
}): Promise<{ seq: number; ts: Date }> {
  try {
    const rows = await db.execute<{ seq: number; ts: Date }>(sql`
      WITH inserted AS (
        INSERT INTO "ResearchEvent" ("runId", "seq", "type", "payload")
        VALUES (
          ${runId},
          COALESCE(
            (SELECT MAX("seq") FROM "ResearchEvent" WHERE "runId" = ${runId}),
            0
          ) + 1,
          ${type},
          ${JSON.stringify(payload)}::json
        )
        RETURNING "seq", "ts"
      ),
      notified AS (
        SELECT pg_notify(${RESEARCH_NOTIFY_CHANNEL}, ${runId})
      )
      SELECT "seq", "ts" FROM inserted
    `);
    const first = rows[0] as { seq: number; ts: Date } | undefined;
    if (!first) {
      throw new Error("INSERT...RETURNING produced no row");
    }
    return first;
  } catch (err) {
    console.error("appendResearchEvent failed", { runId, type, err });
    throw new ChatbotError(
      "bad_request:database",
      "Failed to append research event"
    );
  }
}

export async function listResearchEventsSince({
  runId,
  sinceSeq,
  limit = 1000,
}: {
  runId: string;
  sinceSeq: number;
  limit?: number;
}): Promise<ResearchEvent[]> {
  try {
    return await db
      .select()
      .from(researchEvent)
      .where(
        and(eq(researchEvent.runId, runId), gt(researchEvent.seq, sinceSeq))
      )
      .orderBy(asc(researchEvent.seq))
      .limit(Math.min(Math.max(limit, 1), 5000));
  } catch (err) {
    console.error("listResearchEventsSince failed", { runId, sinceSeq, err });
    throw new ChatbotError(
      "bad_request:database",
      "Failed to list research events"
    );
  }
}

// ── Sub-agents / sources / report ────────────────────────────────────────

export async function listResearchSubagents({
  runId,
}: {
  runId: string;
}): Promise<ResearchSubagent[]> {
  try {
    return await db
      .select()
      .from(researchSubagent)
      .where(eq(researchSubagent.runId, runId))
      .orderBy(asc(researchSubagent.createdAt));
  } catch (err) {
    console.error("listResearchSubagents failed", { runId, err });
    throw new ChatbotError(
      "bad_request:database",
      "Failed to list research sub-agents"
    );
  }
}

export async function listResearchSources({
  runId,
}: {
  runId: string;
}): Promise<ResearchSource[]> {
  try {
    return await db
      .select()
      .from(researchSource)
      .where(eq(researchSource.runId, runId))
      .orderBy(asc(researchSource.citationNum));
  } catch (err) {
    console.error("listResearchSources failed", { runId, err });
    throw new ChatbotError(
      "bad_request:database",
      "Failed to list research sources"
    );
  }
}

export async function getLatestResearchReport({
  runId,
}: {
  runId: string;
}): Promise<ResearchReport | null> {
  try {
    const [row] = await db
      .select()
      .from(researchReport)
      .where(eq(researchReport.runId, runId))
      .orderBy(desc(researchReport.version))
      .limit(1);
    return row ?? null;
  } catch (err) {
    console.error("getLatestResearchReport failed", { runId, err });
    throw new ChatbotError(
      "bad_request:database",
      "Failed to get research report"
    );
  }
}
