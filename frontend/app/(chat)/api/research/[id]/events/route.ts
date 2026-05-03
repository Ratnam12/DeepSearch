import { auth } from "@clerk/nextjs/server";
import type { NextRequest } from "next/server";
import postgres from "postgres";
import {
  getResearchRunById,
  listResearchEventsSince,
} from "@/lib/db/queries-research";
import { ChatbotError } from "@/lib/errors";

// SSE stream of ResearchEvent rows for a given run.
//
// Resumability: clients send `Last-Event-ID: <seq>` on reconnect (the
// browser's EventSource does this automatically). On reconnect we replay
// every event with seq > Last-Event-ID, then subscribe to live updates.
//
// Live updates: the Python worker calls `pg_notify('research_events', <runId>)`
// after each event INSERT. We LISTEN on that channel and filter in
// memory by runId; when notified we re-query rows beyond what we've
// already sent. This keeps the architecture simple (one channel, no
// per-run subscription churn) and the filter cost is negligible at our
// scale.

export const dynamic = "force-dynamic";
export const runtime = "nodejs";
export const maxDuration = 300;

const NOTIFY_CHANNEL = "research_events";
const MAX_TICKS_NO_EVENTS = 60; // 60 * 5s = 5 min watchdog when worker is silent

function sse(part: { id?: number; event?: string; data: unknown }): string {
  const lines: string[] = [];
  if (typeof part.id === "number") lines.push(`id: ${part.id}`);
  if (part.event) lines.push(`event: ${part.event}`);
  lines.push(`data: ${JSON.stringify(part.data)}`);
  return `${lines.join("\n")}\n\n`;
}

type RunSnapshot = {
  id: string;
  status: string;
  userId: string;
  finishedAt: Date | null;
};

const TERMINAL_STATUSES = new Set(["done", "failed", "cancelled"]);

function isTerminal(status: string): boolean {
  return TERMINAL_STATUSES.has(status);
}

export async function GET(
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

  const lastEventId = Number.parseInt(
    request.headers.get("last-event-id") ?? "0",
    10
  );
  const sinceSeq = Number.isFinite(lastEventId) && lastEventId > 0
    ? lastEventId
    : 0;

  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      let lastSent = sinceSeq;
      let closed = false;
      let listener: postgres.ListenMeta | null = null;
      const listenClient = postgres(
        process.env.DATABASE_URL ?? process.env.POSTGRES_URL ?? "",
        { max: 1 }
      );

      const send = (chunk: string) => {
        if (closed) return;
        try {
          controller.enqueue(encoder.encode(chunk));
        } catch {
          closed = true;
        }
      };

      const close = async () => {
        if (closed) return;
        closed = true;
        try {
          if (listener?.unlisten) await listener.unlisten();
        } catch {
          // Channel teardown errors are non-fatal here.
        }
        try {
          await listenClient.end({ timeout: 1 });
        } catch {
          // Best-effort.
        }
        try {
          controller.close();
        } catch {
          // Already closed.
        }
      };

      // Watchdog: heartbeat every 15s so intermediaries don't kill the
      // connection, and abort the stream after MAX_TICKS_NO_EVENTS quiet
      // ticks if the run has been idle and isn't in a terminal state.
      let quietTicks = 0;
      const heartbeat = setInterval(() => {
        if (closed) return;
        send(`: keepalive\n\n`);
      }, 15_000);

      const drain = async (latestRun: RunSnapshot): Promise<boolean> => {
        const events = await listResearchEventsSince({
          runId,
          sinceSeq: lastSent,
        });
        if (events.length > 0) {
          quietTicks = 0;
        } else {
          quietTicks += 1;
        }
        for (const evt of events) {
          send(
            sse({
              id: evt.seq,
              event: evt.type,
              data: { seq: evt.seq, type: evt.type, ts: evt.ts, payload: evt.payload },
            })
          );
          lastSent = evt.seq;
        }
        return isTerminal(latestRun.status);
      };

      // Initial replay + opening status snapshot.
      send(
        sse({
          event: "status",
          data: { status: run.status, runId, lastSent },
        })
      );
      let runState: RunSnapshot = {
        id: run.id,
        status: run.status,
        userId: run.userId,
        finishedAt: run.finishedAt ?? null,
      };
      let terminal = await drain(runState);

      if (terminal) {
        // Already finished — replay then close cleanly.
        clearInterval(heartbeat);
        await close();
        return;
      }

      // Subscribe to live notifications.
      try {
        listener = await listenClient.listen(NOTIFY_CHANNEL, async (payload) => {
          if (closed) return;
          if (payload && payload !== runId) return;
          // Re-fetch run status alongside new events so the loop can exit
          // promptly on terminal transitions, even when the terminal
          // event itself was already drained.
          const latest = await getResearchRunById({ id: runId });
          if (!latest) {
            await close();
            return;
          }
          runState = {
            id: latest.id,
            status: latest.status,
            userId: latest.userId,
            finishedAt: latest.finishedAt ?? null,
          };
          const isTerm = await drain(runState);
          if (isTerm) {
            clearInterval(heartbeat);
            await close();
          }
        });
      } catch (err) {
        console.error("research SSE listen failed", { runId, err });
        // Fall through to polling — the loop below still drains.
      }

      // Polling fallback: even with LISTEN, poll every 5s as a safety
      // net (a dropped notification or a race between INSERT and the
      // listener's setup would otherwise stall the stream until the
      // next NOTIFY). It also handles deployments where Postgres
      // notifications aren't reachable for some reason.
      const poll = setInterval(async () => {
        if (closed) return;
        try {
          const latest = await getResearchRunById({ id: runId });
          if (!latest) {
            clearInterval(poll);
            clearInterval(heartbeat);
            await close();
            return;
          }
          runState = {
            id: latest.id,
            status: latest.status,
            userId: latest.userId,
            finishedAt: latest.finishedAt ?? null,
          };
          const isTerm = await drain(runState);
          if (isTerm) {
            clearInterval(poll);
            clearInterval(heartbeat);
            await close();
          } else if (quietTicks >= MAX_TICKS_NO_EVENTS) {
            // Worker silent for ~5 min — close cleanly so the client
            // reconnects with Last-Event-ID rather than holding open
            // forever. The run row stays in its current state for the
            // worker to resume.
            clearInterval(poll);
            clearInterval(heartbeat);
            await close();
          }
        } catch (err) {
          console.error("research SSE poll failed", { runId, err });
        }
      }, 5_000);

      // Client disconnect cleanup.
      const abortHandler = () => {
        clearInterval(poll);
        clearInterval(heartbeat);
        void close();
      };
      request.signal.addEventListener("abort", abortHandler);
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
    },
  });
}
