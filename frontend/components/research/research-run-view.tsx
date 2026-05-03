"use client";

import { format } from "date-fns";
import { Loader2Icon } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import type {
  ResearchPlan,
  ResearchReport,
  ResearchRun,
  ResearchSubagent,
  ResearchSubQuestion,
} from "@/lib/db/schema";

// Live event payload as emitted by the SSE endpoint. The server stamps
// each one with seq/type/ts and a free-form `payload` object that
// depends on the event type — we keep it as `unknown` here and only
// extract specific keys where we render them.
type StreamEvent = {
  seq: number;
  type: string;
  ts: string;
  payload: Record<string, unknown>;
};

const TERMINAL_STATUSES = new Set(["done", "failed", "cancelled"]);

const STATUS_LABELS: Record<string, string> = {
  queued: "Queued",
  scoping: "Scoping",
  planning: "Planning",
  awaiting_approval: "Awaiting your approval",
  researching: "Researching",
  writing: "Writing report",
  done: "Done",
  failed: "Failed",
  cancelled: "Cancelled",
};

export function ResearchRunView({
  initialRun,
  initialPlan,
  initialSubagents,
  initialReport,
}: {
  initialRun: ResearchRun;
  initialPlan: ResearchPlan | null;
  initialSubagents: ResearchSubagent[];
  initialReport: ResearchReport | null;
}) {
  const [run, setRun] = useState<ResearchRun>(initialRun);
  const [plan, setPlan] = useState<ResearchPlan | null>(initialPlan);
  const [report, setReport] = useState<ResearchReport | null>(initialReport);
  const [events, setEvents] = useState<StreamEvent[]>([]);
  const [approving, setApproving] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [streamReady, setStreamReady] = useState(false);
  const lastSeqRef = useRef<number>(0);

  // ── Live event stream (SSE) ──────────────────────────────────────────
  useEffect(() => {
    if (TERMINAL_STATUSES.has(run.status)) {
      // Run already finished; no reason to open SSE.
      setStreamReady(true);
      return;
    }
    const url = `/api/research/${run.id}/events`;
    const es = new EventSource(url);
    let cancelled = false;

    const onMessage = (e: MessageEvent) => {
      if (cancelled) return;
      try {
        const data = JSON.parse(e.data) as StreamEvent | { status?: string };
        if ("seq" in data) {
          handleStreamEvent(data);
        } else if (data && typeof data.status === "string") {
          setRun((r) => ({ ...r, status: data.status as ResearchRun["status"] }));
        }
      } catch {
        // Ignore unparseable events.
      }
    };

    const handleStreamEvent = (evt: StreamEvent) => {
      lastSeqRef.current = Math.max(lastSeqRef.current, evt.seq);
      setEvents((prev) => {
        if (prev.some((e) => e.seq === evt.seq)) return prev;
        return [...prev, evt].sort((a, b) => a.seq - b.seq);
      });
      // Apply side-effects of recognised event types directly to local
      // state so the UI doesn't have to wait for a re-fetch round-trip.
      switch (evt.type) {
        case "status_changed": {
          const status = evt.payload.status;
          if (typeof status === "string") {
            setRun((r) => ({ ...r, status: status as ResearchRun["status"] }));
          }
          break;
        }
        case "plan_proposed":
        case "plan_approved": {
          // Re-fetch plan + run snapshot — small enough to not be a concern.
          fetch(`/api/research/${run.id}`, { cache: "no-store" })
            .then((r) => r.json())
            .then((snap) => {
              if (snap.run) setRun(snap.run);
              if (snap.plan) setPlan(snap.plan);
            })
            .catch(() => undefined);
          break;
        }
        case "report_written":
        case "run_completed": {
          fetch(`/api/research/${run.id}`, { cache: "no-store" })
            .then((r) => r.json())
            .then((snap) => {
              if (snap.run) setRun(snap.run);
              if (snap.report) setReport(snap.report);
            })
            .catch(() => undefined);
          break;
        }
        default:
          break;
      }
    };

    // Listen on every event type the server emits — EventSource
    // delivers typed events to per-name listeners only. We register a
    // fallback `message` handler for any types the server adds later.
    es.addEventListener("message", onMessage);
    const events = [
      "status",
      "status_changed",
      "run_started",
      "brief_drafted",
      "plan_proposed",
      "plan_approved",
      "research_started",
      "subagent_started",
      "subagent_finished",
      "report_written",
      "run_completed",
      "run_failed",
      "run_cancelled",
    ];
    for (const t of events) es.addEventListener(t, onMessage);
    es.onopen = () => {
      if (!cancelled) setStreamReady(true);
    };
    es.onerror = () => {
      // EventSource auto-reconnects; we just surface a hint via state.
      if (!cancelled) setStreamReady(false);
    };
    return () => {
      cancelled = true;
      es.close();
    };
  }, [run.id, run.status]);

  // ── Plan approval ───────────────────────────────────────────────────
  const onApprovePlan = useCallback(async () => {
    if (!plan || approving) return;
    setApproving(true);
    try {
      const res = await fetch(`/api/research/${run.id}/plan`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      if (!res.ok) {
        const text = await res.text();
        toast.error(`Couldn't approve plan: ${text}`);
        return;
      }
      // Status will flip via the SSE stream.
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "approve failed");
    } finally {
      setApproving(false);
    }
  }, [plan, approving, run.id]);

  const onCancel = useCallback(async () => {
    if (cancelling || TERMINAL_STATUSES.has(run.status)) return;
    setCancelling(true);
    try {
      const res = await fetch(`/api/research/${run.id}/cancel`, {
        method: "POST",
      });
      if (!res.ok) {
        const text = await res.text();
        toast.error(`Couldn't cancel: ${text}`);
        return;
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "cancel failed");
    } finally {
      setCancelling(false);
    }
  }, [cancelling, run.id, run.status]);

  // ── Render ──────────────────────────────────────────────────────────
  const isTerminal = TERMINAL_STATUSES.has(run.status);
  const statusLabel = STATUS_LABELS[run.status] ?? run.status;

  const subQuestions = useMemo<ResearchSubQuestion[]>(() => {
    if (!plan?.subQuestions) return [];
    if (Array.isArray(plan.subQuestions)) return plan.subQuestions;
    return [];
  }, [plan?.subQuestions]);

  return (
    <div className="grid h-dvh grid-rows-[auto_1fr] overflow-hidden">
      <header className="border-b border-border bg-background px-6 py-4">
        <div className="mx-auto flex max-w-5xl items-start justify-between gap-4">
          <div className="min-w-0">
            <p className="text-muted-foreground text-xs">
              Research · {format(new Date(run.createdAt), "MMM d, yyyy 'at' p")}
            </p>
            <h1 className="mt-0.5 line-clamp-2 font-semibold text-lg leading-tight">
              {run.query}
            </h1>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            <StatusPill isTerminal={isTerminal} label={statusLabel} streamReady={streamReady} />
            {!isTerminal && (
              <Button
                onClick={onCancel}
                disabled={cancelling}
                size="sm"
                variant="ghost"
              >
                {cancelling ? "Cancelling…" : "Cancel"}
              </Button>
            )}
          </div>
        </div>
      </header>

      <div className="grid grid-cols-1 gap-6 overflow-hidden lg:grid-cols-[1fr_360px]">
        <main className="min-h-0 overflow-y-auto px-6 py-6">
          <div className="mx-auto max-w-3xl">
            {plan && run.status === "awaiting_approval" && (
              <PlanApprovalCard
                approving={approving}
                onApprove={onApprovePlan}
                plan={plan}
                subQuestions={subQuestions}
              />
            )}

            {report ? (
              <ReportPanel report={report} />
            ) : (
              <ReportPlaceholder run={run} />
            )}
          </div>
        </main>

        <aside className="min-h-0 overflow-y-auto border-border border-l bg-muted/20 px-4 py-4 lg:px-5">
          <h2 className="font-semibold text-muted-foreground text-xs uppercase tracking-wide">
            Live progress
          </h2>
          <Timeline events={events} />
        </aside>
      </div>
    </div>
  );
}

function StatusPill({
  isTerminal,
  label,
  streamReady,
}: {
  isTerminal: boolean;
  label: string;
  streamReady: boolean;
}) {
  return (
    <span
      className={
        isTerminal
          ? "inline-flex items-center gap-1.5 rounded-full bg-muted px-2.5 py-1 font-medium text-[11px] text-muted-foreground"
          : "inline-flex items-center gap-1.5 rounded-full bg-primary/10 px-2.5 py-1 font-medium text-[11px] text-primary"
      }
    >
      {!isTerminal && <Loader2Icon className="size-3 animate-spin" />}
      {label}
      {!isTerminal && !streamReady && <span className="opacity-60">· reconnecting</span>}
    </span>
  );
}

function PlanApprovalCard({
  plan,
  subQuestions,
  approving,
  onApprove,
}: {
  plan: ResearchPlan;
  subQuestions: ResearchSubQuestion[];
  approving: boolean;
  onApprove: () => void;
}) {
  return (
    <div className="mb-6 rounded-lg border border-primary/30 bg-primary/5 p-4">
      <p className="font-medium text-sm">Research plan</p>
      <p className="mt-1 text-muted-foreground text-xs">
        Review and approve before research begins. (Phase-1 plan editing
        UI is read-only; full editing lands with the planner agent.)
      </p>
      {plan.briefMd && (
        <p className="mt-3 whitespace-pre-wrap text-sm">{plan.briefMd}</p>
      )}
      <ol className="mt-3 space-y-1 text-sm">
        {subQuestions.map((q, idx) => (
          <li className="flex gap-2" key={q.id}>
            <span className="shrink-0 font-medium text-muted-foreground">
              {idx + 1}.
            </span>
            <span>
              <span className="font-medium">{q.question}</span>
              {q.rationale && (
                <span className="block text-muted-foreground text-xs">
                  {q.rationale}
                </span>
              )}
            </span>
          </li>
        ))}
      </ol>
      <div className="mt-4 flex justify-end">
        <Button disabled={approving} onClick={onApprove} size="sm">
          {approving ? "Approving…" : "Approve & start research"}
        </Button>
      </div>
    </div>
  );
}

function ReportPanel({ report }: { report: ResearchReport }) {
  return (
    <article className="prose prose-sm dark:prose-invert max-w-none">
      <pre className="whitespace-pre-wrap font-sans text-sm leading-relaxed">
        {report.markdown}
      </pre>
    </article>
  );
}

function ReportPlaceholder({ run }: { run: ResearchRun }) {
  return (
    <div className="rounded-lg border border-border border-dashed p-8 text-center">
      <p className="font-medium text-sm">Report will appear here</p>
      <p className="mt-1 text-muted-foreground text-xs">
        Status: {STATUS_LABELS[run.status] ?? run.status}. The report is
        written once research is complete.
      </p>
    </div>
  );
}

function Timeline({ events }: { events: StreamEvent[] }) {
  if (events.length === 0) {
    return (
      <p className="mt-3 text-muted-foreground text-xs">
        Waiting for the worker to pick up this run…
      </p>
    );
  }
  return (
    <ol className="mt-3 space-y-2.5">
      {events.map((evt) => (
        <li className="flex flex-col gap-0.5" key={evt.seq}>
          <span className="font-medium text-sm">{prettyType(evt.type)}</span>
          <span className="text-[11px] text-muted-foreground">
            {format(new Date(evt.ts), "p")} · seq {evt.seq}
          </span>
          {extractEventDetail(evt) && (
            <span className="line-clamp-3 text-muted-foreground text-xs">
              {extractEventDetail(evt)}
            </span>
          )}
        </li>
      ))}
    </ol>
  );
}

function prettyType(type: string): string {
  return type.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function extractEventDetail(evt: StreamEvent): string | null {
  const p = evt.payload;
  if (typeof p?.subQuestion === "string") return p.subQuestion;
  if (typeof p?.findingMd === "string") return p.findingMd;
  if (typeof p?.briefMd === "string") return p.briefMd;
  if (typeof p?.error === "string") return p.error;
  return null;
}
