"use client";

import { format } from "date-fns";
import { Loader2Icon, PlusIcon, XIcon } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import type {
  ResearchOutlineSection,
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

// In-memory shape for live sub-agent state — derived from `subagent_*`
// SSE events plus the initial DB snapshot. Kept separate from the
// Drizzle row type so we can hydrate it from the event payload before
// the DB row has been re-fetched.
type SubagentLive = {
  id: string;
  subQuestion: string;
  status: "running" | "done" | "failed";
  findingMd: string | null;
  sourceCount: number | null;
  latestAction: string | null;
  latestActionDetail: string | null;
  model: string | null;
  stub: boolean;
};

function subagentFromInitial(row: ResearchSubagent): SubagentLive {
  const rawStatus = row.status;
  const status: SubagentLive["status"] =
    rawStatus === "done" || rawStatus === "failed" ? rawStatus : "running";
  const sources = Array.isArray(row.sources) ? (row.sources as unknown[]) : [];
  return {
    id: row.id,
    subQuestion: row.subQuestion,
    status,
    findingMd: row.findingMd,
    sourceCount: sources.length || null,
    latestAction: null,
    latestActionDetail: null,
    model: row.model,
    stub: false,
  };
}

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
  const [subagents, setSubagents] = useState<Record<string, SubagentLive>>(
    () => Object.fromEntries(
      initialSubagents.map((sa) => [sa.id, subagentFromInitial(sa)])
    )
  );
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
        case "subagent_started": {
          // Hydrate from the event payload so the card appears
          // instantly — no need to wait for a DB round-trip.
          const id = evt.payload.id;
          const sub = evt.payload.subQuestion;
          const model = evt.payload.model;
          if (typeof id !== "string" || typeof sub !== "string") break;
          setSubagents((prev) => ({
            ...prev,
            [id]: {
              id,
              subQuestion: sub,
              status: "running",
              findingMd: null,
              sourceCount: null,
              latestAction: null,
              latestActionDetail: null,
              model: typeof model === "string" ? model : null,
              stub: Boolean(evt.payload.stub),
            },
          }));
          break;
        }
        case "subagent_progress": {
          const id = evt.payload.id;
          const action = evt.payload.action;
          const detail = evt.payload.detail;
          if (typeof id !== "string") break;
          setSubagents((prev) => {
            const cur = prev[id];
            if (!cur) return prev;
            return {
              ...prev,
              [id]: {
                ...cur,
                latestAction: typeof action === "string" ? action : cur.latestAction,
                latestActionDetail:
                  typeof detail === "string" ? detail : cur.latestActionDetail,
              },
            };
          });
          break;
        }
        case "subagent_finished": {
          const id = evt.payload.id;
          const finding = evt.payload.findingMd;
          const sourceCount = evt.payload.sourceCount;
          if (typeof id !== "string") break;
          setSubagents((prev) => ({
            ...prev,
            [id]: {
              ...(prev[id] ?? {
                id,
                subQuestion: "",
                model: null,
                stub: false,
              }),
              id,
              subQuestion: prev[id]?.subQuestion ?? "",
              status: "done",
              findingMd: typeof finding === "string" ? finding : null,
              sourceCount:
                typeof sourceCount === "number" ? sourceCount : null,
              latestAction: null,
              latestActionDetail: null,
              model: prev[id]?.model ?? null,
              stub: prev[id]?.stub ?? false,
            },
          }));
          break;
        }
        case "subagent_failed": {
          const id = evt.payload.id;
          const error = evt.payload.error;
          if (typeof id !== "string") break;
          setSubagents((prev) => {
            const cur = prev[id];
            if (!cur) return prev;
            return {
              ...prev,
              [id]: {
                ...cur,
                status: "failed",
                findingMd:
                  typeof error === "string"
                    ? `_Sub-agent failed: ${error}_`
                    : cur.findingMd,
              },
            };
          });
          break;
        }
        case "report_written":
        case "run_completed": {
          fetch(`/api/research/${run.id}`, { cache: "no-store" })
            .then((r) => r.json())
            .then((snap) => {
              if (snap.run) setRun(snap.run);
              if (snap.report) setReport(snap.report);
              if (Array.isArray(snap.subagents)) {
                setSubagents(
                  Object.fromEntries(
                    (snap.subagents as ResearchSubagent[]).map((sa) => [
                      sa.id,
                      subagentFromInitial(sa),
                    ])
                  )
                );
              }
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
      "scoping_started",
      "brief_drafted",
      "plan_proposed",
      "plan_approved",
      "research_started",
      "research_dispatch",
      "subagent_started",
      "subagent_progress",
      "subagent_finished",
      "subagent_failed",
      "research_complete",
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
  const onApprovePlan = useCallback(
    async (edits?: {
      subQuestions: ResearchSubQuestion[];
      outline: ResearchOutlineSection[];
    }) => {
      if (!plan || approving) return;
      setApproving(true);
      try {
        const res = await fetch(`/api/research/${run.id}/plan`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(edits ?? {}),
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
    },
    [plan, approving, run.id]
  );

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

  const outline = useMemo<ResearchOutlineSection[]>(() => {
    if (!plan?.outline) return [];
    if (Array.isArray(plan.outline)) return plan.outline;
    return [];
  }, [plan?.outline]);

  // Order sub-agents by their plan position so the UI is stable as
  // events arrive — sub-agents that complete first don't jump to the
  // top of the list. Falls back to insertion order for ad-hoc ids.
  const subagentList = useMemo<SubagentLive[]>(() => {
    const planOrder = new Map<string, number>(
      subQuestions.map((q, i) => [q.id, i])
    );
    return Object.values(subagents).sort((a, b) => {
      const ai = planOrder.get(a.id);
      const bi = planOrder.get(b.id);
      if (ai !== undefined && bi !== undefined) return ai - bi;
      if (ai !== undefined) return -1;
      if (bi !== undefined) return 1;
      return a.id.localeCompare(b.id);
    });
  }, [subagents, subQuestions]);

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
                initialOutline={outline}
                initialSubQuestions={subQuestions}
                onApprove={onApprovePlan}
                plan={plan}
                planEvents={events}
              />
            )}

            <SubagentList
              status={run.status}
              subagents={subagentList}
            />

            {report ? (
              <ReportPanel report={report} />
            ) : (
              <ReportPlaceholder run={run} subagentCount={subagentList.length} />
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
  initialSubQuestions,
  initialOutline,
  approving,
  onApprove,
  planEvents,
}: {
  plan: ResearchPlan;
  initialSubQuestions: ResearchSubQuestion[];
  initialOutline: ResearchOutlineSection[];
  approving: boolean;
  onApprove: (edits?: {
    subQuestions: ResearchSubQuestion[];
    outline: ResearchOutlineSection[];
  }) => void;
  planEvents: StreamEvent[];
}) {
  // Surface "this plan came from the deterministic stub, not the
  // model" so a user staring at a generic-looking plan understands
  // why. The planner stamps `stub: true` on plan_proposed events.
  const planProposedEvent = useMemo(
    () => planEvents.find((e) => e.type === "plan_proposed"),
    [planEvents]
  );
  const usedStub = Boolean(planProposedEvent?.payload?.stub);
  const plannerModel =
    typeof planProposedEvent?.payload?.model === "string"
      ? (planProposedEvent.payload.model as string)
      : null;
  // Local working copy so the user's typed edits don't lose focus on
  // each keystroke (which would happen if we re-derived from props).
  // Reset when the plan id/version changes.
  const [draftSubQs, setDraftSubQs] =
    useState<ResearchSubQuestion[]>(initialSubQuestions);
  const [draftOutline, setDraftOutline] =
    useState<ResearchOutlineSection[]>(initialOutline);
  const planSignature = `${plan.runId}:${plan.version}`;
  const planSignatureRef = useRef(planSignature);
  if (planSignatureRef.current !== planSignature) {
    planSignatureRef.current = planSignature;
    setDraftSubQs(initialSubQuestions);
    setDraftOutline(initialOutline);
  }

  const dirty = useMemo(() => {
    if (draftSubQs.length !== initialSubQuestions.length) return true;
    if (draftOutline.length !== initialOutline.length) return true;
    for (let i = 0; i < draftSubQs.length; i += 1) {
      const a = draftSubQs[i];
      const b = initialSubQuestions[i];
      if (a.id !== b.id || a.question !== b.question || a.rationale !== b.rationale)
        return true;
    }
    for (let i = 0; i < draftOutline.length; i += 1) {
      const a = draftOutline[i];
      const b = initialOutline[i];
      if (a.id !== b.id || a.title !== b.title || a.description !== b.description)
        return true;
    }
    return false;
  }, [draftSubQs, draftOutline, initialSubQuestions, initialOutline]);

  const updateSubQ = (index: number, patch: Partial<ResearchSubQuestion>) => {
    setDraftSubQs((prev) =>
      prev.map((q, i) => (i === index ? { ...q, ...patch } : q))
    );
  };

  const removeSubQ = (index: number) => {
    setDraftSubQs((prev) => prev.filter((_, i) => i !== index));
  };

  const addSubQ = () => {
    if (draftSubQs.length >= 8) {
      toast("Up to 8 sub-questions supported.");
      return;
    }
    const idx = draftSubQs.length + 1;
    setDraftSubQs((prev) => [
      ...prev,
      {
        id: `sq${idx}-${Math.random().toString(36).slice(2, 6)}`,
        question: "",
        rationale: "",
      },
    ]);
  };

  const updateSection = (
    index: number,
    patch: Partial<ResearchOutlineSection>
  ) => {
    setDraftOutline((prev) =>
      prev.map((s, i) => (i === index ? { ...s, ...patch } : s))
    );
  };

  const removeSection = (index: number) => {
    setDraftOutline((prev) => prev.filter((_, i) => i !== index));
  };

  const addSection = () => {
    if (draftOutline.length >= 6) {
      toast("Up to 6 outline sections supported.");
      return;
    }
    const idx = draftOutline.length + 1;
    setDraftOutline((prev) => [
      ...prev,
      {
        id: `s${idx}-${Math.random().toString(36).slice(2, 6)}`,
        title: "",
        description: "",
      },
    ]);
  };

  const handleApprove = () => {
    // Strip any whitespace-only fields. Keep server validation simple
    // — anything blank would fail the zod schema on the API.
    const cleanSubQs = draftSubQs
      .map((q) => ({
        ...q,
        question: q.question.trim(),
        rationale: q.rationale?.trim(),
      }))
      .filter((q) => q.question.length > 0);
    const cleanOutline = draftOutline
      .map((s) => ({
        ...s,
        title: s.title.trim(),
        description: s.description?.trim(),
      }))
      .filter((s) => s.title.length > 0);

    if (cleanSubQs.length < 1) {
      toast.error("Add at least one sub-question before approving.");
      return;
    }
    if (cleanOutline.length < 1) {
      toast.error("Keep at least one outline section before approving.");
      return;
    }
    onApprove(dirty ? { subQuestions: cleanSubQs, outline: cleanOutline } : undefined);
  };

  return (
    <div className="mb-6 rounded-lg border border-primary/30 bg-primary/5 p-5">
      <div className="flex items-baseline justify-between gap-3">
        <div className="flex items-baseline gap-2">
          <p className="font-medium text-sm">Research plan</p>
          {plannerModel && !usedStub && (
            <span className="rounded-full bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground">
              {plannerModel}
            </span>
          )}
          {usedStub && (
            <span className="rounded-full bg-amber-500/10 px-1.5 py-0.5 text-[10px] text-amber-700 dark:text-amber-400">
              stub plan (LLM disabled)
            </span>
          )}
        </div>
        <p className="text-muted-foreground text-[11px]">
          Review and edit before research begins.
        </p>
      </div>
      {plan.briefMd && (
        <p className="mt-3 whitespace-pre-wrap text-foreground/90 text-sm leading-relaxed">
          {plan.briefMd}
        </p>
      )}

      <PlanSection title="Sub-questions">
        {draftSubQs.length === 0 && (
          <p className="text-muted-foreground text-xs">
            No sub-questions yet. Add at least one to proceed.
          </p>
        )}
        <ol className="space-y-2.5">
          {draftSubQs.map((q, idx) => (
            <li
              className="grid grid-cols-[18px_1fr_24px] items-start gap-2"
              key={q.id}
            >
              <span className="pt-1.5 font-medium text-muted-foreground text-xs">
                {idx + 1}.
              </span>
              <div className="space-y-1">
                <input
                  className="w-full rounded-md border border-border/60 bg-background px-2 py-1.5 font-medium text-sm focus-visible:border-primary focus-visible:outline-none"
                  onChange={(e) => updateSubQ(idx, { question: e.target.value })}
                  placeholder="Sub-question"
                  type="text"
                  value={q.question}
                />
                <input
                  className="w-full rounded-md border border-transparent bg-transparent px-2 py-1 text-muted-foreground text-xs focus-visible:border-border focus-visible:bg-background focus-visible:outline-none"
                  onChange={(e) =>
                    updateSubQ(idx, { rationale: e.target.value })
                  }
                  placeholder="Why this matters (optional)"
                  type="text"
                  value={q.rationale ?? ""}
                />
              </div>
              <button
                aria-label="Remove sub-question"
                className="mt-1 inline-flex size-6 items-center justify-center rounded-md text-muted-foreground/60 transition-colors hover:bg-muted hover:text-foreground"
                onClick={() => removeSubQ(idx)}
                type="button"
              >
                <XIcon className="size-3.5" />
              </button>
            </li>
          ))}
        </ol>
        <button
          className="mt-1 inline-flex items-center gap-1.5 rounded-md px-1.5 py-1 text-muted-foreground text-xs transition-colors hover:text-foreground"
          onClick={addSubQ}
          type="button"
        >
          <PlusIcon className="size-3" />
          Add sub-question
        </button>
      </PlanSection>

      <PlanSection title="Report outline">
        <ol className="space-y-2">
          {draftOutline.map((sec, idx) => (
            <li
              className="grid grid-cols-[18px_1fr_24px] items-start gap-2"
              key={sec.id}
            >
              <span className="pt-1.5 font-medium text-muted-foreground text-xs">
                {idx + 1}.
              </span>
              <div className="space-y-1">
                <input
                  className="w-full rounded-md border border-border/60 bg-background px-2 py-1.5 font-medium text-sm focus-visible:border-primary focus-visible:outline-none"
                  onChange={(e) => updateSection(idx, { title: e.target.value })}
                  placeholder="Section title"
                  type="text"
                  value={sec.title}
                />
                <input
                  className="w-full rounded-md border border-transparent bg-transparent px-2 py-1 text-muted-foreground text-xs focus-visible:border-border focus-visible:bg-background focus-visible:outline-none"
                  onChange={(e) =>
                    updateSection(idx, { description: e.target.value })
                  }
                  placeholder="What goes in this section (optional)"
                  type="text"
                  value={sec.description ?? ""}
                />
              </div>
              <button
                aria-label="Remove section"
                className="mt-1 inline-flex size-6 items-center justify-center rounded-md text-muted-foreground/60 transition-colors hover:bg-muted hover:text-foreground"
                onClick={() => removeSection(idx)}
                type="button"
              >
                <XIcon className="size-3.5" />
              </button>
            </li>
          ))}
        </ol>
        <button
          className="mt-1 inline-flex items-center gap-1.5 rounded-md px-1.5 py-1 text-muted-foreground text-xs transition-colors hover:text-foreground"
          onClick={addSection}
          type="button"
        >
          <PlusIcon className="size-3" />
          Add section
        </button>
      </PlanSection>

      <div className="mt-5 flex items-center justify-between">
        <p className="text-muted-foreground text-[11px]">
          {dirty ? "Edits will be saved on approval." : "Plan unchanged."}
        </p>
        <Button disabled={approving} onClick={handleApprove} size="sm">
          {approving ? "Approving…" : "Approve & start research"}
        </Button>
      </div>
    </div>
  );
}

function PlanSection({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="mt-4 space-y-2">
      <h3 className="font-medium text-muted-foreground text-xs uppercase tracking-wide">
        {title}
      </h3>
      {children}
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

function ReportPlaceholder({
  run,
  subagentCount,
}: {
  run: ResearchRun;
  subagentCount: number;
}) {
  if (subagentCount > 0 && run.status !== "done") {
    return null;
  }
  return (
    <div className="mt-6 rounded-lg border border-border border-dashed p-8 text-center">
      <p className="font-medium text-sm">Report will appear here</p>
      <p className="mt-1 text-muted-foreground text-xs">
        Status: {STATUS_LABELS[run.status] ?? run.status}. The report is
        written once research is complete.
      </p>
    </div>
  );
}

function SubagentList({
  status,
  subagents,
}: {
  status: ResearchRun["status"];
  subagents: SubagentLive[];
}) {
  // Don't show during early phases — the plan card owns that screen,
  // and the empty list would be visual noise.
  if (status === "queued" || status === "scoping" || status === "planning") {
    return null;
  }
  if (subagents.length === 0) {
    if (status === "researching" || status === "writing") {
      return (
        <div className="mt-6 rounded-lg border border-border border-dashed p-6 text-center text-muted-foreground text-xs">
          Waiting for sub-agents to start…
        </div>
      );
    }
    return null;
  }
  const completed = subagents.filter((s) => s.status === "done").length;
  return (
    <section className="mt-6 space-y-3">
      <div className="flex items-baseline justify-between">
        <h2 className="font-semibold text-sm">
          Sub-agents{" "}
          <span className="font-normal text-muted-foreground">
            · {completed}/{subagents.length} complete
          </span>
        </h2>
      </div>
      <ul className="space-y-3">
        {subagents.map((sa) => (
          <li key={sa.id}>
            <SubagentCard subagent={sa} />
          </li>
        ))}
      </ul>
    </section>
  );
}

const ACTION_LABELS: Record<string, string> = {
  search: "Searching the web",
  scrape: "Reading source",
  retrieve: "Pulling context",
  error: "Error",
};

function SubagentCard({ subagent }: { subagent: SubagentLive }) {
  const [open, setOpen] = useState(false);
  const isRunning = subagent.status === "running";
  const isFailed = subagent.status === "failed";
  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            {isRunning && (
              <Loader2Icon className="size-3.5 shrink-0 animate-spin text-primary" />
            )}
            <p className="line-clamp-2 font-medium text-sm leading-snug">
              {subagent.subQuestion}
            </p>
          </div>
          {(subagent.model || subagent.stub) && (
            <p className="mt-1 text-[11px] text-muted-foreground">
              {subagent.stub
                ? "stub sub-agent (LLM disabled)"
                : (subagent.model ?? "")}
            </p>
          )}
        </div>
        <SubagentBadge
          isFailed={isFailed}
          isRunning={isRunning}
          sourceCount={subagent.sourceCount}
        />
      </div>

      {isRunning && subagent.latestAction && (
        <p className="mt-3 line-clamp-2 text-muted-foreground text-xs">
          <span className="font-medium text-foreground/80">
            {ACTION_LABELS[subagent.latestAction] ?? subagent.latestAction}
          </span>
          {subagent.latestActionDetail ? `: ${subagent.latestActionDetail}` : ""}
        </p>
      )}

      {subagent.findingMd && (
        <div className="mt-3">
          <button
            className="text-muted-foreground text-xs underline-offset-2 hover:text-foreground hover:underline"
            onClick={() => setOpen((v) => !v)}
            type="button"
          >
            {open ? "Hide finding" : "Show finding"}
          </button>
          {open && (
            <pre className="mt-2 whitespace-pre-wrap rounded-md border border-border/70 bg-muted/30 p-3 font-sans text-xs leading-relaxed">
              {subagent.findingMd}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}

function SubagentBadge({
  isRunning,
  isFailed,
  sourceCount,
}: {
  isRunning: boolean;
  isFailed: boolean;
  sourceCount: number | null;
}) {
  if (isFailed) {
    return (
      <span className="shrink-0 rounded-full bg-destructive/10 px-2 py-0.5 font-medium text-[11px] text-destructive">
        failed
      </span>
    );
  }
  if (isRunning) {
    return (
      <span className="shrink-0 rounded-full bg-primary/10 px-2 py-0.5 font-medium text-[11px] text-primary">
        running
      </span>
    );
  }
  return (
    <span className="shrink-0 rounded-full bg-emerald-500/10 px-2 py-0.5 font-medium text-[11px] text-emerald-700 dark:text-emerald-400">
      done{sourceCount ? ` · ${sourceCount} source${sourceCount === 1 ? "" : "s"}` : ""}
    </span>
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
