"use client";

import { AnimatePresence, motion } from "framer-motion";
import {
  AlertTriangleIcon,
  CopyIcon,
  DownloadIcon,
  ExternalLinkIcon,
  Loader2Icon,
  MaximizeIcon,
  PlusIcon,
  TelescopeIcon,
  XIcon,
} from "lucide-react";
import {
  type Dispatch,
  type SetStateAction,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { toast } from "sonner";
import { ResearchReportRenderer } from "@/components/research/report-renderer";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { cn } from "@/lib/utils";
import type {
  ResearchOutlineSection,
  ResearchPlan,
  ResearchReport,
  ResearchRun,
  ResearchSource,
  ResearchSubagent,
  ResearchSubQuestion,
} from "@/lib/db/schema";

// Animation language matches DeepSearchToolGroup: 0.2s with the same
// out-cubic easing curve, height + opacity + small Y offset on
// expand/collapse, fade + scale-in for newly-mounted cards.
const EASE: [number, number, number, number] = [0.22, 1, 0.36, 1];
const FADE_UP = {
  initial: { opacity: 0, y: 4 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -4 },
  transition: { duration: 0.2, ease: EASE },
};
const COLLAPSE = {
  initial: { height: 0, opacity: 0, y: -4 },
  animate: { height: "auto" as const, opacity: 1, y: 0 },
  exit: { height: 0, opacity: 0, y: -4 },
  transition: { duration: 0.2, ease: EASE },
};

// In-chat artifact for a deep-research run.
//
// Rendered when the chat message contains a ``data-research`` part —
// the chat thread becomes the home for research, the way OpenAI does
// it. Compact while running, full report inline once done. Clicking
// "Show progress" opens a side Sheet with the live timeline + plan +
// sub-agent details for users who want to watch the work in flight.
//
// Self-contained: takes only ``runId`` + ``query`` from the message
// part and pulls everything else (run row, plan, sub-agents, sources,
// report, events) directly from the API. That means the parent
// message component doesn't have to know anything about research.

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

type StreamEvent = {
  seq: number;
  type: string;
  ts: string;
  payload: Record<string, unknown>;
};

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

type WorkerStatus = {
  configured: boolean;
  running: boolean;
  error: string | null;
  error_at: string | null;
  started_at: string | null;
  stopped_at: string | null;
};

function subagentFromRow(row: ResearchSubagent): SubagentLive {
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

export function ResearchArtifactCard({
  runId,
  query,
}: {
  runId: string;
  query: string;
}) {
  const [run, setRun] = useState<ResearchRun | null>(null);
  const [plan, setPlan] = useState<ResearchPlan | null>(null);
  const [subagents, setSubagents] = useState<Record<string, SubagentLive>>({});
  const [sources, setSources] = useState<ResearchSource[]>([]);
  const [report, setReport] = useState<ResearchReport | null>(null);
  const [events, setEvents] = useState<StreamEvent[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [sheetOpen, setSheetOpen] = useState(false);
  const [workerStatus, setWorkerStatus] = useState<WorkerStatus | null>(null);
  const initialFetchedRef = useRef(false);

  // ── Initial snapshot fetch ───────────────────────────────────────────
  useEffect(() => {
    if (initialFetchedRef.current) return;
    initialFetchedRef.current = true;
    let cancelled = false;
    void (async () => {
      try {
        const res = await fetch(`/api/research/${runId}`, {
          cache: "no-store",
        });
        if (!res.ok) {
          if (!cancelled) {
            setLoadError(
              res.status === 404
                ? "This research run was deleted."
                : `Couldn't load research run (${res.status}).`
            );
          }
          return;
        }
        const snap = (await res.json()) as {
          run: ResearchRun;
          plan: ResearchPlan | null;
          subagents: ResearchSubagent[];
          sources: ResearchSource[];
          report: ResearchReport | null;
        };
        if (cancelled) return;
        setRun(snap.run);
        setPlan(snap.plan);
        setReport(snap.report);
        setSources(snap.sources ?? []);
        setSubagents(
          Object.fromEntries(
            (snap.subagents ?? []).map((sa) => [sa.id, subagentFromRow(sa)])
          )
        );
      } catch (err) {
        if (!cancelled) {
          setLoadError(
            err instanceof Error ? err.message : "Failed to load research run"
          );
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [runId]);

  // ── Worker-offline probe ─────────────────────────────────────────────
  // If the run sits in 'queued' for more than 12 seconds with zero
  // events, ping the backend's worker-status endpoint. If the worker
  // crashed (e.g. DATABASE_URL missing on Railway) we surface that
  // state in the card so the user gets an honest signal instead of
  // the optimistic "Waiting for the worker…" forever.
  useEffect(() => {
    if (!run) return;
    if (run.status !== "queued") return;
    if (events.length > 0) return;
    const timer = setTimeout(() => {
      void fetch("/api/research/worker-status", { cache: "no-store" })
        .then((r) => r.json())
        .then((status) => setWorkerStatus(status))
        .catch(() => undefined);
    }, 12_000);
    return () => clearTimeout(timer);
  }, [run, events.length]);

  // ── Live SSE subscription ────────────────────────────────────────────
  const isTerminal = run ? TERMINAL_STATUSES.has(run.status) : false;
  useEffect(() => {
    if (!run) return;
    if (isTerminal) return;
    const es = new EventSource(`/api/research/${run.id}/events`);
    let cancelled = false;

    const refreshSnapshot = () => {
      void fetch(`/api/research/${run.id}`, { cache: "no-store" })
        .then((r) => r.json())
        .then((snap) => {
          if (cancelled) return;
          if (snap.run) setRun(snap.run);
          if (snap.plan) setPlan(snap.plan);
          if (snap.report) setReport(snap.report);
          if (Array.isArray(snap.sources)) setSources(snap.sources);
          if (Array.isArray(snap.subagents)) {
            setSubagents(
              Object.fromEntries(
                (snap.subagents as ResearchSubagent[]).map((sa) => [
                  sa.id,
                  subagentFromRow(sa),
                ])
              )
            );
          }
        })
        .catch(() => undefined);
    };

    const onEvent = (e: MessageEvent) => {
      if (cancelled) return;
      try {
        const data = JSON.parse(e.data) as StreamEvent | { status?: string };
        if ("seq" in data) {
          setEvents((prev) =>
            prev.some((e2) => e2.seq === data.seq)
              ? prev
              : [...prev, data].sort((a, b) => a.seq - b.seq)
          );
          handleEvent(data);
        } else if (typeof data.status === "string") {
          setRun((r) =>
            r ? { ...r, status: data.status as ResearchRun["status"] } : r
          );
        }
      } catch {
        // ignore unparseable events
      }
    };

    const handleEvent = (evt: StreamEvent) => {
      switch (evt.type) {
        case "status_changed": {
          const status = evt.payload.status;
          if (typeof status === "string") {
            setRun((r) =>
              r ? { ...r, status: status as ResearchRun["status"] } : r
            );
          }
          break;
        }
        case "plan_proposed":
        case "plan_approved":
          refreshSnapshot();
          break;
        case "subagent_started": {
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
                latestAction:
                  typeof action === "string" ? action : cur.latestAction,
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
          if (typeof id !== "string") break;
          setSubagents((prev) => {
            const cur = prev[id];
            if (!cur) return prev;
            return { ...prev, [id]: { ...cur, status: "failed" } };
          });
          break;
        }
        case "sources_deduped":
        case "report_written":
        case "run_completed":
          refreshSnapshot();
          break;
        default:
          break;
      }
    };

    es.addEventListener("message", onEvent);
    const liveTypes = [
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
      "writer_started",
      "sources_deduped",
      "report_written",
      "run_completed",
      "run_failed",
      "run_cancelled",
    ];
    for (const t of liveTypes) es.addEventListener(t, onEvent);

    return () => {
      cancelled = true;
      es.close();
    };
  }, [run, isTerminal]);

  // ── Derived ──────────────────────────────────────────────────────────
  const subagentList = useMemo<SubagentLive[]>(() => {
    const subQs: ResearchSubQuestion[] = Array.isArray(plan?.subQuestions)
      ? (plan.subQuestions as ResearchSubQuestion[])
      : [];
    const planOrder = new Map<string, number>(
      subQs.map((q, i) => [q.id, i])
    );
    return Object.values(subagents).sort((a, b) => {
      const ai = planOrder.get(a.id);
      const bi = planOrder.get(b.id);
      if (ai !== undefined && bi !== undefined) return ai - bi;
      if (ai !== undefined) return -1;
      if (bi !== undefined) return 1;
      return a.id.localeCompare(b.id);
    });
  }, [subagents, plan?.subQuestions]);

  const completedSubagents = subagentList.filter((s) => s.status === "done").length;
  const totalSubagents = subagentList.length || 0;
  const sourceCount = sources.length;
  const status = run?.status ?? "queued";
  const statusLabel = STATUS_LABELS[status] ?? status;
  const isDone = status === "done";
  const isFailed = status === "failed";
  const isCancelled = status === "cancelled";

  // ── Actions ──────────────────────────────────────────────────────────
  const onApprovePlan = useCallback(
    async (edits?: {
      subQuestions: ResearchSubQuestion[];
      outline: ResearchOutlineSection[];
    }) => {
      if (!plan) return;
      try {
        const res = await fetch(`/api/research/${runId}/plan`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(edits ?? {}),
        });
        if (!res.ok) {
          toast.error(
            (await res.text()) || "Couldn't approve plan, try again."
          );
        }
      } catch (err) {
        toast.error(
          err instanceof Error ? err.message : "Network error approving plan"
        );
      }
    },
    [plan, runId]
  );

  const onCancel = useCallback(async () => {
    if (!run || TERMINAL_STATUSES.has(run.status)) return;
    try {
      const res = await fetch(`/api/research/${run.id}/cancel`, {
        method: "POST",
      });
      if (!res.ok) {
        toast.error((await res.text()) || "Couldn't cancel.");
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Cancel failed");
    }
  }, [run]);

  const onCopy = useCallback(() => {
    if (!report) return;
    void navigator.clipboard
      .writeText(report.markdown)
      .then(() => toast.success("Report markdown copied"))
      .catch(() => toast.error("Couldn't copy to clipboard"));
  }, [report]);

  const onDownload = useCallback(() => {
    if (!report) return;
    const blob = new Blob([report.markdown], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${slugForFilename(query)}.md`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }, [report, query]);

  // ── Render ───────────────────────────────────────────────────────────
  if (loadError) {
    return (
      <motion.div
        animate={FADE_UP.animate}
        className="rounded-lg border border-destructive/30 bg-destructive/5 p-3 text-destructive text-sm"
        initial={FADE_UP.initial}
        transition={FADE_UP.transition}
      >
        <p className="font-medium">Couldn&apos;t load research run</p>
        <p className="text-xs">{loadError}</p>
      </motion.div>
    );
  }

  // While the snapshot is loading, show a stub so the chat doesn't reflow.
  if (!run) {
    return (
      <motion.div
        animate={FADE_UP.animate}
        className="flex items-center gap-2 rounded-lg border border-border bg-card p-3 text-muted-foreground text-sm"
        initial={FADE_UP.initial}
        transition={FADE_UP.transition}
      >
        <Loader2Icon className="size-3.5 animate-spin" />
        Loading research run…
      </motion.div>
    );
  }

  // Done state: render the report inline. The artifact becomes the
  // report once research is complete — chat continues underneath.
  if (isDone && report) {
    return (
      <DoneState
        events={events}
        onCopy={onCopy}
        onDownload={onDownload}
        onOpenSheet={() => setSheetOpen(true)}
        query={query}
        report={report}
        sheetOpen={sheetOpen}
        setSheetOpen={setSheetOpen}
        sources={sources}
        subagentList={subagentList}
        plan={plan}
      />
    );
  }

  // Running / blocked / terminal-non-done states: compact card with a
  // status header, a brief progress line, an inline plan-approval card
  // (only when status='awaiting_approval'), and a "Show progress"
  // button that opens the full sheet.
  const workerOffline =
    workerStatus !== null &&
    workerStatus.configured === true &&
    workerStatus.running === false;

  return (
    <>
      <motion.div
        animate={{ opacity: 1, y: 0, scale: 1 }}
        className="rounded-lg border border-border bg-card overflow-hidden"
        initial={{ opacity: 0, y: 6, scale: 0.985 }}
        transition={{ duration: 0.25, ease: EASE }}
      >
        <div className="flex items-start gap-3 p-4">
          <div
            className={cn(
              "flex size-8 shrink-0 items-center justify-center rounded-md bg-primary/10 text-primary",
              !isFailed && !isCancelled && "animate-pulse"
            )}
          >
            <TelescopeIcon className="size-4" />
          </div>
          <div className="min-w-0 flex-1">
            <div className="flex items-center justify-between gap-2">
              <p className="font-medium text-sm">Research</p>
              <RunningBadge
                status={status}
                isFailed={isFailed}
                isCancelled={isCancelled}
                statusLabel={statusLabel}
              />
            </div>
            <p className="mt-1 line-clamp-2 text-muted-foreground text-xs leading-snug">
              {query}
            </p>
            <AnimatePresence initial={false} mode="wait">
              <motion.div
                animate={FADE_UP.animate}
                exit={FADE_UP.exit}
                initial={FADE_UP.initial}
                key={status}
                transition={FADE_UP.transition}
              >
                <ProgressLine
                  completedSubagents={completedSubagents}
                  sourceCount={sourceCount}
                  status={status}
                  totalSubagents={totalSubagents}
                />
              </motion.div>
            </AnimatePresence>

            {workerOffline && (
              <WorkerOfflineNotice status={workerStatus} />
            )}
          </div>
        </div>

        <AnimatePresence initial={false}>
          {plan && status === "awaiting_approval" && (
            <motion.div
              animate={COLLAPSE.animate}
              exit={COLLAPSE.exit}
              initial={COLLAPSE.initial}
              key={`plan-${plan.runId}-${plan.version}`}
              style={{ overflow: "hidden" }}
              transition={COLLAPSE.transition}
            >
              <PlanApprovalCard
                events={events}
                initialOutline={
                  (Array.isArray(plan.outline)
                    ? (plan.outline as ResearchOutlineSection[])
                    : []) ?? []
                }
                initialSubQuestions={
                  (Array.isArray(plan.subQuestions)
                    ? (plan.subQuestions as ResearchSubQuestion[])
                    : []) ?? []
                }
                onApprove={onApprovePlan}
                plan={plan}
              />
            </motion.div>
          )}
        </AnimatePresence>

        <AnimatePresence initial={false}>
          {(status === "researching" || status === "writing") &&
            (subagentList.length > 0 || events.length > 0) && (
              <motion.div
                animate={COLLAPSE.animate}
                exit={COLLAPSE.exit}
                initial={COLLAPSE.initial}
                style={{ overflow: "hidden" }}
                transition={COLLAPSE.transition}
              >
                <InlineLiveTimeline
                  events={events}
                  subagentList={subagentList}
                />
              </motion.div>
            )}
        </AnimatePresence>

        <div className="flex items-center justify-between border-border/60 border-t bg-muted/30 px-3 py-2">
          <Button
            className="h-7 gap-1.5 px-2 text-[12px]"
            onClick={() => setSheetOpen(true)}
            size="sm"
            variant="ghost"
          >
            <MaximizeIcon className="size-3" />
            Show progress
          </Button>
          {!isFailed && !isCancelled && (
            <Button
              className="h-7 px-2 text-[12px]"
              onClick={onCancel}
              size="sm"
              variant="ghost"
            >
              Cancel
            </Button>
          )}
        </div>
      </motion.div>

      <ProgressSheet
        events={events}
        onApprovePlan={onApprovePlan}
        onCancel={onCancel}
        open={sheetOpen}
        plan={plan}
        run={run}
        setOpen={setSheetOpen}
        sources={sources}
        subagentList={subagentList}
      />
    </>
  );
}

// ── Worker-offline notice ───────────────────────────────────────────────


function WorkerOfflineNotice({ status }: { status: WorkerStatus }) {
  const detail = status.error?.toLowerCase() ?? "";
  const hint = detail.includes("database_url")
    ? "Set DATABASE_URL in your Railway env vars and redeploy."
    : detail.includes("cannot reach backend")
    ? "Backend service is unreachable from the frontend."
    : null;
  return (
    <motion.div
      animate={FADE_UP.animate}
      className="mt-3 flex items-start gap-2 rounded-md border border-amber-500/30 bg-amber-500/5 px-2.5 py-2 text-amber-700 text-xs dark:text-amber-400"
      initial={FADE_UP.initial}
      transition={FADE_UP.transition}
    >
      <AlertTriangleIcon className="size-3.5 shrink-0" />
      <div className="min-w-0">
        <p className="font-medium">Research worker is offline</p>
        {status.error ? (
          <p className="mt-0.5 break-all opacity-90">{status.error}</p>
        ) : null}
        {hint ? <p className="mt-1 opacity-80">{hint}</p> : null}
      </div>
    </motion.div>
  );
}

// ── Inline live timeline (last few events with fade-in) ─────────────────


function InlineLiveTimeline({
  events,
  subagentList,
}: {
  events: StreamEvent[];
  subagentList: SubagentLive[];
}) {
  // Derive a friendlier "current activity" line by combining
  // sub-agent latest actions and the most recent event type. This
  // gives the user something to read while research runs without
  // them having to open the sheet.
  const recentEvents = useMemo(() => events.slice(-5), [events]);
  const runningSubagents = subagentList.filter((s) => s.status === "running");

  return (
    <div className="border-border/60 border-t bg-muted/15 px-4 py-3">
      <div className="flex items-center gap-2">
        <PulsingDot />
        <span className="font-medium text-[11px] text-muted-foreground uppercase tracking-wide">
          Live progress
        </span>
      </div>

      {runningSubagents.length > 0 && (
        <div className="mt-2 space-y-1.5">
          {runningSubagents.slice(0, 3).map((sa) => (
            <motion.div
              animate={{ opacity: 1, x: 0 }}
              className="flex items-start gap-2 text-xs"
              initial={{ opacity: 0, x: -6 }}
              key={sa.id}
              transition={{ duration: 0.2, ease: EASE }}
            >
              <Loader2Icon className="mt-0.5 size-3 shrink-0 animate-spin text-primary" />
              <div className="min-w-0">
                <span className="line-clamp-1 font-medium text-foreground/90">
                  {sa.subQuestion}
                </span>
                {sa.latestAction && (
                  <span className="line-clamp-1 text-[11px] text-muted-foreground">
                    {sa.latestAction}
                    {sa.latestActionDetail ? `: ${sa.latestActionDetail}` : ""}
                  </span>
                )}
              </div>
            </motion.div>
          ))}
        </div>
      )}

      {recentEvents.length > 0 && (
        <ol className="mt-2 space-y-0.5 border-border/40 border-l pl-2.5">
          <AnimatePresence initial={false}>
            {recentEvents.map((evt) => (
              <motion.li
                animate={{ opacity: 1, x: 0 }}
                className="flex items-center gap-2 text-[11px] text-muted-foreground"
                exit={{ opacity: 0 }}
                initial={{ opacity: 0, x: -4 }}
                key={evt.seq}
                transition={{ duration: 0.18, ease: EASE }}
              >
                <span className="size-1 shrink-0 rounded-full bg-muted-foreground/40" />
                <span className="truncate">{prettyType(evt.type)}</span>
              </motion.li>
            ))}
          </AnimatePresence>
        </ol>
      )}
    </div>
  );
}

function PulsingDot() {
  return (
    <span className="relative flex size-2 items-center justify-center">
      <span className="absolute inline-flex size-full animate-ping rounded-full bg-primary/60 opacity-60" />
      <span className="relative inline-flex size-1.5 rounded-full bg-primary" />
    </span>
  );
}

// ── Done state (inline report) ──────────────────────────────────────────


function DoneState({
  events,
  onCopy,
  onDownload,
  onOpenSheet,
  query,
  report,
  sheetOpen,
  setSheetOpen,
  sources,
  subagentList,
  plan,
}: {
  events: StreamEvent[];
  onCopy: () => void;
  onDownload: () => void;
  onOpenSheet: () => void;
  query: string;
  report: ResearchReport;
  sheetOpen: boolean;
  setSheetOpen: Dispatch<SetStateAction<boolean>>;
  sources: ResearchSource[];
  subagentList: SubagentLive[];
  plan: ResearchPlan | null;
}) {
  const writerEvent = useMemo(
    () => [...events].reverse().find((e) => e.type === "report_written"),
    [events]
  );
  const usedStub = Boolean(writerEvent?.payload?.stub);
  const writerModel =
    typeof writerEvent?.payload?.model === "string"
      ? (writerEvent.payload.model as string)
      : null;

  return (
    <>
      <div className="rounded-lg border border-border bg-card">
        <header className="flex items-center justify-between gap-3 border-border/60 border-b px-4 py-3">
          <div className="flex items-center gap-2 min-w-0">
            <div className="flex size-7 shrink-0 items-center justify-center rounded-md bg-emerald-500/10 text-emerald-600 dark:text-emerald-400">
              <TelescopeIcon className="size-3.5" />
            </div>
            <span className="font-medium text-sm">Research report</span>
            {writerModel && !usedStub && (
              <span className="rounded-full bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground">
                {writerModel}
              </span>
            )}
            {usedStub && (
              <span className="rounded-full bg-amber-500/10 px-1.5 py-0.5 text-[10px] text-amber-700 dark:text-amber-400">
                stub report
              </span>
            )}
          </div>
          <div className="flex items-center gap-1">
            <Button
              className="h-7 gap-1.5 px-2 text-[11px]"
              onClick={onCopy}
              size="sm"
              variant="ghost"
            >
              <CopyIcon className="size-3" />
              Copy
            </Button>
            <Button
              className="h-7 gap-1.5 px-2 text-[11px]"
              onClick={onDownload}
              size="sm"
              variant="ghost"
            >
              <DownloadIcon className="size-3" />
              Download
            </Button>
            <Button
              className="h-7 gap-1.5 px-2 text-[11px]"
              onClick={onOpenSheet}
              size="sm"
              variant="ghost"
            >
              <MaximizeIcon className="size-3" />
              Open
            </Button>
          </div>
        </header>

        <div className="p-5">
          <ResearchReportRenderer citations={sources} markdown={report.markdown} />
        </div>

        {sources.length > 0 && (
          <div className="border-border/60 border-t px-5 py-4">
            <SourcesList sources={sources} />
          </div>
        )}
      </div>

      <ProgressSheet
        events={events}
        onApprovePlan={async () => undefined}
        onCancel={async () => undefined}
        open={sheetOpen}
        plan={plan}
        run={null}
        setOpen={setSheetOpen}
        sources={sources}
        subagentList={subagentList}
        // Pass the report so the sheet can also show it.
        report={report}
        query={query}
      />
    </>
  );
}

// ── Compact bits ────────────────────────────────────────────────────────


function RunningBadge({
  status,
  isFailed,
  isCancelled,
  statusLabel,
}: {
  status: string;
  isFailed: boolean;
  isCancelled: boolean;
  statusLabel: string;
}) {
  if (isFailed) {
    return (
      <span className="shrink-0 rounded-full bg-destructive/10 px-2 py-0.5 font-medium text-[11px] text-destructive">
        Failed
      </span>
    );
  }
  if (isCancelled) {
    return (
      <span className="shrink-0 rounded-full bg-muted px-2 py-0.5 font-medium text-[11px] text-muted-foreground">
        Cancelled
      </span>
    );
  }
  return (
    <span className="inline-flex shrink-0 items-center gap-1.5 rounded-full bg-primary/10 px-2 py-0.5 font-medium text-[11px] text-primary">
      <Loader2Icon className="size-3 animate-spin" />
      {statusLabel}
    </span>
  );
}

function ProgressLine({
  status,
  completedSubagents,
  totalSubagents,
  sourceCount,
}: {
  status: string;
  completedSubagents: number;
  totalSubagents: number;
  sourceCount: number;
}) {
  if (status === "queued") {
    return (
      <p className="mt-2 text-muted-foreground text-[11px]">
        Waiting for the worker to pick up this run…
      </p>
    );
  }
  if (status === "scoping" || status === "planning") {
    return (
      <p className="mt-2 text-muted-foreground text-[11px]">
        Drafting a research plan from your query.
      </p>
    );
  }
  if (status === "awaiting_approval") {
    return (
      <p className="mt-2 text-muted-foreground text-[11px]">
        Plan ready. Approve below to start research.
      </p>
    );
  }
  if (status === "researching") {
    if (totalSubagents > 0) {
      return (
        <p className="mt-2 text-muted-foreground text-[11px]">
          {completedSubagents} of {totalSubagents} sub-agents complete
          {sourceCount > 0 ? ` · ${sourceCount} source${sourceCount === 1 ? "" : "s"}` : ""}
        </p>
      );
    }
    return (
      <p className="mt-2 text-muted-foreground text-[11px]">
        Spinning up sub-agents…
      </p>
    );
  }
  if (status === "writing") {
    return (
      <p className="mt-2 text-muted-foreground text-[11px]">
        Writing the final report from sub-agent findings…
      </p>
    );
  }
  return null;
}

// ── Plan approval (lifted from ResearchRunView, simplified) ────────────


function PlanApprovalCard({
  plan,
  initialSubQuestions,
  initialOutline,
  events,
  onApprove,
}: {
  plan: ResearchPlan;
  initialSubQuestions: ResearchSubQuestion[];
  initialOutline: ResearchOutlineSection[];
  events: StreamEvent[];
  onApprove: (edits?: {
    subQuestions: ResearchSubQuestion[];
    outline: ResearchOutlineSection[];
  }) => Promise<void> | void;
}) {
  const planProposed = useMemo(
    () => events.find((e) => e.type === "plan_proposed"),
    [events]
  );
  const usedStub = Boolean(planProposed?.payload?.stub);
  const plannerModel =
    typeof planProposed?.payload?.model === "string"
      ? (planProposed.payload.model as string)
      : null;

  const [draftSubQs, setDraftSubQs] =
    useState<ResearchSubQuestion[]>(initialSubQuestions);
  const [draftOutline, setDraftOutline] =
    useState<ResearchOutlineSection[]>(initialOutline);
  const [approving, setApproving] = useState(false);
  const planSig = `${plan.runId}:${plan.version}`;
  const planSigRef = useRef(planSig);
  if (planSigRef.current !== planSig) {
    planSigRef.current = planSig;
    setDraftSubQs(initialSubQuestions);
    setDraftOutline(initialOutline);
  }

  const dirty = useMemo(() => {
    if (
      draftSubQs.length !== initialSubQuestions.length ||
      draftOutline.length !== initialOutline.length
    )
      return true;
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

  const handleApprove = async () => {
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
      toast.error("Add at least one sub-question.");
      return;
    }
    if (cleanOutline.length < 1) {
      toast.error("Keep at least one outline section.");
      return;
    }
    setApproving(true);
    try {
      await onApprove(
        dirty ? { subQuestions: cleanSubQs, outline: cleanOutline } : undefined
      );
    } finally {
      setApproving(false);
    }
  };

  return (
    <div className="border-border/60 border-t bg-muted/20 p-4">
      <div className="flex items-center gap-2">
        <span className="font-medium text-sm">Plan</span>
        {plannerModel && !usedStub && (
          <span className="rounded-full bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground">
            {plannerModel}
          </span>
        )}
        {usedStub && (
          <span className="rounded-full bg-amber-500/10 px-1.5 py-0.5 text-[10px] text-amber-700 dark:text-amber-400">
            stub plan
          </span>
        )}
      </div>
      {plan.briefMd && (
        <p className="mt-2 whitespace-pre-wrap text-foreground/85 text-xs leading-relaxed">
          {plan.briefMd}
        </p>
      )}
      <div className="mt-3 space-y-2">
        <p className="font-medium text-muted-foreground text-[10px] uppercase tracking-wide">
          Sub-questions
        </p>
        <ol className="space-y-1.5">
          {draftSubQs.map((q, idx) => (
            <li
              className="grid grid-cols-[16px_1fr_22px] items-start gap-2"
              key={q.id}
            >
              <span className="pt-1 font-medium text-muted-foreground text-[11px]">
                {idx + 1}.
              </span>
              <input
                className="w-full rounded-md border border-border/60 bg-background px-2 py-1 text-xs focus-visible:border-primary focus-visible:outline-none"
                onChange={(e) =>
                  setDraftSubQs((prev) =>
                    prev.map((p, i) =>
                      i === idx ? { ...p, question: e.target.value } : p
                    )
                  )
                }
                placeholder="Sub-question"
                type="text"
                value={q.question}
              />
              <button
                aria-label="Remove sub-question"
                className="mt-0.5 inline-flex size-5 items-center justify-center rounded text-muted-foreground/60 transition-colors hover:bg-muted hover:text-foreground"
                onClick={() =>
                  setDraftSubQs((prev) => prev.filter((_, i) => i !== idx))
                }
                type="button"
              >
                <XIcon className="size-3" />
              </button>
            </li>
          ))}
        </ol>
        <button
          className="inline-flex items-center gap-1 px-1 text-muted-foreground text-[11px] hover:text-foreground"
          onClick={() => {
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
          }}
          type="button"
        >
          <PlusIcon className="size-3" />
          Add
        </button>
      </div>
      <div className="mt-4 flex justify-end">
        <Button disabled={approving} onClick={handleApprove} size="sm">
          {approving ? "Approving…" : "Approve & start research"}
        </Button>
      </div>
    </div>
  );
}

// ── Sources list (used by both done state and sheet) ───────────────────


function SourcesList({ sources }: { sources: ResearchSource[] }) {
  return (
    <div>
      <h3 className="font-semibold text-muted-foreground text-xs uppercase tracking-wide">
        Sources ({sources.length})
      </h3>
      <ol className="mt-2 space-y-1">
        {sources.map((s) => {
          const domain = domainFor(s.url);
          const label = s.title ?? domain;
          return (
            <li
              className="flex items-start gap-2 rounded-md p-1.5 text-sm transition-colors hover:bg-muted/40"
              key={s.citationNum}
            >
              <span className="shrink-0 rounded bg-muted px-1.5 py-0.5 font-mono text-[11px] text-muted-foreground">
                [{s.citationNum}]
              </span>
              <a
                className="group min-w-0 flex-1"
                href={s.url}
                rel="noreferrer"
                target="_blank"
              >
                <span className="line-clamp-1 font-medium text-foreground group-hover:underline">
                  {label}
                </span>
                <span className="line-clamp-1 text-[11px] text-muted-foreground">
                  {domain}
                  <ExternalLinkIcon className="ml-1 inline size-3" />
                </span>
              </a>
            </li>
          );
        })}
      </ol>
    </div>
  );
}

// ── Side sheet (full progress view) ────────────────────────────────────


function ProgressSheet({
  open,
  setOpen,
  run,
  plan,
  subagentList,
  sources,
  events,
  onApprovePlan,
  onCancel,
  report,
  query,
}: {
  open: boolean;
  setOpen: Dispatch<SetStateAction<boolean>>;
  run: ResearchRun | null;
  plan: ResearchPlan | null;
  subagentList: SubagentLive[];
  sources: ResearchSource[];
  events: StreamEvent[];
  onApprovePlan: (edits?: {
    subQuestions: ResearchSubQuestion[];
    outline: ResearchOutlineSection[];
  }) => Promise<void> | void;
  onCancel: () => Promise<void> | void;
  report?: ResearchReport;
  query?: string;
}) {
  const status = run?.status ?? (report ? "done" : "queued");
  return (
    <Sheet onOpenChange={setOpen} open={open}>
      <SheetContent
        className="w-full max-w-2xl overflow-y-auto sm:max-w-2xl"
        side="right"
      >
        <SheetHeader className="border-border border-b">
          <SheetTitle className="text-base">
            Research progress
          </SheetTitle>
          <p className="text-muted-foreground text-xs">
            Status: {STATUS_LABELS[status] ?? status}
          </p>
        </SheetHeader>

        <div className="space-y-5 px-4 py-4">
          {plan && (
            <section>
              <h3 className="font-semibold text-muted-foreground text-xs uppercase tracking-wide">
                Plan (v{plan.version})
              </h3>
              {plan.briefMd && (
                <p className="mt-2 whitespace-pre-wrap text-foreground/85 text-sm">
                  {plan.briefMd}
                </p>
              )}
              {Array.isArray(plan.subQuestions) && plan.subQuestions.length > 0 && (
                <ol className="mt-3 space-y-1 text-sm">
                  {(plan.subQuestions as ResearchSubQuestion[]).map((q, i) => (
                    <li className="flex gap-2" key={q.id}>
                      <span className="shrink-0 text-muted-foreground">
                        {i + 1}.
                      </span>
                      <span>{q.question}</span>
                    </li>
                  ))}
                </ol>
              )}
            </section>
          )}

          {subagentList.length > 0 && (
            <section>
              <h3 className="font-semibold text-muted-foreground text-xs uppercase tracking-wide">
                Sub-agents
              </h3>
              <ul className="mt-2 space-y-2">
                {subagentList.map((sa) => (
                  <li className="rounded-md border border-border bg-card p-3 text-sm" key={sa.id}>
                    <div className="flex items-start justify-between gap-2">
                      <p className="line-clamp-2 font-medium">
                        {sa.subQuestion}
                      </p>
                      <SubagentBadge sa={sa} />
                    </div>
                    {sa.status === "running" && sa.latestAction && (
                      <p className="mt-1 line-clamp-2 text-muted-foreground text-xs">
                        {sa.latestAction}
                        {sa.latestActionDetail ? `: ${sa.latestActionDetail}` : ""}
                      </p>
                    )}
                    {sa.findingMd && (
                      <details className="mt-2">
                        <summary className="cursor-pointer text-muted-foreground text-xs hover:text-foreground">
                          Show finding
                        </summary>
                        <pre className="mt-2 whitespace-pre-wrap rounded border border-border/70 bg-muted/30 p-2 font-sans text-xs">
                          {sa.findingMd}
                        </pre>
                      </details>
                    )}
                  </li>
                ))}
              </ul>
            </section>
          )}

          <section>
            <h3 className="font-semibold text-muted-foreground text-xs uppercase tracking-wide">
              Live timeline
            </h3>
            <Timeline events={events} />
          </section>

          {sources.length > 0 && <SourcesList sources={sources} />}

          {report && (
            <section>
              <h3 className="font-semibold text-muted-foreground text-xs uppercase tracking-wide">
                Report
              </h3>
              <div className="mt-2 rounded-md border border-border bg-card p-4">
                <ResearchReportRenderer
                  citations={sources}
                  markdown={report.markdown}
                />
              </div>
            </section>
          )}
        </div>
      </SheetContent>
    </Sheet>
  );
}

function SubagentBadge({ sa }: { sa: SubagentLive }) {
  if (sa.status === "failed") {
    return (
      <span className="shrink-0 rounded-full bg-destructive/10 px-2 py-0.5 font-medium text-[11px] text-destructive">
        failed
      </span>
    );
  }
  if (sa.status === "running") {
    return (
      <span className="shrink-0 rounded-full bg-primary/10 px-2 py-0.5 font-medium text-[11px] text-primary">
        running
      </span>
    );
  }
  return (
    <span className="shrink-0 rounded-full bg-emerald-500/10 px-2 py-0.5 font-medium text-[11px] text-emerald-700 dark:text-emerald-400">
      done
      {sa.sourceCount ? ` · ${sa.sourceCount}` : ""}
    </span>
  );
}

function Timeline({ events }: { events: StreamEvent[] }) {
  if (events.length === 0) {
    return (
      <p className="mt-1 text-muted-foreground text-xs">
        Waiting for the worker to start…
      </p>
    );
  }
  return (
    <ol className="mt-2 space-y-1">
      {events.map((evt) => (
        <li className="flex flex-col text-xs" key={evt.seq}>
          <span className="font-medium">{prettyType(evt.type)}</span>
          <span className="text-[10px] text-muted-foreground">
            seq {evt.seq}
          </span>
        </li>
      ))}
    </ol>
  );
}

// ── Helpers ────────────────────────────────────────────────────────────


function prettyType(type: string): string {
  return type.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function domainFor(url: string): string {
  try {
    const u = new URL(url);
    return u.hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
}

function slugForFilename(query: string): string {
  const slug = query
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 60);
  return slug || "research-report";
}
