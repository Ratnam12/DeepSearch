"use client";

import { AnimatePresence, motion } from "framer-motion";
import {
  AlertTriangleIcon,
  ArrowUpRightIcon,
  CheckCircle2Icon,
  CopyIcon,
  DownloadIcon,
  ExternalLinkIcon,
  FileTextIcon,
  GlobeIcon,
  Loader2Icon,
  SearchIcon,
  SparklesIcon,
  TelescopeIcon,
  XCircleIcon,
} from "lucide-react";
import { useArtifact } from "@/hooks/use-artifact";
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
  ResearchPlan,
  ResearchReport,
  ResearchRun,
  ResearchSource,
  ResearchSubagent,
  ResearchSubQuestion,
} from "@/lib/db/schema";

// In-chat artifact for a deep-research run.
//
// One compact, fixed-height card in the chat. Hover lifts it; click
// opens a side sheet with the full activity feed + final report.
// All variable-height detail (sub-question rows, source cards, the
// report itself) lives in the sheet so the chat doesn't re-flow as
// the run progresses. The sheet's activity feed is OpenAI Deep
// Research-style — natural-language steps with favicon source
// cards, no agent jargon.

const TERMINAL_STATUSES = new Set(["done", "failed", "cancelled"]);

const EASE: [number, number, number, number] = [0.22, 1, 0.36, 1];

export type StreamEvent = {
  seq: number;
  type: string;
  ts: string;
  payload: Record<string, unknown>;
};

export type SubagentLive = {
  id: string;
  subQuestion: string;
  status: "running" | "done" | "failed";
  findingMd: string | null;
  sourceCount: number | null;
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

export function subagentFromRow(row: ResearchSubagent): SubagentLive {
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
    model: row.model,
    stub: false,
  };
}

// ── Activity-feed types ─────────────────────────────────────────────────

export type SearchHit = {
  title: string;
  url: string;
  host: string;
  snippet: string;
};

export type RetrievedChunk = { host: string; url: string; snippet: string };

export type ActivityStep =
  | {
      kind: "phase";
      ts: string;
      label: string;
      icon: "spark" | "globe" | "file" | "check";
    }
  | { kind: "task"; ts: string; subQuestion: string }
  | {
      kind: "search";
      ts: string;
      subQuestion: string | null;
      query: string;
      results: SearchHit[];
    }
  | {
      kind: "read";
      ts: string;
      subQuestion: string | null;
      url: string;
      host: string;
      ok: boolean;
      chunks: number | null;
    }
  | {
      kind: "retrieve";
      ts: string;
      subQuestion: string | null;
      query: string;
      chunks: RetrievedChunk[];
    };

// ────────────────────────────────────────────────────────────────────────
// Top-level component
// ────────────────────────────────────────────────────────────────────────

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
                : `Couldn't load research (${res.status}).`
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
          events?: StreamEvent[];
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
        // Seed the events list from the snapshot so terminal runs
        // (which never open an SSE connection) still render the full
        // activity history. The SSE branch below dedupes by seq.
        if (Array.isArray(snap.events) && snap.events.length > 0) {
          setEvents(snap.events);
        }
      } catch (err) {
        if (!cancelled) {
          setLoadError(
            err instanceof Error ? err.message : "Failed to load research"
          );
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [runId]);

  // ── Worker-offline probe ─────────────────────────────────────────────
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
        // ignore
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
              model: typeof model === "string" ? model : null,
              stub: Boolean(evt.payload.stub),
            },
          }));
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

  const status = run?.status ?? "queued";
  const isDone = status === "done";
  const isFailed = status === "failed";
  const isCancelled = status === "cancelled";

  const activitySteps = useMemo<ActivityStep[]>(
    () => buildActivitySteps(events, subagentList),
    [events, subagentList]
  );

  const liveLine = useMemo(
    () => deriveLiveLine(status, activitySteps),
    [status, activitySteps]
  );

  const reportPreview = useMemo(
    () => (report ? extractReportPreview(report.markdown) : null),
    [report]
  );

  const elapsedLabel = useMemo(() => {
    if (!run) return null;
    const started = run.startedAt ? new Date(run.startedAt).getTime() : null;
    const ended = run.finishedAt
      ? new Date(run.finishedAt).getTime()
      : null;
    if (!started || !ended) return null;
    const sec = Math.max(1, Math.round((ended - started) / 1000));
    if (sec < 60) return `${sec}s`;
    return `${Math.round(sec / 60)}m`;
  }, [run]);

  // ── Actions ──────────────────────────────────────────────────────────
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
      .then(() => toast.success("Report copied"))
      .catch(() => toast.error("Couldn't copy"));
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

  // Done runs open the artifact pane (right-side, full-height,
  // tabbed Report/Sources/Activity); in-progress runs open the
  // lighter side sheet so the user can watch the live activity feed
  // without committing the full-screen real estate. These two hooks
  // MUST stay above the early-return guards below — React's hook
  // ordering is positional and putting them after a conditional
  // return causes "Rendered more hooks than during the previous
  // render" (#310) on the second render.
  const { setArtifact } = useArtifact();
  const onCardClick = useCallback(() => {
    if (!run) return;
    if (isDone && report) {
      setArtifact({
        documentId: run.id,
        kind: "research" as const,
        title: query.length > 80 ? `${query.slice(0, 79)}…` : query,
        content: report.markdown,
        status: "idle",
        isVisible: true,
        boundingBox: { top: 0, left: 0, width: 0, height: 0 },
      });
      return;
    }
    setSheetOpen(true);
  }, [isDone, report, run, query, setArtifact]);

  // ── Render ───────────────────────────────────────────────────────────
  if (loadError) {
    return (
      <div className="rounded-xl border border-destructive/30 bg-destructive/5 p-3 text-destructive text-sm">
        <p className="font-medium">Couldn&apos;t load research</p>
        <p className="text-xs opacity-80">{loadError}</p>
      </div>
    );
  }

  if (!run) {
    return (
      <div className="flex h-[88px] items-center gap-2 rounded-xl border border-border/60 bg-card px-4 text-muted-foreground text-sm">
        <Loader2Icon className="size-3.5 animate-spin" />
        Loading research…
      </div>
    );
  }

  const workerOffline =
    workerStatus !== null &&
    workerStatus.configured === true &&
    workerStatus.running === false;

  return (
    <>
      <ResearchPreviewCard
        elapsedLabel={elapsedLabel}
        isCancelled={isCancelled}
        isDone={isDone}
        isFailed={isFailed}
        liveLine={liveLine}
        onOpen={onCardClick}
        query={query}
        reportPreview={reportPreview}
        sourceCount={sources.length}
        workerOffline={workerOffline ? workerStatus : null}
      />

      <ResearchSheet
        activitySteps={activitySteps}
        elapsedLabel={elapsedLabel}
        isCancelled={isCancelled}
        isDone={isDone}
        isFailed={isFailed}
        onCancel={onCancel}
        onCopy={onCopy}
        onDownload={onDownload}
        open={sheetOpen}
        query={query}
        report={report}
        setOpen={setSheetOpen}
        sources={sources}
        status={status}
        workerOffline={workerOffline ? workerStatus : null}
      />
    </>
  );
}

// ────────────────────────────────────────────────────────────────────────
// In-chat preview card
// ────────────────────────────────────────────────────────────────────────

function ResearchPreviewCard({
  elapsedLabel,
  isCancelled,
  isDone,
  isFailed,
  liveLine,
  onOpen,
  query,
  reportPreview,
  sourceCount,
  workerOffline,
}: {
  elapsedLabel: string | null;
  isCancelled: boolean;
  isDone: boolean;
  isFailed: boolean;
  liveLine: string;
  onOpen: () => void;
  query: string;
  reportPreview: string | null;
  sourceCount: number;
  workerOffline: WorkerStatus | null;
}) {
  const headline = isDone
    ? "Research complete"
    : isFailed
    ? "Couldn't finish this research"
    : isCancelled
    ? "Research cancelled"
    : "Researching";

  const meta = isDone
    ? [
        sourceCount > 0
          ? `${sourceCount} source${sourceCount === 1 ? "" : "s"}`
          : null,
        elapsedLabel,
      ]
        .filter(Boolean)
        .join(" · ")
    : null;

  return (
    <motion.button
      animate={{ opacity: 1, y: 0 }}
      aria-label={`Open ${headline.toLowerCase()}`}
      className={cn(
        "group relative w-full max-w-[640px] cursor-pointer overflow-hidden rounded-xl border bg-card text-left",
        "transition-[transform,box-shadow,border-color] duration-200 ease-out",
        "hover:-translate-y-0.5 hover:shadow-lg hover:shadow-black/[0.06]",
        isFailed
          ? "border-destructive/30 hover:border-destructive/50"
          : isDone
          ? "border-emerald-500/25 hover:border-emerald-500/50 dark:border-emerald-400/20 dark:hover:border-emerald-400/40"
          : "border-border/70 hover:border-primary/40"
      )}
      initial={{ opacity: 0, y: 4 }}
      onClick={onOpen}
      transition={{ duration: 0.2, ease: EASE }}
      type="button"
    >
      <div className="flex items-start gap-3 p-4">
        <PreviewIcon
          isCancelled={isCancelled}
          isDone={isDone}
          isFailed={isFailed}
        />
        <div className="min-w-0 flex-1">
          <div className="flex items-center justify-between gap-2">
            <p className="truncate font-medium text-[13.5px] text-foreground">
              {headline}
              {meta && (
                <span className="ml-1.5 font-normal text-[12px] text-muted-foreground">
                  · {meta}
                </span>
              )}
            </p>
            <ArrowUpRightIcon
              aria-hidden
              className="size-3.5 shrink-0 text-muted-foreground/40 transition-colors group-hover:text-foreground/70"
            />
          </div>
          <p className="mt-1 line-clamp-2 text-[12.5px] text-muted-foreground leading-snug">
            {query}
          </p>
          <div className="mt-2.5 min-h-[32px] text-[11.5px] leading-snug">
            {isDone && reportPreview ? (
              <span className="line-clamp-2 text-foreground/75">
                {reportPreview}
              </span>
            ) : isFailed || isCancelled ? (
              <span className="text-muted-foreground/70">
                {isFailed
                  ? "Open to see what went wrong."
                  : "You stopped this run."}
              </span>
            ) : (
              <span className="flex items-center gap-1.5 text-muted-foreground">
                <PulsingDot />
                <span className="truncate">{liveLine}</span>
              </span>
            )}
          </div>
        </div>
      </div>
      {workerOffline && <WorkerOfflineNotice compact status={workerOffline} />}
    </motion.button>
  );
}

export function PreviewIcon({
  isCancelled,
  isDone,
  isFailed,
}: {
  isCancelled: boolean;
  isDone: boolean;
  isFailed: boolean;
}) {
  if (isFailed) {
    return (
      <span className="flex size-8 shrink-0 items-center justify-center rounded-md bg-destructive/10 text-destructive">
        <XCircleIcon className="size-4" />
      </span>
    );
  }
  if (isCancelled) {
    return (
      <span className="flex size-8 shrink-0 items-center justify-center rounded-md bg-muted text-muted-foreground">
        <XCircleIcon className="size-4" />
      </span>
    );
  }
  if (isDone) {
    return (
      <span className="flex size-8 shrink-0 items-center justify-center rounded-md bg-emerald-500/10 text-emerald-600 dark:text-emerald-400">
        <CheckCircle2Icon className="size-4" />
      </span>
    );
  }
  return (
    <span className="relative flex size-8 shrink-0 items-center justify-center rounded-md bg-primary/10 text-primary">
      <TelescopeIcon className="size-4" />
      <span className="-bottom-0.5 -right-0.5 absolute size-2 rounded-full bg-primary">
        <span className="absolute inset-0 animate-ping rounded-full bg-primary/60" />
      </span>
    </span>
  );
}

function PulsingDot() {
  return (
    <span className="relative inline-flex size-2 items-center justify-center">
      <span className="absolute inline-flex size-full animate-ping rounded-full bg-primary/50 opacity-60" />
      <span className="relative inline-flex size-1.5 rounded-full bg-primary" />
    </span>
  );
}

// ────────────────────────────────────────────────────────────────────────
// Side sheet (full activity feed + report)
// ────────────────────────────────────────────────────────────────────────

function ResearchSheet({
  activitySteps,
  elapsedLabel,
  isCancelled,
  isDone,
  isFailed,
  onCancel,
  onCopy,
  onDownload,
  open,
  query,
  report,
  setOpen,
  sources,
  status,
  workerOffline,
}: {
  activitySteps: ActivityStep[];
  elapsedLabel: string | null;
  isCancelled: boolean;
  isDone: boolean;
  isFailed: boolean;
  onCancel: () => Promise<void> | void;
  onCopy: () => void;
  onDownload: () => void;
  open: boolean;
  query: string;
  report: ResearchReport | null;
  setOpen: Dispatch<SetStateAction<boolean>>;
  sources: ResearchSource[];
  status: string;
  workerOffline: WorkerStatus | null;
}) {
  const [activityOpen, setActivityOpen] = useState(!isDone);
  useEffect(() => {
    if (isDone) setActivityOpen(false);
  }, [isDone]);

  const headline = isDone
    ? "Research complete"
    : isFailed
    ? "Couldn't finish this research"
    : isCancelled
    ? "Research cancelled"
    : "Researching";

  return (
    <Sheet onOpenChange={setOpen} open={open}>
      <SheetContent
        className="flex w-full flex-col gap-0 p-0 sm:max-w-[760px]"
        side="right"
      >
        <SheetHeader className="space-y-3 border-border border-b px-6 py-5">
          <div className="flex items-center justify-between gap-3">
            <SheetTitle className="flex items-center gap-2 text-[15px]">
              <PreviewIcon
                isCancelled={isCancelled}
                isDone={isDone}
                isFailed={isFailed}
              />
              <span>{headline}</span>
              {elapsedLabel && (
                <span className="font-normal text-muted-foreground text-xs">
                  · {elapsedLabel}
                </span>
              )}
            </SheetTitle>
            {isDone && report && (
              <div className="flex items-center gap-1">
                <Button
                  className="h-8 gap-1.5 px-2.5 text-xs"
                  onClick={onCopy}
                  size="sm"
                  variant="ghost"
                >
                  <CopyIcon className="size-3.5" />
                  Copy
                </Button>
                <Button
                  className="h-8 gap-1.5 px-2.5 text-xs"
                  onClick={onDownload}
                  size="sm"
                  variant="ghost"
                >
                  <DownloadIcon className="size-3.5" />
                  Download
                </Button>
              </div>
            )}
            {!isDone && !isFailed && !isCancelled && (
              <Button
                className="h-8 px-2.5 text-xs"
                onClick={onCancel}
                size="sm"
                variant="ghost"
              >
                Stop
              </Button>
            )}
          </div>
          <p className="text-[13px] text-muted-foreground leading-relaxed">
            {query}
          </p>
        </SheetHeader>

        <div className="flex-1 overflow-y-auto">
          {workerOffline && (
            <div className="px-6 pt-5">
              <WorkerOfflineNotice status={workerOffline} />
            </div>
          )}

          {isDone && report && (
            <section className="px-6 pt-6 pb-4">
              <ResearchReportRenderer
                citations={sources}
                markdown={report.markdown}
              />
            </section>
          )}

          {sources.length > 0 && isDone && (
            <section className="border-border/60 border-t px-6 py-5">
              <SourcesList sources={sources} />
            </section>
          )}

          <section
            className={cn(
              "px-6 py-5",
              isDone && "border-border/60 border-t bg-muted/15"
            )}
          >
            <button
              aria-expanded={activityOpen}
              className="-mx-1 flex w-full items-center justify-between rounded-md px-1 py-1 text-left transition-colors hover:bg-muted/40"
              onClick={() => setActivityOpen((v) => !v)}
              type="button"
            >
              <h3 className="font-semibold text-[11px] text-muted-foreground uppercase tracking-[0.08em]">
                {isDone ? "How this report was researched" : "Activity"}
              </h3>
              <span className="text-[11px] text-muted-foreground">
                {activitySteps.length} step
                {activitySteps.length === 1 ? "" : "s"}
                {activityOpen ? " · hide" : " · show"}
              </span>
            </button>
            <AnimatePresence initial={false}>
              {activityOpen && (
                <motion.div
                  animate={{ height: "auto", opacity: 1 }}
                  className="overflow-hidden"
                  exit={{ height: 0, opacity: 0 }}
                  initial={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.18, ease: EASE }}
                >
                  <ActivityFeed
                    inProgress={!isDone && !isFailed && !isCancelled}
                    status={status}
                    steps={activitySteps}
                  />
                </motion.div>
              )}
            </AnimatePresence>
          </section>
        </div>
      </SheetContent>
    </Sheet>
  );
}

// ────────────────────────────────────────────────────────────────────────
// Activity feed
// ────────────────────────────────────────────────────────────────────────

export function ActivityFeed({
  inProgress,
  status,
  steps,
}: {
  inProgress: boolean;
  status: string;
  steps: ActivityStep[];
}) {
  if (steps.length === 0) {
    return (
      <p className="mt-3 flex items-center gap-2 text-[13px] text-muted-foreground">
        <Loader2Icon className="size-3.5 animate-spin" />
        {status === "queued"
          ? "Waiting for the research worker to pick this up…"
          : "Getting started…"}
      </p>
    );
  }
  return (
    <ol className="mt-3 space-y-3">
      {steps.map((step, i) => (
        <ActivityStepRow
          isLast={i === steps.length - 1}
          key={`${step.kind}-${step.ts}-${i}`}
          showTrailingPulse={inProgress && i === steps.length - 1}
          step={step}
        />
      ))}
    </ol>
  );
}

function ActivityStepRow({
  isLast,
  showTrailingPulse,
  step,
}: {
  isLast: boolean;
  showTrailingPulse: boolean;
  step: ActivityStep;
}) {
  const Icon = stepIcon(step);
  return (
    <li className="relative flex gap-3">
      <div className="relative flex flex-col items-center">
        <span
          className={cn(
            "flex size-6 shrink-0 items-center justify-center rounded-full border bg-background",
            stepIconClass(step)
          )}
        >
          {showTrailingPulse ? (
            <Loader2Icon className="size-3 animate-spin" />
          ) : (
            <Icon className="size-3" />
          )}
        </span>
        {!isLast && <span aria-hidden className="mt-1 w-px flex-1 bg-border" />}
      </div>
      <div className="min-w-0 flex-1 pb-2">
        <ActivityStepBody step={step} />
      </div>
    </li>
  );
}

function ActivityStepBody({ step }: { step: ActivityStep }) {
  if (step.kind === "phase") {
    return (
      <p className="font-medium text-[13px] text-foreground/85">{step.label}</p>
    );
  }
  if (step.kind === "task") {
    return (
      <p className="font-medium text-[13px] text-foreground/85">
        Looking into{" "}
        <span className="text-foreground">{lowerFirst(step.subQuestion)}</span>
      </p>
    );
  }
  if (step.kind === "search") {
    return (
      <div className="space-y-2">
        <p className="text-[13px] text-foreground/85">
          Searched{" "}
          <span className="font-medium text-foreground">“{step.query}”</span>
        </p>
        {step.results.length > 0 && <SourceCardList results={step.results} />}
      </div>
    );
  }
  if (step.kind === "read") {
    return (
      <p className="text-[13px] text-foreground/85">
        {step.ok ? "Read" : "Tried to read"}{" "}
        <a
          className="font-medium text-foreground underline-offset-2 hover:underline"
          href={step.url}
          rel="noreferrer"
          target="_blank"
        >
          {step.host || step.url}
        </a>
        {step.ok && step.chunks
          ? ` · ${step.chunks} section${step.chunks === 1 ? "" : "s"} indexed`
          : ""}
        {!step.ok && " · couldn't extract content"}
      </p>
    );
  }
  if (step.kind === "retrieve") {
    return (
      <div className="space-y-1.5">
        <p className="text-[13px] text-foreground/85">Pulled relevant excerpts</p>
        {step.chunks.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {step.chunks.map((c, i) => (
              <a
                className="rounded-full border border-border/60 bg-muted/30 px-2 py-0.5 text-[11px] text-muted-foreground hover:bg-muted/60 hover:text-foreground"
                href={c.url}
                key={`${c.url}-${i}`}
                rel="noreferrer"
                target="_blank"
              >
                {c.host}
              </a>
            ))}
          </div>
        )}
      </div>
    );
  }
  return null;
}

function stepIcon(step: ActivityStep) {
  if (step.kind === "phase") {
    if (step.icon === "globe") return GlobeIcon;
    if (step.icon === "file") return FileTextIcon;
    if (step.icon === "check") return CheckCircle2Icon;
    return SparklesIcon;
  }
  if (step.kind === "task") return SparklesIcon;
  if (step.kind === "search") return SearchIcon;
  if (step.kind === "read") return GlobeIcon;
  if (step.kind === "retrieve") return FileTextIcon;
  return SparklesIcon;
}

function stepIconClass(step: ActivityStep): string {
  if (step.kind === "task") return "border-primary/30 text-primary";
  return "border-border/70 text-muted-foreground";
}

function SourceCardList({ results }: { results: SearchHit[] }) {
  return (
    <div className="grid gap-1.5">
      {results.slice(0, 4).map((r) => (
        <a
          className={cn(
            "flex items-start gap-2 rounded-lg border border-border/40 bg-muted/15 px-2.5 py-1.5",
            "no-underline transition-colors hover:bg-muted/40 hover:border-border/70"
          )}
          href={r.url}
          key={r.url}
          rel="noopener noreferrer"
          target="_blank"
        >
          <img
            alt=""
            className="mt-0.5 size-3.5 shrink-0 rounded-sm"
            loading="lazy"
            onError={(e) => {
              (e.currentTarget as HTMLImageElement).style.display = "none";
            }}
            referrerPolicy="no-referrer"
            src={`https://www.google.com/s2/favicons?domain=${r.host}&sz=32`}
          />
          <div className="min-w-0 flex-1">
            <p className="line-clamp-1 font-medium text-[12.5px] text-foreground/90">
              {r.title}
            </p>
            <p className="line-clamp-1 text-[11px] text-muted-foreground/70">
              {r.host}
              {r.snippet ? ` · ${r.snippet}` : ""}
            </p>
          </div>
        </a>
      ))}
      {results.length > 4 && (
        <p className="pl-1 text-[11px] text-muted-foreground/60">
          +{results.length - 4} more result{results.length - 4 === 1 ? "" : "s"}
        </p>
      )}
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────
// Sources list (final, deduped)
// ────────────────────────────────────────────────────────────────────────

export function SourcesList({ sources }: { sources: ResearchSource[] }) {
  return (
    <div>
      <h3 className="font-semibold text-[11px] text-muted-foreground uppercase tracking-[0.08em]">
        Sources ({sources.length})
      </h3>
      <ol className="mt-3 grid gap-1.5">
        {sources.map((s) => {
          const domain = domainFor(s.url);
          const label = s.title ?? domain;
          return (
            <li key={s.citationNum}>
              <a
                className={cn(
                  "flex items-start gap-2 rounded-lg border border-border/40 bg-muted/10 px-2.5 py-2",
                  "no-underline transition-colors hover:bg-muted/30 hover:border-border/70"
                )}
                href={s.url}
                rel="noreferrer"
                target="_blank"
              >
                <span className="mt-0.5 shrink-0 rounded bg-muted px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
                  {s.citationNum}
                </span>
                <img
                  alt=""
                  className="mt-0.5 size-3.5 shrink-0 rounded-sm"
                  loading="lazy"
                  onError={(e) => {
                    (e.currentTarget as HTMLImageElement).style.display = "none";
                  }}
                  referrerPolicy="no-referrer"
                  src={`https://www.google.com/s2/favicons?domain=${domain}&sz=32`}
                />
                <div className="min-w-0 flex-1">
                  <p className="line-clamp-1 font-medium text-[12.5px] text-foreground/90">
                    {label}
                  </p>
                  <p className="line-clamp-1 text-[11px] text-muted-foreground/70">
                    {domain}
                    <ExternalLinkIcon className="ml-1 inline size-3 align-[-1px]" />
                  </p>
                </div>
              </a>
            </li>
          );
        })}
      </ol>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────
// Worker offline notice
// ────────────────────────────────────────────────────────────────────────

function WorkerOfflineNotice({
  compact,
  status,
}: {
  compact?: boolean;
  status: WorkerStatus;
}) {
  const detail = status.error?.toLowerCase() ?? "";
  const hint = detail.includes("database_url")
    ? "Set DATABASE_URL on the backend service."
    : detail.includes("cannot reach backend")
    ? "Backend service is unreachable."
    : null;
  return (
    <div
      className={cn(
        "flex items-start gap-2 border-amber-500/30 bg-amber-500/5 px-3 py-2 text-amber-700 text-xs dark:text-amber-400",
        compact ? "border-t" : "rounded-md border"
      )}
    >
      <AlertTriangleIcon className="mt-0.5 size-3.5 shrink-0" />
      <div className="min-w-0">
        <p className="font-medium">Research worker is offline</p>
        {status.error ? (
          <p className="mt-0.5 break-all opacity-90">{status.error}</p>
        ) : null}
        {hint ? <p className="mt-1 opacity-80">{hint}</p> : null}
      </div>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────
// Activity-feed builder + helpers
// ────────────────────────────────────────────────────────────────────────

export function buildActivitySteps(
  events: StreamEvent[],
  subagents: SubagentLive[]
): ActivityStep[] {
  const steps: ActivityStep[] = [];
  const subQById = new Map<string, string>();
  for (const sa of subagents) subQById.set(sa.id, sa.subQuestion);

  for (const evt of events) {
    const id = typeof evt.payload?.id === "string" ? evt.payload.id : null;
    const subQ = id ? subQById.get(id) ?? null : null;

    switch (evt.type) {
      case "brief_drafted":
        steps.push({
          icon: "spark",
          kind: "phase",
          label: "Framed the question",
          ts: evt.ts,
        });
        break;
      case "plan_proposed":
        steps.push({
          icon: "spark",
          kind: "phase",
          label: "Mapped out angles to investigate",
          ts: evt.ts,
        });
        break;
      case "research_started":
        steps.push({
          icon: "globe",
          kind: "phase",
          label: "Searching the web",
          ts: evt.ts,
        });
        break;
      case "subagent_started": {
        const sq =
          typeof evt.payload?.subQuestion === "string"
            ? evt.payload.subQuestion
            : null;
        if (sq) {
          if (id) subQById.set(id, sq);
          steps.push({ kind: "task", subQuestion: sq, ts: evt.ts });
        }
        break;
      }
      case "subagent_progress": {
        const action =
          typeof evt.payload?.action === "string" ? evt.payload.action : null;
        if (!action) break;
        if (action === "search") {
          const query =
            typeof evt.payload.detail === "string" ? evt.payload.detail : "";
          const rawResults = Array.isArray(evt.payload.results)
            ? (evt.payload.results as Array<Record<string, unknown>>)
            : [];
          const results: SearchHit[] = rawResults
            .map((r) => ({
              host: String(r.host ?? hostOf(String(r.url ?? ""))),
              snippet: String(r.snippet ?? ""),
              title: String(r.title ?? ""),
              url: String(r.url ?? ""),
            }))
            .filter((r) => r.url);
          steps.push({
            kind: "search",
            query,
            results,
            subQuestion: subQ,
            ts: evt.ts,
          });
        } else if (action === "scrape") {
          const url =
            typeof evt.payload.url === "string"
              ? evt.payload.url
              : typeof evt.payload.detail === "string"
              ? evt.payload.detail
              : "";
          const ok =
            typeof evt.payload.ok === "boolean" ? evt.payload.ok : true;
          const chunks =
            typeof evt.payload.chunks === "number"
              ? (evt.payload.chunks as number)
              : null;
          const host =
            typeof evt.payload.host === "string"
              ? (evt.payload.host as string)
              : hostOf(url);
          steps.push({
            chunks,
            host,
            kind: "read",
            ok,
            subQuestion: subQ,
            ts: evt.ts,
            url,
          });
        } else if (action === "retrieve") {
          const query =
            typeof evt.payload.detail === "string" ? evt.payload.detail : "";
          const rawChunks = Array.isArray(evt.payload.chunks)
            ? (evt.payload.chunks as Array<Record<string, unknown>>)
            : [];
          const chunks: RetrievedChunk[] = rawChunks.map((c) => ({
            host: String(c.host ?? ""),
            snippet: String(c.snippet ?? ""),
            url: String(c.url ?? ""),
          }));
          steps.push({
            chunks,
            kind: "retrieve",
            query,
            subQuestion: subQ,
            ts: evt.ts,
          });
        }
        break;
      }
      case "writer_started":
        steps.push({
          icon: "file",
          kind: "phase",
          label: "Writing the report",
          ts: evt.ts,
        });
        break;
      case "report_written":
        steps.push({
          icon: "check",
          kind: "phase",
          label: "Report ready",
          ts: evt.ts,
        });
        break;
      default:
        break;
    }
  }

  return steps;
}

function deriveLiveLine(status: string, steps: ActivityStep[]): string {
  for (let i = steps.length - 1; i >= 0; i -= 1) {
    const s = steps[i];
    if (s.kind === "search") return `Searching “${truncate(s.query, 60)}”`;
    if (s.kind === "read") return `Reading ${s.host || "a source"}`;
    if (s.kind === "retrieve") return "Pulling relevant excerpts";
    if (s.kind === "task")
      return `Looking into ${truncate(lowerFirst(s.subQuestion), 60)}`;
    if (s.kind === "phase") return s.label;
  }
  if (status === "queued") return "Starting up…";
  if (status === "scoping" || status === "planning")
    return "Mapping out angles to investigate";
  if (status === "writing") return "Writing the report";
  return "Getting started…";
}

export function extractReportPreview(markdown: string): string | null {
  const lines = markdown.split("\n");
  for (const raw of lines) {
    const line = raw.trim();
    if (!line) continue;
    if (line.startsWith("#")) continue;
    if (line.startsWith(">")) continue;
    if (line.startsWith("|")) continue;
    if (line.startsWith("```")) continue;
    const clean = line
      .replace(/\[\d+\]/g, "")
      .replace(/[*_`]/g, "")
      .replace(/\s+/g, " ")
      .trim();
    if (clean.length < 20) continue;
    return truncate(clean, 220);
  }
  return null;
}

function truncate(s: string, n: number): string {
  if (s.length <= n) return s;
  return `${s.slice(0, n - 1).trimEnd()}…`;
}

function lowerFirst(s: string): string {
  if (!s) return s;
  return s[0].toLowerCase() + s.slice(1);
}

function domainFor(url: string): string {
  try {
    const u = new URL(url);
    return u.hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
}

function hostOf(url: string): string {
  return domainFor(url);
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
