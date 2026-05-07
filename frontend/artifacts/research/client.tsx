"use client";

import {
  ActivityIcon,
  CopyIcon,
  DownloadIcon,
  FileTextIcon,
  Link2Icon,
  Loader2Icon,
} from "lucide-react";
import { type ReactNode, useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import { Artifact } from "@/components/chat/create-artifact";
import { ResearchReportRenderer } from "@/components/research/report-renderer";
import {
  ActivityFeed,
  buildActivitySteps,
  SourcesList,
  type StreamEvent,
  subagentFromRow,
  type SubagentLive,
} from "@/components/research/research-artifact-card";
import { cn } from "@/lib/utils";
import type {
  ResearchPlan,
  ResearchReport,
  ResearchRun,
  ResearchSource,
  ResearchSubagent,
} from "@/lib/db/schema";

// Right-pane artifact for a finished deep-research run.
//
// Opens when the user clicks the inline ResearchPreviewCard for a
// completed run. Three tabs:
//   - Report   → the markdown report rendered with citations
//   - Sources  → favicon list of every source the report cites
//   - Activity → the chronological feed of what each sub-agent did
//
// While a run is *in progress* the user sees the lighter side sheet
// from research-artifact-card.tsx, not this. The artifact pane is
// reserved for terminal runs where the report is the focal point.

type ResearchArtifactMetadata = {
  loaded: boolean;
  loadError: string | null;
  run: ResearchRun | null;
  plan: ResearchPlan | null;
  report: ResearchReport | null;
  sources: ResearchSource[];
  subagents: SubagentLive[];
  events: StreamEvent[];
};

const initialMetadata: ResearchArtifactMetadata = {
  loaded: false,
  loadError: null,
  run: null,
  plan: null,
  report: null,
  sources: [],
  subagents: [],
  events: [],
};

export const researchArtifact = new Artifact<
  "research",
  ResearchArtifactMetadata
>({
  kind: "research",
  description:
    "Deep-research report with sources and the full activity log of how it was researched.",
  initialize: async ({ documentId, setMetadata }) => {
    setMetadata(initialMetadata);
    try {
      const res = await fetch(`/api/research/${documentId}`, {
        cache: "no-store",
      });
      if (!res.ok) {
        setMetadata((m) => ({
          ...m,
          loaded: true,
          loadError:
            res.status === 404
              ? "This research run was deleted."
              : `Couldn't load research (${res.status}).`,
        }));
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
      setMetadata({
        loaded: true,
        loadError: null,
        run: snap.run,
        plan: snap.plan,
        report: snap.report,
        sources: snap.sources ?? [],
        subagents: (snap.subagents ?? []).map(subagentFromRow),
        events: snap.events ?? [],
      });
    } catch (err) {
      setMetadata((m) => ({
        ...m,
        loaded: true,
        loadError:
          err instanceof Error ? err.message : "Failed to load research",
      }));
    }
  },
  onStreamPart: () => {
    // Research runs stream via the per-run SSE endpoint, not the
    // chat data stream — the artifact only opens once a run is
    // already done, so there's nothing to merge in here.
  },
  content: ResearchArtifactBody,
  actions: [
    {
      icon: <CopyIcon size={16} />,
      description: "Copy the report markdown",
      onClick: ({ content }) => {
        if (!content) {
          toast.error("Nothing to copy yet.");
          return;
        }
        void navigator.clipboard
          .writeText(content)
          .then(() => toast.success("Report copied"))
          .catch(() => toast.error("Couldn't copy"));
      },
    },
    {
      icon: <DownloadIcon size={16} />,
      description: "Download the report as Markdown",
      onClick: ({ content, metadata }) => {
        if (!content) {
          toast.error("Nothing to download yet.");
          return;
        }
        const blob = new Blob([content], { type: "text/markdown" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        const slug = slugForFilename(metadata?.run?.query ?? "research");
        a.download = `${slug}.md`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
      },
    },
  ],
  toolbar: [],
});

type Tab = "report" | "sources" | "activity";

function ResearchArtifactBody({
  content,
  isLoading,
  metadata,
}: {
  content: string;
  isLoading: boolean;
  metadata: ResearchArtifactMetadata;
}) {
  const [tab, setTab] = useState<Tab>("report");
  const meta = metadata ?? initialMetadata;

  // Re-anchor at "report" if the user re-opens the artifact for a
  // different run mid-session.
  const runId = meta.run?.id ?? null;
  useEffect(() => {
    setTab("report");
  }, [runId]);

  const activitySteps = useMemo(
    () => buildActivitySteps(meta.events, meta.subagents),
    [meta.events, meta.subagents]
  );

  if (isLoading || !meta.loaded) {
    return (
      <div className="flex h-full items-center justify-center text-muted-foreground text-sm">
        <Loader2Icon className="mr-2 size-4 animate-spin" />
        Loading research…
      </div>
    );
  }

  if (meta.loadError) {
    return (
      <div className="px-6 py-8 text-destructive text-sm">
        <p className="font-medium">Couldn&apos;t load research</p>
        <p className="text-xs opacity-80">{meta.loadError}</p>
      </div>
    );
  }

  const sourceCount = meta.sources.length;
  const stepCount = activitySteps.length;

  return (
    <div className="flex h-full flex-col">
      <div className="sticky top-0 z-10 flex items-center gap-1 border-border/60 border-b bg-background/85 px-6 py-2 backdrop-blur">
        <TabButton
          active={tab === "report"}
          icon={<FileTextIcon className="size-3.5" />}
          label="Report"
          onClick={() => setTab("report")}
        />
        <TabButton
          active={tab === "sources"}
          icon={<Link2Icon className="size-3.5" />}
          label={`Sources${sourceCount > 0 ? ` · ${sourceCount}` : ""}`}
          onClick={() => setTab("sources")}
        />
        <TabButton
          active={tab === "activity"}
          icon={<ActivityIcon className="size-3.5" />}
          label={`Activity${stepCount > 0 ? ` · ${stepCount}` : ""}`}
          onClick={() => setTab("activity")}
        />
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-6">
        {tab === "report" && <ReportTab content={content} meta={meta} />}
        {tab === "sources" && <SourcesTab sources={meta.sources} />}
        {tab === "activity" && (
          <ActivityTab inProgress={false} steps={activitySteps} />
        )}
      </div>
    </div>
  );
}

function TabButton({
  active,
  icon,
  label,
  onClick,
}: {
  active: boolean;
  icon: ReactNode;
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      className={cn(
        "inline-flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-[12.5px] font-medium",
        "transition-colors duration-150",
        active
          ? "bg-muted text-foreground"
          : "text-muted-foreground hover:bg-muted/50 hover:text-foreground"
      )}
      onClick={onClick}
      type="button"
    >
      {icon}
      {label}
    </button>
  );
}

function ReportTab({
  content,
  meta,
}: {
  content: string;
  meta: ResearchArtifactMetadata;
}) {
  const markdown = content || meta.report?.markdown || "";
  if (!markdown) {
    return (
      <p className="py-12 text-center text-muted-foreground text-sm">
        No report yet.
      </p>
    );
  }
  return (
    <div className="mx-auto max-w-3xl">
      <ResearchReportRenderer citations={meta.sources} markdown={markdown} />
    </div>
  );
}

function SourcesTab({ sources }: { sources: ResearchSource[] }) {
  if (sources.length === 0) {
    return (
      <p className="py-12 text-center text-muted-foreground text-sm">
        No sources cited.
      </p>
    );
  }
  return (
    <div className="mx-auto max-w-3xl">
      <SourcesList sources={sources} />
    </div>
  );
}

function ActivityTab({
  inProgress,
  steps,
}: {
  inProgress: boolean;
  steps: ReturnType<typeof buildActivitySteps>;
}) {
  return (
    <div className="mx-auto max-w-3xl">
      <ActivityFeed inProgress={inProgress} status="done" steps={steps} />
    </div>
  );
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
