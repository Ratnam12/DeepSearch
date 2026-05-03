import { auth } from "@clerk/nextjs/server";
import { format } from "date-fns";
import Link from "next/link";
import { redirect } from "next/navigation";
import { listResearchRunsByUserId } from "@/lib/db/queries-research";
import type { ResearchRun } from "@/lib/db/schema";

export const dynamic = "force-dynamic";

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

const TERMINAL = new Set(["done", "failed", "cancelled"]);

export default async function ResearchListPage() {
  const { userId } = await auth();
  if (!userId) {
    redirect("/sign-in");
  }

  const runs = await listResearchRunsByUserId({ userId, limit: 50 });

  return (
    <div className="mx-auto flex h-dvh w-full max-w-3xl flex-col gap-4 px-6 py-10">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="font-semibold text-2xl tracking-tight">Research</h1>
          <p className="text-muted-foreground text-sm">
            Long-form research runs with a planner, sub-agents, and a
            cited report. Click a run to follow its live progress.
          </p>
        </div>
      </header>

      {runs.length === 0 ? (
        <EmptyState />
      ) : (
        <ul className="flex flex-col gap-2">
          {runs.map((run) => (
            <li key={run.id}>
              <RunRow run={run} />
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function EmptyState() {
  return (
    <div className="rounded-lg border border-border border-dashed p-10 text-center">
      <p className="font-medium text-base">No research runs yet</p>
      <p className="mx-auto mt-1 max-w-md text-muted-foreground text-sm">
        Phase 1 of DeepSearch&apos;s deep-research agent ships the
        scaffolding (durable runs, plan approval, live progress
        streaming). Real agents land in the next phase.
      </p>
    </div>
  );
}

function RunRow({ run }: { run: ResearchRun }) {
  const statusLabel = STATUS_LABELS[run.status] ?? run.status;
  const isTerminal = TERMINAL.has(run.status);
  return (
    <Link
      className="block rounded-lg border border-border bg-card p-4 transition-colors duration-150 hover:bg-muted/50"
      href={`/research/${run.id}`}
    >
      <div className="flex items-start justify-between gap-3">
        <p className="line-clamp-2 font-medium text-sm leading-snug">
          {run.query}
        </p>
        <span
          className={
            isTerminal
              ? "shrink-0 rounded-full bg-muted px-2 py-0.5 font-medium text-[11px] text-muted-foreground"
              : "shrink-0 rounded-full bg-primary/10 px-2 py-0.5 font-medium text-[11px] text-primary"
          }
        >
          {statusLabel}
        </span>
      </div>
      <p className="mt-2 text-muted-foreground text-xs">
        {format(new Date(run.createdAt), "MMM d, yyyy 'at' p")}
      </p>
    </Link>
  );
}
