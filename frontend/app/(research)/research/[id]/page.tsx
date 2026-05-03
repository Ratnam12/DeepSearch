import { auth } from "@clerk/nextjs/server";
import { redirect } from "next/navigation";
import {
  getLatestResearchPlan,
  getLatestResearchReport,
  getResearchRunById,
  listResearchSources,
  listResearchSubagents,
} from "@/lib/db/queries-research";
import { ResearchRunView } from "@/components/research/research-run-view";

export const dynamic = "force-dynamic";

export default async function ResearchRunPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const { userId } = await auth();
  if (!userId) {
    redirect("/sign-in");
  }

  const run = await getResearchRunById({ id });
  if (!run || run.userId !== userId) {
    return (
      <div className="mx-auto flex h-dvh w-full max-w-3xl flex-col items-center justify-center gap-3 px-6 py-10 text-center">
        <h1 className="font-semibold text-xl">Research run not found</h1>
        <p className="text-muted-foreground text-sm">
          It may have been deleted, or it belongs to a different account.
        </p>
      </div>
    );
  }

  const [plan, subagents, sources, report] = await Promise.all([
    getLatestResearchPlan({ runId: id }),
    listResearchSubagents({ runId: id }),
    listResearchSources({ runId: id }),
    getLatestResearchReport({ runId: id }),
  ]);

  return (
    <ResearchRunView
      initialPlan={plan ?? null}
      initialReport={report ?? null}
      initialRun={run}
      initialSources={sources}
      initialSubagents={subagents}
    />
  );
}
