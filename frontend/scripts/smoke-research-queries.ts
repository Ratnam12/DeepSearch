/**
 * Smoke test for the research queries module against the live Neon DB.
 *
 * This exercises the same code paths the Next.js API routes use, but
 * without going through Clerk-gated HTTP — so we get fast feedback on
 * the data layer without spinning up a dev server.
 *
 *   npx tsx scripts/smoke-research-queries.ts
 *
 * Cleans up its own rows on success or failure.
 */
import { config } from "dotenv";
import postgres from "postgres";

// Load env BEFORE importing the queries module — that module
// instantiates its `postgres` client at import time using
// DATABASE_URL, so we need the env wired up first. ESM static imports
// are hoisted, so we use a dynamic import below.
config({ path: ".env.development.local" });
config({ path: ".env.local", override: false });

type QueriesModule = typeof import("../lib/db/queries-research");

const TEST_USER_ID = `test-user-smoke-${process.pid}`;

function log(label: string, ok: boolean, extra?: string) {
  const tag = ok ? "ok" : "FAIL";
  console.log(`  [${tag}] ${label}${extra ? `: ${extra}` : ""}`);
}

async function main() {
  const url =
    process.env.DATABASE_URL ?? process.env.POSTGRES_URL ?? "";
  if (!url) throw new Error("DATABASE_URL/POSTGRES_URL not set");
  const cleanup = postgres(url, { max: 1 });
  let runId: string | null = null;
  let failures = 0;
  const fail = (label: string, extra: string) => {
    log(label, false, extra);
    failures += 1;
  };

  const {
    appendResearchEvent,
    approveResearchPlan,
    cancelResearchRun,
    createResearchRun,
    getLatestResearchPlan,
    getLatestResearchReport,
    getResearchRunById,
    listResearchEventsSince,
    listResearchRunsByUserId,
    listResearchSources,
    listResearchSubagents,
    saveResearchPlan,
    updateResearchRunStatus,
  }: QueriesModule = await import("../lib/db/queries-research");

  try {
    // ── createResearchRun ────────────────────────────────────────────
    const run = await createResearchRun({
      userId: TEST_USER_ID,
      query: "smoke test query",
    });
    runId = run.id;
    if (run.status !== "queued") fail("createResearchRun status", run.status);
    else log("createResearchRun -> queued", true);

    // ── getResearchRunById ──────────────────────────────────────────
    const fetched = await getResearchRunById({ id: runId });
    if (!fetched || fetched.id !== runId) fail("getResearchRunById", "missing");
    else log("getResearchRunById", true);

    // ── listResearchRunsByUserId ────────────────────────────────────
    const list = await listResearchRunsByUserId({ userId: TEST_USER_ID });
    if (!list.some((r) => r.id === runId)) fail("listResearchRunsByUserId", `no row for ${runId}`);
    else log("listResearchRunsByUserId includes new run", true);

    // ── updateResearchRunStatus ─────────────────────────────────────
    await updateResearchRunStatus({ id: runId, status: "scoping" });
    const afterStatus = await getResearchRunById({ id: runId });
    if (afterStatus?.status !== "scoping") fail("updateResearchRunStatus", afterStatus?.status ?? "missing");
    else log("updateResearchRunStatus -> scoping", true);

    // ── saveResearchPlan + getLatestResearchPlan ────────────────────
    await saveResearchPlan({
      runId,
      version: 1,
      briefMd: "smoke brief",
      subQuestions: [{ id: "sq1", question: "what?", rationale: "because" }],
      outline: [{ id: "s1", title: "intro", description: "" }],
    });
    const plan = await getLatestResearchPlan({ runId });
    if (!plan || plan.version !== 1) fail("getLatestResearchPlan", `version=${plan?.version}`);
    else log("saveResearchPlan + getLatestResearchPlan", true);

    // ── approveResearchPlan (no edits) ──────────────────────────────
    await approveResearchPlan({ runId, version: 1 });
    const approvedPlan = await getLatestResearchPlan({ runId });
    if (!approvedPlan?.approvedAt) fail("approveResearchPlan", "approvedAt is null");
    else log("approveResearchPlan stamps approvedAt", true);

    // ── approveResearchPlan (with user edits) ──────────────────────
    // Mirrors the path the plan-approval card takes when the user
    // tweaks sub-questions / outline before clicking approve.
    await approveResearchPlan({
      runId,
      version: 1,
      subQuestions: [
        { id: "sq-edit", question: "edited question", rationale: "user added" },
        { id: "sq-edit2", question: "second", rationale: "still here" },
      ],
      outline: [{ id: "s-edit", title: "edited section", description: "" }],
    });
    const editedPlan = await getLatestResearchPlan({ runId });
    const editedSubQs = (editedPlan?.subQuestions ?? []) as Array<{
      id: string;
      question: string;
    }>;
    if (
      editedSubQs.length !== 2 ||
      editedSubQs[0]?.question !== "edited question"
    )
      fail("approveResearchPlan persists user edits", JSON.stringify(editedSubQs));
    else log("approveResearchPlan persists user edits", true);

    // ── appendResearchEvent (atomic seq allocation) ────────────────
    const evtA = await appendResearchEvent({
      runId,
      type: "smoke_a",
      payload: { i: 1 },
    });
    const evtB = await appendResearchEvent({
      runId,
      type: "smoke_b",
      payload: { i: 2 },
    });
    if (evtA.seq !== 1 || evtB.seq !== 2)
      fail("appendResearchEvent seq", `${evtA.seq}, ${evtB.seq}`);
    else log("appendResearchEvent allocates seq=1,2", true);

    // ── listResearchEventsSince ─────────────────────────────────────
    const since0 = await listResearchEventsSince({ runId, sinceSeq: 0 });
    const since1 = await listResearchEventsSince({ runId, sinceSeq: 1 });
    if (since0.length !== 2 || since1.length !== 1)
      fail("listResearchEventsSince counts", `${since0.length}, ${since1.length}`);
    else log("listResearchEventsSince filters by seq", true);

    // ── listResearchSubagents / listResearchSources / report all empty ─
    const subs = await listResearchSubagents({ runId });
    const sources = await listResearchSources({ runId });
    const report = await getLatestResearchReport({ runId });
    if (subs.length || sources.length || report)
      fail("empty queries should be empty", `${subs.length}/${sources.length}/${!!report}`);
    else log("empty queries are empty", true);

    // ── cancelResearchRun ───────────────────────────────────────────
    const cancelled = await cancelResearchRun({ id: runId, userId: TEST_USER_ID });
    if (!cancelled) fail("cancelResearchRun returned false", "");
    const finalRun = await getResearchRunById({ id: runId });
    if (finalRun?.status !== "cancelled") fail("cancel sets status", finalRun?.status ?? "?");
    else log("cancelResearchRun -> cancelled", true);

    // ── Authorization guard: cancel for wrong user must be no-op ────
    const otherCancelled = await cancelResearchRun({
      id: runId,
      userId: "test-user-other",
    });
    if (otherCancelled)
      fail("cancelResearchRun ignores wrong user", "false-positive cancel");
    else log("cancelResearchRun rejects wrong user", true);
  } finally {
    if (runId) {
      await cleanup`DELETE FROM "ResearchRun" WHERE "id" = ${runId}::uuid`;
    }
    await cleanup.end({ timeout: 5 });
  }

  if (failures > 0) {
    console.error(`\n${failures} check(s) failed`);
    process.exit(1);
  }
  console.log("\nall checks ok");
  // Explicit exit because the queries module's postgres client is
  // module-scoped and never .end()'d, so the event loop has idle DB
  // connections keeping it alive after the script's work is done.
  process.exit(0);
}

main().catch((err) => {
  console.error(err);
  process.exit(2);
});
