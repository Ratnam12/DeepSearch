/**
 * Verifies the research-* tables exist on Neon and reports their column
 * counts. Used as a smoke test after `pnpm db:migrate`.
 *
 *   npx tsx scripts/verify-research-schema.ts
 */
import { config } from "dotenv";
import postgres from "postgres";

config({ path: ".env.development.local" });
config({ path: ".env.local", override: false });

const url = process.env.DATABASE_URL ?? process.env.POSTGRES_URL;
if (!url) {
  console.error("DATABASE_URL/POSTGRES_URL not set");
  process.exit(1);
}

const EXPECTED = [
  "ResearchEvent",
  "ResearchPlan",
  "ResearchReport",
  "ResearchRun",
  "ResearchSource",
  "ResearchSubagent",
] as const;

const sql = postgres(url, { max: 1 });

async function main() {
  const rows = await sql<{ table_name: string; col_count: number }[]>`
    SELECT table_name, count(*)::int AS col_count
    FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name = ANY(${EXPECTED as readonly string[] as string[]})
    GROUP BY table_name
    ORDER BY table_name;
  `;
  for (const r of rows) console.log("  -", r.table_name, "cols:", r.col_count);
  if (rows.length !== EXPECTED.length) {
    console.error(`EXPECTED ${EXPECTED.length} tables, got ${rows.length}`);
    process.exit(2);
  }
  console.log("ok");
  await sql.end();
}

main().catch((err) => {
  console.error(err);
  process.exit(3);
});
