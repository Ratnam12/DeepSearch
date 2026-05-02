import { config } from "dotenv";
import { defineConfig } from "drizzle-kit";

// Load both env files so the CLI works regardless of which one the user
// pulled secrets into. Vercel-managed Neon writes to .env.development.local;
// hand-edited overrides typically live in .env.local.
config({ path: ".env.development.local" });
config({ path: ".env.local", override: false });

export default defineConfig({
  schema: "./lib/db/schema.ts",
  out: "./lib/db/migrations",
  dialect: "postgresql",
  dbCredentials: {
    url: process.env.DATABASE_URL ?? process.env.POSTGRES_URL ?? "",
  },
});
