# DeepSearch — Frontend

Next.js 16 + AI SDK 6 frontend for the DeepSearch agentic research assistant.

## Stack

- **Next.js 16** (App Router, Turbopack)
- **AI SDK 6** (`useChat`, UI Message Stream Protocol)
- **Clerk** for authentication
- **Drizzle ORM + Neon Postgres** for chat history persistence
- **Vercel Blob** for image attachments
- **shadcn/ui + Tailwind CSS 4** for the interface

## Backend

The Python FastAPI agent lives in [`../backend/`](../backend) and is deployed
on Railway. The Next.js app proxies chat requests to its `/chat` endpoint
and pipes the AI SDK UI Message Stream Protocol response back to the
client.

## Local development

```bash
# from the repo root, link to the Vercel project (one-time)
vercel link        # Team: ratnamsingh1201-8407s-projects, Project: deep-search

# from /frontend/, pull env vars and install
vercel env pull .env.development.local
pnpm install

# apply the latest Drizzle migrations to Neon
pnpm db:migrate

# run the dev server
pnpm dev
```

The dev server listens on http://localhost:3000. Unauthenticated requests
redirect to `/sign-in`; sign in with a Clerk account to reach the chat UI.

## Required env vars

Pulled from Vercel via `vercel env pull`:

- `DATABASE_URL` (Neon Postgres)
- `BLOB_READ_WRITE_TOKEN` (Vercel Blob)
- `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`, `CLERK_SECRET_KEY`
- `NEXT_PUBLIC_API_URL` (Railway backend URL)

## Project layout

- `app/(chat)/` — main chat UI, history sidebar, artifact panel
- `app/(chat)/api/chat/` — proxy route that forwards to the FastAPI backend
- `app/sign-in/`, `app/sign-up/` — Clerk-hosted auth pages
- `lib/db/` — Drizzle schema, queries, migrations
- `lib/ai/` — model registry, prompts, document tools
- `components/` — shadcn-based UI components
- `proxy.ts` — Clerk middleware (auth gate)
