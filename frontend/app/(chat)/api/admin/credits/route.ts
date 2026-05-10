import { z } from "zod";
import { getAdminContext } from "@/lib/admin";
import { listUserCredits, upsertUserCredits } from "@/lib/db/queries";
import { ChatbotError } from "@/lib/errors";

// Admin-only. GET returns every row in UserCredits (capped at 200 in
// the query layer) so the /admin page can show a "users with overrides"
// table without a per-row fetch. POST upserts one row by userId — the
// admin UI fills userId from the search endpoint above.
export async function GET() {
  const ctx = await getAdminContext();
  if (!ctx) {
    return new ChatbotError("forbidden:auth").toResponse();
  }
  const rows = await listUserCredits();
  return Response.json({ credits: rows });
}

const upsertSchema = z.object({
  userId: z.string().min(1).max(255),
  email: z.string().email().max(320).nullable().optional(),
  // Negative values are nonsense (the LIMIT is already the floor) so
  // clamp the lower bound here. Upper bound is loose — admin already
  // has full DB access if they need to grant a million credits.
  bonusChat: z.number().int().min(0).max(100_000),
  bonusDeepSearch: z.number().int().min(0).max(100_000),
  notes: z.string().max(2000).nullable().optional(),
});

export async function POST(request: Request) {
  const ctx = await getAdminContext();
  if (!ctx) {
    return new ChatbotError("forbidden:auth").toResponse();
  }

  let body: z.infer<typeof upsertSchema>;
  try {
    body = upsertSchema.parse(await request.json());
  } catch {
    return new ChatbotError(
      "bad_request:api",
      "invalid request body"
    ).toResponse();
  }

  const row = await upsertUserCredits({
    userId: body.userId,
    email: body.email ?? null,
    bonusChat: body.bonusChat,
    bonusDeepSearch: body.bonusDeepSearch,
    notes: body.notes ?? null,
  });

  return Response.json({ credits: row });
}
