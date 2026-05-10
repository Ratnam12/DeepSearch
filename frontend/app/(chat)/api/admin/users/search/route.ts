import { clerkClient } from "@clerk/nextjs/server";
import { getAdminContext } from "@/lib/admin";
import {
  FREE_CHAT_MESSAGE_LIMIT,
  FREE_DEEPSEARCH_LIMIT,
} from "@/lib/constants";
import { getLifetimeUserMessageCount, getUserCredits } from "@/lib/db/queries";
import { countResearchRunsByUserId } from "@/lib/db/queries-research";
import { ChatbotError } from "@/lib/errors";

// Admin lookup. Given a (case-insensitive) email, finds the matching
// Clerk user, then joins in the user's lifetime usage + any existing
// bonus credits — one round-trip from the admin UI is enough to render
// "Alice Smith · 18/20 chat · 2/2 deepsearch · +5 chat bonus".
//
// Returns null `user` when no Clerk account matches; the admin UI shows
// a "user hasn't signed up yet" hint in that case.
export async function GET(request: Request) {
  const ctx = await getAdminContext();
  if (!ctx) {
    return new ChatbotError("forbidden:auth").toResponse();
  }

  const { searchParams } = new URL(request.url);
  const rawEmail = searchParams.get("email")?.trim();
  if (!rawEmail) {
    return new ChatbotError(
      "bad_request:api",
      "email query param required"
    ).toResponse();
  }
  const email = rawEmail.toLowerCase();

  const client = await clerkClient();
  const list = await client.users.getUserList({
    emailAddress: [email],
    limit: 5,
  });

  // Clerk's emailAddress filter is exact-match but not case-insensitive
  // in every region — defensive check against the lowercased local copy.
  const match = list.data.find((u) =>
    u.emailAddresses.some((e) => e.emailAddress.toLowerCase() === email)
  );

  if (!match) {
    return Response.json({ user: null });
  }

  const [chatUsed, deepSearchUsed, credits] = await Promise.all([
    getLifetimeUserMessageCount({ userId: match.id }),
    countResearchRunsByUserId({ userId: match.id }),
    getUserCredits({ userId: match.id }),
  ]);

  return Response.json({
    user: {
      userId: match.id,
      email:
        match.emailAddresses.find((e) => e.id === match.primaryEmailAddressId)
          ?.emailAddress ??
        match.emailAddresses[0]?.emailAddress ??
        email,
      fullName:
        [match.firstName, match.lastName].filter(Boolean).join(" ") || null,
      imageUrl: match.imageUrl,
      createdAt: match.createdAt,
      usage: {
        chat: {
          used: chatUsed,
          baseLimit: FREE_CHAT_MESSAGE_LIMIT,
          bonus: credits?.bonusChat ?? 0,
        },
        deepSearch: {
          used: deepSearchUsed,
          baseLimit: FREE_DEEPSEARCH_LIMIT,
          bonus: credits?.bonusDeepSearch ?? 0,
        },
      },
      notes: credits?.notes ?? null,
      lastUpdatedAt: credits?.updatedAt ?? null,
    },
  });
}
