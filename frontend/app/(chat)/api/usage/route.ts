import { auth } from "@clerk/nextjs/server";
import {
  FREE_CHAT_MESSAGE_LIMIT,
  FREE_DEEPSEARCH_LIMIT,
} from "@/lib/constants";
import { getLifetimeUserMessageCount, getUserCredits } from "@/lib/db/queries";
import { countResearchRunsByUserId } from "@/lib/db/queries-research";

// Read-only quota status for the LimitReachedDialog. The composer SWRs
// this on focus + after every send so the modal can fire as soon as the
// next attempt would tip the user over the cap. Server-side enforcement
// still lives in /api/chat so a tampered client can't bypass.
//
// Effective limit = FREE_*_LIMIT + UserCredits.bonus*. The admin can
// raise either bucket per-user from /admin without redeploying.
export async function GET() {
  const { userId } = await auth();
  if (!userId) {
    return Response.json(
      {
        signedIn: false,
        chat: { used: 0, limit: FREE_CHAT_MESSAGE_LIMIT },
        deepSearch: { used: 0, limit: FREE_DEEPSEARCH_LIMIT },
      },
      { status: 200 }
    );
  }

  const [chatUsed, deepSearchUsed, credits] = await Promise.all([
    getLifetimeUserMessageCount({ userId }),
    countResearchRunsByUserId({ userId }),
    getUserCredits({ userId }),
  ]);

  const chatLimit = FREE_CHAT_MESSAGE_LIMIT + (credits?.bonusChat ?? 0);
  const deepSearchLimit =
    FREE_DEEPSEARCH_LIMIT + (credits?.bonusDeepSearch ?? 0);

  return Response.json({
    signedIn: true,
    chat: { used: chatUsed, limit: chatLimit },
    deepSearch: { used: deepSearchUsed, limit: deepSearchLimit },
  });
}
