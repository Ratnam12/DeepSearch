import { auth } from "@clerk/nextjs/server";

import { forkChat } from "@/lib/db/queries";
import { ChatbotError } from "@/lib/errors";

export async function POST(request: Request): Promise<Response> {
  const { userId } = await auth();
  if (!userId) {
    return new ChatbotError("unauthorized:chat").toResponse();
  }

  let sourceChatId: string;
  try {
    const body = await request.json();
    if (typeof body?.sourceChatId !== "string" || !body.sourceChatId) {
      return new ChatbotError("bad_request:api").toResponse();
    }
    sourceChatId = body.sourceChatId;
  } catch {
    return new ChatbotError("bad_request:api").toResponse();
  }

  try {
    const chatId = await forkChat({ sourceChatId, userId });
    return Response.json({ chatId });
  } catch (err) {
    if (err instanceof ChatbotError) {
      return err.toResponse();
    }
    return new ChatbotError("offline:chat").toResponse();
  }
}
