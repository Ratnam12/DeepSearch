// Phase 2 stub. Phase 3 rewrites this as the Clerk-auth proxy that forwards
// UI messages to the FastAPI /chat endpoint and pipes the AI SDK UI Message
// Stream Protocol response straight back to the `useChat` hook on the client.

import { auth } from "@clerk/nextjs/server";
import { deleteChatById, getChatById } from "@/lib/db/queries";
import { ChatbotError } from "@/lib/errors";

export const maxDuration = 60;

export async function POST() {
  return new ChatbotError(
    "offline:chat",
    "Chat is being rewired for the new backend. Try again shortly."
  ).toResponse();
}

export async function DELETE(request: Request) {
  const { searchParams } = new URL(request.url);
  const id = searchParams.get("id");

  if (!id) {
    return new ChatbotError("bad_request:api").toResponse();
  }

  const { userId } = await auth();

  if (!userId) {
    return new ChatbotError("unauthorized:chat").toResponse();
  }

  const chat = await getChatById({ id });

  if (chat?.userId !== userId) {
    return new ChatbotError("forbidden:chat").toResponse();
  }

  const deletedChat = await deleteChatById({ id });

  return Response.json(deletedChat, { status: 200 });
}
