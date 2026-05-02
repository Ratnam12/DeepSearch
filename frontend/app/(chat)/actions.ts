"use server";

import { auth } from "@clerk/nextjs/server";
import type { UIMessage } from "ai";
import { cookies } from "next/headers";
import type { VisibilityType } from "@/components/chat/visibility-selector";
import {
  deleteMessagesByChatIdAfterTimestamp,
  getChatById,
  getMessageById,
  updateChatVisibilityById,
} from "@/lib/db/queries";
import { getTextFromMessage } from "@/lib/utils";

const TITLE_MAX_CHARS = 60;

export async function saveChatModelAsCookie(model: string) {
  const cookieStore = await cookies();
  cookieStore.set("chat-model", model);
}

export async function generateTitleFromUserMessage({
  message,
}: {
  message: UIMessage;
}): Promise<string> {
  // Title generation deliberately does not call any LLM — DeepSearch routes
  // every model call through the FastAPI/OpenRouter backend, so reaching
  // for Vercel AI Gateway here would require a separate billing setup we
  // don't need. A truncated copy of the first user message is a perfectly
  // good chat title and matches what most chat UIs do as their initial
  // label until a human renames the thread.
  const text = getTextFromMessage(message).trim().replace(/\s+/g, " ");
  if (!text) {
    return "New chat";
  }
  if (text.length <= TITLE_MAX_CHARS) {
    return text;
  }
  // Try to break on the nearest word boundary.
  const truncated = text.slice(0, TITLE_MAX_CHARS);
  const lastSpace = truncated.lastIndexOf(" ");
  const head = lastSpace > 30 ? truncated.slice(0, lastSpace) : truncated;
  return `${head}…`;
}

export async function deleteTrailingMessages({ id }: { id: string }) {
  const { userId } = await auth();
  if (!userId) {
    throw new Error("Unauthorized");
  }

  const [message] = await getMessageById({ id });
  if (!message) {
    throw new Error("Message not found");
  }

  const chat = await getChatById({ id: message.chatId });
  if (!chat || chat.userId !== userId) {
    throw new Error("Unauthorized");
  }

  await deleteMessagesByChatIdAfterTimestamp({
    chatId: message.chatId,
    timestamp: message.createdAt,
  });
}

export async function updateChatVisibility({
  chatId,
  visibility,
}: {
  chatId: string;
  visibility: VisibilityType;
}) {
  const { userId } = await auth();
  if (!userId) {
    throw new Error("Unauthorized");
  }

  const chat = await getChatById({ id: chatId });
  if (!chat || chat.userId !== userId) {
    throw new Error("Unauthorized");
  }

  await updateChatVisibilityById({ chatId, visibility });
}
