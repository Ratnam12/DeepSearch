import { auth } from "@clerk/nextjs/server";
import { ipAddress } from "@vercel/functions";
import {
  convertToModelMessages,
  createUIMessageStream,
  createUIMessageStreamResponse,
} from "ai";
import { after } from "next/server";
import { generateTitleFromUserMessage } from "../../actions";
import {
  deleteChatById,
  getChatById,
  getMessagesByChatId,
  saveChat,
  saveDocument,
  saveMessages,
  updateChatTitleById,
} from "@/lib/db/queries";
import type { DBMessage } from "@/lib/db/schema";
import { ChatbotError } from "@/lib/errors";
import { checkIpRateLimit } from "@/lib/ratelimit";
import type { ChatMessage } from "@/lib/types";
import { convertToUIMessages, generateUUID } from "@/lib/utils";
import { type PostRequestBody, postRequestBodySchema } from "./schema";

export const maxDuration = 60;

const BACKEND_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/+$/, "") ??
  "http://localhost:8000";

// Forward UI messages to the FastAPI agent and pipe the AI SDK UI Message
// Stream Protocol response back to the client. The FastAPI side already
// emits protocol-compliant parts (see backend/main.py::_ui_message_stream),
// so we only need to:
//   1. Authenticate via Clerk and rate-limit
//   2. Persist the chat + user message before forwarding (so retries don't
//      lose context if the backend errors out mid-stream)
//   3. Translate AI SDK UIMessages -> OpenAI-format messages
//      (convertToModelMessages handles multimodal file parts -> image_url)
//   4. Re-emit each FastAPI part through createUIMessageStream so the
//      onFinish hook receives a structured assistant message we can save
export async function POST(request: Request) {
  let requestBody: PostRequestBody;
  try {
    requestBody = postRequestBodySchema.parse(await request.json());
  } catch {
    return new ChatbotError("bad_request:api").toResponse();
  }

  const { id, message, messages, selectedVisibilityType } = requestBody;

  const { userId } = await auth();
  if (!userId) {
    return new ChatbotError("unauthorized:chat").toResponse();
  }

  await checkIpRateLimit(ipAddress(request));

  // ── Load (or create) the chat row ──────────────────────────────────────
  const existingChat = await getChatById({ id });
  let messagesFromDb: DBMessage[] = [];
  let titlePromise: Promise<string> | null = null;

  if (existingChat) {
    if (existingChat.userId !== userId) {
      return new ChatbotError("forbidden:chat").toResponse();
    }
    messagesFromDb = await getMessagesByChatId({ id });
  } else if (message?.role === "user") {
    await saveChat({
      id,
      userId,
      title: "New chat",
      visibility: selectedVisibilityType,
    });
    titlePromise = generateTitleFromUserMessage({ message });
  } else {
    return new ChatbotError("not_found:chat").toResponse();
  }

  // ── Build the full UIMessages array we'll send to the model ────────────
  // For the tool-approval continuation case the frontend sends `messages`
  // (the entire history with approval responses applied). DeepSearch tools
  // auto-execute, so this branch is currently unused — we still honour the
  // shape so the schema stays in sync with the frontend.
  const isToolApprovalFlow = Boolean(messages);
  const uiMessages: ChatMessage[] = isToolApprovalFlow
    ? (messages as ChatMessage[])
    : [...convertToUIMessages(messagesFromDb), message as ChatMessage];

  // ── Persist the new user message immediately ───────────────────────────
  if (message?.role === "user" && !isToolApprovalFlow) {
    await saveMessages({
      messages: [
        {
          id: message.id,
          chatId: id,
          role: "user",
          parts: message.parts,
          attachments: [],
          createdAt: new Date(),
        },
      ],
    });
  }

  // ── Translate to OpenAI-format messages and forward ────────────────────
  // AI SDK's convertToModelMessages handles multimodal file parts (Vercel
  // Blob URLs from /api/files/upload) by converting them to OpenAI's
  // image_url shape. The FastAPI agent passes them through to OpenRouter.
  const modelMessages = await convertToModelMessages(uiMessages);

  let backendResponse: Response;
  try {
    backendResponse = await fetch(`${BACKEND_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id, messages: modelMessages }),
    });
  } catch (err) {
    console.error("FastAPI fetch failed", { id, err });
    return new ChatbotError("offline:chat").toResponse();
  }

  if (!backendResponse.ok || !backendResponse.body) {
    console.error("FastAPI returned non-OK", {
      id,
      status: backendResponse.status,
    });
    return new ChatbotError("offline:chat").toResponse();
  }

  // ── Wrap the FastAPI body in createUIMessageStream so onFinish runs ────
  const stream = createUIMessageStream<ChatMessage>({
    originalMessages: uiMessages,
    execute: async ({ writer }) => {
      const reader = backendResponse.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) {
            break;
          }
          buffer += decoder.decode(value, { stream: true });

          // SSE events are separated by blank lines (\n\n).
          const events = buffer.split("\n\n");
          buffer = events.pop() ?? "";

          for (const event of events) {
            const trimmed = event.trim();
            if (!trimmed.startsWith("data:")) {
              continue;
            }
            const payload = trimmed.slice(5).trim();
            if (payload === "[DONE]" || payload.length === 0) {
              continue;
            }
            try {
              const part = JSON.parse(payload);
              writer.write(part);

              // Persist artifacts to the documents table so they show up
              // in the user's history alongside chats. The frontend
              // DataStreamHandler renders them in the side panel from the
              // same `data-artifact` part.
              if (
                part?.type === "data-artifact" &&
                part?.data &&
                typeof part.data.id === "string"
              ) {
                const artifact = part.data as {
                  id: string;
                  kind: "text" | "code" | "sheet";
                  title: string;
                  content: string;
                };
                after(
                  saveDocument({
                    id: artifact.id,
                    title: artifact.title,
                    kind: artifact.kind,
                    content: artifact.content,
                    userId,
                  }).catch((err) =>
                    console.error("Failed to persist artifact", {
                      id: artifact.id,
                      err,
                    })
                  )
                );
              }
            } catch {
              console.warn("Skipping unparseable stream part", { payload });
            }
          }
        }
      } finally {
        reader.releaseLock();
      }

      // Generate and broadcast the chat title for newly-created chats.
      if (titlePromise) {
        try {
          const title = await titlePromise;
          writer.write({ type: "data-chat-title", data: title });
          // Don't block the stream on the DB write.
          after(updateChatTitleById({ chatId: id, title }));
        } catch (err) {
          console.warn("Title generation failed", { id, err });
        }
      }
    },
    generateId: generateUUID,
    onFinish: async ({ messages: finishedMessages }) => {
      // finishedMessages is the full conversation. Save anything not
      // already persisted — we already wrote the user message above, so
      // in practice this is the assistant turn(s) generated by FastAPI.
      const previouslySavedIds = new Set(
        uiMessages.map((m) => m.id).filter(Boolean)
      );
      const newMessages = finishedMessages.filter(
        (m) => !previouslySavedIds.has(m.id)
      );

      if (newMessages.length === 0) {
        return;
      }

      try {
        await saveMessages({
          messages: newMessages.map((m) => ({
            id: m.id,
            chatId: id,
            role: m.role,
            parts: m.parts,
            attachments: [],
            createdAt: new Date(),
          })),
        });
      } catch (err) {
        console.error("Failed to persist assistant messages", { id, err });
      }
    },
    onError: (error) => {
      console.error("Stream error", { id, error });
      return "Something went wrong while streaming the response.";
    },
  });

  return createUIMessageStreamResponse({ stream });
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
