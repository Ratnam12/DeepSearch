import { auth } from "@clerk/nextjs/server";
import { ipAddress } from "@vercel/functions";
import { convertToModelMessages } from "ai";
import { after } from "next/server";
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
import { generateTitleFromUserMessage } from "../../actions";
import { type PostRequestBody, postRequestBodySchema } from "./schema";

// 5 minutes — the Vercel Hobby + Fluid Compute ceiling. DeepSearch's
// agent loop (web_search -> Playwright scrape -> retrieve_chunks, often
// looped 2-3x) can take 60-180s for complex multi-hop queries. The
// chatbot template hardcoded `maxDuration = 60` against the pre-Fluid
// Hobby cap; we get the full 300s for free.
export const maxDuration = 300;

const BACKEND_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/+$/, "") ??
  "http://localhost:8000";

// Pinned slightly under maxDuration so we surface a clean offline:chat
// error to the client instead of letting Vercel kill the function with
// no response.
const BACKEND_TIMEOUT_MS = 290_000;

export async function POST(request: Request) {
  let requestBody: PostRequestBody;
  try {
    requestBody = postRequestBodySchema.parse(await request.json());
  } catch {
    return new ChatbotError("bad_request:api").toResponse();
  }

  const { id, message, messages, selectedChatModel, selectedVisibilityType } =
    requestBody;

  const { userId } = await auth();
  if (!userId) {
    return new ChatbotError("unauthorized:chat").toResponse();
  }

  await checkIpRateLimit(ipAddress(request));

  // ── Load (or create) the chat row ──────────────────────────────────────
  const existingChat = await getChatById({ id });
  let messagesFromDb: DBMessage[] = [];
  let titleToBroadcast: string | null = null;

  if (existingChat) {
    if (existingChat.userId !== userId) {
      return new ChatbotError("forbidden:chat").toResponse();
    }
    messagesFromDb = await getMessagesByChatId({ id });
  } else if (message?.role === "user") {
    // Title comes from a deterministic truncation of the user's first
    // message. We compute it before the stream so the sidebar updates in
    // the same response cycle.
    titleToBroadcast = await generateTitleFromUserMessage({ message });
    await saveChat({
      id,
      userId,
      title: titleToBroadcast,
      visibility: selectedVisibilityType,
    });
  } else {
    return new ChatbotError("not_found:chat").toResponse();
  }

  // ── Build the full UIMessages array we'll send to the model ────────────
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
  const modelMessages = await convertToModelMessages(uiMessages);

  let backendResponse: Response;
  try {
    backendResponse = await fetch(`${BACKEND_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        id,
        messages: modelMessages,
        // Forward the user's model pick so the FastAPI agent can override
        // its default complexity-based router. The backend treats this as
        // an advisory hint — unknown model ids fall back to the default.
        model: selectedChatModel,
      }),
      signal: AbortSignal.timeout(BACKEND_TIMEOUT_MS),
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

  // ── Direct stream pass-through with a tee for persistence ──────────────
  // FastAPI already emits the AI SDK UI Message Stream Protocol verbatim
  // (see backend/main.py::_ui_message_stream), so the simplest correct
  // proxy is to forward the body bytes unchanged. Wrapping it in
  // createUIMessageStream forced us to await the full body before flushing
  // — fine for short replies, fatal for the 30-50s research loop. The tee
  // duplicates the body so we can persist artifacts and the assistant
  // message in the background without slowing the user-facing stream.
  const [forwardStream, observerStream] = backendResponse.body.tee();

  const persistencePromise = persistFromStream({
    stream: observerStream,
    chatId: id,
    userId,
    knownMessageIds: new Set(uiMessages.map((m) => m.id).filter(Boolean)),
    titleToBroadcast,
  });
  // Don't block the response on the DB writes — Vercel keeps the function
  // alive for the after() handlers until they resolve.
  after(persistencePromise);

  return new Response(forwardStream, {
    headers: {
      "content-type": "text/event-stream",
      "x-vercel-ai-ui-message-stream": "v1",
      "cache-control": "no-cache, no-transform",
      "x-accel-buffering": "no",
      connection: "keep-alive",
    },
  });
}

// Loose shape since we read from JSON. We runtime-check fields before use
// rather than relying on the union narrowing — there's no upstream
// guarantee about what FastAPI emits beyond `type` being a string.
type StreamPart = {
  type: string;
  messageId?: string;
  id?: string;
  delta?: string;
  toolCallId?: string;
  toolName?: string;
  input?: unknown;
  output?: unknown;
  data?: {
    id?: string;
    kind?: "text" | "code" | "sheet";
    title?: string;
    content?: string;
  };
  errorText?: string;
};

// Walks the SSE stream the FastAPI agent is producing, accumulates the
// assistant message's parts, persists artifacts as they arrive, and saves
// the final assistant message on completion. Runs entirely in the
// background via after() so the user-facing stream is never blocked on a
// database round-trip.
async function persistFromStream({
  stream,
  chatId,
  userId,
  knownMessageIds,
  titleToBroadcast,
}: {
  stream: ReadableStream<Uint8Array>;
  chatId: string;
  userId: string;
  knownMessageIds: Set<string>;
  titleToBroadcast: string | null;
}) {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  let assistantMessageId: string | null = null;
  const assistantParts: Array<Record<string, unknown>> = [];
  // Group text-delta events into a single text part keyed by the
  // text-start id so the persisted message reads like one paragraph
  // instead of one part per token.
  const textBuffers = new Map<string, string>();
  // Same for streaming tool calls — pair input + output by toolCallId.
  const toolInputs = new Map<
    string,
    { toolName: string; input: unknown; output?: unknown }
  >();

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      buffer += decoder.decode(value, { stream: true });
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

        let part: StreamPart;
        try {
          part = JSON.parse(payload) as StreamPart;
        } catch {
          continue;
        }

        switch (part.type) {
          case "start":
            if (typeof part.messageId === "string") {
              assistantMessageId = part.messageId;
            }
            break;
          case "text-start":
            if (typeof part.id === "string") {
              textBuffers.set(part.id, "");
            }
            break;
          case "text-delta":
            if (typeof part.id === "string" && typeof part.delta === "string") {
              textBuffers.set(
                part.id,
                (textBuffers.get(part.id) ?? "") + part.delta
              );
            }
            break;
          case "text-end": {
            if (typeof part.id !== "string") {
              break;
            }
            const text = textBuffers.get(part.id) ?? "";
            if (text) {
              assistantParts.push({ type: "text", text });
            }
            textBuffers.delete(part.id);
            break;
          }
          case "tool-input-available":
            if (
              typeof part.toolCallId === "string" &&
              typeof part.toolName === "string"
            ) {
              toolInputs.set(part.toolCallId, {
                toolName: part.toolName,
                input: part.input,
              });
            }
            break;
          case "tool-output-available": {
            if (typeof part.toolCallId !== "string") {
              break;
            }
            const existing = toolInputs.get(part.toolCallId);
            if (existing) {
              existing.output = part.output;
              assistantParts.push({
                type: `tool-${existing.toolName}`,
                toolCallId: part.toolCallId,
                state: "output-available",
                input: existing.input,
                output: part.output,
              });
            }
            break;
          }
          case "data-artifact": {
            const artifact = part.data;
            if (
              artifact &&
              typeof artifact.id === "string" &&
              typeof artifact.title === "string" &&
              typeof artifact.content === "string" &&
              (artifact.kind === "text" ||
                artifact.kind === "code" ||
                artifact.kind === "sheet")
            ) {
              try {
                await saveDocument({
                  id: artifact.id,
                  title: artifact.title,
                  kind: artifact.kind,
                  content: artifact.content,
                  userId,
                });
              } catch (err) {
                console.error("Failed to persist artifact", {
                  id: artifact.id,
                  err,
                });
              }
              assistantParts.push({
                type: "data-artifact",
                id: artifact.id,
                data: artifact,
              });
            }
            break;
          }
          default:
            // Forward-compat: ignore anything we don't recognise.
            break;
        }
      }
    }
  } catch (err) {
    console.error("Failed reading FastAPI stream for persistence", {
      chatId,
      err,
    });
  } finally {
    reader.releaseLock();
  }

  // Flush any text that didn't get an explicit text-end (the agent loop
  // can be cut short by the Vercel timeout — better to save what we have
  // than nothing).
  for (const [, text] of textBuffers) {
    if (text) {
      assistantParts.push({ type: "text", text });
    }
  }

  if (assistantParts.length === 0) {
    return;
  }

  // Use a fresh UUID for the DB row regardless of what FastAPI sent in
  // the `start` event — FastAPI emits `msg_<hex>` (AI SDK convention)
  // which isn't valid for the Postgres `uuid` column on Message_v2 and
  // would fail the insert with a 22P02 invalid input syntax error.
  // (`assistantMessageId` and `knownMessageIds` are still useful for
  // future client-side dedup, but the DB doesn't need them.)
  void assistantMessageId;
  void knownMessageIds;
  const finalId = generateUUID();

  try {
    await saveMessages({
      messages: [
        {
          id: finalId,
          chatId,
          role: "assistant",
          parts: assistantParts as DBMessage["parts"],
          attachments: [],
          createdAt: new Date(),
        },
      ],
    });
  } catch (err) {
    console.error("Failed to persist assistant message", { chatId, err });
  }

  if (titleToBroadcast) {
    try {
      await updateChatTitleById({ chatId, title: titleToBroadcast });
    } catch (err) {
      console.error("Failed to update chat title", { chatId, err });
    }
  }
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
