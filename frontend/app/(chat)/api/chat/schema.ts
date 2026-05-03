import { z } from "zod";

const textPartSchema = z.object({
  type: z.enum(["text"]),
  // Research queries often include pasted abstracts, code, or
  // multi-paragraph context — the chatbot template's 2000-char cap
  // (~300 words) was tuned for casual chat. Bumped to ~5000 words.
  text: z.string().min(1).max(32_000),
});

const filePartSchema = z.object({
  type: z.enum(["file"]),
  mediaType: z.enum([
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/heic",
    "image/heif",
    "application/pdf",
  ]),
  displayUrl: z.string().min(1).optional(),
  filename: z.string().min(1).max(255).optional(),
  name: z.string().min(1).max(255).optional(),
  pageCount: z.number().int().positive().optional(),
  pathname: z.string().min(1).max(1024).optional(),
  url: z.string().url(),
}).refine((part) => part.filename || part.name, {
  message: "File part must include a filename",
});

const partSchema = z.union([textPartSchema, filePartSchema]);

const userMessageSchema = z.object({
  id: z.string().uuid(),
  role: z.enum(["user"]),
  parts: z.array(partSchema),
});

const toolApprovalMessageSchema = z.object({
  id: z.string(),
  role: z.enum(["user", "assistant"]),
  parts: z.array(z.record(z.unknown())),
});

export const postRequestBodySchema = z.object({
  id: z.string().uuid(),
  message: userMessageSchema.optional(),
  messages: z.array(toolApprovalMessageSchema).optional(),
  selectedChatModel: z.string(),
  selectedVisibilityType: z.enum(["public", "private"]),
  // When the user submits with the DeepSearch toggle on we fork the
  // request into a research-run flow: /api/chat creates the run row
  // synchronously, emits a single ``data-research`` part with the
  // runId, persists the assistant message, and returns. The actual
  // research happens in the background worker; the chat artifact
  // card subscribes to the run's SSE stream from the client.
  deepSearch: z.boolean().optional(),
});

export type PostRequestBody = z.infer<typeof postRequestBodySchema>;
