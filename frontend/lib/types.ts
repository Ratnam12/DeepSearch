import type { InferUITool, UIMessage } from "ai";
import { z } from "zod";
import type { ArtifactKind } from "@/components/chat/artifact";
import type { createDocument } from "./ai/tools/create-document";
import type { editDocument } from "./ai/tools/edit-document";
import type { updateDocument } from "./ai/tools/update-document";
import type { Suggestion } from "./db/schema";

export const messageMetadataSchema = z.object({
  createdAt: z.string(),
});

export type MessageMetadata = z.infer<typeof messageMetadataSchema>;

type createDocumentTool = InferUITool<ReturnType<typeof createDocument>>;
type editDocumentTool = InferUITool<ReturnType<typeof editDocument>>;
type updateDocumentTool = InferUITool<ReturnType<typeof updateDocument>>;

// DeepSearch tools live on the FastAPI backend; the Next.js proxy forwards
// them as part-streamed events, but TypeScript only needs to know about the
// tools whose UI bridges still live in this codebase (the document handlers
// for artifacts).
export type ChatTools = {
  createDocument: createDocumentTool;
  editDocument: editDocumentTool;
  updateDocument: updateDocumentTool;
};

export type CustomUIDataTypes = {
  textDelta: string;
  imageDelta: string;
  sheetDelta: string;
  codeDelta: string;
  suggestion: Suggestion;
  appendMessage: string;
  id: string;
  title: string;
  kind: ArtifactKind;
  clear: null;
  finish: null;
  "chat-title": string;
  // DeepSearch artifact payload from the FastAPI `create_artifact` tool —
  // surfaced to the UI as a `data-artifact` part by the chat proxy.
  artifact: {
    id: string;
    kind: "text" | "code" | "sheet";
    title: string;
    content: string;
  };
  // [N] → source URL mapping emitted after each retrieve_chunks tool
  // call. The frontend rewrites bracketed citations in the assistant's
  // streamed answer into clickable links pointing to the source.
  citations: Array<{
    index: number;
    url: string;
    score?: number;
  }>;
};

export type ChatMessage = UIMessage<
  MessageMetadata,
  CustomUIDataTypes,
  ChatTools
>;

export type Attachment = {
  name: string;
  url: string;
  contentType: string;
  pathname?: string;
  modelUrl?: string;
  pageCount?: number;
};
