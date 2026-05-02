"use client";
import type { UseChatHelpers } from "@ai-sdk/react";
import type { Vote } from "@/lib/db/schema";
import type { ChatMessage } from "@/lib/types";
import { cn, sanitizeText } from "@/lib/utils";
import { MessageContent, MessageResponse } from "../ai-elements/message";
import { Shimmer } from "../ai-elements/shimmer";
import { useDataStream } from "./data-stream-provider";
import { DeepSearchArtifactCard } from "./deepsearch-artifact-card";
import { DeepSearchMark } from "./deepsearch-mark";
import { DeepSearchToolGroup } from "./deepsearch-tool-group";
import type { DeepSearchToolPart } from "./deepsearch-tool-step";
import { DocumentPreview } from "./document-preview";
import { MessageActions } from "./message-actions";
import { MessageReasoning } from "./message-reasoning";
import { PreviewAttachment } from "./preview-attachment";

// Tool parts that get the special grouped/collapsible treatment. The
// document tools (createDocument / updateDocument) keep their existing
// inline-card UI since those are the actual artifact previews, not
// research steps.
const STEP_TOOL_TYPES = new Set([
  "tool-web_search",
  "tool-retrieve_chunks",
  "tool-scrape_and_index",
  "tool-create_artifact",
]);

function isStepTool(part: { type: string }): boolean {
  if (part.type === "dynamic-tool") {
    return true;
  }
  if (STEP_TOOL_TYPES.has(part.type)) {
    return true;
  }
  // Any unknown `tool-*` that isn't one of our document tools is treated
  // as a research step too — we'd rather group too eagerly than show
  // raw JSON for a tool the UI doesn't have a custom card for.
  if (
    part.type.startsWith("tool-") &&
    part.type !== "tool-createDocument" &&
    part.type !== "tool-updateDocument" &&
    part.type !== "tool-editDocument"
  ) {
    return true;
  }
  return false;
}

// Rewrites bracketed citations (`[1]`, `[2]`) in markdown text into
// clickable links pointing at their source URL. The mapping comes from
// `data-citations` parts emitted after each retrieve_chunks tool call.
// Markdown link text uses escaped square brackets so streamdown still
// renders the visible label as `[N]`.
function linkifyCitations(
  text: string,
  citations: Map<number, string>
): string {
  if (citations.size === 0) {
    return text;
  }
  return text.replace(/\[(\d+)\]/g, (match, idxStr) => {
    const idx = Number.parseInt(idxStr, 10);
    const url = citations.get(idx);
    return url ? `[\\[${idx}\\]](${url})` : match;
  });
}

function collectCitations(
  parts: ReadonlyArray<{ type: string; data?: unknown }>
): Map<number, string> {
  const map = new Map<number, string>();
  for (const part of parts) {
    if (part.type !== "data-citations" || !Array.isArray(part.data)) {
      continue;
    }
    for (const item of part.data as Array<{
      index?: number;
      url?: string;
    }>) {
      if (
        typeof item.index === "number" &&
        typeof item.url === "string" &&
        item.url
      ) {
        map.set(item.index, item.url);
      }
    }
  }
  return map;
}

const PurePreviewMessage = ({
  addToolApprovalResponse: _addToolApprovalResponse,
  chatId,
  message,
  vote,
  isLoading,
  setMessages: _setMessages,
  regenerate: _regenerate,
  isReadonly,
  requiresScrollPadding: _requiresScrollPadding,
  onEdit,
}: {
  addToolApprovalResponse: UseChatHelpers<ChatMessage>["addToolApprovalResponse"];
  chatId: string;
  message: ChatMessage;
  vote: Vote | undefined;
  isLoading: boolean;
  setMessages: UseChatHelpers<ChatMessage>["setMessages"];
  regenerate: UseChatHelpers<ChatMessage>["regenerate"];
  isReadonly: boolean;
  requiresScrollPadding: boolean;
  onEdit?: (message: ChatMessage) => void;
}) => {
  const attachmentsFromMessage = message.parts.filter(
    (part) => part.type === "file"
  );

  useDataStream();

  const isUser = message.role === "user";
  const isAssistant = message.role === "assistant";

  const hasAnyContent = message.parts?.some(
    (part) =>
      (part.type === "text" && part.text?.trim().length > 0) ||
      (part.type === "reasoning" &&
        "text" in part &&
        part.text?.trim().length > 0) ||
      part.type.startsWith("tool-")
  );
  const isThinking = isAssistant && isLoading && !hasAnyContent;

  const attachments = attachmentsFromMessage.length > 0 && (
    <div
      className="flex flex-row justify-end gap-2"
      data-testid={"message-attachments"}
    >
      {attachmentsFromMessage.map((attachment) => (
        <PreviewAttachment
          attachment={{
            name:
              attachment.filename ??
              ("name" in attachment && typeof attachment.name === "string"
                ? attachment.name
                : "file"),
            contentType: attachment.mediaType,
            pageCount:
              "pageCount" in attachment &&
              typeof attachment.pageCount === "number"
                ? attachment.pageCount
                : undefined,
            url:
              "displayUrl" in attachment &&
              typeof attachment.displayUrl === "string"
                ? attachment.displayUrl
                : attachment.url,
          }}
          key={attachment.url}
        />
      ))}
    </div>
  );

  const mergedReasoning = message.parts?.reduce(
    (acc, part) => {
      if (part.type === "reasoning" && part.text?.trim().length > 0) {
        return {
          text: acc.text ? `${acc.text}\n\n${part.text}` : part.text,
          isStreaming: "state" in part ? part.state === "streaming" : false,
          rendered: false,
        };
      }
      return acc;
    },
    { text: "", isStreaming: false, rendered: false }
  ) ?? { text: "", isStreaming: false, rendered: false };

  // Pre-pass: group consecutive research-step tool parts into a single
  // collapsible block. Whether the block defaults to expanded depends on
  // whether any text part appears later in the message — once the answer
  // starts streaming, the steps fold up to a one-line summary so the
  // answer is the focal point.
  //
  // We pre-filter `data-citations` (consumed only as the [N]→URL map for
  // linkifyCitations) and `data-artifact` (rendered once at the end of
  // the message) so they don't break consecutive-tool runs into many
  // small groups.
  const allParts = message.parts ?? [];
  const citations = collectCitations(allParts);
  const artifactParts = allParts.filter((p) => p.type === "data-artifact");
  const rawParts = allParts.filter(
    (p) => p.type !== "data-citations" && p.type !== "data-artifact"
  );
  type RenderItem =
    | { kind: "single"; part: ChatMessage["parts"][number]; index: number }
    | {
        kind: "tool-group";
        tools: DeepSearchToolPart[];
        firstIndex: number;
        collapsed: boolean;
      };

  const renderItems: RenderItem[] = [];
  for (let i = 0; i < rawParts.length; i += 1) {
    const part = rawParts[i];
    if (isStepTool(part)) {
      const groupStart = i;
      const groupTools: DeepSearchToolPart[] = [];
      while (i < rawParts.length && isStepTool(rawParts[i])) {
        groupTools.push(rawParts[i] as DeepSearchToolPart);
        i += 1;
      }
      // Look ahead to see if there's substantive text content following
      // this group anywhere in the message — that triggers the auto-
      // collapse so the answer takes focus.
      const collapsed = rawParts
        .slice(i)
        .some(
          (p) =>
            (p.type === "text" && p.text?.trim().length > 0) ||
            (p.type === "reasoning" &&
              "text" in p &&
              (p as { text?: string }).text?.trim().length)
        );
      renderItems.push({
        kind: "tool-group",
        tools: groupTools,
        firstIndex: groupStart,
        collapsed,
      });
      i -= 1; // outer loop will i += 1
      continue;
    }
    renderItems.push({ kind: "single", part, index: i });
  }

  const parts = renderItems.map((item) => {
    if (item.kind === "tool-group") {
      const firstId =
        (item.tools[0] as { toolCallId?: string }).toolCallId ??
        `${message.id}-group-${item.firstIndex}`;
      return (
        <DeepSearchToolGroup
          collapsed={item.collapsed}
          key={`tool-group-${firstId}`}
          tools={item.tools}
        />
      );
    }

    const { part, index } = item;
    const { type } = part;
    const key = `message-${message.id}-part-${index}`;

    if (type === "reasoning") {
      if (!mergedReasoning.rendered && mergedReasoning.text) {
        mergedReasoning.rendered = true;
        return (
          <MessageReasoning
            isLoading={isLoading || mergedReasoning.isStreaming}
            key={key}
            reasoning={mergedReasoning.text}
          />
        );
      }
      return null;
    }

    if (type === "text") {
      const rendered =
        message.role === "assistant"
          ? linkifyCitations(sanitizeText(part.text), citations)
          : sanitizeText(part.text);
      return (
        <MessageContent
          className={cn("text-[13px] leading-[1.65]", {
            "w-fit max-w-[min(80%,56ch)] overflow-hidden break-words rounded-2xl rounded-br-lg border border-border/30 bg-gradient-to-br from-secondary to-muted px-3.5 py-2 shadow-[var(--shadow-card)]":
              message.role === "user",
          })}
          data-testid="message-content"
          key={key}
        >
          <MessageResponse>{rendered}</MessageResponse>
        </MessageContent>
      );
    }

    if (type === "tool-createDocument") {
      const { toolCallId } = part;

      if (part.output && "error" in part.output) {
        return (
          <div
            className="rounded-lg border border-red-200 bg-red-50 p-4 text-red-500 dark:bg-red-950/50"
            key={toolCallId}
          >
            Error creating document: {String(part.output.error)}
          </div>
        );
      }

      return (
        <DocumentPreview
          isReadonly={isReadonly}
          key={toolCallId}
          result={part.output}
        />
      );
    }

    if (type === "tool-updateDocument") {
      const { toolCallId } = part;

      if (part.output && "error" in part.output) {
        return (
          <div
            className="rounded-lg border border-red-200 bg-red-50 p-4 text-red-500 dark:bg-red-950/50"
            key={toolCallId}
          >
            Error updating document: {String(part.output.error)}
          </div>
        );
      }

      return (
        <div className="relative" key={toolCallId}>
          <DocumentPreview
            args={{ ...part.output, isUpdate: true }}
            isReadonly={isReadonly}
            result={part.output}
          />
        </div>
      );
    }

    return null;
  });

  // Inline artifact cards rendered once at the end of the assistant
  // message — clicking opens the side panel, the only entry point after
  // a reload (data-stream-handler doesn't run on persisted parts).
  const artifactCards = artifactParts
    .map((part, idx) => {
      const dataPart = part as {
        data?: {
          id?: string;
          kind?: "text" | "code" | "sheet";
          title?: string;
          content?: string;
        };
      };
      const artifact = dataPart.data;
      if (
        !artifact ||
        typeof artifact.id !== "string" ||
        typeof artifact.title !== "string" ||
        typeof artifact.content !== "string" ||
        (artifact.kind !== "text" &&
          artifact.kind !== "code" &&
          artifact.kind !== "sheet")
      ) {
        return null;
      }
      return (
        <DeepSearchArtifactCard
          artifact={{
            id: artifact.id,
            kind: artifact.kind,
            title: artifact.title,
            content: artifact.content,
          }}
          key={`artifact-${artifact.id}-${idx}`}
        />
      );
    })
    .filter(Boolean);

  const actions = !isReadonly && (
    <MessageActions
      chatId={chatId}
      isLoading={isLoading}
      key={`action-${message.id}`}
      message={message}
      onEdit={onEdit ? () => onEdit(message) : undefined}
      vote={vote}
    />
  );

  const content = isThinking ? (
    <div className="flex h-[calc(13px*1.65)] items-center text-[13px] leading-[1.65]">
      <Shimmer className="font-medium" duration={1}>
        Thinking...
      </Shimmer>
    </div>
  ) : (
    <>
      {attachments}
      {parts}
      {artifactCards}
      {actions}
    </>
  );

  return (
    <div
      className={cn(
        "group/message w-full",
        !isAssistant && "animate-[fade-up_0.25s_cubic-bezier(0.22,1,0.36,1)]"
      )}
      data-role={message.role}
      data-testid={`message-${message.role}`}
    >
      <div
        className={cn(
          isUser ? "flex flex-col items-end gap-2" : "flex items-start gap-3"
        )}
      >
        {isAssistant && (
          <div className="flex h-[calc(13px*1.65)] shrink-0 items-center">
            <div className="flex size-7 items-center justify-center rounded-lg bg-muted/60 ring-1 ring-border/50">
              <DeepSearchMark size={16} />
            </div>
          </div>
        )}
        {isAssistant ? (
          <div className="flex min-w-0 flex-1 flex-col gap-2">{content}</div>
        ) : (
          content
        )}
      </div>
    </div>
  );
};

export const PreviewMessage = PurePreviewMessage;

export const ThinkingMessage = () => {
  return (
    <div
      className="group/message w-full"
      data-role="assistant"
      data-testid="message-assistant-loading"
    >
      <div className="flex items-start gap-3">
        <div className="flex h-[calc(13px*1.65)] shrink-0 items-center">
          <div className="flex size-7 items-center justify-center rounded-lg bg-muted/60 ring-1 ring-border/50">
            <DeepSearchMark size={16} />
          </div>
        </div>

        <div className="flex h-[calc(13px*1.65)] items-center text-[13px] leading-[1.65]">
          <Shimmer className="font-medium" duration={1}>
            Thinking...
          </Shimmer>
        </div>
      </div>
    </div>
  );
};
